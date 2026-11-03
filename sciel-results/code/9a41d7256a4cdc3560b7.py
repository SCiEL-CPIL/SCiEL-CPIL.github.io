import time
import random
import math
import numpy as np

def construct(q, n, radius):
    deadline = time.monotonic() + 115.0
    total = q ** n
    rng = random.Random(42)

    # 1. Precompute neighbors and ball matrix
    # neigh[i] is a list of indices covered by codeword i
    # ball_matrix[i, j] is 1 if j is in neigh[i]
    neigh = []
    for idx in range(total):
        neighbors = [idx]
        for i in range(n):
            pow_q_i = q ** i
            digit = (idx // pow_q_i) % q
            for new_digit in range(q):
                if new_digit != digit:
                    new_idx = idx + (new_digit - digit) * pow_q_i
                    neighbors.append(new_idx)
        neigh.append(neighbors)
    
    # Build boolean matrix for fast vector operations
    ball_matrix = np.zeros((total, total), dtype=np.uint8)
    for c in range(total):
        for u in neigh[c]:
            ball_matrix[c, u] = 1

    # Helper: Convert index to word
    def idx_to_word(idx):
        word = [0] * n
        for i in range(n):
            word[i] = idx % q
            idx //= q
        return word

    # Helper: Check if a code covers all words
    # Returns counts array
    def compute_counts(code):
        counts = np.zeros(total, dtype=np.int32)
        for c in code:
            counts += ball_matrix[c]
        return counts

    # Phase 1: Greedy Initialization
    def greedy_init(seed):
        r = random.Random(seed)
        counts = np.zeros(total, dtype=np.int32)
        code = []
        
        # We want to cover all 'total' words
        # remaining_mask: 1 if uncovered
        remaining_mask = np.ones(total, dtype=np.uint8)
        
        while remaining_mask.sum() > 0:
            # Calculate gain for each candidate
            # gain[c] = number of currently uncovered words covered by c
            gains = ball_matrix @ remaining_mask
            
            # Pick candidate with max gain
            best_gain = int(gains.max())
            if best_gain <= 0:
                break
            
            candidates = np.where(gains == best_gain)[0]
            # Random tie-breaking
            best_c = int(candidates[r.randint(0, len(candidates) - 1)])
            
            code.append(best_c)
            counts += ball_matrix[best_c]
            remaining_mask = (counts == 0).astype(np.uint8)
            
        return code, counts

    # Phase 2: Local Search with ILS
    def local_search(code, counts, time_limit):
        code_set = set(code)
        best_code = code[:]
        best_size = len(code)
        best_counts = counts.copy()
        
        # SA parameters
        temp = 10.0
        cool_rate = 0.999
        min_temp = 0.01
        
        # Helper: Is codeword c removable?
        def is_removable(c):
            # All neighbors of c must be covered by at least 2 codewords
            # i.e., counts[u] > 1 for all u in neigh[c]
            return np.all(counts[neigh[c]] > 1)
        
        # Helper: Is swap c -> d valid? (Maintains cover)
        def is_valid_swap(c, d):
            # Words covered by c but not by d must be covered by others
            # unique_old = neigh[c] \ neigh[d]
            # We need counts[u] > 1 for all u in unique_old
            # Note: counts includes c. So if counts[u] == 1 and u in unique_old, removing c breaks cover.
            # So we need counts[u] >= 2 for all u in (neigh[c] \ neigh[d])
            # This is equivalent to: min(counts[neigh[c] & ~neigh[d]]) >= 2
            # Using numpy:
            mask_c = ball_matrix[c]
            mask_d = ball_matrix[d]
            unique_mask = mask_c & (~mask_d)
            if unique_mask.any():
                return np.all(counts[unique_mask] >= 2)
            return True
            
        # Helper: Perform swap
        def do_swap(c, d):
            code[code.index(c)] = d
            code_set.discard(c)
            code_set.add(d)
            counts -= ball_matrix[c]
            counts += ball_matrix[d]

        # Helper: Perform delete
        def do_delete(c):
            code.remove(c)
            code_set.discard(c)
            counts -= ball_matrix[c]

        # Helper: Perform add
        def do_add(d):
            code.append(d)
            code_set.add(d)
            counts += ball_matrix[d]

        # Helper: Repair cover (Greedy)
        def repair():
            nonlocal counts, code, code_set
            remaining_mask = (counts == 0).astype(np.uint8)
            while remaining_mask.sum() > 0:
                gains = ball_matrix @ remaining_mask
                best_gain = int(gains.max())
                if best_gain <= 0:
                    break
                candidates = np.where(gains == best_gain)[0]
                best_c = int(candidates[rng.randint(0, len(candidates) - 1)])
                if best_c not in code_set:
                    do_add(best_c)
                remaining_mask = (counts == 0).astype(np.uint8)

        start_time = time.monotonic()
        no_improve_count = 0
        
        while time.monotonic() < time_limit and temp > min_temp:
            move_type = rng.random()
            
            if move_type < 0.5:
                # Try to delete a redundant word
                if code:
                    idx = rng.randint(0, len(code) - 1)
                    c = code[idx]
                    if is_removable(c):
                        do_delete(c)
                        if len(code) < best_size:
                            best_size = len(code)
                            best_code = code[:]
                            best_counts = counts.copy()
                            no_improve_count = 0
                        else:
                            no_improve_count += 1
            elif move_type < 0.9:
                # Try to swap a word
                if code:
                    idx = rng.randint(0, len(code) - 1)
                    c = code[idx]
                    d = rng.randint(0, total - 1)
                    if d != c and d not in code_set:
                        if is_valid_swap(c, d):
                            do_swap(c, d)
                            # Size doesn't change, but might enable deletion
                            # Check for immediate deletion opportunities
                            for i in range(len(code) - 1, -1, -1):
                                if is_removable(code[i]):
                                    do_delete(code[i])
                                    if len(code) < best_size:
                                        best_size = len(code)
                                        best_code = code[:]
                                        best_counts = counts.copy()
                                        no_improve_count = 0
                                    break
                            else:
                                no_improve_count += 1
            else:
                # Perturbation: Add a random word (increases size, allows escape)
                d = rng.randint(0, total - 1)
                if d not in code_set:
                    # Accept with probability based on temperature
                    # Delta E = +1 (size increase)
                    if rng.random() < math.exp(-1.0 / temp):
                        do_add(d)
                        no_improve_count += 1

            temp *= cool_rate
            
            # If stuck, do a stronger perturbation: Random Restart of Local Search
            if no_improve_count > 200:
                # Remove k random words and repair
                k = rng.randint(2, 5)
                if len(code) > k:
                    remove_indices = rng.sample(range(len(code)), k)
                    for idx in sorted(remove_indices, reverse=True):
                        c = code[idx]
                        do_delete(c)
                    repair()
                    no_improve_count = 0
                    # Reset temperature slightly to allow more exploration
                    temp = max(temp, 1.0)

        # Final cleanup: Remove all redundant words
        changed = True
        while changed:
            changed = False
            for i in range(len(code) - 1, -1, -1):
                if is_removable(code[i]):
                    do_delete(code[i])
                    changed = True
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = code[:]
                        best_counts = counts.copy()

        return best_code, best_counts, best_size

    # Main Logic
    overall_best_code = None
    overall_best_size = total + 1
    
    # Multiple restarts
    num_restarts = 0
    while time.monotonic() < deadline - 10:
        seed = 42 + num_restarts
        code, counts = greedy_init(seed)
        
        if not code:
            continue
            
        # Quick redundancy removal
        code_set = set(code)
        counts = compute_counts(code)
        changed = True
        while changed:
            changed = False
            for i in range(len(code) - 1, -1, -1):
                c = code[i]
                if np.all(counts[neigh[c]] > 1):
                    code.pop(i)
                    code_set.discard(c)
                    counts -= ball_matrix[c]
                    changed = True
        
        if len(code) < overall_best_size:
            overall_best_size = len(code)
            overall_best_code = code[:]
            
        # Allocate time for local search
        time_remaining = deadline - time.monotonic() - 5
        if time_remaining < 5:
            break
            
        # Run local search
        time_budget = min(20.0, time_remaining)
        code_ls, counts_ls, size_ls = local_search(code, counts, time.monotonic() + time_budget)
        
        if size_ls < overall_best_size:
            overall_best_size = size_ls
            overall_best_code = code_ls[:]
            
        num_restarts += 1
        # If we found 73, we can stop early or try to find 72/71
        if overall_best_size <= 73:
             # Continue searching for a few more seconds to see if we can break 73
             if overall_best_size < 73:
                 break
             elif num_restarts > 5:
                 break

    if overall_best_code is None:
        # Fallback
        code, counts = greedy_init(42)
        overall_best_code = code
        overall_best_size = len(code)

    # Convert to words
    result = []
    for idx in overall_best_code:
        result.append(idx_to_word(idx))
    
    return result
