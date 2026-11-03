import time
import random
import math
import heapq
import numpy as np

def construct(q, n, radius):
    deadline = time.monotonic() + 110.0
    total = q ** n
    rng = random.Random(42)

    # Precompute neighbors (ball of radius 1 for ternary, but generalize)
    # For radius=1, neighbors are the word itself + words differing in exactly one position
    # Precompute as list of lists for fast access
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

    # Build ball_matrix as uint8 for fast vectorized operations
    # ball_matrix[c, u] = 1 if u is in ball(c)
    ball_matrix = np.zeros((total, total), dtype=np.uint8)
    for c in range(total):
        for u in neigh[c]:
            ball_matrix[c, u] = 1

    def greedy_init(seed, time_limit):
        """Vectorized greedy set cover with random tie-breaking."""
        r = random.Random(seed)
        counts = np.zeros(total, dtype=np.int32)
        code = []
        
        while time.monotonic() < time_limit:
            remaining_mask = (counts == 0).astype(np.uint8)
            if remaining_mask.sum() == 0:
                break
            
            gains = ball_matrix @ remaining_mask
            best_gain = int(gains.max())
            if best_gain <= 0:
                break
            
            candidates = np.where(gains == best_gain)[0]
            if len(candidates) > 1:
                best_c = int(candidates[r.randint(0, len(candidates) - 1)])
            else:
                best_c = int(candidates[0])
            
            code.append(best_c)
            counts += ball_matrix[best_c]
        
        # Fallback: cover any remaining uncovered words
        remaining_mask = (counts == 0)
        if remaining_mask.any():
            code_set = set(code)
            while remaining_mask.any() and time.monotonic() < time_limit:
                uncovered = np.where(remaining_mask)[0]
                u = int(uncovered[0])
                covering = np.where(ball_matrix[:, u] == 1)[0]
                for nc in covering:
                    nc = int(nc)
                    if nc not in code_set:
                        code.append(nc)
                        code_set.add(nc)
                        counts += ball_matrix[nc]
                        remaining_mask = (counts == 0)
                        break
        
        return code, counts

    def remove_redundant(code, counts, time_limit):
        """Aggressively remove all redundant codewords using a priority-queue-like approach."""
        code_set = set(code)
        changed = True
        while changed and time.monotonic() < time_limit:
            changed = False
            # Iterate in reverse to avoid index issues when removing
            for i in range(len(code) - 1, -1, -1):
                c = code[i]
                # Check if any word in ball(c) has count == 1
                ball_words = neigh[c]
                if np.all(counts[ball_words] > 1):
                    code.pop(i)
                    code_set.discard(c)
                    counts[ball_matrix[c] == 1] -= 1
                    changed = True
        return code, counts, code_set

    def local_search_k_swap(code, counts, code_set, k, time_limit, rng):
        """
        Iterated Local Search with k-swap perturbations.
        Remove k random codewords, add k new ones (greedily or randomly),
        then do aggressive redundancy removal.
        """
        best_code = code[:]
        best_size = len(code)
        best_counts = counts.copy()
        
        while time.monotonic() < time_limit:
            # Perturbation: remove k random codewords
            if len(code) < k:
                break
            
            # Choose k random indices to remove
            remove_indices = rng.sample(range(len(code)), k)
            removed = []
            for idx in sorted(remove_indices, reverse=True):
                c = code[idx]
                code.pop(idx)
                code_set.discard(c)
                counts[ball_matrix[c] == 1] -= 1
                removed.append(c)
            
            # Add k new codewords: pick from neighbors of removed or random
            new_codewords = []
            for _ in range(k):
                if rng.random() < 0.6:
                    # Pick a random neighbor of a removed codeword
                    if removed:
                        base = removed[rng.randint(0, len(removed) - 1)]
                        candidates = [x for x in neigh[base] if x not in code_set]
                        if candidates:
                            new_c = candidates[rng.randint(0, len(candidates) - 1)]
                        else:
                            new_c = rng.randint(0, total - 1)
                    else:
                        new_c = rng.randint(0, total - 1)
                else:
                    new_c = rng.randint(0, total - 1)
                
                if new_c not in code_set:
                    code.append(new_c)
                    code_set.add(new_c)
                    counts[ball_matrix[new_c] == 1] += 1
                    new_codewords.append(new_c)
            
            # Aggressive redundancy removal
            code, counts, code_set = remove_redundant(code, counts, time_limit)
            
            # Check coverage: if any word is uncovered, repair
            zero_mask = (counts == 0)
            if zero_mask.any():
                # Repair: add codewords to cover uncovered words
                uncovered = np.where(zero_mask)[0]
                for u in uncovered:
                    u = int(u)
                    covering = np.where(ball_matrix[:, u] == 1)[0]
                    added = False
                    for nc in covering:
                        nc = int(nc)
                        if nc not in code_set:
                            code.append(nc)
                            code_set.add(nc)
                            counts[ball_matrix[nc] == 1] += 1
                            added = True
                            break
                    if not added:
                        # Should not happen, but just in case
                        break
                
                # After repair, do another redundancy removal pass
                code, counts, code_set = remove_redundant(code, counts, time_limit)
            
            # Update best
            if len(code) < best_size:
                best_size = len(code)
                best_code = code[:]
                best_counts = counts.copy()
        
        return best_code, best_counts, best_size

    def sa_polish(code, counts, time_limit):
        """Simulated annealing polish for fine-tuning."""
        code_set = set(code)
        best_code = code[:]
        best_size = len(code)
        best_counts = counts.copy()
        
        temp = 5.0
        cool_rate = 0.9995
        min_temp = 0.01
        
        while time.monotonic() < time_limit and temp > min_temp:
            move_type = rng.random()
            
            if move_type < 0.4:
                # Try remove
                if len(code) > 0:
                    idx = rng.randint(0, len(code) - 1)
                    c = code[idx]
                    ball_words = neigh[c]
                    if np.all(counts[ball_words] > 1):
                        code.pop(idx)
                        code_set.discard(c)
                        counts[ball_matrix[c] == 1] -= 1
                        if len(code) < best_size:
                            best_size = len(code)
                            best_code = code[:]
                            best_counts = counts.copy()
                        
            elif move_type < 0.9:
                # Try swap
                if len(code) > 0:
                    idx = rng.randint(0, len(code) - 1)
                    old_c = code[idx]
                    if rng.random() < 0.7:
                        new_c = neigh[old_c][rng.randint(1, len(neigh[old_c]) - 1)]
                    else:
                        new_c = rng.randint(0, total - 1)
                    
                    if new_c == old_c or new_c in code_set:
                        continue
                    
                    old_ball = ball_matrix[old_c]
                    new_ball = ball_matrix[new_c]
                    unique_old = (old_ball == 1) & (new_ball == 0) & (counts == 1)
                    uncovered = int(unique_old.sum())
                    
                    if uncovered == 0:
                        counts[old_ball == 1] -= 1
                        counts[new_ball == 1] += 1
                        code[idx] = new_c
                        code_set.discard(old_c)
                        code_set.add(new_c)
                        if len(code) < best_size:
                            best_size = len(code)
                            best_code = code[:]
                            best_counts = counts.copy()
                    else:
                        if uncovered <= 3 and rng.random() < math.exp(-uncovered / temp):
                            counts[old_ball == 1] -= 1
                            counts[new_ball == 1] += 1
                            code[idx] = new_c
                            code_set.discard(old_c)
                            code_set.add(new_c)
                            # Repair uncovered
                            zero_mask = (counts == 0)
                            if zero_mask.any():
                                zero_indices = np.where(zero_mask)[0]
                                for u in zero_indices:
                                    u = int(u)
                                    covering = np.where(ball_matrix[:, u] == 1)[0]
                                    for nc in covering:
                                        nc = int(nc)
                                        if nc not in code_set:
                                            code.append(nc)
                                            code_set.add(nc)
                                            counts[ball_matrix[nc] == 1] += 1
                                            break
            else:
                # Try add (perturbation)
                new_c = rng.randint(0, total - 1)
                if new_c not in code_set:
                    if rng.random() < math.exp(-1.5 / temp):
                        code.append(new_c)
                        code_set.add(new_c)
                        counts[ball_matrix[new_c] == 1] += 1
            
            temp *= cool_rate
        
        # Final redundancy removal
        code, counts, code_set = remove_redundant(code, counts, time_limit)
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]
            best_counts = counts.copy()
        
        return best_code, best_counts, best_size

    # Multi-restart strategy
    overall_best_code = None
    overall_best_size = total + 1
    
    # Phase 1: Multiple greedy inits with different seeds
    num_greedy = 8
    time_per_greedy = 2.0
    
    for i in range(num_greedy):
        if time.monotonic() > deadline - 20:
            break
        code, counts = greedy_init(42 + i * 13, time.monotonic() + time_per_greedy)
        if not code:
            continue
        
        # Remove redundancy
        code, counts, code_set = remove_redundant(code, counts, time.monotonic() + 0.5)
        
        if len(code) < overall_best_size:
            overall_best_size = len(code)
            overall_best_code = code[:]
    
    # Phase 2: ILS with k-swaps on the best solution found
    if overall_best_code is not None and time.monotonic() < deadline - 10:
        code = overall_best_code[:]
        code_set = set(code)
        counts = np.zeros(total, dtype=np.int32)
        for c in code:
            counts[ball_matrix[c] == 1] += 1
        
        # Run ILS with k=1 and k=2
        for k in [1, 2]:
            if time.monotonic() > deadline - 5:
                break
            time_budget = min(15.0, deadline - time.monotonic() - 5)
            code, counts, size = local_search_k_swap(code, counts, code_set, k, time.monotonic() + time_budget, rng)
            if size < overall_best_size:
                overall_best_size = size
                overall_best_code = code[:]
        
        # Phase 3: SA polish on the best
        if time.monotonic() < deadline - 5:
            code = overall_best_code[:]
            code_set = set(code)
            counts = np.zeros(total, dtype=np.int32)
            for c in code:
                counts[ball_matrix[c] == 1] += 1
            
            time_budget = min(20.0, deadline - time.monotonic() - 3)
            code, counts, size = sa_polish(code, counts, time.monotonic() + time_budget)
            if size < overall_best_size:
                overall_best_size = size
                overall_best_code = code[:]
    
    # If we haven't found anything, do one more greedy
    if overall_best_code is None:
        code, counts = greedy_init(42, time.monotonic() + 10)
        overall_best_code = code
        overall_best_size = len(code)
    
    # Convert to words
    result = []
    for idx in overall_best_code:
        word = [0] * n
        temp_idx = idx
        for i in range(n):
            word[i] = temp_idx % q
            temp_idx //= q
        result.append(word)
    
    return result
