import time
import numpy as np

def construct(q, n, radius):
    t_start = time.time()
    deadline = 148.0
    space_size = q ** n
    nb_count = 1 + n * (q - 1)

    # Precompute neighbors
    q_powers = [1] * (n + 1)
    for i in range(1, n + 1):
        q_powers[i] = q_powers[i - 1] * q

    def idx_to_word(idx):
        w = [0] * n
        for pos in range(n):
            w[pos] = (idx // q_powers[n - 1 - pos]) % q
        return w

    neighbors = np.empty((space_size, nb_count), dtype=np.int32)
    for idx in range(space_size):
        w = idx_to_word(idx)
        nb_list = [idx]
        for pos in range(n):
            for new_d in range(q):
                if new_d == w[pos]:
                    continue
                delta = (new_d - w[pos]) * q_powers[n - 1 - pos]
                nb_list.append(idx + delta)
        neighbors[idx] = nb_list

    def build_cov(code_indices):
        cov = np.zeros(space_size, dtype=np.int32)
        for c in code_indices:
            cov[neighbors[c]] += 1
        return cov

    def greedy_cover(seed=0):
        rng = np.random.default_rng(seed)
        uncovered = np.ones(space_size, dtype=bool)
        code = []
        while np.any(uncovered) and (time.time() - t_start) < deadline - 15.0:
            gains = np.sum(uncovered[neighbors], axis=1)
            perturb = rng.random(space_size) * 0.001
            gains_pert = gains + perturb
            best = np.argmax(gains_pert)
            if gains[best] == 0:
                break
            code.append(int(best))
            uncovered[neighbors[best]] = False
        return code

    def greedy_repair(code_list, cov, deadline_time):
        uncovered_mask = cov == 0
        while np.any(uncovered_mask) and (time.time() - t_start) < deadline_time:
            gains = np.sum(uncovered_mask[neighbors], axis=1)
            best = np.argmax(gains)
            if gains[best] == 0:
                break
            code_list.append(int(best))
            cov[neighbors[best]] += 1
            uncovered_mask = cov == 0
        return code_list, cov

    def polish(code_list, cov, deadline_time):
        changed = True
        while changed and (time.time() - t_start) < deadline_time:
            changed = False
            for i in range(len(code_list)):
                c = code_list[i]
                if np.min(cov[neighbors[c]]) >= 2:
                    code_list.pop(i)
                    cov[neighbors[c]] -= 1
                    changed = True
                    break
        return code_list, cov

    # Phase 1: Multiple greedy seeds to find a good initial solution
    best_code = None
    best_size = space_size + 1

    for seed in range(15):
        if (time.time() - t_start) > deadline - 60.0:
            break
        code = greedy_cover(seed)
        cov = build_cov(code)
        if not np.all(cov > 0):
            code, cov = greedy_repair(code, cov, deadline - 40.0)
        code, cov = polish(code, cov, deadline - 30.0)
        if len(code) < best_size:
            best_code = code[:]
            best_size = len(code)

    if best_code is None:
        best_code = greedy_cover(0)
        best_size = len(best_code)

    # Phase 2: Fixed-size ILS with permutation representation
    # The permutation is a random ordering of all words.
    # Code = first k elements of the permutation.
    # Moves: swap two positions in the permutation.
    # If k is fixed, swapping within the first k doesn't change the code.
    # Swapping between first k and rest changes the code.
    # Actually, for fixed size k, we want to swap a word IN the code with one OUT.
    
    # Better: represent code as a set, and use 1-swap (remove one, add one) and 2-swap (remove two, add two)
    # with error repair if needed.
    
    rng = np.random.default_rng(42)
    all_words = np.arange(space_size)
    
    # Start from best_code found in Phase 1
    cur_code = best_code[:]
    cur_cov = build_cov(cur_code)
    cur_size = len(cur_code)
    
    # Adaptive target: start from cur_size, try to reduce
    # Time-slice: use remaining time for ILS
    il_start = time.time()
    il_deadline = deadline - 5.0
    
    # Target sizes to try, from current best down
    target_sizes = list(range(cur_size, max(150, cur_size - 50), -1))
    # Add some targets below
    for t in range(max(150, cur_size - 50), 150, -1):
        target_sizes.append(t)
    
    # ILS loop
    for restart in range(200):
        if (time.time() - t_start) > il_deadline:
            break
        
        # Choose target size based on progress
        progress = (time.time() - il_start) / max(1e-9, il_deadline - il_start)
        # Start with larger targets, move to smaller
        idx = min(len(target_sizes) - 1, int(progress * len(target_sizes) * 1.5))
        target_k = target_sizes[idx]
        
        # If target_k >= cur_size, skip (we want to reduce)
        if target_k >= cur_size:
            target_k = cur_size - 1
        if target_k < 150:
            target_k = 150
        
        # Initialize: take first target_k from a random permutation
        perm = rng.permutation(space_size)
        code = list(perm[:target_k])
        cov = build_cov(code)
        
        # Repair if not covering
        if not np.all(cov > 0):
            code, cov = greedy_repair(code, cov, il_deadline - 5.0)
            if len(code) > target_k + 20:  # Too many repairs, skip
                continue
        
        # Local search: 1-swap and 2-swap
        improved = True
        max_iters = 200
        for it in range(max_iters):
            if (time.time() - t_start) > il_deadline:
                break
            
            found_swap = False
            
            # Try 1-swap: remove one from code, add one not in code
            if len(code) > target_k:
                code_set = set(code)
                # Find words not in code
                not_in_code = np.array([w for w in all_words if w not in code_set])
                if len(not_in_code) > 0:
                    # For each word in code, check if removing it and adding a random word from not_in_code helps
                    # This is expensive; do a few random trials
                    for trial in range(10):
                        if (time.time() - t_start) > il_deadline:
                            break
                        rem_idx = rng.integers(len(code))
                        rem_word = code[rem_idx]
                        add_word = int(not_in_code[rng.integers(len(not_in_code))])
                        
                        new_code = code[:rem_idx] + code[rem_idx+1:] + [add_word]
                        new_cov = build_cov(new_code)
                        
                        if np.all(new_cov > 0) and len(new_code) < len(code):
                            code = new_code
                            cov = new_cov
                            found_swap = True
                            break
                        elif np.all(new_cov > 0) and len(new_code) == len(code) and len(code) > target_k:
                            # Same size but we want to reduce; accept if it enables further reduction
                            # Actually, for fixed target, accept same-size swaps that improve "quality"
                            # Use a simple heuristic: prefer swaps that increase minimum coverage
                            old_min_cov = np.min(cov)
                            new_min_cov = np.min(new_cov)
                            if new_min_cov >= old_min_cov:
                                code = new_code
                                cov = new_cov
                                found_swap = True
                                break
            
            # Try to reduce size if above target
            if len(code) > target_k:
                # Polish: remove redundant words
                code, cov = polish(code, cov, il_deadline - 5.0)
                if len(code) < len(best_code) if best_code is not None else False:
                    pass
            
            if not found_swap:
                # Kick: random perturbation
                if len(code) > 0:
                    kick_size = min(3, len(code))
                    kick_indices = rng.choice(len(code), size=kick_size, replace=False)
                    code_set = set(code)
                    not_in_code = np.array([w for w in all_words if w not in code_set])
                    for ki in kick_indices:
                        if len(not_in_code) > 0:
                            new_word = int(not_in_code[rng.integers(len(not_in_code))])
                            code[ki] = new_word
                    
                    cov = build_cov(code)
                    if not np.all(cov > 0):
                        code, cov = greedy_repair(code, cov, il_deadline - 5.0)
            
            # Check if we improved
            if len(code) < cur_size:
                cur_size = len(code)
                cur_code = code[:]
                cur_cov = cov
                if best_code is None or cur_size < len(best_code):
                    best_code = cur_code[:]
                    best_size = cur_size
                improved = True
        
        # If we've been at the same size for a while, break
        if (time.time() - t_start) > il_deadline - 2.0:
            break
    
    # Final polish
    if (time.time() - t_start) < deadline - 3.0:
        final_code = best_code[:] if best_code is not None else cur_code[:]
        final_cov = build_cov(final_code)
        final_code, final_cov = polish(final_code, final_cov, deadline - 2.0)
        if len(final_code) < best_size:
            best_code = final_code
            best_size = len(final_code)
    
    # Final verification
    if best_code is None or len(best_code) == 0:
        best_code = greedy_cover(0)
    
    final_cov = build_cov(best_code)
    if not np.all(final_cov > 0):
        fc = greedy_cover(0)
        fcov = build_cov(fc)
        if np.all(fcov > 0):
            best_code = fc
        else:
            best_code = list(range(space_size))
    
    return [idx_to_word(c) for c in best_code]
