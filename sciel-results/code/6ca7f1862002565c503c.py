import itertools
import time
import numpy as np

def construct(q, n, radius):
    t_start = time.time()
    deadline = 115.0  # safety margin before 120s limit
    
    # Build initial direct-sum code: tetracode x all tails
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n-4)]
    m0 = len(initial)
    
    # Precompute powers of q
    q_powers = [1] * (n + 1)
    for i in range(1, n + 1):
        q_powers[i] = q_powers[i-1] * q
    
    space_size = q ** n
    
    # Map word to index
    def word_to_idx(w):
        idx = 0
        for d in w:
            idx = idx * q + d
        return idx
    
    # Precompute neighbors for radius=1: each word and all single-digit changes
    all_neighbors = {}
    for idx in range(space_size):
        neighbors = [idx]
        for pos in range(n):
            digit = (idx // q_powers[n-1-pos]) % q
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                delta = (new_digit - digit) * q_powers[n-1-pos]
                neighbors.append(idx + delta)
        all_neighbors[idx] = np.array(neighbors, dtype=np.int32)
    
    # Initial coverage
    cov = np.zeros(space_size, dtype=np.int32)
    for c in initial:
        c_idx = word_to_idx(c)
        cov[all_neighbors[c_idx]] += 1
    
    # Greedy removal pass
    code = initial[:]
    removed = [False] * m0
    changed = True
    while changed and (time.time() - t_start) < deadline:
        changed = False
        for i in range(m0):
            if removed[i]:
                continue
            c_idx = word_to_idx(code[i])
            nbs = all_neighbors[c_idx]
            if np.min(cov[nbs]) >= 2:
                removed[i] = True
                cov[nbs] -= 1
                changed = True
    
    best_code = [code[i] for i in range(m0) if not removed[i]]
    best_size = len(best_code)
    
    # Verify
    cov_check = np.zeros(space_size, dtype=np.int32)
    for c in best_code:
        c_idx = word_to_idx(c)
        cov_check[all_neighbors[c_idx]] += 1
    if np.any(cov_check == 0):
        return initial
    
    # Local search: try replacing codewords to improve coverage efficiency
    # We'll do random restarts with different initial perturbations
    np.random.seed(42)
    
    for restart in range(20):
        if time.time() - t_start > deadline - 5:
            break
        
        # Start from current best
        cur_code = best_code[:]
        cur_size = len(cur_code)
        
        # Rebuild coverage
        cur_cov = np.zeros(space_size, dtype=np.int32)
        for c in cur_code:
            c_idx = word_to_idx(c)
            cur_cov[all_neighbors[c_idx]] += 1
        
        active = list(range(cur_size))
        last_improve = time.time()
        
        # Iterative removal with random ordering
        for iteration in range(100):
            if time.time() - t_start > deadline - 3:
                break
            if not active:
                break
            
            # Shuffle active indices
            np.random.shuffle(active)
            
            removed_this_pass = False
            for idx in active:
                if time.time() - t_start > deadline - 3:
                    break
                
                c = cur_code[idx]
                c_idx = word_to_idx(c)
                nbs = all_neighbors[c_idx]
                
                if np.min(cur_cov[nbs]) >= 2:
                    cur_cov[nbs] -= 1
                    active.remove(idx)
                    removed_this_pass = True
                    last_improve = time.time()
            
            if not removed_this_pass:
                # Try random swap: remove one codeword, add another
                # Find a word that's only covered by one codeword (if any)
                # Actually, let's try: for each codeword, see if removing it and adding
                # a random uncovered word improves things
                
                # Try to find a better replacement
                found_swap = False
                for idx in active[:min(5, len(active))]:
                    if time.time() - t_start > deadline - 3:
                        break
                    
                    c = cur_code[idx]
                    c_idx = word_to_idx(c)
                    nbs = all_neighbors[c_idx]
                    
                    # What words would become uncovered if we remove this?
                    would_uncover = []
                    for nb in nbs:
                        if cur_cov[nb] == 1:
                            would_uncover.append(nb)
                    
                    if not would_uncover:
                        continue
                    
                    # Try to find a word that covers all would_uncover words
                    # This is a set cover subproblem, so let's just try random candidates
                    for trial in range(10):
                        new_word_idx = np.random.randint(0, space_size)
                        new_word_nbs = all_neighbors[new_word_idx]
                        
                        # Check if new word covers all would_uncover
                        all_covered = True
                        for uw in would_uncover:
                            if uw not in new_word_nbs:
                                all_covered = False
                                break
                        
                        if all_covered:
                            # Do the swap
                            cur_cov[nbs] -= 1
                            cur_cov[new_word_nbs] += 1
                            # Replace the codeword
                            new_word = [(new_word_idx // q_powers[n-1-pos]) % q for pos in range(n)]
                            cur_code[idx] = new_word
                            found_swap = True
                            last_improve = time.time()
                            break
                    
                    if found_swap:
                        break
            
            if time.time() - last_improve > 2.0:
                break
        
        if len(active) < best_size:
            best_code = [cur_code[idx] for idx in active]
            best_size = len(active)
    
    # Final verification
    cov_final = np.zeros(space_size, dtype=np.int32)
    for c in best_code:
        c_idx = word_to_idx(c)
        cov_final[all_neighbors[c_idx]] += 1
    
    if np.any(cov_final == 0):
        return initial
    
    return best_code
