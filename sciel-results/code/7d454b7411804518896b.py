import time
import numpy as np
import random

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 115.0
    
    total = q ** n
    ball_size = 1 + n * (q - 1)  # 1 + 7*2 = 15 for q=3, n=7
    
    # Precompute place values
    base = np.zeros(n, dtype=np.int32)
    base[-1] = 1
    for i in range(n - 2, -1, -1):
        base[i] = base[i + 1] * q
    
    # Precompute all words
    words = np.zeros((total, n), dtype=np.int32)
    for i in range(total):
        temp = i
        for j in range(n - 1, -1, -1):
            words[i, j] = temp % q
            temp //= q
    
    # Precompute ball for each word: indices within Hamming distance 1
    # For radius 1, ball size is 1 + n*(q-1) = 15
    ball_indices = np.zeros((total, ball_size), dtype=np.int32)
    for i in range(total):
        w = words[i]
        ball_indices[i, 0] = i
        idx = 1
        for coord in range(n):
            orig = w[coord]
            for v in range(q):
                if v == orig:
                    continue
                new_idx = i + (v - orig) * base[coord]
                ball_indices[i, idx] = new_idx
                idx += 1
    
    # Greedy construction with randomization
    def greedy_build_random():
        covered = np.zeros(total, dtype=bool)
        code = []
        remaining = total
        
        while remaining > 0:
            # Find best codeword: one that covers most new points
            # Use random tie-breaking for diversity
            best_c = -1
            max_new = -1
            # Sample a subset for speed, or use full scan
            # For 2187 words, full scan is fine
            for c in range(total):
                ball = ball_indices[c]
                new_cov = np.sum(~covered[ball])
                if new_cov > max_new:
                    max_new = new_cov
                    best_c = c
                    if max_new == ball_size:
                        break
            
            code.append(best_c)
            covered[ball_indices[best_c]] = True
            remaining = np.sum(~covered)
        
        return code
    
    # Simulated annealing
    def simulated_annealing(initial_code):
        code_set = set(initial_code)
        cnt = np.zeros(total, dtype=np.int32)
        for c in code_set:
            cnt[ball_indices[c]] += 1
        
        best_code = list(code_set)
        best_size = len(best_code)
        current_size = len(best_code)
        
        # Temperature schedule
        T = 1.0
        T_min = 0.01
        cooling = 0.9995
        
        iter_count = 0
        max_iters = 50000
        
        while time.time() < deadline - 2.0 and T > T_min:
            iter_count += 1
            
            # Choose a move type
            move_type = random.random()
            
            if move_type < 0.3:
                # Removal move
                if len(code_set) <= 1:
                    T *= cooling
                    continue
                c_remove = random.choice(list(code_set))
                ball_rem = ball_indices[c_remove]
                
                # Check if removal is valid
                if np.min(cnt[ball_rem]) >= 2:
                    # Valid removal
                    code_set.remove(c_remove)
                    cnt[ball_rem] -= 1
                    current_size -= 1
                    
                    if current_size < best_size:
                        best_size = current_size
                        best_code = list(code_set)
                else:
                    # Invalid, accept with probability based on T
                    # Penalty = number of points that would become uncovered
                    uncovered = np.sum(cnt[ball_rem] == 1)
                    delta_E = uncovered  # cost increase
                    if random.random() < np.exp(-delta_E / T):
                        code_set.remove(c_remove)
                        cnt[ball_rem] -= 1
                        current_size -= 1
                        # Note: this makes solution invalid temporarily
                        # We'll try to fix it
            elif move_type < 0.7:
                # Swap move
                if len(code_set) == 0:
                    T *= cooling
                    continue
                c_remove = random.choice(list(code_set))
                c_add = random.randint(0, total - 1)
                
                if c_add in code_set or c_add == c_remove:
                    T *= cooling
                    continue
                
                ball_rem = ball_indices[c_remove]
                ball_add = ball_indices[c_add]
                
                # Check validity
                # Points in ball_rem but not in ball_add need cnt >= 2
                # Compute intersection
                in_both = np.intersect1d(ball_rem, ball_add)
                in_rem_only = np.setdiff1d(ball_rem, ball_add, assume_unique=True)
                
                valid = True
                if len(in_rem_only) > 0:
                    if np.min(cnt[in_rem_only]) < 2:
                        valid = False
                
                if valid:
                    code_set.remove(c_remove)
                    code_set.add(c_add)
                    cnt[ball_rem] -= 1
                    cnt[ball_add] += 1
                    # Size unchanged
                else:
                    # Compute penalty
                    if len(in_rem_only) > 0:
                        uncovered = np.sum(cnt[in_rem_only] == 1)
                    else:
                        uncovered = 0
                    delta_E = uncovered
                    if random.random() < np.exp(-delta_E / T):
                        code_set.remove(c_remove)
                        code_set.add(c_add)
                        cnt[ball_rem] -= 1
                        cnt[ball_add] += 1
            else:
                # Insertion move (to recover from invalid states)
                # Find an uncovered point and add a codeword covering it
                uncovered_mask = (cnt == 0)
                if np.any(uncovered_mask):
                    uncovered_points = np.where(uncovered_mask)[0]
                    target = random.choice(uncovered_points)
                    # Find codewords that cover this target
                    # A codeword c covers target if target is in ball_indices[c]
                    # Equivalent: c is in ball_indices[target]
                    candidates = ball_indices[target]
                    c_add = random.choice(candidates)
                    if c_add not in code_set:
                        code_set.add(c_add)
                        cnt[ball_indices[c_add]] += 1
                        current_size += 1
                        if current_size > best_size + 10:
                            # Too large, revert
                            code_set.remove(c_add)
                            cnt[ball_indices[c_add]] -= 1
                            current_size -= 1
                
                T *= cooling
                continue
            
            # Check if we improved
            if current_size < best_size:
                best_size = current_size
                best_code = list(code_set)
            
            T *= cooling
        
        return best_code
    
    # Multi-start approach
    best_code = None
    best_size = float('inf')
    
    # First greedy
    initial = greedy_build_random()
    best_code = simulated_annealing(initial)
    best_size = len(best_code)
    
    # Try more starts if time permits
    start_times = []
    for start_i in range(1, 10):
        if time.time() > deadline - 20.0:
            break
        initial = greedy_build_random()
        code = simulated_annealing(initial)
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
    
    # Final conversion
    result = []
    for idx in best_code:
        w = []
        temp = idx
        for i in range(n):
            w.append(temp % q)
            temp //= q
        w.reverse()
        result.append(w)
    
    return result
