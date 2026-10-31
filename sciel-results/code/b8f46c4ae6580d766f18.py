import numpy as np
import time
import random

N = 50

def compute_ratio(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 < 1e-300:
        return 0.0
    l2_sq = np.sum(h * h)
    hh = np.convolve(h, h, mode='full')
    hh_max = np.max(hh)
    if hh_max < 1e-300:
        return 0.0
    return l2_sq / (l1 * hh_max)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    best_h = np.zeros(N, dtype=np.float64)
    best_score = 0.0

    def update_best(h, s):
        nonlocal best_h, best_score
        if s > best_score:
            best_score = s
            best_h = h.copy()

    # 1. Initial Candidates
    # Single spike
    h = np.zeros(N)
    h[0] = 1.0
    update_best(h, compute_ratio(h))

    # Uniform
    h = np.ones(N)
    update_best(h, compute_ratio(h))

    # 2. Greedy Sidon Construction
    # We want to maximize the number of distinct sums i+j.
    # For a set S of size K, if all K^2 sums are distinct, max conv is 1/K^2 (normalized).
    # Ratio becomes (1/K) / (1/K^2) = K.
    # Max K for N=50 is around 7.
    
    def greedy_sidon(max_elements, max_attempts=100):
        best_set = []
        for _ in range(max_attempts):
            S = []
            # Random start
            start = random.randint(0, N-1)
            S.append(start)
            
            for _ in range(max_elements):
                if time.time() > deadline - 10:
                    return S
                # Find candidate that minimizes collisions
                best_candidate = None
                best_collisions = float('inf')
                
                # To speed up, we can sample candidates or check all
                # N=50 is small, checking all is fine.
                for c in range(N):
                    if c in S:
                        continue
                    # Count collisions
                    collisions = 0
                    new_S = S + [c]
                    # Check sums
                    # Efficient way: maintain set of sums
                    # For simplicity in this inner loop, just compute sums
                    sums = set()
                    valid = True
                    for i in new_S:
                        for j in new_S:
                            s_val = i + j
                            if s_val in sums:
                                valid = False
                                break
                            sums.add(s_val)
                        if not valid:
                            break
                    
                    # If valid, it's a Sidon set extension.
                    # But we might want to allow some collisions if we can't find a perfect one?
                    # The prompt suggests "minimum collision".
                    # Let's count collisions properly.
                    collisions = 0
                    sums_dict = {}
                    for i in new_S:
                        for j in new_S:
                            s_val = i + j
                            if s_val in sums_dict:
                                collisions += 1
                            else:
                                sums_dict[s_val] = 1
                    
                    if collisions < best_collisions:
                        best_collisions = collisions
                        best_candidate = c
                        if best_collisions == 0:
                            break
                
                if best_candidate is None:
                    break
                S.append(best_candidate)
                if best_collisions == 0:
                    # If we can keep it perfect, great.
                    # But maybe we should break if we hit max_elements?
                    pass
            if len(S) > len(best_set):
                best_set = S
            # If we found a perfect Sidon set of size > 5, it's likely optimal for this structure
            if len(best_set) >= 7:
                break
        return best_set

    # Try to find a good Sidon set
    sidon_set = greedy_sidon(10, 200)
    
    if len(sidon_set) > 1:
        h = np.zeros(N)
        for idx in sidon_set:
            h[idx] = 1.0
        s = compute_ratio(h)
        update_best(h, s)

    # 3. Local Search / Refinement
    # We will perform local search on the best candidate found so far.
    # The search space is continuous (heights) and discrete (support).
    
    current_h = best_h.copy()
    current_score = best_score
    no_improve_count = 0
    
    while time.time() < deadline - 5.0:
        improved = False
        
        # Discrete moves: Add, Remove, Move
        # 1. Try adding a mass to a random empty index
        empty_indices = [i for i in range(N) if current_h[i] < 1e-9]
        if empty_indices:
            idx = random.choice(empty_indices)
            old_val = current_h[idx]
            current_h[idx] = 1.0 # Arbitrary scale, ratio is scale invariant
            new_score = compute_ratio(current_h)
            if new_score > current_score + 1e-12:
                current_score = new_score
                improved = True
            else:
                current_h[idx] = old_val
                no_improve_count += 1
        
        # 2. Try removing mass from a random non-empty index
        non_empty_indices = [i for i in range(N) if current_h[i] > 1e-9]
        if len(non_empty_indices) > 1:
            idx = random.choice(non_empty_indices)
            old_val = current_h[idx]
            current_h[idx] = 0.0
            new_score = compute_ratio(current_h)
            if new_score > current_score + 1e-12:
                current_score = new_score
                improved = True
            else:
                current_h[idx] = old_val
                no_improve_count += 1
        
        # 3. Try moving mass from i to j
        if len(non_empty_indices) >= 2:
            i, j = random.sample(non_empty_indices, 2)
            old_i, old_j = current_h[i], current_h[j]
            current_h[i] = 0.0
            current_h[j] = old_i
            new_score = compute_ratio(current_h)
            if new_score > current_score + 1e-12:
                current_score = new_score
                improved = True
            else:
                current_h[i] = old_i
                current_h[j] = old_j
                no_improve_count += 1
        
        # Continuous moves: Perturb weights
        # Pick a non-empty index and scale it by a factor
        if non_empty_indices:
            idx = random.choice(non_empty_indices)
            factor = random.uniform(0.5, 1.5)
            old_val = current_h[idx]
            current_h[idx] *= factor
            new_score = compute_ratio(current_h)
            if new_score > current_score + 1e-12:
                current_score = new_score
                improved = True
            else:
                current_h[idx] = old_val
                no_improve_count += 1

        if improved:
            no_improve_count = 0
            update_best(current_h, current_score)
        else:
            if no_improve_count > 2000:
                # Restart from a slightly perturbed best or a new random candidate
                # Let's try a new random sparse candidate
                k = random.randint(2, 8)
                indices = random.sample(range(N), k)
                h = np.zeros(N)
                for idx in indices:
                    h[idx] = 1.0
                s = compute_ratio(h)
                if s > current_score:
                    current_h = h.copy()
                    current_score = s
                    update_best(current_h, current_score)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    if no_improve_count > 5000:
                        break

    return list(best_h)
