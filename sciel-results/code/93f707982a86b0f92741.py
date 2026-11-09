import numpy as np
import time
import random

def construct():
    N = 51975
    # Find all divisors of N greater than 1
    divisors = []
    for i in range(3, N + 1, 2):
        if N % i == 0:
            divisors.append(i)
    
    # Sort divisors: Smaller moduli cover more residues (N/m is larger).
    # In a greedy covering problem, it's often beneficial to place the "big" sets first
    # to establish the backbone, then fill gaps with smaller sets.
    # Alternatively, larger moduli are more specific.
    # Let's stick to the current behavior or try sorting by size descending/ascending.
    # The previous program used unsorted (ascending from 3).
    # Let's try sorting by descending size to see if "specific" classes help?
    # Actually, for covering, large moduli (few residues) are less efficient per class.
    # Small moduli (many residues) are more efficient.
    # So ascending order (small m first) is generally good for greedy.
    
    # Greedy Initialization
    classes = []
    covered = np.zeros(N, dtype=np.int8)
    score = N
    used_moduli = set()
    available = set(divisors)
    
    # Precompute indices for each divisor to speed up slice operations?
    # Numpy slicing covered[r::m] is quite fast.
    
    while available:
        best_m = None
        best_r = None
        best_gain = -1
        
        # Optimization: Iterate over available moduli.
        # To speed up, we can prioritize moduli that have high potential.
        # But we must check all to find the true greedy best.
        
        for m in available:
            # Find r that maximizes gain
            # Gain = count of zeros in covered[r::m]
            
            # We can compute this for all r at once?
            # For a fixed m, we want to find r such that sum(covered[r::m] == 0) is max.
            
            # Let's loop r.
            min_nonzero = N
            best_r_for_m = 0
            
            # Optimization: 
            # If m is small, we have many r's but each slice is long.
            # If m is large, we have many r's but each slice is short.
            
            # We can break early if we find a perfect cover for this m (min_nonzero == 0).
            
            for r in range(m):
                nz = np.count_nonzero(covered[r::m])
                if nz < min_nonzero:
                    min_nonzero = nz
                    best_r_for_m = r
                    if min_nonzero == 0:
                        break
            
            total_len = (N - 1 - best_r_for_m) // m + 1
            gain = total_len - min_nonzero
            
            if gain > best_gain:
                best_gain = gain
                best_m = m
                best_r = best_r_for_m
        
        if best_gain <= 0:
            break
            
        classes.append([best_m, best_r])
        used_moduli.add(best_m)
        available.remove(best_m)
        
        covered[best_r::best_m] += 1
        score -= best_gain

    # Local Search
    start_time = time.time()
    deadline = start_time + 110 # Safety margin
    
    best_score = score
    best_classes = [c[:] for c in classes]
    
    curr_classes = [c[:] for c in classes]
    curr_covered = covered.copy()
    curr_score = score
    curr_used_moduli = set(used_moduli)
    
    # We will use a "First Improvement" strategy with random restarts of the search order
    # or random sampling of moves.
    # Given the structure, changing residues is the most likely to yield gains.
    
    # To improve on the current program, we will implement a more systematic local search:
    # 1. For each class, try all possible residues and pick the best one (Hill Climbing).
    # 2. Repeat until no improvement or time limit.
    # 3. Add randomization to escape local optima.
    
    # However, trying ALL residues for ALL classes every time is O(num_classes * sum(m_i) * N/m_i)
    # sum(m_i) is the number of variables. N/m_i is the cost of eval.
    # This might be slow if done repeatedly.
    
    # Let's use a hybrid:
    # - Randomly pick a class.
    # - Try all residues for that class (or a subset if m is huge, but m <= 51975).
    # - If improvement, accept.
    
    # Actually, for small m, trying all r is cheap (m is small, N/m is large but m is small loop).
    # For large m, m is large, N/m is small.
    # Cost per class evaluation: sum_{r in 0..m-1} O(N/m) = O(N).
    # So evaluating all residues for one class is O(N).
    # We have ~40 classes. So one full pass is O(40*N) ~ 2M operations.
    # This is very fast in NumPy.
    
    # Strategy:
    # 1. Full Pass: For each class i, find the best r. Update if better.
    # 2. Repeat until no improvement in a full pass.
    # 3. If stuck, try adding/removing classes or random restarts.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Shuffle class order to avoid bias?
        # indices = list(range(len(curr_classes)))
        # random.shuffle(indices)
        
        for i in range(len(curr_classes)):
            if time.time() > deadline:
                break
                
            m, r = curr_classes[i]
            
            # Find best r for this m
            best_r_for_m = r
            best_gain = -1 # We want to maximize gain (minimize lost)
            
            # Current score contribution of this class?
            # It's complex to isolate.
            # Instead, let's just calculate the delta for changing r to r_new.
            # Delta = Lost - Gained.
            # We want Delta < 0.
            
            # We can calculate the "quality" of a residue r for modulus m as:
            # Q(r) = - (Count of x in r::m where covered[x] == 1) + (Count of x in r::m where covered[x] == 0)
            # Wait, if we change r to r_new:
            # We lose coverage on old_r::m (where covered==1).
            # We gain coverage on new_r::m (where covered==0).
            # Note: covered includes the current class.
            # So for the current class, covered[x] >= 1 for x in old_r::m.
            # If we remove it, those x with covered[x]==1 become 0.
            # If we add new_r::m, those x with covered[x]==0 become 1.
            
            # Let's define a "score" for a candidate residue r_cand given the current state (with class i present):
            # We simulate removing class i and adding class i with r_cand.
            
            # Actually, simpler:
            # Let C be the covered array.
            # Let A_old = old_r::m.
            # Let A_new = new_r::m.
            # If A_old == A_new, delta = 0.
            # If disjoint (which they are if r != r_new):
            #   Lost = count(C[x] == 1 for x in A_old)
            #   Gained = count(C[x] == 0 for x in A_new)
            #   Delta = Lost - Gained
            
            # We want to minimize Delta.
            
            best_delta = 0 # Current delta is 0 (no change)
            best_new_r = r
            
            for r_new in range(m):
                if r_new == r:
                    continue
                
                old_slice = curr_covered[r::m]
                new_slice = curr_covered[r_new::m]
                
                # Optimization: We can precompute the counts for all r_new?
                # No, curr_covered changes.
                
                lost = np.count_nonzero(old_slice == 1)
                gained = np.count_nonzero(new_slice == 0)
                
                delta = lost - gained
                
                if delta < best_delta:
                    best_delta = delta
                    best_new_r = r_new
            
            if best_delta < 0:
                # Apply move
                curr_covered[r::m] -= 1
                curr_covered[best_new_r::m] += 1
                curr_classes[i][1] = best_new_r
                curr_score += best_delta
                improved = True
                
                if curr_score < best_score:
                    best_score = curr_score
                    best_classes = [c[:] for c in curr_classes]
                    
                # If we improved, we should probably restart the scan?
                # Or continue?
                # Continuing is faster, but might miss interactions.
                # Let's continue for now, but reset 'improved' logic?
                # No, if we improved, we keep going.
                
    # If we still have time, try adding/removing classes
    if time.time() < deadline - 5:
        # Try adding unused moduli
        for _ in range(10): # Limit attempts
            if time.time() > deadline:
                break
            unused = [m for m in divisors if m not in curr_used_moduli]
            if not unused:
                break
            
            best_m_new = None
            best_r_new = None
            best_gain = -1
            
            for m_new in unused:
                # Find best r for this new modulus
                min_nonzero = N
                best_r_for_m = 0
                for r_new in range(m_new):
                    nz = np.count_nonzero(curr_covered[r_new::m_new])
                    if nz < min_nonzero:
                        min_nonzero = nz
                        best_r_for_m = r_new
                        if min_nonzero == 0:
                            break
                
                total_len = (N - 1 - best_r_for_m) // m_new + 1
                gain = total_len - min_nonzero
                
                if gain > best_gain:
                    best_gain = gain
                    best_m_new = m_new
                    best_r_new = best_r_for_m
            
            if best_gain > 0:
                curr_covered[best_r_new::best_m_new] += 1
                curr_classes.append([best_m_new, best_r_new])
                curr_used_moduli.add(best_m_new)
                curr_score -= best_gain
                if curr_score < best_score:
                    best_score = curr_score
                    best_classes = [c[:] for c in curr_classes]
            else:
                break # No more beneficial additions

    return best_classes
