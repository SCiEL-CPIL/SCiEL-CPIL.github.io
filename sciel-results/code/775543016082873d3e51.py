import random
import numpy as np

def construct():
    # Parameters
    N_range = 129 # Sums from 128 to 256 inclusive
    S_min = 128
    S_max = 256
    U = 128 # Universe [0, 128]
    
    # Target size K
    # We want to balance K and MaxMult.
    # Smaller K -> fewer pairs -> lower max mult, but harder to cover all sums.
    # Larger K -> more pairs -> higher max mult, easier to cover.
    # Let's try a range of K.
    
    best_max_mult = float('inf')
    best_A = None
    
    # Helper to calculate multiplicities
    def get_multiplicities(A_set):
        # A_set is a set of ints
        # Returns dict S -> count
        counts = {s: 0 for s in range(S_min, S_max + 1)}
        A_list = list(A_set)
        n = len(A_list)
        for i in range(n):
            a = A_list[i]
            for j in range(i, n):
                b = A_list[j]
                s = a + b
                if S_min <= s <= S_max:
                    counts[s] += 1
        return counts

    # Helper to check coverage
    def is_covered(counts):
        return all(c > 0 for c in counts.values())

    # New Approach: Score-based Heuristic mimicking ADMM/BQP relaxation
    def score_based_search(K, iterations=200):
        # Initialize scores randomly or uniformly
        scores = np.ones(U + 1)
        
        # Initial A: top K scores (random tie-break)
        # Add small noise to break ties
        scores += np.random.uniform(0, 0.1, U + 1)
        A = set(np.argsort(scores)[-K:])
        
        # Ensure coverage initially? If not, the pressure will fix it.
        
        current_counts = get_multiplicities(A)
        if not is_covered(current_counts):
            # If not covered, we need to be more aggressive in pressure
            pass
            
        for it in range(iterations):
            # 1. Calculate Pressure for each sum
            # We want to minimize max(counts).
            # Let target = max(1, min(counts.values()))? No, we want to flatten.
            # Let's define pressure based on deviation from a target or just penalty for high counts.
            # Penalty for high counts: quadratic?
            # Penalty for zero counts: linear large penalty.
            
            pressures = np.zeros(S_max - S_min + 1)
            max_c = max(current_counts.values()) if current_counts else 0
            min_c = min(current_counts.values()) if current_counts else 0
            
            for s in range(S_min, S_max + 1):
                c = current_counts[s]
                if c == 0:
                    # Strong penalty to increase this sum
                    pressures[s - S_min] = 100.0 
                elif c > max_c * 0.8: # Top 20% of counts
                    # Penalty to decrease
                    pressures[s - S_min] = -1.0 * (c - max_c * 0.5)
                else:
                    pressures[s - S_min] = 0.0
            
            # 2. Update Scores for each element x in [0, U]
            # Score(x) += sum_{y in A, x+y in [S_min, S_max]} pressure[x+y]
            # This is O(U * K). U=128, K~60. 128*60 = 7680 ops per iteration.
            # 200 iterations -> 1.5M ops. Fast.
            
            new_scores = scores.copy()
            A_list = list(A)
            
            for x in range(U + 1):
                delta = 0.0
                for y in A_list:
                    s = x + y
                    if S_min <= s <= S_max:
                        delta += pressures[s - S_min]
                new_scores[x] += 0.1 * delta # Learning rate
                
                # Regularization: penalize elements not in A? Or encourage diversity?
                # If x not in A, small penalty to keep scores bounded?
                # Actually, we just select top K.
            
            scores = new_scores
            
            # 3. Projection: Select top K scores
            # Add small noise to allow exploration
            scores_noisy = scores + np.random.normal(0, 0.01, U + 1)
            top_K_indices = np.argsort(scores_noisy)[-K:]
            new_A = set(top_K_indices)
            
            # Check if new A is valid (covered)
            new_counts = get_multiplicities(new_A)
            
            if is_covered(new_counts):
                new_max = max(new_counts.values())
                old_max = max(current_counts.values())
                
                # Accept if better or equal (with some probability for exploration)
                if new_max < old_max:
                    A = new_A
                    current_counts = new_counts
                elif new_max == old_max:
                    # Secondary objective: minimize sum of squares (variance)
                    new_sq = sum(c*c for c in new_counts.values())
                    old_sq = sum(c*c for c in current_counts.values())
                    if new_sq < old_sq:
                        A = new_A
                        current_counts = new_counts
                    elif random.random() < 0.1:
                        A = new_A
                        current_counts = new_counts
            else:
                # If not covered, revert or try to fix?
                # Revert to previous valid state if we had one.
                # For simplicity, if we never had a valid state, keep going.
                # But if we had a valid state and this is invalid, revert.
                if is_covered(current_counts):
                    pass # Keep old A
                else:
                    # Force coverage by adding elements that cover missing sums?
                    # This is complex. Let's just skip this update.
                    pass
        
        return A, max(current_counts.values()) if current_counts else float('inf')

    # Local Search Polish
    def local_search_polish(initial_A, steps=1000):
        A = set(initial_A)
        counts = get_multiplicities(A)
        if not is_covered(counts):
            return None, float('inf')
        
        current_max = max(counts.values())
        current_sq_sum = sum(c*c for c in counts.values())
        
        for step in range(steps):
            a = random.choice(list(A))
            candidates = [x for x in range(U + 1) if x not in A]
            if not candidates: break
            b = random.choice(candidates)
            
            # Calculate change
            affected_sums = set()
            s_aa = 2 * a
            if S_min <= s_aa <= S_max: affected_sums.add(s_aa)
            for x in A:
                if x == a: continue
                s_ax = a + x
                if S_min <= s_ax <= S_max: affected_sums.add(s_ax)
                
            s_bb = 2 * b
            if S_min <= s_bb <= S_max: affected_sums.add(s_bb)
            for x in A:
                if x == a: continue
                s_bx = b + x
                if S_min <= s_bx <= S_max: affected_sums.add(s_bx)
                
            new_counts = {}
            for s in affected_sums:
                old_c = counts[s]
                change = 0
                if s == 2 * a:
                    change -= 1
                for x in A:
                    if x == a: continue
                    if s == a + x:
                        change -= 1
                    if s == b + x:
                        change += 1
                if s == 2 * b:
                    change += 1
                new_counts[s] = old_c + change
            
            valid = True
            for s in affected_sums:
                if new_counts[s] < 0:
                    valid = False
                    break
                if new_counts[s] == 0 and counts[s] > 0:
                    valid = False
                    break
            
            if not valid:
                continue
                
            new_max = current_max
            new_sq_sum = current_sq_sum
            
            # Recalculate max and sq for affected sums
            # We need to know if the global max changed.
            # If current_max was at an affected sum, it might decrease.
            # If a new sum becomes larger than current_max, it increases.
            
            # Simple check:
            max_affected_old = max(counts[s] for s in affected_sums)
            max_affected_new = max(new_counts[s] for s in affected_sums)
            
            # If the old max was unique and in affected, and decreased, new_max might drop.
            # If the new max in affected is higher than current_max, new_max rises.
            
            # To be safe and fast, just recalculate global max if there's a chance.
            # Since K is small, recalc is cheap? No, O(K^2).
            # Let's just do a quick check.
            
            # If any affected sum becomes 0, invalid (already checked).
            # If any affected sum > current_max, new_max = that sum.
            # Else if the sum that was current_max is affected and decreased, we need to find the new global max.
            
            potential_new_max = max(current_max, max_affected_new)
            
            # If current_max was at an affected sum, and it decreased, we need to check if it's still the max.
            # This is getting complex. Let's just accept if local improvement is clear.
            
            # Heuristic: Accept if new_sq_sum < current_sq_sum and new_max <= current_max
            # Or if new_max < current_max
            
            # Calculate new_sq_sum
            for s in affected_sums:
                new_sq_sum += new_counts[s]*new_counts[s] - counts[s]*counts[s]
            
            # Determine new_max
            if max_affected_new > current_max:
                new_max = max_affected_new
            else:
                # Check if the element that held current_max is still current_max
                # This is hard to track without hist.
                # Let's just use a histogram approach for polish as in original.
                pass
            
            # Let's stick to the original polish logic which was robust, 
            # but driven by the score-based init.
            
            # Re-implementing robust polish check:
            temp_counts = counts.copy()
            for s in affected_sums:
                temp_counts[s] = new_counts[s]
                
            new_max_val = max(temp_counts.values())
            
            accept = False
            if new_max_val < current_max:
                accept = True
            elif new_max_val == current_max and new_sq_sum < current_sq_sum:
                accept = True
            elif new_max_val == current_max and new_sq_sum == current_sq_sum:
                if random.random() < 0.5:
                    accept = True
            
            if accept:
                for s in affected_sums:
                    counts[s] = new_counts[s]
                current_max = new_max_val
                current_sq_sum = new_sq_sum
                A.remove(a)
                A.add(b)
                
        return A, current_max

    # Main loop
    best_max = float('inf')
    best_A = None
    
    # Try a few K values
    for K in [55, 60, 65, 70]:
        for trial in range(5):
            A_init, max_init = score_based_search(K, iterations=100)
            
            if A_init is None: continue
            
            # Polish
            A_res, max_res = local_search_polish(A_init, steps=500)
            
            if A_res is not None and max_res < best_max:
                best_max = max_res
                best_A = A_res
                
            if best_max <= 10:
                break
        if best_max <= 10:
            break

    if best_A is None:
        # Fallback to seed
        return list(range(64, 129))
        
    return list(best_A)
