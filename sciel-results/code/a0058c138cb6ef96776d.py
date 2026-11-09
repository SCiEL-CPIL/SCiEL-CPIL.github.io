"""Program for integer sum-product extremal sets using structured smooth-number APs and factor-aware local search."""
import time
import random
import math
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 55.0  # Safety margin
    n = 20
    max_val = 10000
    
    # Precompute indices for fast evaluation
    i_idx, j_idx = np.triu_indices(n)
    
    def evaluate(A):
        """Compute max(|A+A|, |A*A|) for set A using NumPy for speed."""
        # Ensure A is a sorted numpy array of int64
        A_arr = np.array(sorted(A), dtype=np.int64)
        sums = A_arr[i_idx] + A_arr[j_idx]
        prods = A_arr[i_idx] * A_arr[j_idx]
        
        # Count unique values
        # Using np.unique is O(m log m) where m = n(n+1)/2 = 210
        unique_sums = np.unique(sums)
        unique_prods = np.unique(prods)
        
        return max(len(unique_sums), len(unique_prods))

    best_set = None
    best_score = float('inf')
    
    def update_best(cand):
        nonlocal best_set, best_score
        if len(set(cand)) != n or max(cand) > max_val or min(cand) < 1:
            return False
        score = evaluate(cand)
        if score < best_score:
            best_score = score
            best_set = list(sorted(cand))
            return True
        return False

    candidates = []
    
    # 1. Pure Geometric Progressions (Small bases)
    for start in range(1, 10):
        for ratio in range(2, 6):
            cand = []
            val = start
            for i in range(n):
                if val > max_val:
                    break
                cand.append(val)
                val *= ratio
            if len(cand) == n:
                candidates.append(sorted(cand))
                
    # 2. Pure Arithmetic Progressions (Small steps)
    for start in range(1, 15):
        for step in range(1, 20):
            cand = [start + i * step for i in range(n)]
            if max(cand) <= max_val:
                candidates.append(sorted(cand))
                
    # 3. Smooth Number APs: APs where start and step are "smooth" (2^a * 3^b)
    # This helps in creating multiplicative collisions
    smooth_nums = []
    for a in range(0, 14):
        for b in range(0, 14):
            val = (2 ** a) * (3 ** b)
            if 1 <= val <= max_val:
                smooth_nums.append(val)
    smooth_nums = sorted(set(smooth_nums))
    
    # Generate APs with smooth start and step
    for a in smooth_nums[:50]: # Limit to smaller smooths for start
        for d in smooth_nums[:50]: # Limit to smaller smooths for step
            if d == 0: continue
            cand = [a + i * d for i in range(n)]
            if max(cand) <= max_val and min(cand) >= 1:
                candidates.append(sorted(cand))
                
    # 4. Hybrid: Union of two short APs
    # Try to interleave or concatenate
    for k in range(8, 13): # Split sizes
        gp_len = n - k
        if gp_len < 4: continue
        
        # AP part
        for a in range(1, 20):
            for d in range(1, 15):
                ap_part = [a + i * d for i in range(k)]
                if max(ap_part) > max_val: continue
                
                # GP part (or another AP with different structure)
                # Let's try GP part with small ratio
                for b in range(1, 20):
                    for r in range(2, 6):
                        gp_part = []
                        val = b
                        for i in range(gp_len):
                            if val > max_val: break
                            gp_part.append(val)
                            val *= r
                        if len(gp_part) != gp_len: continue
                        
                        cand = ap_part + gp_part
                        if len(set(cand)) == n and max(cand) <= max_val and min(cand) >= 1:
                            candidates.append(sorted(cand))
                            
    # 5. Random Smooth Sets: Sample n distinct smooth numbers
    # This often yields small product sets
    for _ in range(20):
        if len(smooth_nums) >= n:
            cand = random.sample(smooth_nums, n)
            if max(cand) <= max_val:
                candidates.append(sorted(cand))
                
    # Evaluate all structured candidates
    for cand in candidates:
        update_best(cand)
        
    if best_set is None:
        best_set = list(range(1, 21))
        best_score = evaluate(best_set)
        
    # Phase 2: Local Search with Factor-Aware Moves
    current_set = list(best_set)
    current_score = best_score
    
    T = 12.0
    cooling_rate = 0.999
    min_T = 0.5
    
    # Precompute for fast move generation
    # We will maintain a set of current values for O(1) membership check
    current_set_arr = np.array(current_set, dtype=np.int64)
    
    while time.time() < deadline:
        if T < min_T:
            T = min_T
            
        new_set = list(current_set)
        move_type = random.random()
        
        if move_type < 0.30:
            # Nudge one element slightly
            idx = random.randint(0, n - 1)
            old_val = new_set[idx]
            for _ in range(10):
                delta = random.randint(-10, 10)
                if delta == 0: continue
                new_val = old_val + delta
                if 1 <= new_val <= max_val and new_val not in new_set:
                    new_set[idx] = new_val
                    break
            if new_set == current_set: continue
                    
        elif move_type < 0.50:
            # Factor-Aware Replace: 
            # Pick an element x. Pick another element z in A.
            # Try to replace x with a multiple of z or z + small_delta
            # This encourages shared factors, reducing |A*A|
            idx = random.randint(0, n - 1)
            old_val = new_set[idx]
            z_idx = random.randint(0, n - 1)
            z_val = new_set[z_idx]
            
            # Try candidates based on z_val
            cands = []
            # Multiples
            for k in range(2, 6):
                v = z_val * k
                if 1 <= v <= max_val and v != old_val and v not in new_set:
                    cands.append(v)
            # Near multiples (z_val * k + delta)
            for k in range(1, 5):
                for delta in [-2, -1, 1, 2]:
                    v = z_val * k + delta
                    if 1 <= v <= max_val and v != old_val and v not in new_set:
                        cands.append(v)
                        
            if cands:
                new_set[idx] = random.choice(cands)
            else:
                # Fallback to random nudge
                for _ in range(5):
                    delta = random.randint(-15, 15)
                    if delta == 0: continue
                    new_val = old_val + delta
                    if 1 <= new_val <= max_val and new_val not in new_set:
                        new_set[idx] = new_val
                        break
                if new_set == current_set: continue

        elif move_type < 0.70:
            # Additive-Aware Replace:
            # Pick x. Pick z in A.
            # Try to replace x with z + delta (small delta)
            # This encourages additive collisions, reducing |A+A|
            idx = random.randint(0, n - 1)
            old_val = new_set[idx]
            z_idx = random.randint(0, n - 1)
            z_val = new_set[z_idx]
            
            cands = []
            for delta in range(-5, 6):
                if delta == 0: continue
                v = z_val + delta
                if 1 <= v <= max_val and v != old_val and v not in new_set:
                    cands.append(v)
                
            if cands:
                new_set[idx] = random.choice(cands)
            else:
                for _ in range(5):
                    delta = random.randint(-15, 15)
                    if delta == 0: continue
                    new_val = old_val + delta
                    if 1 <= new_val <= max_val and new_val not in new_set:
                        new_set[idx] = new_val
                        break
                if new_set == current_set: continue

        elif move_type < 0.85:
            # Swap two elements
            i, j = random.sample(range(n), 2)
            new_set[i], new_set[j] = new_set[j], new_set[i]
            
        else:
            # Random Replace with bias towards small numbers
            idx = random.randint(0, n - 1)
            old_val = new_set[idx]
            if random.random() < 0.7:
                new_val = random.randint(1, 100)
            else:
                new_val = random.randint(1, 500)
            attempts = 0
            while (new_val == old_val or new_val in new_set) and attempts < 15:
                if random.random() < 0.7:
                    new_val = random.randint(1, 100)
                else:
                    new_val = random.randint(1, 500)
                attempts += 1
            if new_val == old_val or new_val in new_set:
                continue
            new_set[idx] = new_val
        
        new_score = evaluate(new_set)
        
        delta_score = new_score - current_score
        if delta_score < 0:
            current_set = new_set
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_set = list(current_set)
        else:
            if random.random() < math.exp(-delta_score / T):
                current_set = new_set
                current_score = new_score
                
        T *= cooling_rate
        
    return best_set
# EVOLVE-BLOCK-END
