import random
import numpy as np
from scipy.optimize import differential_evolution
import time

def construct():
        n = 128
        size = n + 1
        target_low = 128
        target_high = 256
        num_targets = target_high - target_low + 1 # 129

        def objective(x):
            # x is float array [0,1]
            # Threshold to binary
            b = (x > 0.5).astype(np.int8)
            
            # Convolution
            c = np.convolve(b, b)
            
            # Extract relevant sums [128, 256]
            c_relevant = c[128:257]
            
            # Calculate unordered pair counts
            # c[s] counts ordered pairs (a,b) with a+b=s.
            # Unordered count u[s] = (c[s] + (1 if s is even and b[s//2]==1 else 0)) / 2
            s_vals = np.arange(target_low, target_high + 1)
            is_even = (s_vals % 2 == 0)
            mid_indices = s_vals // 2
            # mid_indices might be out of bounds if s//2 > 128?
            # Max s = 256, s//2 = 128. Index 128 is valid (size 129).
            # Min s = 128, s//2 = 64. Index 64 is valid.
            
            d_vals = np.zeros(num_targets, dtype=np.int8)
            d_vals[is_even] = b[mid_indices[is_even]]
            
            counts = (c_relevant + d_vals) / 2.0
            
            # Check coverage
            uncovered = np.sum(counts == 0)
            
            if uncovered > 0:
                # Heavy penalty
                return 10000.0 + uncovered * 100.0 + np.sum(counts) # encourage some mass
            
            max_count = np.max(counts)
            # Add small penalty for size to prefer smaller sets? 
            # The task doesn't explicitly minimize size, only max multiplicity.
            # However, smaller sets generally have lower max multiplicity if structured well.
            # But a larger set can distribute the sums better.
            # Let's stick to minimizing max_count.
            return max_count

        start_time = time.time()
        
        # Better initialization for DE?
        # DE generates its own init.
        # We can run DE for a while.
        
        try:
            res = differential_evolution(
                objective, 
                bounds=[(0, 1)] * size, 
                popsize=30, 
                maxiter=200, 
                tol=1e-5, 
                seed=42,
                polish=False, # We will do our own polish
                init='latinhypercube',
                workers=1
            )
        except Exception:
            res = None

        if res is not None and res.fun < 10000:
            best_x = res.x
        else:
            # Fallback to a known good structure if DE fails
            # A = [0, 1, ..., 128] is bad.
            # A = [64, ..., 128] is the previous init.
            best_x = np.zeros(size)
            best_x[64:129] = 1.0

        current_A = set(np.where(best_x > 0.5)[0].tolist())
        
        # Vectorized Local Search Polish
        def get_metrics(A_set):
            b = np.zeros(size, dtype=np.int8)
            for a in A_set:
                b[a] = 1
            c = np.convolve(b, b)
            c_relevant = c[128:257]
            s_vals = np.arange(128, 257)
            is_even = (s_vals % 2 == 0)
            mid_indices = s_vals // 2
            d_vals = np.zeros(129, dtype=np.int8)
            d_vals[is_even] = b[mid_indices[is_even]]
            counts = (c_relevant + d_vals) / 2.0
            uncovered = np.sum(counts == 0)
            max_count = np.max(counts) if uncovered == 0 else 1000
            return max_count, uncovered, counts

        current_max, current_uncovered, _ = get_metrics(current_A)
        
        if current_uncovered > 0:
             # Fallback
             current_A = set(range(64, 129))
             current_max, current_uncovered, _ = get_metrics(current_A)
             
        best_max = current_max
        best_A = set(current_A)
        
        time_limit = 55
        candidates_to_add = [i for i in range(size) if i not in current_A]
        
        # Local Search: Try removing, adding, swapping
        # We can make this faster by pre-computing counts?
        # For now, re-computing convolution is fast enough for ~10k iterations.
        
        for _ in range(5000):
            if time.time() - start_time > time_limit:
                break
                
            # 1. Try removing an element
            if len(current_A) > 1:
                elem = random.choice(list(current_A))
                new_A = current_A - {elem}
                new_max, new_uncovered, _ = get_metrics(new_A)
                if new_uncovered == 0 and new_max < current_max:
                    current_A = new_A
                    current_max = new_max
                    if current_max < best_max:
                        best_max = current_max
                        best_A = set(current_A)
                        candidates_to_add = [i for i in range(size) if i not in current_A]
                        continue

            # 2. Try swapping
            if candidates_to_add:
                elem = random.choice(list(current_A))
                add_elem = random.choice(candidates_to_add)
                new_A = current_A - {elem} | {add_elem}
                new_max, new_uncovered, _ = get_metrics(new_A)
                if new_uncovered == 0 and new_max < current_max:
                    current_A = new_A
                    current_max = new_max
                    if current_max < best_max:
                        best_max = current_max
                        best_A = set(current_A)
                        candidates_to_add = [i for i in range(size) if i not in current_A]
                    continue

            # 3. Try adding an element (if it doesn't increase max count)
            # This might help if we are stuck, but usually we want to minimize max count.
            # Adding an element increases the number of pairs, potentially increasing max count.
            # However, it might allow removing others later?
            # The task is to minimize max count.
            # Let's stick to removal and swap.

        return list(best_A)
