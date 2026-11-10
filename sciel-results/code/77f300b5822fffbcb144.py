import numpy as np
import time
from itertools import combinations

def kissing_number11() -> np.ndarray:
        d = 11
        start_time = time.time()
        deadline = 580

        def fallback():
            points = []
            for i in range(d):
                for j in range(i + 1, d):
                    for s1 in [-1, 1]:
                        for s2 in [-1, 1]:
                            v = np.zeros(d, dtype=np.int64)
                            v[i] = s1
                            v[j] = s2
                            points.append(v)
            return np.array(points, dtype=np.int64)

        def solve_with_weight(w):
            vectors = []
            for support in combinations(range(d), w):
                for signs in range(2 ** w):
                    v = np.zeros(d, dtype=np.int64)
                    for k_idx, pos in enumerate(support):
                        v[pos] = 1 if (signs >> k_idx) & 1 else -1
                    vectors.append(v)
            
            mat = np.array(vectors, dtype=np.int16)
            n = len(mat)
            
            # Compute dot products
            # Use int16 to save memory
            dots = mat @ mat.T
            conflict = dots > w  # Wait, condition is d <= 2?
            
            # Re-evaluate condition for general weight w
            # R^2 = w
            # ||u-v||^2 = 2w - 2d >= w  => w >= 2d => d <= w/2
            # For w=4, d <= 2.
            # For w=5, d <= 2.5 => d <= 2.
            # For w=3, d <= 1.5 => d <= 1.
            # For w=2, d <= 1.
            
            # So conflict if d > floor(w/2)
            max_dot = w // 2
            conflict = dots > max_dot
            
            best_selected = []
            best_size = 0
            rng = np.random.default_rng(42)
            
            while time.time() - start_time < deadline:
                available = np.ones(n, dtype=bool)
                selected = []
                
                while available.any():
                    if time.time() - start_time > deadline:
                        break
                    
                    conflict_counts = conflict[:, available].sum(axis=1)
                    conflict_counts[~available] = n + 1
                    
                    min_val = np.min(conflict_counts[available])
                    candidates = np.where(conflict_counts == min_val)[0]
                    idx = int(rng.choice(candidates))
                    
                    selected.append(idx)
                    
                    avail_conflicts = conflict[idx] & available
                    available[avail_conflicts] = False
                    available[idx] = False
                
                if len(selected) > best_size:
                    best_size = len(selected)
                    best_selected = selected.copy()
            
            if best_size > 0:
                return mat[best_selected].astype(np.int64), best_size
            return None, 0

        try:
            # Try weight 5
            res5, size5 = solve_with_weight(5)
            if res5 is not None and size5 > 240:
                return res5
            
            # Try weight 4
            res4, size4 = solve_with_weight(4)
            if res4 is not None and size4 > 220:
                return res4
                
        except Exception:
            pass

        return fallback()
