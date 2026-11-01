"""Dual-view construction: define U via Sidon-like difference structure, minimize additive energy."""
import numpy as np
import math
import random

def construct():
    """
    Construct U using dual-view approach:
    - Start with structured sets (APs) that minimize |U+U|
    - Use difference-set-guided perturbations to increase |U-U|
    - Score = 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)
    """
    
    def compute_score_fast(U):
        """Compute score using numpy for speed."""
        if not U or 0 not in U:
            return 0.0
        U_arr = np.array(sorted(set(U)))
        max_u = int(U_arr.max())
        if max_u == 0:
            return 0.0
        
        # Vectorized sum and diff computation
        sums = U_arr[:, None] + U_arr[None, :]
        diffs = U_arr[:, None] - U_arr[None, :]
        
        n_sum = len(np.unique(sums))
        n_diff = len(np.unique(diffs))
        
        if n_sum == 0:
            return 0.0
        log_denom = math.log(2 * max_u + 1)
        if log_denom <= 0:
            return 0.0
        return 1.0 + math.log(n_diff / n_sum) / log_denom
    
    def local_search_optimized(U, max_iters=100, max_val=500):
        """Optimized local search with numpy-based evaluation."""
        U = sorted(set(U))
        if 0 not in U:
            U = sorted([0] + U)
        
        best_U = list(U)
        best_score = compute_score_fast(U)
        
        for iteration in range(max_iters):
            improved = False
            U_set = set(best_U)
            n = len(best_U)
            
            # Try moving each element (except 0)
            for i in range(n):
                if best_U[i] == 0:
                    continue
                
                elem = best_U[i]
                best_new_val = None
                best_new_score = best_score
                
                # Try nearby values
                for new_val in range(max(1, elem - 10), min(max_val + 1, elem + 11)):
                    if new_val == elem or new_val in U_set:
                        continue
                    
                    new_U = list(best_U)
                    new_U[i] = new_val
                    new_U = sorted(set(new_U))
                    if 0 not in new_U or len(new_U) < 2:
                        continue
                    
                    new_score = compute_score_fast(new_U)
                    if new_score > best_new_score + 1e-10:
                        best_new_score = new_score
                        best_new_val = new_val
                
                if best_new_val is not None:
                    best_U[i] = best_new_val
                    best_U = sorted(set(best_U))
                    best_score = best_new_score
                    improved = True
                    break
            
            # Try adding a new element
            if not improved:
                best_candidate = None
                best_candidate_score = best_score
                
                # Generate candidates with bias toward creating new differences
                candidates = set()
                for elem in best_U:
                    for d in range(-5, 6):
                        v = elem + d
                        if 1 <= v <= max_val:
                            candidates.add(v)
                for _ in range(40):
                    candidates.add(random.randint(1, max_val))
                
                for candidate in sorted(candidates):
                    if candidate in U_set:
                        continue
                    
                    new_U = sorted(best_U + [candidate])
                    new_score = compute_score_fast(new_U)
                    if new_score > best_candidate_score + 1e-10:
                        best_candidate_score = new_score
                        best_candidate = candidate
                
                if best_candidate is not None:
                    best_U = sorted(best_U + [best_candidate])
                    best_score = best_candidate_score
                    improved = True
            
            # Try removing an element
            if not improved:
                for i in range(n):
                    if best_U[i] == 0:
                        continue
                    new_U = [x for j, x in enumerate(best_U) if j != i]
                    if len(new_U) < 2:
                        continue
                    new_score = compute_score_fast(new_U)
                    if new_score > best_score + 1e-10:
                        best_score = new_score
                        best_U = new_U
                        improved = True
                        break
            
            if not improved:
                break
        
        return best_U, best_score
    
    def build_structured_set(base, n, max_val=400):
        """Build a set with structured core + sparse perturbations."""
        core = [i * base for i in range(n)]
        U = list(core)
        U_set = set(core)
        
        # Greedily add elements that maximize score
        max_iters = 25
        for _ in range(max_iters):
            best_candidate = None
            best_score = -float('inf')
            
            for candidate in range(max(core[-1] + 1, 1), min(max_val + 1, core[-1] + 150)):
                if candidate in U_set:
                    continue
                
                new_U = sorted(U + [candidate])
                new_score = compute_score_fast(new_U)
                if new_score > best_score:
                    best_score = new_score
                    best_candidate = candidate
            
            if best_candidate is None:
                break
            
            U.append(best_candidate)
            U_set.add(best_candidate)
        
        return sorted(U)
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Structured AP cores with perturbations
    for base in [3, 4, 5, 6, 7, 8, 10, 12]:
        for n in range(4, 11):
            core = [i * base for i in range(n)]
            U = list(core)
            for i in range(1, len(U)):
                U[i] += random.randint(-2, 2)
            U = sorted(set(U))
            if 0 not in U:
                U = [0] + U
            if len(U) >= 2:
                candidates.append(U)
    
    # 2. Geometric-like sequences
    candidates.append([0, 1, 3, 9, 27, 81, 243])
    candidates.append([0, 1, 3, 7, 15, 31, 63, 127])
    candidates.append([0, 1, 4, 16, 64, 256])
    candidates.append([0, 2, 5, 12, 29, 66, 149])
    candidates.append([0, 1, 2, 5, 11, 23, 47, 95])
    
    # 3. Mian-Chowla / Sidon sequences
    candidates.append([0, 1, 4, 10, 18, 23, 31, 40, 50, 61, 73, 86, 100])
    candidates.append([0, 1, 3, 7, 12, 20, 30, 44, 60, 79])
    candidates.append([0, 1, 4, 10, 18, 23, 31, 40, 50, 61])
    candidates.append([0, 1, 4, 10, 18, 23, 31, 40, 50])
    
    # 4. Hybrid: AP core + sparse large elements
    for base in [4, 5, 6, 7, 8]:
        for n in range(3, 7):
            U = build_structured_set(base, n, max_val=350)
            if len(U) >= 2:
                candidates.append(U)
    
    # 5. Random constructions with structure bias
    random.seed(42)
    for trial in range(12):
        n = random.randint(5, 10)
        max_val = random.randint(100, 250)
        U = [0]
        U_set = {0}
        
        core_size = random.randint(2, 4)
        base = random.randint(3, 7)
        for i in range(core_size):
            val = i * base + random.randint(0, 1)
            if val not in U_set and val > 0:
                U.append(val)
                U_set.add(val)
        
        for _ in range(n - len(U)):
            best_candidate = None
            best_score = -float('inf')
            
            for _ in range(25):
                candidate = random.randint(1, max_val)
                if candidate in U_set:
                    continue
                
                new_U = sorted(U + [candidate])
                new_score = compute_score_fast(new_U)
                if new_score > best_score:
                    best_score = new_score
                    best_candidate = candidate
            
            if best_candidate is not None and best_score > -float('inf'):
                U.append(best_candidate)
                U_set.add(best_candidate)
        
        U = sorted(U)
        candidates.append(U)
    
    # Run optimized local search on each candidate
    best_U = [0, 1]
    best_score = 0.0
    
    for U in candidates:
        U = sorted(set(U))
        if 0 not in U:
            U = [0] + U
        if len(U) < 2:
            continue
        searched_U, searched_score = local_search_optimized(U, max_iters=60, max_val=400)
        if searched_score > best_score:
            best_score = searched_score
            best_U = searched_U
    
    return best_U

# EVOLVE-BLOCK-START
# (The construct function above is the main entry point)
# EVOLVE-BLOCK-END
