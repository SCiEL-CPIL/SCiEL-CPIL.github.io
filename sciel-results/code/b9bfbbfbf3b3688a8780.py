import numpy as np
import time
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    D = 11
    indices = list(range(D))
    
    def solve_norm4():
        """Find max independent set among norm-4 vectors using vectorized conflict checks."""
        # Generate all vectors with exactly 4 nonzero entries of ±1
        vectors = []
        for comb in combinations(indices, 4):
            for signs in product([1, -1], repeat=4):
                v = np.zeros(D, dtype=np.int8)
                for idx, val in zip(comb, signs):
                    v[idx] = val
                vectors.append(v)
        
        vectors = np.array(vectors)
        n = len(vectors)
        
        if n == 0:
            return np.array([], dtype=np.int64)
        
        # Precompute dot product matrix
        dots = vectors.astype(np.int16) @ vectors.astype(np.int16).T
        
        # For norm^2 = 4: need ||u-v||^2 >= 4
        # ||u-v||^2 = 8 - 2*u.v >= 4 => u.v <= 2
        max_dot_allowed = 2
        
        # Conflict matrix: True if pair conflicts
        conflict = dots > max_dot_allowed
        
        best_set = []
        best_size = 0
        
        # Strategy 1: Multiple greedy passes with different orderings
        n_restarts = 50
        for restart in range(n_restarts):
            if time.time() > deadline - 15:
                break
            
            # Random permutation
            perm = np.random.permutation(n)
            
            # Track which vectors are still available
            available = np.ones(n, dtype=bool)
            selected = []
            
            for idx in perm:
                if time.time() > deadline - 10:
                    break
                
                if not available[idx]:
                    continue
                
                # Check if this vector conflicts with any selected
                # Vectorized: check against all selected at once
                if len(selected) == 0:
                    selected.append(idx)
                    available[idx] = False
                    # Remove conflicting vectors from available
                    conflict_mask = conflict[idx]
                    available[conflict_mask] = False
                else:
                    # Check against selected vectors
                    sel_arr = np.array(selected)
                    if not np.any(conflict[idx, sel_arr]):
                        selected.append(idx)
                        available[idx] = False
                        conflict_mask = conflict[idx]
                        available[conflict_mask] = False
            
            if len(selected) > best_size:
                best_size = len(selected)
                best_set = selected[:]
        
        # Strategy 2: Local search improvement on best solution found
        if best_size > 0 and time.time() < deadline - 5:
            current_set = set(best_set)
            current_size = len(best_set)
            
            # Try to add unselected vectors that don't conflict with current set
            # Then try swaps to improve
            improved = True
            max_iterations = 200
            iteration = 0
            
            while improved and iteration < max_iterations and time.time() < deadline - 2:
                improved = False
                iteration += 1
                
                # Try adding any vector not in current_set that doesn't conflict
                for idx in range(n):
                    if idx in current_set:
                        continue
                    
                    # Check if idx conflicts with any in current_set
                    sel_arr = list(current_set)
                    if not np.any(conflict[idx, sel_arr]):
                        current_set.add(idx)
                        current_size += 1
                        improved = True
                        break
                
                # If no addition possible, try swaps
                if not improved:
                    for idx_in in list(current_set):
                        for idx_out in range(n):
                            if idx_out in current_set:
                                continue
                            
                            # Check if idx_out conflicts with (current_set - {idx_in})
                            remaining = [x for x in current_set if x != idx_in]
                            if not np.any(conflict[idx_out, remaining]):
                                current_set.remove(idx_in)
                                current_set.add(idx_out)
                                improved = True
                                break
                        if improved:
                            break
            
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = list(current_set)
        
        if best_set:
            return vectors[best_set].astype(np.int64)
        return np.array([], dtype=np.int64)
    
    def solve_norm3():
        """Find max independent set among norm-3 vectors."""
        vectors = []
        for comb in combinations(indices, 3):
            for signs in product([1, -1], repeat=3):
                v = np.zeros(D, dtype=np.int8)
                for idx, val in zip(comb, signs):
                    v[idx] = val
                vectors.append(v)
        
        vectors = np.array(vectors)
        n = len(vectors)
        
        if n == 0:
            return np.array([], dtype=np.int64)
        
        dots = vectors.astype(np.int16) @ vectors.astype(np.int16).T
        
        # For norm^2 = 3: need ||u-v||^2 >= 3
        # ||u-v||^2 = 6 - 2*u.v >= 3 => u.v <= 1.5 => u.v <= 1
        max_dot_allowed = 1
        
        conflict = dots > max_dot_allowed
        
        best_set = []
        best_size = 0
        
        n_restarts = 30
        for restart in range(n_restarts):
            if time.time() > deadline - 20:
                break
            
            perm = np.random.permutation(n)
            available = np.ones(n, dtype=bool)
            selected = []
            
            for idx in perm:
                if time.time() > deadline - 15:
                    break
                
                if not available[idx]:
                    continue
                
                if len(selected) == 0:
                    selected.append(idx)
                    available[idx] = False
                    available[conflict[idx]] = False
                else:
                    sel_arr = np.array(selected)
                    if not np.any(conflict[idx, sel_arr]):
                        selected.append(idx)
                        available[idx] = False
                        available[conflict[idx]] = False
            
            if len(selected) > best_size:
                best_size = len(selected)
                best_set = selected[:]
        
        if best_set:
            return vectors[best_set].astype(np.int64)
        return np.array([], dtype=np.int64)
    
    def solve_norm2():
        """Find max independent set among norm-2 vectors (baseline: 220)."""
        vectors = []
        for comb in combinations(indices, 2):
            for signs in product([1, -1], repeat=2):
                v = np.zeros(D, dtype=np.int8)
                for idx, val in zip(comb, signs):
                    v[idx] = val
                vectors.append(v)
        
        vectors = np.array(vectors)
        n = len(vectors)
        
        if n == 0:
            return np.array([], dtype=np.int64)
        
        dots = vectors.astype(np.int16) @ vectors.astype(np.int16).T
        
        # For norm^2 = 2: need ||u-v||^2 >= 2
        # ||u-v||^2 = 4 - 2*u.v >= 2 => u.v <= 1
        max_dot_allowed = 1
        
        conflict = dots > max_dot_allowed
        
        best_set = []
        best_size = 0
        
        n_restarts = 20
        for restart in range(n_restarts):
            if time.time() > deadline - 25:
                break
            
            perm = np.random.permutation(n)
            available = np.ones(n, dtype=bool)
            selected = []
            
            for idx in perm:
                if time.time() > deadline - 20:
                    break
                
                if not available[idx]:
                    continue
                
                if len(selected) == 0:
                    selected.append(idx)
                    available[idx] = False
                    available[conflict[idx]] = False
                else:
                    sel_arr = np.array(selected)
                    if not np.any(conflict[idx, sel_arr]):
                        selected.append(idx)
                        available[idx] = False
                        available[conflict[idx]] = False
            
            if len(selected) > best_size:
                best_size = len(selected)
                best_set = selected[:]
        
        if best_set:
            return vectors[best_set].astype(np.int64)
        return np.array([], dtype=np.int64)
    
    # Try norm 4 first (most candidates, potentially best result)
    result4 = np.array([], dtype=np.int64)
    if time.time() < deadline - 30:
        result4 = solve_norm4()
    
    # Try norm 3
    result3 = np.array([], dtype=np.int64)
    if time.time() < deadline - 15:
        result3 = solve_norm3()
    
    # Try norm 2 as fallback
    result2 = np.array([], dtype=np.int64)
    if time.time() < deadline - 5:
        result2 = solve_norm2()
    
    # Pick the best result
    results = [r for r in [result4, result3, result2] if len(r) > 0]
    
    if not results:
        # Absolute fallback: just the unit vectors
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
    
    best = max(results, key=len)
    return best.astype(np.int64)
