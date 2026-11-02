import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = 110.0
    
    D = 11
    
    # Generate candidates: weight 3 vectors
    candidates = []
    indices = list(range(D))
    # Precompute combinations of 3 indices
    supports = []
    for i in range(D):
        for j in range(i+1, D):
            for k in range(j+1, D):
                supports.append((i, j, k))
                
    for i, j, k in supports:
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                for s3 in [1, -1]:
                    v = [0]*D
                    v[i] = s1
                    v[j] = s2
                    v[k] = s3
                    candidates.append(v)
                    
    candidates = np.array(candidates, dtype=np.int8)
    N = len(candidates)
    
    # Precompute dot products
    # Dot product matrix
    # Dots = candidates @ candidates.T
    # This is 1320x1320, ~1.7M ops. Fast.
    Dots = candidates @ candidates.T
    
    # Conflict if dot >= 2
    # We want to find Max Independent Set in graph where edge exists if Dots[i,j] >= 2
    # Note: Dots[i,i] = 3. We don't care about self.
    
    # Initialize set
    best_set = []
    best_size = 0
    
    num_restarts = 50
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Greedy Init
        current_set = set()
        order = np.random.permutation(N)
        
        for idx in order:
            if time.time() > deadline:
                break
            # Check if idx conflicts with any in current_set
            # Get indices of current set
            if not current_set:
                current_set.add(idx)
                continue
                
            # Vectorized check
            # Dots[idx, list(current_set)]
            # If any >= 2, conflict
            # This is slow if current_set is large.
            # Better: maintain a bitmask of conflicts?
            # Or just check.
            
            # Optimization: 
            # If we maintain a boolean array 'in_set', we can check 
            # np.any(Dots[idx, np.where(in_set)[0]] >= 2)
            
            # Let's use a simpler check for now, optimize if needed.
            # Given N=1320, and set size likely < 500, this is okay.
            
            conflicts = np.any(Dots[idx, list(current_set)] >= 2)
            if not conflicts:
                current_set.add(idx)
                
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = list(current_set)
            
        # Local Search
        # We want to improve current_set
        # 2-swap: remove 1, add 2
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to add a vector
            # Find candidates not in set
            # This is expensive. 
            # Alternative: iterate over all not in set, try to add.
            
            # To make it faster, we can pick a random vector in set to remove,
            # and see if we can add 2.
            
            # Simpler: Try to add any vector.
            added = False
            for idx in np.random.permutation(N):
                if idx in current_set:
                    continue
                conflicts = np.any(Dots[idx, list(current_set)] >= 2)
                if not conflicts:
                    current_set.add(idx)
                    added = True
                    improved = True
                    break # Restart search for more additions
            
            if not added:
                # Try 2-swap
                # Pick a random v in set to remove
                if current_set:
                    v_to_remove = random.choice(list(current_set))
                    temp_set = current_set.copy()
                    temp_set.remove(v_to_remove)
                    
                    # Find 2 vectors u1, u2 not in temp_set that don't conflict
                    # and don't conflict with temp_set
                    # and don't conflict with each other
                    
                    # This is O(N^2) if done naively.
                    # We can find all "compatible" vectors with temp_set first.
                    # compatible = [i for i in range(N) if i not in temp_set and not np.any(Dots[i, list(temp_set)] >= 2)]
                    # Then find pair in compatible that don't conflict.
                    
                    # Optimization:
                    # 1. Compute mask of compatible with temp_set
                    # 2. Among those, find pair with dot < 2.
                    
                    # Let's try a few random swaps
                    for _ in range(10): # Limit attempts per removal
                        if time.time() > deadline:
                            break
                        u1 = random.randint(0, N-1)
                        u2 = random.randint(0, N-1)
                        if u1 == u2:
                            continue
                        if u1 in temp_set or u2 in temp_set:
                            continue
                        
                        # Check u1, u2 against temp_set
                        c1 = np.any(Dots[u1, list(temp_set)] >= 2)
                        if c1:
                            continue
                        c2 = np.any(Dots[u2, list(temp_set)] >= 2)
                        if c2:
                            continue
                        
                        # Check u1 vs u2
                        if Dots[u1, u2] >= 2:
                            continue
                        
                        # Success
                        current_set = temp_set
                        current_set.add(u1)
                        current_set.add(u2)
                        improved = True
                        break
                        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = list(current_set)
            
    if not best_set:
        # Fallback
        return np.array([[1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
        
    result = candidates[best_set]
    return result.astype(np.int64)
