import numpy as np
import itertools
import random

def construct(n):
    if n < 7:
        return standard_construct(n)
    
    symbols = list(range(1, n + 1))
    perms = list(itertools.permutations(symbols))
    N = len(perms)
    
    # Precompute overlap matrix
    # O[i][j] = max k such that perms[i][-k:] == perms[j][:k]
    # We can store this as a numpy array of uint8
    O = np.zeros((N, N), dtype=np.uint8)
    
    # Optimization: Use string representation or tuple slicing
    # To make it faster, convert perms to strings?
    # "1234567"
    # Slicing strings is fast.
    str_perms = ["".join(map(str, p)) for p in perms]
    
    for i in range(N):
        pi = str_perms[i]
        for j in range(N):
            pj = str_perms[j]
            # Check k from n-1 down to 0
            # pi[-k:] == pj[:k]
            # Max k is n-1
            for k in range(n - 1, -1, -1):
                if k == 0:
                    O[i, j] = 0
                    break
                if pi[-k:] == pj[:k]:
                    O[i, j] = k
                    break

    # Initial path: Random permutation of indices
    path = list(range(N))
    random.shuffle(path)
    
    # Calculate initial total overlap
    total_overlap = 0
    for i in range(1, N):
        total_overlap += O[path[i-1], path[i]]
        
    best_total_overlap = total_overlap
    best_path = path[:]
    
    # Simulated Annealing / Hill Climbing
    # Time limit: 100 seconds
    import time
    start_time = time.time()
    max_time = 100
    
    # We will do swaps
    # To make it faster, we can try random swaps and accept if better
    # Or use a more structured search like 2-opt
    
    # Let's use a simple local search with random restarts if stuck?
    # Or just continuous swaps.
    
    # Since N=5040, a full pass of adjacent swaps is N steps.
    # We can do multiple passes.
    
    improved = True
    while time.time() - start_time < max_time:
        improved = False
        # Randomize start point to avoid local minima?
        # Or just iterate through indices
        for i in range(1, N):
            # Swap path[i-1] and path[i]
            # Neighbors: A = path[i-2] (if i>1), B = path[i+1] (if i<N-1)
            
            idx_a = i - 2
            idx_b = i + 1
            
            p_i = path[i]
            p_im1 = path[i-1]
            
            # Get neighbors
            if idx_a >= 0:
                A = path[idx_a]
            else:
                A = -1 # Sentinel
            
            if idx_b < N:
                B = path[idx_b]
            else:
                B = -1 # Sentinel
            
            # Calculate old overlap contribution
            old_ov = 0
            if A != -1:
                old_ov += O[A, p_im1]
            old_ov += O[p_im1, p_i]
            if B != -1:
                old_ov += O[p_i, B]
                
            # Calculate new overlap contribution
            # Order becomes ... A, p_i, p_im1, B ...
            new_ov = 0
            if A != -1:
                new_ov += O[A, p_i]
            new_ov += O[p_i, p_im1]
            if B != -1:
                new_ov += O[p_im1, B]
                
            if new_ov > old_ov:
                # Swap
                path[i], path[i-1] = p_im1, p_i
                total_overlap += (new_ov - old_ov)
                improved = True
                if total_overlap > best_total_overlap:
                    best_total_overlap = total_overlap
                    best_path = path[:]
            
        # If no improvement in a full pass, we might be stuck in a local minimum
        # We can try a random perturbation or restart
        if not improved:
            # Try a random swap to escape
            i = random.randint(1, N-1)
            p_i = path[i]
            p_im1 = path[i-1]
            idx_a = i - 2
            idx_b = i + 1
            A = path[idx_a] if idx_a >= 0 else -1
            B = path[idx_b] if idx_b < N else -1
            
            old_ov = 0
            if A != -1: old_ov += O[A, p_im1]
            old_ov += O[p_im1, p_i]
            if B != -1: old_ov += O[p_i, B]
            
            new_ov = 0
            if A != -1: new_ov += O[A, p_i]
            new_ov += O[p_i, p_im1]
            if B != -1: new_ov += O[p_im1, B]
            
            # Accept with probability exp(-delta/T) or just accept if not too bad?
            # For simplicity, let's just do a few random restarts if stuck
            # Or implement proper Simulated Annealing
            pass
            
    # Construct final word from best_path
    word = list(perms[best_path[0]])
    for i in range(1, N):
        ov = O[best_path[i-1], best_path[i]]
        word.extend(perms[best_path[i]][ov:])
        
    return "".join(map(str, word))

def standard_construct(n):
    pass
