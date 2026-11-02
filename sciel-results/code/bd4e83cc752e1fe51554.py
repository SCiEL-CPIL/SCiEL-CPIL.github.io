"""Vectorized stochastic greedy for ternary covering codes K_3(7,1)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _precompute_neighbors(q, n, radius):
    """Precompute neighbors for all words using NumPy-friendly structure."""
    N = q ** n
    max_nbrs = 1
    if radius == 1:
        max_nbrs = 1 + n * (q - 1)
    elif radius == 2:
        # Approximate upper bound
        max_nbrs = 1 + n * (q - 1) + (n * (n - 1) // 2) * (q - 1) ** 2
    else:
        # Fallback to general enumeration if radius > 2 (not expected for this task)
        max_nbrs = q ** n 

    nbrs_flat = np.zeros((N, max_nbrs), dtype=np.int32)
    nbrs_len = np.zeros(N, dtype=np.int32)
    
    # Helper to convert idx to word and back
    # For speed, we unroll the loop for n=7, q=3
    if q == 3 and n == 7:
        for idx in range(N):
            word = [0] * 7
            tmp = idx
            word[0] = tmp % 3; tmp //= 3
            word[1] = tmp % 3; tmp //= 3
            word[2] = tmp % 3; tmp //= 3
            word[3] = tmp % 3; tmp //= 3
            word[4] = tmp % 3; tmp //= 3
            word[5] = tmp % 3; tmp //= 3
            word[6] = tmp % 3; tmp //= 3
            
            nbrs = [idx]
            # Radius 1 neighbors
            for pos in range(7):
                old = word[pos]
                for v in range(3):
                    if v != old:
                        new_word = word.copy()
                        new_word[pos] = v
                        new_idx = 0
                        mul = 1
                        # Unrolled conversion
                        new_idx = new_word[0]
                        new_idx += new_word[1] * 3
                        new_idx += new_word[2] * 9
                        new_idx += new_word[3] * 27
                        new_idx += new_word[4] * 81
                        new_idx += new_word[5] * 243
                        new_idx += new_word[6] * 729
                        nbrs.append(new_idx)
                        
            nbrs_flat[idx, :len(nbrs)] = nbrs
            nbrs_len[idx] = len(nbrs)
    else:
        # General case (slower, but functional)
        for idx in range(N):
            word = [0] * n
            tmp = idx
            for i in range(n):
                word[i] = tmp % q
                tmp //= q
            nbrs = [idx]
            # Generate radius 1 neighbors
            for pos in range(n):
                old = word[pos]
                for v in range(q):
                    if v != old:
                        new_word = word.copy()
                        new_word[pos] = v
                        new_idx = 0
                        mul = 1
                        for i in range(n):
                            new_idx += new_word[i] * mul
                            mul *= q
                        nbrs.append(new_idx)
            nbrs_flat[idx, :len(nbrs)] = nbrs
            nbrs_len[idx] = len(nbrs)
            
    return nbrs_flat, nbrs_len

def _run_greedy(nbrs_flat, nbrs_len, N, random_seed=None):
    """Run a single stochastic greedy pass."""
    if random_seed is not None:
        np.random.seed(random_seed)
        
    covered = np.zeros(N, dtype=bool)
    gain = nbrs_len.copy()
    code = []
    
    while True:
        max_gain = np.max(gain)
        if max_gain == 0:
            break
            
        # Randomized selection: pick from candidates with gain close to max
        # Use a small window to allow randomness but keep quality high
        threshold = max_gain - 1 
        candidates = np.where(gain >= threshold)[0]
        
        if len(candidates) == 0:
            idx = np.argmax(gain)
        else:
            idx = np.random.choice(candidates)
            
        code.append(idx)
        
        # Update
        nbrs = nbrs_flat[idx, :nbrs_len[idx]]
        for v in nbrs:
            if not covered[v]:
                covered[v] = True
                nbrs_v = nbrs_flat[v, :nbrs_len[v]]
                gain[nbrs_v] -= 1
                
    return code

def construct(q, n, radius):
    # Precompute neighbors
    nbrs_flat, nbrs_len = _precompute_neighbors(q, n, radius)
    N = q ** n
    
    best_code = None
    best_size = float('inf')
    
    start_time = time.time()
    time_limit = 110.0
    
    seed = 42
    while time.time() - start_time < time_limit:
        code = _run_greedy(nbrs_flat, nbrs_len, N, random_seed=seed)
        if len(code) < best_size:
            best_size = len(code)
            best_code = code.copy()
        seed += 1
        
    # Convert indices to words
    res = []
    for idx in best_code:
        word = [0] * n
        tmp = idx
        for i in range(n):
            word[i] = tmp % q
            tmp //= q
        res.append(word)
        
    return res
# EVOLVE-BLOCK-END
