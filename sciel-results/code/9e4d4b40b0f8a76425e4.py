def construct():
    import numpy as np
    from math import gcd
    import random
    
    n = 128
    # 1. Precompute GCD matrix
    gcd_mat = np.zeros((n+1, n+1), dtype=np.uint8)
    for i in range(1, n+1):
        for j in range(1, n+1):
            gcd_mat[i, j] = gcd(i, j)
            
    # 2. Precompute Bad Triple Tensor
    # bad[i, j, k] = 1 if i,j,k distinct and gcd(i,j)==gcd(i,k)==gcd(j,k)
    # We can compute this efficiently.
    # For each i, j, k? 128^3 is 2M. 
    # Looping in Python for 2M items is slow (~1s). 
    # We can vectorize this too.
    
    # Let's use vectorized logic for bad tensor
    # GCDs are small (<=128).
    # bad[i,j,k] = (gcd_mat[i,j] == gcd_mat[i,k]) & (gcd_mat[i,j] == gcd_mat[j,k]) & (i != j) & (i != k) & (j != k)
    
    # Create index arrays
    idx = np.arange(1, n+1)
    # This will create 3D arrays of size 128x128x128
    # Memory: 128^3 * 1 byte * 3 (for gcds) + bools. ~6MB. OK.
    
    # To save memory/time, we can compute bad on the fly in the check?
    # No, precomputation is better.
    
    # Vectorized Bad Tensor Creation
    # i, j, k ranges 1..128
    # We'll use broadcasting.
    # g_ij = gcd_mat[i, j]
    # g_ik = gcd_mat[i, k]
    # g_jk = gcd_mat[j, k]
    
    # This is tricky with standard broadcasting for 3 indices without explicit loops or einsum.
    # Let's just use a loop for precomputation, it's one-time cost.
    # 2M iterations in Python is ~1-2 seconds. Acceptable if total time < 60s.
    
    bad_tensor = np.zeros((n+1, n+1, n+1), dtype=np.uint8)
    for i in range(1, n+1):
        for j in range(1, n+1):
            if i == j: continue
            for k in range(1, n+1):
                if i == k or j == k: continue
                if gcd_mat[i, j] == gcd_mat[i, k] == gcd_mat[j, k]:
                    bad_tensor[i, j, k] = 1
                    
    # 3. Define validity and score functions using NumPy
    def get_bad_count(s_bool):
        # s_bool is 1D array of length n+1, index 0 unused
        # Sum bad_tensor[i,j,k] where s_bool[i]=s_bool[j]=s_bool[k]=1
        # Use einsum
        return np.einsum('i,j,k,ijk->', s_bool, s_bool, s_bool, bad_tensor)
        
    def is_valid(s_bool):
        return get_bad_count(s_bool) == 0

    # 4. Greedy Initialization
    random.seed(42)
    s_bool = np.zeros(n+1, dtype=np.uint8)
    # Try to add elements in random order
    candidates = list(range(1, n+1))
    random.shuffle(candidates)
    for x in candidates:
        s_bool[x] = 1
        if not is_valid(s_bool):
            s_bool[x] = 0
        # Optional: if valid, keep. If not, remove.
        # This is a simple greedy.
        
    best_s = s_bool.copy()
    best_size = np.sum(best_s)
    
    # 5. Simulated Annealing
    temp = 5.0
    cooling = 0.999
    min_temp = 0.01
    max_iter = 20000 # Can be high because checks are fast
    
    current_s = best_s.copy()
    current_score = len([1 for i in range(1,n+1) if current_s[i]]) # Size
    # Note: We only care about valid sets for the "best" tracking.
    # For SA, we allow invalid sets to escape local optima.
    # Score = Size - 100 * BadCount
    
    def calc_score(s):
        size = np.sum(s)
        bad = get_bad_count(s)
        return size - 100 * bad
        
    current_score = calc_score(current_s)
    
    for _ in range(max_iter):
        if temp < min_temp:
            break
            
        # Random Move
        move_type = random.choice(['add', 'remove', 'swap'])
        new_s = current_s.copy()
        
        if move_type == 'add':
            # Pick random 0
            zeros = np.where(current_s == 0)[0]
            if len(zeros) == 0: continue
            x = random.choice(zeros)
            if x == 0: continue
            new_s[x] = 1
        elif move_type == 'remove':
            ones = np.where(current_s == 1)[0]
            if len(ones) == 0: continue
            x = random.choice(ones)
            new_s[x] = 0
        else: # swap
            ones = np.where(current_s == 1)[0]
            zeros = np.where(current_s == 0)[0]
            if len(ones) == 0 or len(zeros) == 0: continue
            u = random.choice(ones)
            v = random.choice(zeros)
            if v == 0: continue
            new_s[u] = 0
            new_s[v] = 1
            
        new_score = calc_score(new_s)
        delta = new_score - current_score
        
        if delta > 0:
            current_s = new_s
            current_score = new_score
            # Check if valid and better than best
            if is_valid(current_s):
                size = np.sum(current_s)
                if size > best_size:
                    best_size = size
                    best_s = current_s.copy()
        else:
            # Metropolis
            if random.random() < np.exp(delta / temp):
                current_s = new_s
                current_score = new_score
                
        temp *= cooling
        
    # Final check: ensure best_s is valid
    if not is_valid(best_s):
        # Fallback to a simple valid set
        # e.g. all primes? No, primes form bad triples.
        # Just return a small valid set
        fallback = np.zeros(n+1, dtype=np.uint8)
        fallback[1] = 1
        fallback[2] = 1
        return list(np.where(fallback == 1)[0])
        
    return list(np.where(best_s == 1)[0])
