import numpy as np
import time

N = 100
DEADLINE = 145.0  # 5s margin

def construct():
    start_time = time.time()
    
    # Initialize with Zadoff-Chu like sequence
    # z_n = exp(1j * pi * n^2 / N)
    # s_n = sign(Re(z_n))
    n = np.arange(N)
    z = np.exp(1j * np.pi * n**2 / N)
    s = np.sign(np.real(z))
    
    # Ensure sum is 0
    # Count ones and minus ones
    # We need 50 ones and 50 minus ones
    # If not, flip some bits
    for _ in range(10):
        sum_s = np.sum(s)
        if sum_s == 0:
            break
        # Flip a random bit to reduce sum
        # If sum > 0, flip a 1 to -1
        # If sum < 0, flip a -1 to 1
        if sum_s > 0:
            idxs = np.where(s == 1)[0]
        else:
            idxs = np.where(s == -1)[0]
        if len(idxs) > 0:
            idx = np.random.choice(idxs)
            s[idx] *= -1
    
    # Function to compute A(k) and O(k)
    def compute_A_O(s):
        A = np.zeros(N)
        for k in range(N):
            A[k] = np.sum(s * np.roll(s, k))
        O = N/4.0 - A/4.0
        return A, O
    
    A, O = compute_A_O(s)
    best_max_O = np.max(O)
    best_s = s.copy()
    
    # Local search
    # Precompute neighbors? No, just update A incrementally.
    # A[k] = sum s_j s_{j+k}
    # Flip s_i:
    # A_new[k] = A_old[k] - 2 s_i (s_{i-k} + s_{i+k})
    
    # We will keep A updated.
    # But flipping one bit changes A for all k.
    # O(k) = N/4 - A(k)/4
    # Max O is what we want to minimize.
    
    # To speed up, we can just recompute O from scratch if N is small (100).
    # 100 * 100 = 10000 ops per evaluation.
    # 10000 evaluations in 140s is ~70 evals/sec.
    # We can do much better with incremental updates.
    
    # Let's use incremental updates.
    # A is an array of length N.
    # When we flip s[i], we update A.
    
    current_A = A.copy()
    current_max_O = best_max_O
    
    # Helper to update A when flipping index i
    def flip_and_update(i, s, A):
        # s[i] flips sign
        # A[k] involves s[i] in two ways:
        # 1. j = i, term s[i] s[i+k]
        # 2. j+k = i => j = i-k, term s[i-k] s[i]
        # So A[k] changes by -2 s[i] s[i+k] - 2 s[i-k] s[i]
        # Note: s[i] in the formula is the OLD value
        old_s_i = s[i]
        delta = -2 * old_s_i * (np.roll(s, -i) + np.roll(s, i)) # This is not quite right for all k
        # Let's do it element by element for A
        # A[k] = sum_j s[j] s[j+k]
        # Term with j=i: s[i] s[i+k]
        # Term with j+k=i => j=i-k: s[i-k] s[i]
        # So for each k, A[k] -= 2 * s[i] * s[i+k] + 2 * s[i-k] * s[i]
        # Wait, s[i-k] is the same as s[(i-k)%N]
        
        # Vectorized update for A
        # We need to update A[k] for all k.
        # A[k] depends on s[i] if i is in the sum.
        # s[i] appears in A[k] as s[i] s[i+k] (j=i) and s[i-k] s[i] (j=i-k).
        # So A[k] changes by -2 s[i] s[i+k] - 2 s[i-k] s[i].
        # Let's compute the change for all k at once.
        # term1[k] = s[i] * s[(i+k)%N]
        # term2[k] = s[(i-k)%N] * s[i]
        # delta_A[k] = -2 * (term1[k] + term2[k])
        
        idx1 = (i + np.arange(N)) % N
        idx2 = (i - np.arange(N)) % N
        term1 = old_s_i * s[idx1]
        term2 = s[idx2] * old_s_i
        A -= 2 * (term1 + term2)
        
        # Flip s[i]
        s[i] *= -1

    # Local search loop
    no_improvement_count = 0
    while time.time() < DEADLINE:
        improved = False
        # Try flipping each bit
        # To avoid bias, shuffle indices
        indices = np.random.permutation(N)
        for i in indices:
            if time.time() > DEADLINE:
                break
            
            old_max_O = np.max(N/4.0 - current_A/4.0)
            
            # Flip
            flip_and_update(i, s, current_A)
            
            new_max_O = np.max(N/4.0 - current_A/4.0)
            
            if new_max_O < old_max_O - 1e-10:
                improved = True
                no_improvement_count = 0
                if new_max_O < best_max_O:
                    best_max_O = new_max_O
                    best_s = s.copy()
            else:
                # Revert
                flip_and_update(i, s, current_A)
                no_improvement_count += 1
        
        if not improved:
            # Try random restarts or larger jumps
            if np.random.random() < 0.1:
                # Flip a random pair
                i1, i2 = np.random.choice(N, 2, replace=False)
                flip_and_update(i1, s, current_A)
                flip_and_update(i2, s, current_A)
                new_max_O = np.max(N/4.0 - current_A/4.0)
                if new_max_O < best_max_O:
                    best_max_O = new_max_O
                    best_s = s.copy()
                else:
                    flip_and_update(i2, s, current_A)
                    flip_and_update(i1, s, current_A)
            
            if no_improvement_count > 1000:
                break
                
    # Convert s to h
    h = 0.5 + 0.5 * best_s
    return h
