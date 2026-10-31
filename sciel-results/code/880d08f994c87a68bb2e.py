"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from numpy.fft import fft, ifft
import time

def _autoconv(x):
    """Compute linear autoconvolution via FFT."""
    n = len(x)
    size = 2 * n - 1
    size2 = 1
    while size2 < size:
        size2 *= 2
    F = fft(x, size2)
    A = ifft(F * np.conj(F)).real
    return A[:size]

def _score(x):
    """Compute max(|autoconv|) / (sum(x))^2. Lower is better."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.inf
    A = _autoconv(x)
    m = np.max(np.abs(A))
    return m / (s * s)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    rng = np.random.default_rng(42)
    
    # 1. Generate diverse initializations
    # The key insight is that "all ones" or simple random signs get stuck in local minima 
    # where the autoconvolution peak is dominated by the DC component (sum^2).
    # Chirp and block patterns provide structure that allows sign flips to reduce 
    # the high-frequency peaks of the autoconvolution without immediately destroying 
    # the sum magnitude too drastically.
    
    inits = []
    
    # Init 1: Chirp (quadratic phase)
    x_chirp = np.exp(1j * (np.pi * np.arange(N)**2 / N)).real
    x_chirp[x_chirp >= 0] = 1.0
    x_chirp[x_chirp < 0] = -1.0
    inits.append(x_chirp)
    
    # Init 2: Block structure (alternating large blocks)
    x_block = np.zeros(N)
    block_size = 16
    for i in range(0, N, block_size * 2):
        x_block[i:i+block_size] = 1.0
        x_block[i+block_size:i+2*block_size] = -1.0
    # Ensure length N
    if len(x_block) < N:
        x_block = np.concatenate([x_block, np.ones(N - len(x_block))])
    x_block = x_block[:N]
    inits.append(x_block)
    
    # Init 3: Random
    inits.append(rng.choice([-1.0, 1.0], size=N))
    
    # Init 4: Low frequency sinusoid
    x_sine = np.sign(np.sin(2 * np.pi * np.arange(N) * 3 / N))
    inits.append(x_sine)
    
    best_x = None
    best_score = np.inf
    
    # 2. Simulated Annealing
    # Objective: Minimize max(|autoconv|) / (sum(x))^2
    # Move: Flip a random sign at a random index.
    # Delta calculation:
    # Flipping x[i] from s to -s changes x by -2*s at index i.
    # This is a rank-1 update to the autoconvolution? No, autoconv is quadratic.
    # A[i] = sum_k x[k] x[i-k]
    # Delta A = -2*s * (x * e_i) + (-2*s * e_i) * x + 4*s^2 * e_i * e_i? 
    # Actually, it's easier to just recompute the score if we don't optimize the delta.
    # However, recomputing FFT (O(N log N)) for each step might be slow.
    # N=400. FFT size 1024. ~1000 ops. 10000 steps = 10^7 ops. Python loop overhead dominates.
    # We need a faster delta or fewer steps.
    
    # Let's estimate the number of steps we can do.
    # If we do 100 steps per second (conservative due to Python overhead in _score), 
    # we have ~14000 steps.
    # We can afford to recompute the full autoconvolution for each candidate if we keep the overhead low.
    
    # To speed up, we can precompute the FFT of the current x? 
    # No, flipping a bit changes the FFT significantly.
    # However, we can compute the change in autoconvolution analytically?
    # Let x' = x - 2*s_i * e_i.
    # A' = x' * x' = (x - 2 s e_i) * (x - 2 s e_i) = x*x - 4s (x*e_i + e_i*x) + 4s^2 e_i*e_i
    # x*e_i is the shift of x. e_i*x is the shift of x.
    # So A'[j] = A[j] - 4 s_i (x[j-i] + x[i-j]) + 4 s_i^2 delta_{j,i} (if linear conv defined appropriately)
    # Wait, linear autoconvolution A[j] = sum_{k} x[k] x[j-k].
    # If we flip x[i], let delta_k = -2 s_i if k=i else 0.
    # A'[j] = sum_k (x[k]+delta_k)(x[j-k]+delta_{j-k})
    # = A[j] + sum_k delta_k x[j-k] + sum_k x[k] delta_{j-k} + sum_k delta_k delta_{j-k}
    # Term 1: -2 s_i x[j-i]
    # Term 2: -2 s_i x[i-j]  (Note: x[k] is non-zero only if k is in range. x[i-j] is valid if i-j in range)
    # Term 3: 4 s_i^2 if j-i == i => j=2i. And also if i == j-i => j=2i. 
    # So the change is local!
    # A'[j] = A[j] - 2 s_i (x[j-i] + x[i-j]) + 4 s_i^2 (1 if j==2i else 0)
    # This allows O(N) update for the autoconvolution vector instead of O(N log N) FFT.
    # Even better, we only need to check if the *max absolute value* changes significantly.
    # But we need the exact max to be sure.
    
    # Let's implement the O(N) delta update for speed.
    
    def get_delta_autoconv(x, A, i, s):
        """
        Compute the change in autoconvolution if x[i] is flipped from s to -s.
        Returns the new A vector.
        """
        n = len(x)
        size = 2 * n - 1
        A_new = A.copy()
        
        # s is the current value at x[i]. We flip it to -s.
        # delta = -2*s
        # Change in A[j] = -2*s * (x[j-i] + x[i-j]) + 4*s^2 * (1 if j==2i else 0)
        # Note: x[k] is 0 for k < 0 or k >= n.
        
        # Iterate over j in [0, size-1]
        # x[j-i] is non-zero if 0 <= j-i < n  => i <= j < i+n
        # x[i-j] is non-zero if 0 <= i-j < n  => i-n < j <= i
        
        # Range of j affected is [max(0, i-n+1), min(size-1, i+n-1)]
        # Since size = 2n-1, and i in [0, n-1]:
        # j_min = max(0, i - n + 1)
        # j_max = min(size - 1, i + n - 1)
        
        # Actually, simpler:
        # x[j-i] contributes for j in [i, i+n-1]
        # x[i-j] contributes for j in [i-n+1, i]
        
        # We can just loop j from 0 to size-1 and check bounds, or slice.
        # Slicing is faster in numpy.
        
        # Part 1: -2*s * x[j-i]
        # Valid j: i <= j < i+n
        j_start1 = i
        j_end1 = i + n
        # Clip to A size
        j_start1 = max(0, j_start1)
        j_end1 = min(size, j_end1)
        
        if j_start1 < j_end1:
            # x indices: j - i => [j_start1 - i, j_end1 - i - 1]
            idx_start = j_start1 - i
            idx_end = j_end1 - i
            # x is length n. idx must be in [0, n-1].
            # Since j in [i, i+n-1], j-i in [0, n-1]. Correct.
            A_new[j_start1:j_end1] -= 2.0 * s * x[idx_start:idx_end]
            
        # Part 2: -2*s * x[i-j]
        # Valid j: i-n+1 <= j <= i
        j_start2 = i - n + 1
        j_end2 = i + 1
        # Clip
        j_start2 = max(0, j_start2)
        j_end2 = min(size, j_end2)
        
        if j_start2 < j_end2:
            # x indices: i - j.
            # j goes from j_start2 to j_end2-1.
            # i - j goes from i - (j_end2-1) to i - j_start2.
            # Let's just compute the vector of x[i-j] for these j's.
            js = np.arange(j_start2, j_end2)
            idxs = i - js
            # Filter valid idxs? 
            # By construction of j range, idxs should be in [0, n-1].
            # j >= i-n+1 => i-j <= n-1.
            # j <= i => i-j >= 0.
            vals = x[idxs]
            A_new[j_start2:j_end2] -= 2.0 * s * vals
            
        # Part 3: +4*s^2 at j=2i
        j_center = 2 * i
        if 0 <= j_center < size:
            A_new[j_center] += 4.0 * s * s
            
        return A_new

    # We need to maintain A and s (sum) efficiently.
    # Sum change: s_new = s - 2*s_i
    
    # Annealing parameters
    T_start = 1.0
    T_end = 0.01
    alpha = 0.995 # cooling rate
    
    for x_init in inits:
        x = x_init.copy()
        s = np.sum(x)
        A = _autoconv(x)
        current_score = _score(x)
        
        if current_score < best_score:
            best_score = current_score
            best_x = x.copy()
            
        T = T_start
        
        # Number of steps per initialization
        # Let's aim for ~3000 steps per init to stay within time.
        # 4 inits * 3000 = 12000 steps.
        # Each step: O(N) delta update + O(N) max abs check? 
        # np.max(np.abs(A)) is O(N).
        # Total O(N^2) per step? No, O(N) per step.
        # N=400. 400 ops per step. 12000 steps = 5e6 ops. Very fast in NumPy.
        
        steps_per_init = 3000
        
        for step in range(steps_per_init):
            if time.time() > deadline:
                break
                
            # Pick random index to flip
            i = rng.integers(0, N)
            s_i = x[i]
            
            # Calculate delta
            # New sum
            s_new = s - 2.0 * s_i
            
            # Check if sum is too small (score inf)
            if abs(s_new) < 1e-12:
                # Reject this move
                T *= alpha
                continue
                
            # Calculate new A
            # We don't want to allocate a new array every time if possible, but A is small (799).
            # Allocation is fine.
            A_new = get_delta_autoconv(x, A, i, s_i)
            
            # Calculate new score
            m_new = np.max(np.abs(A_new))
            score_new = m_new / (s_new * s_new)
            
            # Acceptance probability
            if score_new < current_score:
                # Accept
                x[i] = -s_i
                s = s_new
                A = A_new
                current_score = score_new
            else:
                # Metropolis criterion
                # Delta E = score_new - current_score (positive)
                delta_E = score_new - current_score
                # Prob = exp(-delta_E / T)
                # If T is small, prob is 0.
                if T > 0:
                    # Avoid overflow
                    if delta_E / T < 50:
                        p = np.exp(-delta_E / T)
                    else:
                        p = 0.0
                    
                    if rng.random() < p:
                        x[i] = -s_i
                        s = s_new
                        A = A_new
                        current_score = score_new
            
            # Update best
            if current_score < best_score:
                best_score = current_score
                best_x = x.copy()
                
            # Cool down
            T *= alpha
            if T < T_end:
                T = T_end
                # Optionally break early if converged? 
                # Let's just finish the steps to explore more.
        
        if time.time() > deadline:
            break

    # Final check
    if best_x is None:
        # Fallback to random
        return rng.choice([-1.0, 1.0], size=N)
        
    return best_x

# EVOLVE-BLOCK-END
