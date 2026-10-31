"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.fft import fft, ifft

N = 400
TIME_LIMIT = 110.0  # Safety margin before 120s kill

# EVOLVE-BLOCK-START

def _compute_score(x):
    """
    Compute the objective value: max |autoconv(x)| / (sum(x))^2
    Lower is better.
    """
    if x is None:
        return np.inf
    
    # Ensure x is a numpy array
    x = np.asarray(x)
    
    # Compute autoconvolution using FFT
    # Length of linear convolution is 2*N - 1
    n_fft = 1 << (2 * N - 1).bit_length()
    
    # Pad x to n_fft
    x_padded = np.zeros(n_fft)
    x_padded[:N] = x
    
    # FFT, square, inverse FFT
    X = fft(x_padded)
    X2 = X * X
    autoconv = np.real(ifft(X2))[:2*N - 1]
    
    # Maximum absolute value
    max_abs = np.max(np.abs(autoconv))
    
    # Squared integral (sum)
    s = np.sum(x)
    s_sq = s * s
    
    if s_sq == 0:
        return np.inf
        
    return max_abs / s_sq


def _thue_morse(n):
    """Generate Thue-Morse sequence of length n."""
    # Thue-Morse: t(n) = parity of number of 1s in binary representation of n
    # t(n) = t(n >> 1) ^ (n & 1)
    # Or simply: t(n) = bin(n).count('1') % 2
    # Vectorized approach:
    idx = np.arange(n)
    # Count set bits
    # Using bit operations
    # t = 0
    # for i in range(8):
    #     t ^= (idx >> i) & 1
    # return t % 2
    
    # Faster vectorized:
    x = idx.copy()
    x = x ^ (x >> 1)
    x = x ^ (x >> 2)
    x = x ^ (x >> 4)
    x = x ^ (x >> 8)
    x = x ^ (x >> 16)
    x = x ^ (x >> 32)
    # Now x has the parity in the LSB
    return (x & 1).astype(np.int8)


def _construct_thue_morse_signed():
    """
    Construct a signed sequence based on Thue-Morse.
    Map 0 -> +1, 1 -> -1.
    """
    tm = _thue_morse(N)
    # Map 0->1, 1->-1
    x = 1 - 2 * tm
    return x


def _local_search(x, max_iters=5000, time_limit=110.0):
    """
    Greedy local search to minimize the score.
    """
    start_time = time.time()
    best_x = x.copy()
    best_score = _compute_score(best_x)
    
    # Pre-compute current autoconv for quick updates?
    # Updating autoconv for a single flip is O(N).
    # Score update is O(1) if we track max, but max can change anywhere.
    # However, flipping x[k] changes autoconv at indices k and N-1-k (roughly).
    # Actually, autoconv[k] = sum x[i]x[i+k].
    # Flipping x[j] changes autoconv[k] for all k where i=j or i+k=j.
    # This is complex to update incrementally efficiently without O(N) work per flip.
    # Given N=400, O(N) per flip is 400 ops. 5000 iters * 400 = 2M ops. Very fast.
    
    for _ in range(max_iters):
        if time.time() - start_time > time_limit:
            break
            
        improved = False
        # Randomly pick a few indices to try flipping
        # Or try all indices? 400 flips * 400 cost = 160k per full pass.
        # Let's do full passes.
        
        # To make it faster, we can just try random flips
        # Or we can do a sweep.
        
        # Let's try a random restart / random flip strategy
        # Pick a random index
        idx = np.random.randint(0, N)
        
        # Flip it
        old_val = best_x[idx]
        best_x[idx] = -old_val
        
        new_score = _compute_score(best_x)
        
        if new_score < best_score:
            best_score = new_score
            improved = True
        else:
            # Revert
            best_x[idx] = old_val
            
        # If no improvement in a long time, maybe random restart or perturb?
        # For now, just keep going.
        
    return best_x, best_score


def construct():
    """
    Construct the signed step heights.
    Returns a sequence of signed step heights (numpy array).
    """
    start_time = time.time()
    
    # 1. Generate initial Thue-Morse sequence
    x_tm = _construct_thue_morse_signed()
    
    # Check score
    score_tm = _compute_score(x_tm)
    
    # 2. Try some other simple structures?
    # Alternating: 1, -1, 1, -1...
    x_alt = np.ones(N)
    x_alt[1::2] = -1
    score_alt = _compute_score(x_alt)
    
    # 3. Random initializations
    best_x = x_tm
    best_score = score_tm
    
    if score_alt < best_score:
        best_x = x_alt
        best_score = score_alt
    
    # Try a few random initializations
    for _ in range(5):
        if time.time() - start_time > TIME_LIMIT - 10:
            break
        x_rand = np.random.choice([-1, 1], size=N)
        score_rand = _compute_score(x_rand)
        if score_rand < best_score:
            best_x = x_rand
            best_score = score_rand
            
    # 4. Local search from the best candidate
    remaining_time = TIME_LIMIT - (time.time() - start_time)
    if remaining_time > 5:
        # Estimate max iterations
        # Each iter takes ~100us? 400 FFT ops is fast.
        # Let's just let it run until time limit.
        best_x, best_score = _local_search(best_x, max_iters=100000, time_limit=remaining_time)
        
    # Final check
    final_score = _compute_score(best_x)
    
    # If time is running out, return what we have
    return best_x

# EVOLVE-BLOCK-END
