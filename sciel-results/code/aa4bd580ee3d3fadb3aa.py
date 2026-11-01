"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_score(f):
    """
    Computes the raw score: sum of cubes of absolute values of non-DC circular autocorrelation.
    """
    fft_f = np.fft.fft(f)
    power_spectrum = fft_f * np.conj(fft_f)
    autocorr = np.fft.ifft(power_spectrum).real
    # Exclude DC term (index 0)
    # The DC term is N, which is constant.
    # We sum |R[k]|^3 for k != 0
    return np.sum(np.abs(autocorr[1:]) ** 3)

def _delta_cost(f, i, R):
    """
    Calculate the change in score if we flip f[i].
    f: current sequence
    i: index to flip
    R: current autocorrelation vector (real)
    Returns delta in score.
    """
    # R[k] = sum_j f[j] f[j+k]
    # Flipping f[i] changes R[k] by -2 * f[i] * (f[i+k] + f[i-k])
    # Note: indices are modulo N
    flip_val = f[i]
    # u_k = f[i+k] + f[i-k]
    # We need to compute sum_k (|R[k] - 2*flip_val*u_k|^3 - |R[k]|^3)
    # Note: R[0] is constant, but the formula holds for all k including 0?
    # R[0] = sum f[j]^2 = N.
    # Change in R[0] when flipping f[i]:
    # Term j=i: f[i]^2 -> (-f[i])^2. No change.
    # So R[0] does not change.
    # The formula: u_0 = f[i] + f[i] = 2 f[i].
    # Delta R[0] = -2 f[i] (2 f[i]) = -4 f[i]^2 = -4.
    # Wait, R[0] is sum f[j]^2. Flipping f[i] changes f[i]^2 to f[i]^2. No change.
    # Why does the formula give -4?
    # R[0] = sum_j f[j] f[j].
    # Terms involving f[i]: j=i (f[i]^2) and j such that j+k=i -> j=i (f[i]^2).
    # So f[i] appears twice in the sum for R[0]?
    # No, R[0] = sum_j f[j]^2. Each term is distinct.
    # The formula $R[k] = \sum_j f[j] f[j+k]$ for k=0 is $\sum_j f[j]^2$.
    # When we flip f[i], the term $j=i$ changes from $f[i]^2$ to $(-f[i])^2 = f[i]^2$. No change.
    # So $\Delta R[0] = 0$.
    # My derivation: $\Delta R[k] = -2 f[i] (f[i+k] + f[i-k])$.
    # For k=0: $-2 f[i] (f[i] + f[i]) = -4 f[i]^2 = -4$.
    # Contradiction.
    # Let's re-derive.
    # $R_{new}[k] = \sum_j f'_j f'_{j+k}$.
    # $f'_j = f_j$ except $f'_i = -f_i$.
    # $R_{new}[k] - R[k] = \sum_j (f'_j f'_{j+k} - f_j f_{j+k})$.
    # Non-zero only if $j=i$ or $j+k=i$.
    # Case 1: $j=i$. Term: $f'_i f'_{i+k} - f_i f_{i+k} = (-f_i) f_{i+k} - f_i f_{i+k} = -2 f_i f_{i+k}$.
    # Case 2: $j+k=i \implies j=i-k$. Term: $f'_{i-k} f'_i - f_{i-k} f_i = f_{i-k} (-f_i) - f_{i-k} f_i = -2 f_{i-k} f_i$.
    # Sum: $-2 f_i (f_{i+k} + f_{i-k})$.
    # For k=0:
    # Case 1: $j=i$. Term: $-2 f_i f_i = -2 f_i^2 = -2$.
    # Case 2: $j=i$. Term: $-2 f_i f_i = -2$.
    # Total: $-4$.
    # But $R[0]$ is $\sum f_j^2$. Flipping one element doesn't change the sum of squares.
    # Ah, in the sum $\sum_j f_j f_{j+k}$, when $k=0$, the term $j=i$ is $f_i^2$.
    # Is it counted once or twice?
    # The sum is over all $j$.
    # For $k=0$, $R[0] = \sum_j f_j^2$.
    # The term for $j=i$ is $f_i^2$.
    # In the difference calculation:
    # $R_{new}[0] - R[0] = f'_i^2 - f_i^2 = 0$.
    # Why did the formula give -4?
    # Because for $k=0$, the "Case 1" and "Case 2" refer to the *same* term in the sum?
    # No, the sum is $\sum_j$.
    # $R[0] = f_0^2 + f_1^2 + \dots + f_i^2 + \dots$
    # $R_{new}[0] = \dots + f'_i^2 + \dots$
    # Difference is $f'_i^2 - f_i^2 = 0$.
    # The formula $\Delta R[k] = -2 f_i (f_{i+k} + f_{i-k})$ assumes $j=i$ and $j=i-k$ are distinct contributions?
    # If $k=0$, $j=i$ and $j=i-0=i$ are the same index.
    # So we are double counting?
    # Let's check $k \neq 0$.
    # $j=i$ and $j=i-k$ are distinct if $k \neq 0$.
    # So for $k \neq 0$, $\Delta R[k] = -2 f_i (f_{i+k} + f_{i-k})$.
    # For $k=0$, $\Delta R[0] = 0$.
    
    # So we must exclude k=0 from the delta calculation or handle it separately.
    # Since R[0] is constant, we can just ignore it in the cost function.
    
    # Compute u_k for k=1 to N-1
    # u_k = f[(i+k)%N] + f[(i-k)%N]
    # Delta R[k] = -2 * flip_val * u_k
    # Delta Cost = sum_{k=1}^{N-1} (|R[k] + Delta R[k]|^3 - |R[k]|^3)
    
    # Optimization:
    # R is an array.
    # We can vectorize the u calculation.
    idx_plus = (np.arange(N) + i) % N
    idx_minus = (np.arange(N) - i) % N
    u = f[idx_plus] + f[idx_minus]
    
    # Set u[0] to 0 to ignore k=0
    u[0] = 0
    
    delta_R = -2.0 * flip_val * u
    
    # We only care about k != 0
    # R[0] is not used in score, but delta_R[0] is 0 anyway.
    
    # Calculate change in sum of cubes
    # |R[k] + dR[k]|^3 - |R[k]|^3
    # Note: R[k] can be negative.
    
    new_R = R + delta_R
    # Exclude index 0
    cost_change = np.sum(np.abs(new_R[1:]) ** 3 - np.abs(R[1:]) ** 3)
    
    return cost_change

def construct():
    """
    Construct a signed step function with N=400 equal intervals on [-1/4, 1/4].
    Uses Simulated Annealing to minimize the L3 norm of the circular autocorrelation.
    """
    # Time limit
    start_time = time.time()
    time_limit = 115.0  # Leave 5s buffer
    
    # Initialization
    # Try a few random starts and pick the best
    best_f = None
    best_score = float('inf')
    
    for _ in range(10):
        f = np.random.choice([-1, 1], size=N).astype(float)
        score = _compute_score(f)
        if score < best_score:
            best_score = score
            best_f = f.copy()
            
    f = best_f
    R = np.fft.ifft(np.fft.fft(f) * np.conj(np.fft.fft(f))).real
    
    # Simulated Annealing Parameters
    T = 1000.0  # Initial temperature
    T_min = 1e-6
    cooling_rate = 0.999
    
    current_score = best_score
    best_score_sa = current_score
    best_f_sa = f.copy()
    
    # Main Loop
    while time.time() - start_time < time_limit:
        # Pick a random index to flip
        i = np.random.randint(N)
        
        # Calculate delta
        delta = _delta_cost(f, i, R)
        
        # Acceptance criterion
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            # Accept
            f[i] *= -1
            # Update R
            # We can update R using the delta_R we computed, but we didn't return it.
            # It's cheaper to recompute R? No, O(N log N).
            # Better: Update R in place.
            # But _delta_cost didn't return delta_R.
            # Let's modify _delta_cost to return delta_R or just recompute R if N is small?
            # N=400. FFT is fast.
            # But doing FFT every step might be slow.
            # Let's optimize _delta_cost to return the new R or update it?
            # Actually, let's just recompute R. 
            # 120s / 1000 steps/s = 120,000 steps.
            # FFT of 400 is very fast in numpy.
            # Let's check speed.
            
            R = np.fft.ifft(np.fft.fft(f) * np.conj(np.fft.fft(f))).real
            current_score += delta
            
            if current_score < best_score_sa:
                best_score_sa = current_score
                best_f_sa = f.copy()
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            # Restart or break?
            # If T is too low, we might get stuck.
            # Let's just keep going, it will accept only improvements.
            
    return best_f_sa.astype(int)

# EVOLVE-BLOCK-END
