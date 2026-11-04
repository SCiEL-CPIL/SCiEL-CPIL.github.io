"""
Strategy: Direct Binary Optimization via Simulated Annealing + Greedy Polish.

The current program optimizes continuous variables and projects to signs. 
This strategy treats the problem as a binary search problem directly.
We use Simulated Annealing (SA) to escape local minima in the sign space,
using a differentiable surrogate for the peak sidelobe metric (LogSumExp of 
squared magnitudes of the autoconvolution side-lobes). 
Finally, we apply a greedy bit-flip polish to minimize the exact metric.
"""

import numpy as np
import time
import scipy.optimize

N = 400

# EVOLVE-BLOCK-START

def _autoconv_full(x):
    """Full linear autoconvolution of x (length 2N-1)."""
    n = len(x)
    size = 1 << (2 * n - 1).bit_length()
    F = np.fft.rfft(x, n=size)
    conv = np.fft.irfft(F * F, n=size)
    return conv[:2 * n - 1]

def _score_offdiag(x):
    """
    Compute the metric: max(|autoconv|) for k != 0 / (sum(x))^2
    We want to minimize this.
    """
    integral = np.sum(x)
    if abs(integral) < 1e-12:
        return 1e12
    conv = _autoconv_full(x)
    if len(conv) <= 1:
        return 0.0
    max_abs_offdiag = np.max(np.abs(conv[1:]))
    return max_abs_offdiag / (integral * integral)

def _sa_cost(x, beta):
    """
    Surrogate cost for Simulated Annealing.
    Uses LogSumExp to approximate the max of absolute autoconv side-lobes.
    This is differentiable and smoother than the exact max.
    We normalize by sum(x)^2 to keep the scale invariant property in mind, 
    but for SA, we can just minimize the raw sidelobe energy/peak 
    if we assume sum is roughly fixed or handle scaling carefully.
    
    Actually, since the score is scale-invariant, we can fix sum(x) ~ 0 
    or just minimize the raw peak sidelobe level (PSL) of the normalized signal.
    Let's normalize x to have unit L2 norm or unit sum?
    If sum(x) is small, the score blows up. 
    For binary sequences, sum(x) = (#plus - #minus).
    To keep score low, we want |sum(x)| large and max(|conv[1:]|) small.
    
    Let's define the SA cost as:
    C(x) = max(|conv[1:]|) / (sum(x)^2 + epsilon)
    But max is not differentiable.
    Use LogSumExp(beta * |conv[1:]|) as a proxy for max(|conv[1:]|).
    
    Cost = LSE / (sum(x)^2 + 1e-6)
    """
    s = np.sum(x)
    if abs(s) < 1e-6:
        return 1e10
        
    # Compute autoconv
    n = len(x)
    size = 1 << (2 * n - 1).bit_length()
    F = np.fft.rfft(x, n=size)
    # Power spectrum
    P = np.abs(F)**2
    # Inverse FFT to get autoconv
    conv = np.fft.irfft(P, n=size)
    conv = conv[:2 * n - 1]
    
    # Side lobes
    side_lobs = np.abs(conv[1:])
    
    # LogSumExp approximation of max
    m = np.max(side_lobs)
    if m < 1e-15:
        lse = 0.0
    else:
        lse = m + np.log(np.sum(np.exp(beta * (side_lobs - m)))) / beta
        
    return lse / (s * s)

def _greedy_polish(x, deadline):
    """
    Greedy local search: flip bit i if it reduces the exact score.
    Repeat until no improvement or deadline.
    """
    n = len(x)
    current = x.copy()
    current_score = _score_offdiag(current)
    best_score = current_score
    best_x = current.copy()
    
    size = 1 << (2 * n - 1).bit_length()
    
    # Pre-compute FFT size
    no_improve = 0
    max_no_improve = 2000
    
    # We will iterate through indices in a random or deterministic order.
    # Random order helps avoid cycles.
    indices = np.arange(n)
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Shuffle indices occasionally to break symmetry
        if no_improve > 100:
            np.random.shuffle(indices)
            
        improved = False
        for i in indices:
            if time.time() > deadline - 1.0:
                break
                
            old_val = current[i]
            current[i] = -old_val
            
            s = np.sum(current)
            if abs(s) < 1e-12:
                new_score = np.inf
            else:
                H = np.fft.rfft(current, size)
                ac = np.fft.irfft(H * H, size)[:2 * n - 1]
                max_abs = np.max(np.abs(ac[1:]))
                new_score = max_abs / (s * s)
                
            if new_score < current_score:
                current_score = new_score
                no_improve = 0
                improved = True
                if new_score < best_score:
                    best_score = new_score
                    best_x = current.copy()
            else:
                current[i] = old_val
                no_improve += 1
                
        if not improved:
            # If no single flip improved, try double flips? Or break?
            # For now, break if no single flip works, as double flips are expensive.
            # But we might be stuck in a local min. 
            # Let's allow a few more rounds of shuffling before giving up on this state.
            if no_improve > 1000:
                break

    return best_x, best_score

def construct():
    t0 = time.time()
    # Safety margin: 150s limit. Use 145s.
    deadline = t0 + 145.0
    
    rng = np.random.default_rng(42)
    
    # Initialize with a balanced sequence
    # Start with random signs
    x = rng.choice([-1.0, 1.0], size=N)
    
    # Ensure sum is not zero (though for N=400, it's unlikely to be exactly 0, 
    # but could be small)
    # If sum is 0, score is inf. 
    # We want |sum| to be large.
    # Let's bias the initial sequence or check.
    # Actually, for random signs, sum ~ N(0, sqrt(N)). 
    # If sum is small, we might want to flip a few bits to increase |sum|.
    
    # Phase 1: Simulated Annealing
    # We want to minimize _sa_cost.
    # SA parameters
    T_start = 1.0
    T_end = 1e-4
    alpha = 0.95
    T = T_start
    
    best_x_sa = x.copy()
    best_score_sa = _score_offdiag(best_x_sa)
    current_x = x.copy()
    current_cost = _sa_cost(current_x, beta=50.0)
    
    # We need to estimate the scale of the cost.
    # _sa_cost is roughly PSL / S^2.
    # PSL ~ O(N), S^2 ~ O(N^2). So cost ~ O(1/N).
    # Let's check the initial cost.
    
    # To make SA effective, we need to accept moves with probability exp(-delta/T).
    # Delta is the change in cost.
    
    # Number of steps
    # Each step involves an FFT (O(N log N)).
    # N=400. FFT is fast.
    # Let's run for a fixed number of steps or time.
    
    steps = 0
    max_steps = 20000 # Adjust based on timing
    
    # We will monitor time.
    while time.time() < deadline - 30.0: # Leave 30s for polish
        # Choose a random index to flip
        i = int(rng.integers(0, N))
        old_val = current_x[i]
        current_x[i] = -old_val
        
        # Calculate new cost
        new_cost = _sa_cost(current_x, beta=50.0)
        
        delta = new_cost - current_cost
        
        if delta < 0 or rng.random() < np.exp(-delta / T):
            # Accept move
            current_cost = new_cost
            
            # Update best
            # Check exact score occasionally to track best exact solution
            if steps % 100 == 0:
                exact_score = _score_offdiag(current_x)
                if exact_score < best_score_sa:
                    best_score_sa = exact_score
                    best_x_sa = current_x.copy()
        else:
            # Reject move
            current_x[i] = old_val
            
        # Cool down
        T *= alpha
        if T < T_end:
            T = T_end
            # If we hit T_end, we might want to restart with higher T? 
            # Or just switch to polish.
            # Let's break if T is very low and no improvement recently.
            
        steps += 1
        
        # Early stop if T is very low and we've had many steps
        if T < 1e-3 and steps > 5000:
            break

    # Phase 2: Greedy Polish from the best SA result
    best_x_final, best_score_final = _greedy_polish(best_x_sa, deadline - 5.0)
    
    # Also try polishing the initial random sequence? 
    # Maybe SA got stuck. 
    # Let's try polishing a few other candidates if time permits.
    
    # Generate a few more diverse candidates
    candidates = [best_x_final]
    
    # 1. Hanning window projected
    h = np.hanning(N)
    h_sign = np.sign(h)
    candidates.append(h_sign)
    
    # 2. M-sequence
    def _m_sequence(n, k):
        polys = {
            6: [5, 3, 0],
            7: [6, 5, 3, 0],
            8: [7, 5, 3, 2, 0],
        }
        if k not in polys:
            return np.sign(np.sin(np.arange(n) * 2 * np.pi / n * 3))[:n]
        
        taps = polys[k]
        state = [0] * k
        state[0] = 1
        seq = []
        for _ in range(n):
            fb = 0
            for t in taps:
                fb ^= state[t]
            for i in range(k - 1, 0, -1):
                state[i] = state[i - 1]
            state[0] = fb
            seq.append(1.0 if state[0] else -1.0)
        return np.array(seq, dtype=np.float64)
        
    candidates.append(_m_sequence(N, 7))
    candidates.append(_m_sequence(N, 8))
    
    # Polish them
    for h in candidates[1:]:
        if time.time() > deadline - 10.0:
            break
        h_polished, score_polished = _greedy_polish(h, deadline - 10.0)
        if score_polished < best_score_final:
            best_score_final = score_polished
            best_x_final = h_polished
            
    # Final polish on the best
    if time.time() < deadline - 1.0:
        best_x_final, best_score_final = _greedy_polish(best_x_final, deadline - 1.0)
        
    return best_x_final

# EVOLVE-BLOCK-END
