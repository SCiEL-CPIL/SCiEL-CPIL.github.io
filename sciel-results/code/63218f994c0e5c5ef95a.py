"""
Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Strategy: Simulated Annealing with discrete quantization.
Instead of continuous gradient descent, we discretize the height values 
to a fine grid and use a stochastic global optimization method. 
This avoids local minima of smooth approximations and directly 
optimizes the non-smooth max objective.
"""
import time
import numpy as np
import random

N = 100
_DEADLINE = time.time() + 145  # Safety margin before 150s kill

# Quantization levels
# We need sum(h) = N/2 = 50.
# If we use 256 levels, values are k/255.
# Let's use a slightly finer grid if time permits, but 256 is a good start.
Q_LEVELS = 256
Q_STEP = 1.0 / (Q_LEVELS - 1)

def _linear_overlap_values(h):
    """
    Compute the linear overlap values C_k for k=0..N-1.
    C_k = (2/N) * (N/2 - sum_{i=k}^{N-1} h_i h_{i-k})
    Vectorized using FFT for speed.
    """
    n = len(h)
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs

def _score(h):
    """Compute the maximum linear overlap."""
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))

def _fix_sum_discrete(h, target_sum):
    """
    Adjust discrete h to have sum exactly target_sum (within float precision).
    Since we are working with discrete steps, we adjust by moving mass 
    between elements while keeping them on the grid if possible, 
    or just projecting to the closest feasible continuous point 
    and then re-quantizing? 
    Actually, for the annealing, it's better to work in continuous space 
    but snap to grid for evaluation? No, let's work in integer indices 
    for the annealing to ensure exact sum constraints are easier to handle 
    via swaps.
    
    Let's represent h as integer indices q_i in [0, Q_LEVELS-1].
    Sum constraint: sum(q_i) = Target_Q.
    Target_Q = target_sum / Q_STEP.
    If Target_Q is not integer, we have a problem.
    N/2 = 50. Q_STEP = 1/255. 50 * 255 = 12750. Integer. Good.
    """
    pass

def construct():
    t0 = time.time()
    
    # Target sum in continuous space
    target_sum = N / 2.0
    
    # Target sum in discrete index space
    # q_i = h_i / Q_STEP
    # sum(h_i) = Q_STEP * sum(q_i) = target_sum
    # sum(q_i) = target_sum / Q_STEP
    target_q_sum = int(round(target_sum / Q_STEP))
    
    # Initial state: Random assignment of q_i such that sum is target_q_sum
    # We can start with a structured pattern for better initial score
    # e.g., half 0s and half 1s (in q space: 0 and Q_LEVELS-1)
    # Or a cosine wave mapped to q space.
    
    def _generate_initial_state():
        # Cosine wave mapped to discrete indices
        i = np.arange(N)
        # h_i = 0.5 + 0.5 * cos(2*pi*i/N)
        # q_i = h_i / Q_STEP
        h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
        q_cos = np.round(h_cos / Q_STEP).astype(int)
        
        # Adjust sum to match target_q_sum
        current_sum = np.sum(q_cos)
        diff = target_q_sum - current_sum
        
        # Adjust by changing values of elements
        # If diff > 0, increase some q_i. If diff < 0, decrease.
        # We pick elements randomly to adjust.
        indices = np.arange(N)
        random.shuffle(indices)
        
        idx_iter = 0
        while diff != 0 and idx_iter < N * 10:
            idx = indices[idx_iter % N]
            idx_iter += 1
            if diff > 0:
                if q_cos[idx] < Q_LEVELS - 1:
                    q_cos[idx] += 1
                    diff -= 1
            else:
                if q_cos[idx] > 0:
                    q_cos[idx] -= 1
                    diff += 1
                    
        return q_cos

    best_q = _generate_initial_state()
    best_h = best_q * Q_STEP
    best_score = _score(best_h)
    
    # Simulated Annealing
    # State is q (integer array)
    # Move: Pick two random indices i, j. 
    # To maintain sum, we must move mass from one to another? 
    # No, simple swap of values q_i, q_j maintains sum.
    # But swap might not explore the space well if values are similar.
    # Better move: Pick i, j. Increase q_i by 1, decrease q_j by 1.
    # This maintains sum.
    # Constraint: 0 <= q_i < Q_LEVELS.
    
    current_q = best_q.copy()
    current_h = current_q * Q_STEP
    current_score = _score(current_h)
    
    T = 0.1  # Initial temperature
    T_min = 1e-6
    alpha = 0.9995  # Cooling rate
    
    # Precompute overlap for current state to speed up? 
    # FFT is fast enough for N=100. 
    # Let's just recompute.
    
    while time.time() < _DEADLINE - 5:
        if T < T_min:
            break
            
        # Generate move
        i = random.randint(0, N - 1)
        j = random.randint(0, N - 1)
        
        if i == j:
            continue
            
        # Try to move 1 unit from i to j (or vice versa)
        # We need to check bounds.
        # Let's pick a random direction
        if random.random() < 0.5:
            # Increase i, decrease j
            if current_q[i] < Q_LEVELS - 1 and current_q[j] > 0:
                old_i = current_q[i]
                old_j = current_q[j]
                current_q[i] += 1
                current_q[j] -= 1
                
                new_h = current_q * Q_STEP
                new_score = _score(new_h)
                
                # Acceptance criterion
                if new_score < current_score or random.random() < np.exp((current_score - new_score) / T):
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_q = current_q.copy()
                        best_h = current_q * Q_STEP
                else:
                    # Reject
                    current_q[i] = old_i
                    current_q[j] = old_j
        else:
            # Decrease i, increase j
            if current_q[i] > 0 and current_q[j] < Q_LEVELS - 1:
                old_i = current_q[i]
                old_j = current_q[j]
                current_q[i] -= 1
                current_q[j] += 1
                
                new_h = current_q * Q_STEP
                new_score = _score(new_h)
                
                if new_score < current_score or random.random() < np.exp((current_score - new_score) / T):
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_q = current_q.copy()
                        best_h = current_q * Q_STEP
                else:
                    current_q[i] = old_i
                    current_q[j] = old_j
                    
        T *= alpha
        
        # If we stagnate, we might want to restart or increase T?
        # For now, simple cooling.
        
    # Final refinement: Local search on the best continuous point?
    # The discrete solution is quantized. We can try to "de-quantize" 
    # by doing a few steps of gradient descent on the continuous values 
    # starting from the best discrete solution.
    
    if best_h is not None and time.time() < _DEADLINE - 10:
        h_ref = best_h.copy()
        # Simple gradient step?
        # Let's just return the discrete solution scaled back.
        # It's valid and has exact sum (within float precision of multiplication).
        
    # Ensure sum is exactly N/2 in floating point
    # sum(best_h) might be slightly off due to Q_STEP * int.
    # Q_STEP = 1/255. 
    # sum(q_i) = 12750.
    # sum(h_i) = 12750 / 255 = 50.0 exactly?
    # 12750 / 255 = 50.
    # In float64, 1/255 is not exact. 
    # So sum(h_i) might be 50.00000000000001.
    # We need to fix this.
    
    diff = N/2.0 - np.sum(best_h)
    if abs(diff) > 1e-12:
        # Adjust one element
        # Find element with most room
        # To keep it on the "grid" is hard, so we just nudge.
        # But the task says "step heights h ... bounded between 0 and 1".
        # It doesn't say they must be on a grid.
        # So we can just add the residual to one element.
        idx = np.argmax(best_h) if diff < 0 else np.argmin(best_h)
        best_h[idx] += diff
        # Check bounds
        if best_h[idx] < 0: best_h[idx] = 0
        if best_h[idx] > 1: best_h[idx] = 1
        
    return best_h
