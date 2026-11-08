"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_max_overlap(h):
    """
    Compute the maximum overlap of h with shifted 1-h.
    h is a 1D array of step heights.
    The overlap at shift k is sum_i h_i * (1 - h_{i+k}) for valid i.
    Actually, the task says "max overlap of h with shifted 1-h".
    Let g(x) = 1 - h(x).
    Overlap at shift k: O_k = sum_i h_i * g_{i+k} = sum_i h_i * (1 - h_{i+k})
    We want to minimize max_k O_k.
    Note: sum_i h_i = N/2.
    sum_i g_i = N - N/2 = N/2.
    """
    # To handle circular shifts or linear shifts?
    # "shifted 1-h" usually implies circular convolution in these types of periodic problems 
    # or linear convolution. Given "step heights on [0,2]" and periodic nature of Erdos problems,
    # circular is likely. Let's assume circular shifts.
    # O_k = sum_{i=0}^{N-1} h[i] * (1 - h[(i+k)%N])
    
    # Vectorized computation using FFT for speed
    # O_k = sum h[i] - sum h[i]h[i+k]
    # sum h[i] is constant N/2.
    # So minimizing max O_k is equivalent to maximizing min autocorrelation sum h[i]h[i+k].
    # Wait, O_k = N/2 - A_k, where A_k = sum h[i]h[i+k].
    # Minimizing max O_k is equivalent to minimizing (N/2 - min A_k) ? No.
    # max O_k = max (N/2 - A_k) = N/2 - min A_k.
    # So we want to MAXIMIZE the minimum autocorrelation value min_k A_k.
    
    # Autocorrelation via FFT
    # A_k = sum h[i] h[i+k] is circular autocorrelation.
    # FFT of h, square magnitude, inverse FFT gives circular autocorrelation.
    H = np.fft.fft(h)
    A = np.real(np.fft.ifft(H * np.conj(H)))
    # A[0] is sum h^2. A[k] is sum h[i]h[i+k].
    
    # We want to maximize min(A).
    # The score in the evaluator is "upper bound on C5". 
    # The prompt says: "score is the upper bound on C5".
    # And "minimize the maximum overlap".
    # The evaluator measures the raw value (lower is better).
    # So we should return the max overlap value? Or the constructed sequence?
    # "Interface: construct() -> sequence of step heights... score is the upper bound on C5"
    # Usually, the evaluator computes the score from the returned sequence.
    # The prompt says "The evaluator measures the raw value (lower is better)."
    # This implies the evaluator calculates the overlap from my returned h and returns that as the score.
    # So I just need to return the best h.
    
    # Let's verify the objective function to optimize locally.
    # Objective to minimize: max_k (N/2 - A_k) = N/2 - min_k A_k.
    # So I want to maximize min_k A_k.
    
    return np.max(N/2 - A)

def _objective(h):
    """
    Objective function to minimize.
    Returns the maximum overlap.
    """
    # Ensure h is in [0,1] and sums to N/2
    # The evaluator likely checks constraints.
    # For local optimization, we assume h is valid.
    
    H = np.fft.fft(h)
    A = np.real(np.fft.ifft(H * np.conj(H)))
    max_overlap = np.max(N/2 - A)
    return max_overlap

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Strategy:
    # 1. Initialize with a smooth function that has low autocorrelation variance.
    #    A ramp function (sawtooth) has specific autocorrelation properties.
    #    A constant function 0.5 has A_k = N/4 for all k.
    #    Max overlap = N/2 - N/4 = N/4 = 25.
    #    Can we do better?
    #    If we make h vary, A_0 = sum h^2 increases (since sum h is fixed, variance increases sum of squares).
    #    A_0 is always the maximum of A.
    #    The minimum of A will likely decrease if we introduce structure.
    #    We want to flatten A_k as much as possible.
    #    
    #    Let's try a "balanced" sequence.
    #    Start with 0.5 everywhere.
    #    Perturb with small random noise, project to [0,1] and sum N/2.
    #    Use gradient descent or simulated annealing on the continuous domain.
    
    h = np.full(N, 0.5)
    best_h = h.copy()
    best_score = _objective(best_h)
    
    # Gradient Descent / Optimization
    # We want to maximize min(A). This is non-smooth.
    # We can use a soft-min approximation: -log(sum(exp(-A_k / T)))
    # Or just use a random search / hill climbing with small steps.
    
    # Let's try a simple local search with random perturbations.
    # Since N=100 is small, we can afford many iterations.
    
    rng = np.random.default_rng(42)
    
    # Try different initial patterns
    # Pattern 1: Constant 0.5
    # Pattern 2: Alternating 0.25, 0.75
    # Pattern 3: Ramp
    
    candidates = []
    candidates.append(np.full(N, 0.5))
    
    # Alternating
    alt = np.zeros(N)
    alt[0::2] = 0.25
    alt[1::2] = 0.75
    candidates.append(alt)
    
    # Ramp
    ramp = np.linspace(0.2, 0.8, N)
    # Normalize to sum N/2
    ramp = ramp - np.mean(ramp) + 0.5
    ramp = np.clip(ramp, 0, 1)
    # Adjust sum
    diff = N/2 - np.sum(ramp)
    ramp += diff / N
    ramp = np.clip(ramp, 0, 1)
    candidates.append(ramp)
    
    # Sine wave
    sine = 0.5 + 0.3 * np.sin(2 * np.pi * np.arange(N) / N)
    sine = np.clip(sine, 0, 1)
    diff = N/2 - np.sum(sine)
    sine += diff / N
    sine = np.clip(sine, 0, 1)
    candidates.append(sine)
    
    # Pick the best candidate
    for c in candidates:
        score = _objective(c)
        if score < best_score:
            best_score = score
            best_h = c.copy()
    
    # Local Search
    # We will perform random perturbations and accept if score improves.
    # To ensure sum constraint, we perturb pairs or adjust after clipping.
    # A better way: work in a parameter space that guarantees constraints?
    # Or just project: clip to [0,1], then adjust to sum N/2 by adding a constant?
    # Adding a constant might push out of [0,1].
    
    # Let's use a simple gradient-free local search.
    # Step size decay.
    
    current_h = best_h.copy()
    current_score = best_score
    step_size = 0.05
    iterations = 0
    max_iterations = 50000
    
    while time.time() < deadline and iterations < max_iterations:
        # Generate random perturbation
        # Perturb a random subset of indices
        num_perturb = rng.integers(1, 5)
        idxs = rng.choice(N, num_perturb, replace=False)
        
        new_h = current_h.copy()
        for idx in idxs:
            delta = rng.uniform(-step_size, step_size)
            new_h[idx] += delta
        
        # Project to [0, 1]
        new_h = np.clip(new_h, 0, 1)
        
        # Adjust sum to N/2
        # This is the tricky part. Simple projection:
        # Subtract mean error.
        error = np.sum(new_h) - N/2
        if error != 0:
            # Distribute error proportionally? Or just add constant?
            # Adding constant might violate [0,1].
            # If we add constant, we might need to clip again, which changes sum.
            # Iterative projection (Dykstra) or simple:
            # Try to adjust. If it fails, reject.
            new_h -= error / N
            if np.any(new_h < 0) or np.any(new_h > 1):
                # Fallback: try smaller adjustment or reject
                # Let's try a smarter adjustment:
                # Only adjust elements that have room.
                # For now, reject if simple projection fails.
                # Or, use a different perturbation strategy: swap mass.
                # Let's stick to rejecting invalid projections for simplicity and robustness.
                pass
            else:
                # Check if sum is close enough
                if abs(np.sum(new_h) - N/2) < 1e-9:
                    new_score = _objective(new_h)
                    if new_score < current_score:
                        current_h = new_h
                        current_score = new_score
                        if current_score < best_score:
                            best_score = current_score
                            best_h = current_h.copy()
        iterations += 1
        
        # Decay step size
        if iterations % 1000 == 0:
            step_size *= 0.95
    
    return best_h

# EVOLVE-BLOCK-END
