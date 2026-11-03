"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Construct a signed step function (sequence of signed step heights) that minimizes
    the maximum absolute autoconvolution divided by the squared integral (sum).
    
    Strategy:
    1. The metric is scale-invariant, so we can optimize over real-valued vectors.
    2. We use gradient descent with a soft-max surrogate for the maximum absolute autocorrelation term.
    3. We optimize the vector x directly.
    4. We handle the division by (sum(x))^2 by incorporating it into the loss or by normalizing x.
       Normalizing x to have sum 1 simplifies the objective to minimizing max |c_k|.
       However, if sum(x) approaches 0, the metric explodes. We must keep sum(x) away from 0.
       We can add a penalty for small sum(x) or simply constrain the optimization.
       Alternatively, we optimize the raw metric using gradients.
       
    Let's use the normalized approach:
    Fix sum(x) = N (or 1). Minimize max |c_k|.
    If we fix sum(x) = N, the all-ones solution gives max |c_k| = N-1.
    Score = (N-1)/N^2.
    
    We will use Adam optimizer on x, with a projection step to maintain sum(x) = N?
    Or just add a strong penalty term: (sum(x) - N)^2.
    
    Actually, simpler:
    Minimize L(x) = soft_max_abs(c[1:]) / (sum(x)^2 + eps)
    Use Adam.
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Initialize x
    # Try all ones
    x = np.ones(N)
    
    # Also try random
    np.random.seed(42)
    x_rand = np.random.randn(N)
    # Ensure sum is not zero
    if abs(np.sum(x_rand)) < 1e-6:
        x_rand += 1.0
        
    best_x = x.copy()
    best_score = float('inf')
    
    def compute_autocorr(x):
        """Compute linear autoconvolution."""
        n = len(x)
        size = 1
        while size < 2 * n - 1:
            size *= 2
        fx = np.fft.rfft(x, size)
        # Autoconvolution is inverse FFT of |F|^2
        ac = np.fft.irfft(fx * np.conj(fx), size)[:2 * n - 1]
        return ac

    def score(x):
        s = np.sum(x)
        if abs(s) < 1e-12:
            return float('inf')
        c = compute_autocorr(x)
        # Exclude k=0
        if len(c) > 1:
            return float(np.max(np.abs(c[1:]))) / (s * s)
        return 0.0

    def soft_max_abs(c_excl, beta=100.0):
        """Differentiable approximation of max(|c|)."""
        abs_c = np.abs(c_excl)
        m = np.max(abs_c)
        if m < 1e-15:
            return 0.0
        # log-sum-exp trick
        return m + np.log(np.sum(np.exp(beta * (abs_c - m)))) / beta

    def objective(x):
        s = np.sum(x)
        if abs(s) < 1e-12:
            return 1e10
        c = compute_autocorr(x)
        c_excl = c[1:]
        sm = soft_max_abs(c_excl)
        return sm / (s * s)

    # Simple Gradient Descent with Adam-like updates
    # We'll implement a basic Adam optimizer
    
    def adam_step(x, grads, lr, beta1, beta2, eps, m, v, t):
        m[:] = beta1 * m + (1 - beta1) * grads
        v[:] = beta2 * v + (1 - beta2) * (grads ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        x[:] -= lr * m_hat / (np.sqrt(v_hat) + eps)
        return m, v, t + 1

    def gradient_descent(x_init, iterations=2000, lr=0.01):
        x = x_init.copy()
        m = np.zeros_like(x)
        v = np.zeros_like(x)
        t = 1
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        
        best_local = x.copy()
        best_local_score = objective(x)
        
        for i in range(iterations):
            if time.time() > deadline:
                break
            
            # Compute gradient numerically or analytically?
            # Analytical gradient of soft_max_abs / s^2 is complex.
            # Let's use numerical gradient (central difference) for robustness, 
            # but it's O(N) evaluations. N=400, 2000 iters -> 800k evals.
            # Each eval is O(N log N). 800k * 400 * 9 ~ 2.8e9 ops. Too slow.
            
            # We need analytical gradient or a faster method.
            # Let's stick to a simpler heuristic: Random Perturbation + Accept if better.
            # Or use the fact that we can just try a few specific constructions.
            
            # Given the time constraint and complexity, let's try a few specific candidates:
            # 1. All ones
            # 2. Random
            # 3. Sinusoidal with offset
            # 4. Chebyshev-like (minimax polynomial)
            
            # Let's break out of the loop and test candidates directly.
            break
            
        return best_local, best_local_score

    # Candidate 1: All ones
    c1 = np.ones(N)
    s1 = score(c1)
    
    # Candidate 2: Random
    c2 = np.random.randn(N)
    # Normalize to sum N?
    c2 = c2 / np.sum(c2) * N
    s2 = score(c2)
    
    # Candidate 3: Sinusoid
    t = np.linspace(0, 2*np.pi, N, endpoint=False)
    c3 = 1 + 0.5 * np.cos(t)
    s3 = score(c3)
    
    # Candidate 4: Triangle wave
    c4 = 1 + 0.5 * (2 * np.abs(2 * (t / (2*np.pi) - np.floor(t / (2*np.pi) + 0.5))) - 1)
    s4 = score(c4)
    
    # Candidate 5: Chebyshev T_N scaled?
    # Chebyshev polynomials have minimax property.
    # x_i = T_N(2*i/N - 1)?
    # Let's just pick the best of these.
    
    candidates = [c1, c2, c3, c4]
    scores = [s1, s2, s3, s4]
    
    best_idx = np.argmin(scores)
    best_x = candidates[best_idx]
    best_score = scores[best_idx]
    
    # Try to improve best_x with local search (sign flips if we threshold, or small perturbations)
    # Since we are allowed real values, small perturbations are better.
    
    x = best_x.copy()
    current_score = best_score
    
    # Gradient-free local search: random small perturbations
    for i in range(1000):
        if time.time() > deadline:
            break
        # Perturb a random subset
        mask = np.random.rand(N) < 0.1
        perturb = np.random.randn(N) * 0.01
        x_new = x + mask * perturb
        
        s_new = score(x_new)
        if s_new < current_score:
            x = x_new
            current_score = s_new
            if current_score < best_score:
                best_score = current_score
                best_x = x.copy()
        else:
            # Revert
            pass
            
    return best_x
# EVOLVE-BLOCK-END
