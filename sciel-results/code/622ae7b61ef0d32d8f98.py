"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Find the 24 coefficients in {+1,-1} that minimize the sup norm of P(z) on |z|=1.
    Uses simulated annealing with multiple restarts and golden-section refinement.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin
    
    N = 24
    sqrt_N = np.sqrt(N)
    
    # Coarse grid for fast evaluation during search
    n_coarse = 4096
    theta_coarse = np.linspace(0, 2*np.pi, n_coarse, endpoint=False)
    exp_mat_coarse = np.exp(1j * np.outer(np.arange(N), theta_coarse))
    
    def eval_coarse(coeffs):
        """Fast evaluation on coarse grid. Returns max |P(z)|/sqrt(N)."""
        vals = exp_mat_coarse.T @ coeffs
        return np.max(np.abs(vals)) / sqrt_N
    
    # Golden-section refinement for accurate peak finding
    def refine_peak(coeffs, theta0, half_width=0.1):
        """Refine the peak near theta0 using golden-section search."""
        # First find a tighter interval by sampling
        n_samples = 100
        thetas = np.linspace(theta0 - half_width, theta0 + half_width, n_samples)
        # Wrap thetas to [0, 2*pi)
        thetas = thetas % (2 * np.pi)
        
        def abs_p(theta):
            z = np.exp(1j * theta)
            # Compute P(z) = sum c_k * z^k
            powers = np.power(z, np.arange(N))
            return np.abs(np.sum(coeffs * powers)) / sqrt_N
        
        vals = np.array([abs_p(t) for t in thetas])
        idx = np.argmax(vals)
        
        # Golden-section search around the best sample
        a = thetas[max(0, idx-1)]
        b = thetas[min(n_samples-1, idx+1)]
        
        if b <= a:
            # Fallback: use wider interval
            a = (theta0 - half_width) % (2*np.pi)
            b = (theta0 + half_width) % (2*np.pi)
        
        gr = (np.sqrt(5) - 1) / 2  # golden ratio
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        
        fc = abs_p(c)
        fd = abs_p(d)
        
        for _ in range(60):
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = abs_p(c)
            else:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = abs_p(d)
        
        return max(fc, fd)
    
    def eval_accurate(coeffs):
        """Accurate evaluation: find top peaks and refine each."""
        vals = exp_mat_coarse.T @ coeffs
        abs_vals = np.abs(vals)
        
        # Find top 10 peaks
        top_idx = np.argsort(abs_vals)[-10:]
        
        best = 0
        for idx in top_idx:
            theta_peak = theta_coarse[idx]
            refined = refine_peak(coeffs, theta_peak, half_width=np.pi/n_coarse)
            best = max(best, refined)
        
        return best
    
    # Simulated annealing
    def simulated_annealing(initial_coeffs, t0=1.0, t_min=0.01, cooling=0.995, max_iter=20000):
        """Simulated annealing to minimize sup norm."""
        current = initial_coeffs.copy()
        current_val = eval_coarse(current)
        best = current.copy()
        best_val = current_val
        
        T = t0
        rng = np.random.RandomState()
        
        for it in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            # Pick a random bit to flip
            i = rng.randint(N)
            new_coeffs = current.copy()
            new_coeffs[i] *= -1
            new_val = eval_coarse(new_coeffs)
            
            delta = new_val - current_val
            if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
                current = new_coeffs
                current_val = new_val
                if current_val < best_val:
                    best = current.copy()
                    best_val = current_val
            
            T *= cooling
            if T < t_min:
                break
        
        return best, best_val
    
    # Generate initial candidates
    def make_rudin_shapiro(n):
        """Generate Rudin-Shapiro sequence of length >= n."""
        p, q = np.array([1.0]), np.array([1.0])
        while len(p) < n:
            p, q = np.concatenate([p, q]), np.concatenate([p, -q])
        return p[:n].copy()
    
    candidates = []
    
    # Rudin-Shapiro and variants
    rs = make_rudin_shapiro(N)
    candidates.append(rs)
    candidates.append(-rs)
    candidates.append(rs[::-1].copy())
    candidates.append(-rs[::-1].copy())
    
    # Some random starts
    rng_init = np.random.RandomState(12345)
    for _ in range(10):
        candidates.append(rng_init.choice([-1.0, 1.0], size=N))
    
    # Run simulated annealing from each candidate
    best_overall = None
    best_overall_val = float('inf')
    
    for i, cand in enumerate(candidates):
        if time.time() > deadline - 10.0:
            break
        
        t0 = 0.5 + 0.3 * (i % 5) / 5.0  # Vary initial temperature
        cooling = 0.998 - 0.001 * (i % 3)
        
        result, val = simulated_annealing(cand, t0=t0, cooling=cooling, max_iter=30000)
        
        if val < best_overall_val:
            best_overall = result
            best_overall_val = val
    
    # Final polish: greedy local search from the best solution
    improved = True
    while improved:
        improved = False
        if time.time() > deadline - 5.0:
            break
        for i in range(N):
            new_coeffs = best_overall.copy()
            new_coeffs[i] *= -1
            val = eval_coarse(new_coeffs)
            if val < best_overall_val - 1e-12:
                best_overall = new_coeffs
                best_overall_val = val
                improved = True
    
    # Accurate evaluation of the best solution
    final_val = eval_accurate(best_overall)
    
    # Convert to list of ints
    result = [1 if c > 0 else -1 for c in best_overall]
    assert len(result) == N
    return result
# EVOLVE-BLOCK-END
