"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import math

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s kill

    N = 24
    sqrt_N = math.sqrt(N)

    # Initial candidate: Golay-Rudin-Shapiro truncated to 24
    p, q = [1], [1]
    while len(p) < N:
        p, q = p + q, p + [-c for c in q]
    best_coeffs = np.array(p[:N], dtype=np.int8)
    
    # Pre-compute powers for fast evaluation at specific angles
    # We will use a dense grid for initial bounds and analytic refinement

    def get_precise_sup_norm(coeffs):
        """
        High-accuracy sup-norm.
        1. Dense grid scan (8192 points) to find candidate peaks.
        2. For each candidate, perform Newton's method on the derivative to find the exact local max.
        """
        # 1. Dense Grid
        n_grid = 8192
        angles = 2 * np.pi * np.arange(n_grid) / n_grid
        zs = np.exp(1j * angles)
        
        # Fast eval on grid
        # coeffs is length N. P(z) = sum c_k z^k
        # We can use FFT if N was a power of 2, but N=24. 
        # Direct multiplication is fast enough for 8192 * 24 ops.
        vals = np.zeros(n_grid, dtype=complex)
        for k in range(N):
            vals += coeffs[k] * (zs ** k)
            
        abs_vals = np.abs(vals)
        
        # Find top candidates
        # We look for local maxima on the grid to identify regions to refine
        # Simple approach: take top 10 values from grid
        top_indices = np.argsort(abs_vals)[-10:]
        
        max_val = 0.0
        for idx in top_indices:
            # Refine around angles[idx]
            # Newton's method on f(theta) = |P(e^{i theta})|^2
            # f'(theta) = 2 Re( P'(e^{i theta}) * conj(P(e^{i theta})) * i )
            # Actually, let's just do a fine golden section or Newton on the angle
            
            # Initial guess
            theta = angles[idx]
            
            # Newton iteration
            for _ in range(20):
                z = np.exp(1j * theta)
                # P(z)
                P_val = 0.0
                P_deriv = 0.0
                for k in range(N):
                    P_val += coeffs[k] * (z ** k)
                    if k > 0:
                        P_deriv += k * coeffs[k] * (z ** (k - 1))
                
                # Derivative of |P(z)|^2 with respect to theta
                # d/dtheta |P|^2 = Re( d|P|^2/dz * dz/dtheta )
                # d|P|^2/dz = P'(z) * conj(P(z))
                # dz/dtheta = i z
                # So d/dtheta = Re( P'(z) * conj(P(z)) * i z )
                
                if abs(P_val) < 1e-12:
                    break
                
                d_theta = np.real(P_deriv * np.conj(P_val) * 1j * z)
                d2_theta = 0.0 # Second derivative is complex to compute, use finite diff or just step
                
                # Since we want max, we move against gradient
                # Use a simple adaptive step or fixed small step if Newton is unstable
                # Newton: theta_new = theta - d/f''
                # Let's use a simpler approach: local golden section in a small window
                # This is more robust than Newton for |P| which isn't smooth at zeros (but we are at peaks)
                
                # Actually, let's just do 50 iterations of golden section in [theta - pi/8192, theta + pi/8192]
                break # Skip Newton, do GS below
                
            # Golden Section Search
            a = theta - 2 * np.pi / n_grid
            b = theta + 2 * np.pi / n_grid
            gr = (5 ** 0.5 - 1) / 2
            c = b - gr * (b - a)
            d = a + gr * (b - a)
            
            def eval_abs(angle):
                z = np.exp(1j * angle)
                val = 0.0
                for k in range(N):
                    val += coeffs[k] * (z ** k)
                return abs(val)
                
            fc = eval_abs(c)
            fd = eval_abs(d)
            
            for _ in range(50):
                if fc > fd:
                    b = d
                    d = c
                    fd = fc
                    c = b - gr * (b - a)
                    fc = eval_abs(c)
                else:
                    a = c
                    c = d
                    fc = fd
                    d = a + gr * (b - a)
                    fd = eval_abs(d)
                    
            mid = (a + b) / 2
            fmid = eval_abs(mid)
            max_val = max(max_val, fmid)
            
        return max_val

    best_score = get_precise_sup_norm(best_coeffs) / sqrt_N

    # Branch and Bound
    # State: (coeffs_prefix, current_best_sup_norm_estimate)
    # We need an upper bound for the remaining coefficients.
    # If we have fixed k coefficients, and remaining N-k are free.
    # P(z) = P_fixed(z) + P_free(z)
    # |P(z)| <= |P_fixed(z)| + |P_free(z)|
    # |P_free(z)| <= sum |c_j| = N-k
    # So |P(z)| <= |P_fixed(z)| + N-k
    # This bound is very loose.
    
    # Better bound:
    # Max over z of |P_fixed(z) + P_free(z)|
    # This is hard to compute exactly without solving the full problem.
    
    # Alternative: Use the "energy" or L2 norm?
    # Parseval: sum |c_k|^2 = N.
    # sup norm >= L2 norm = sqrt(N).
    
    # Let's try a different strategy: 
    # The optimal polynomials for small N are often related to Barker sequences or specific algebraic structures.
    # For N=24, the known optimum is 1.2583758 * sqrt(24) = 6.155...
    
    # Since B&B with loose bounds is unlikely to prune enough in 120s for N=24 (2^24 = 16M nodes),
    # and precise evaluation is expensive, let's switch to a **high-resolution Simulated Annealing** 
    # with a **fast analytical local optimizer** for the neighborhood.
    
    # Wait, the directive says "MATERIALLY DIFFERENT". 
    # The current program uses coordinate descent (greedy local search).
    # B&B is structurally different.
    # Even if it doesn't find the global optimum, it explores the space differently.
    # However, to be effective, we need a tighter bound.
    
    # Tighter Bound Idea:
    # For the remaining coefficients, we can choose them to minimize the sup norm of the sum.
    # This is still hard.
    
    # Let's stick to a **Systematic Search with Pruning** using a **Relaxed Sup Norm**.
    # We can use the fact that if the current partial sum already exceeds the best known global max 
    # plus the maximum possible contribution of the remaining terms (which is just sum of magnitudes, i.e., count),
    # we can prune.
    # Best known max ~ 6.15.
    # Max remaining contribution for k=10 is 14.
    # 6.15 + 14 = 20.15. 
    # This bound is never tight enough to prune early.
    
    # Therefore, pure B&B is not viable for N=24 in 120s with this bound.
    
    # Let's try **Iterated Local Search (ILS)** with **Random Restarts** and **Multi-Start**.
    # This is a different family from single-start coordinate descent.
    # It uses a global diversification mechanism.
    
    def fast_sup_norm(coeffs):
        """Fast sup-norm estimate on a moderately dense grid."""
        n_grid = 2048
        angles = 2 * np.pi * np.arange(n_grid) / n_grid
        zs = np.exp(1j * angles)
        vals = np.zeros(n_grid, dtype=complex)
        for k in range(N):
            vals += coeffs[k] * (zs ** k)
        return np.max(np.abs(vals))

    def local_search(start_coeffs):
        """
        Deterministic local search.
        1. Try single flips.
        2. Try double flips if stuck.
        """
        current = start_coeffs.copy()
        current_val = fast_sup_norm(current)
        
        # Phase 1: Single flips
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            best_flip_idx = -1
            best_flip_val = current_val
            best_trial = None
            
            for i in range(N):
                trial = current.copy()
                trial[i] = -trial[i]
                trial_val = fast_sup_norm(trial)
                if trial_val < best_flip_val - 1e-9:
                    best_flip_val = trial_val
                    best_flip_idx = i
                    best_trial = trial
            
            if best_flip_idx >= 0:
                current = best_trial
                current_val = best_flip_val
                improved = True
        
        # Phase 2: Double flips (pairwise swaps)
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            best_pair = (-1, -1)
            best_pair_val = current_val
            best_trial = None
            
            # Limit pairs to avoid O(N^2) explosion if too slow, but N=24 is small (276 pairs)
            for i in range(N):
                for j in range(i+1, N):
                    trial = current.copy()
                    trial[i] = -trial[i]
                    trial[j] = -trial[j]
                    trial_val = fast_sup_norm(trial)
                    if trial_val < best_pair_val - 1e-9:
                        best_pair_val = trial_val
                        best_pair = (i, j)
                        best_trial = trial
            
            if best_pair[0] >= 0:
                current = best_trial
                current_val = best_pair_val
                improved = True
                
        return current, current_val

    # Main ILS Loop
    global_best_coeffs = best_coeffs.copy()
    global_best_val = get_precise_sup_norm(global_best_coeffs)
    global_best_score = global_best_val / sqrt_N
    
    # Pre-compute grid for fast eval
    n_grid_fast = 4096
    angles_fast = 2 * np.pi * np.arange(n_grid_fast) / n_grid_fast
    zs_fast = np.exp(1j * angles_fast)
    
    def fast_sup_norm_v2(coeffs):
        vals = np.zeros(n_grid_fast, dtype=complex)
        for k in range(N):
            vals += coeffs[k] * (zs_fast ** k)
        return np.max(np.abs(vals))

    # Restart ILS
    while time.time() < deadline - 20.0:
        # Generate perturbation of global best
        # Perturb by flipping 2-5 random coefficients
        perturbed = global_best_coeffs.copy()
        n_flips = np.random.randint(2, 6)
        indices = np.random.choice(N, size=n_flips, replace=False)
        for idx in indices:
            perturbed[idx] = -perturbed[idx]
            
        # Local search from perturbed
        local_best, local_val = local_search(perturbed)
        
        # If local best is better, update global
        if local_val < global_best_val - 1e-9:
            global_best_coeffs = local_best
            global_best_val = local_val
            global_best_score = local_val / sqrt_N
            
    # Final precise evaluation
    final_val = get_precise_sup_norm(global_best_coeffs)
    return global_best_coeffs.astype(int).tolist()
# EVOLVE-BLOCK-END
