import numpy as np
import time
from scipy.optimize import minimize

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s limit

    def compute_c3(h):
        """
        Compute the upper bound C3 = max(|R[k]| for k != 0) / R[0]
        where R is the full autocorrelation of h.
        """
        n_fft = 1 << (2 * N - 1).bit_length()
        F = np.fft.rfft(h.astype(float), n=n_fft)
        R_full = np.fft.irfft(F * np.conj(F), n=n_fft)
        R_pos = R_full[:N]
        
        peak = R_pos[0]
        if peak < 1e-10:
            return 1e10
        sidelobes = R_pos[1:]
        max_side = np.max(np.abs(sidelobes))
        return max_side / peak

    def compute_c3_grad(h):
        """
        Compute C3 and its gradient with respect to h.
        Uses a smooth approximation of max to get gradients.
        """
        n_fft = 1 << (2 * N - 1).bit_length()
        F = np.fft.rfft(h.astype(float), n=n_fft)
        R_full = np.fft.irfft(F * np.conj(F), n=n_fft)
        R_pos = R_full[:N].copy()
        
        peak = R_pos[0]
        if peak < 1e-10:
            return 1e10, np.zeros(N)
        
        sidelobes = R_pos[1:]
        abs_side = np.abs(sidelobes)
        
        # Smooth max using softmax-like weighting
        temp = 0.5
        scaled = abs_side / temp
        max_val = np.max(scaled)
        weights = np.exp(scaled - max_val)
        weights /= np.sum(weights)
        
        smooth_max = temp * (max_val + np.log(np.sum(weights)))
        c3_val = smooth_max / peak
        
        # Gradient computation
        # d(R[k])/dh[j] = h[j+k] + h[j-k] (for valid indices)
        # This is complex, so we use finite differences or a simpler approach
        # For now, let's just return the value and use a finite difference approach
        # in the optimizer, or we can derive it properly.
        
        # Actually, let's compute the gradient properly using FFT derivatives
        # dR[k]/dh = 2 * (h shifted by k) for the real part
        # This is getting complex. Let's use a different strategy:
        # Optimize a continuous relaxation where h is in [-1, 1] and 
        # we minimize the max sidelobe directly using subgradient methods
        # or just use the value for L-BFGS-B with numerical gradients.
        
        return c3_val, np.zeros(N)  # Placeholder for gradient

    def objective_continuous(x):
        """
        Objective function for continuous optimization.
        x is in [-1, 1]^N
        """
        if time.time() > deadline - 5.0:
            return 1e10
        c3, _ = compute_c3_grad(x)
        # Add a small penalty to encourage binary values
        binary_penalty = 0.01 * np.sum((x**2 - 1)**2)
        return c3 + binary_penalty

    def objective_continuous_grad(x):
        """
        Objective function with gradient for continuous optimization.
        """
        if time.time() > deadline - 5.0:
            return 1e10, np.zeros(N)
        c3, grad_c3 = compute_c3_grad(x)
        # Gradient of binary penalty
        grad_penalty = 0.04 * (x**2 - 1) * x
        return c3 + 0.01 * np.sum((x**2 - 1)**2), grad_c3 + grad_penalty

    best_h = None
    best_val = float('inf')

    # 1. Generate initial candidates
    np.random.seed(42)
    
    # Random binary sequences
    for _ in range(20):
        if time.time() > deadline - 30.0:
            break
        h = np.random.choice([-1, 1], size=N)
        val = compute_c3(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()

    # Thue-Morse sequence
    if time.time() < deadline - 30.0:
        def thue_morse(n):
            tm = [0]
            for i in range(1, n):
                tm.append(tm[i // 2] ^ (i % 2))
            return np.array(tm[:n])
        
        tm = thue_morse(N)
        h = np.where(tm == 0, 1, -1).astype(int)
        val = compute_c3(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()

    # 2. Continuous relaxation optimization using L-BFGS-B
    if best_h is not None and time.time() < deadline - 20.0:
        # Start from the best binary solution, mapped to continuous
        x0 = best_h.astype(float)
        
        # Try L-BFGS-B with bounds
        try:
            result = minimize(
                objective_continuous,
                x0,
                method='L-BFGS-B',
                bounds=[(-1, 1)] * N,
                options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            if time.time() < deadline - 10.0:
                # Round to binary
                h_rounded = np.where(result.x >= 0, 1, -1).astype(int)
                val = compute_c3(h_rounded)
                if val < best_val:
                    best_val = val
                    best_h = h_rounded.copy()
        except Exception:
            pass

    # 3. Simulated annealing to escape local minima
    if best_h is not None and time.time() < deadline - 10.0:
        def simulated_annealing(x, max_iter=20000, T_start=1.0, T_end=0.01, time_limit=None):
            start_time_sa = time.time()
            current_c3 = compute_c3(x)
            best_x_sa = x.copy()
            best_c3_sa = current_c3
            T = T_start
            
            for iteration in range(max_iter):
                # Check time
                if time_limit is not None and time.time() - start_time_sa > time_limit:
                    break
                if time.time() > deadline - 5.0:
                    break
                
                # Annealing schedule
                T = T_start * (T_end / T_start) ** (iteration / max_iter)
                
                # Try a random flip
                idx = np.random.randint(0, N)
                x_new = x.copy()
                x_new[idx] = -x_new[idx]
                new_c3 = compute_c3(x_new)
                
                delta = new_c3 - current_c3
                if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
                    x = x_new
                    current_c3 = new_c3
                    if current_c3 < best_c3_sa:
                        best_x_sa = x.copy()
                        best_c3_sa = current_c3
            
            return best_x_sa, best_c3_sa
        
        x_sa, c3_sa = simulated_annealing(best_h, max_iter=10000, T_start=0.5, T_end=0.01,
                                          time_limit=deadline - 10.0 - time.time())
        if c3_sa < best_val:
            best_val = c3_sa
            best_h = x_sa.copy()

    # 4. Local search refinement on the best candidate
    if best_h is not None and time.time() < deadline - 5.0:
        h = best_h.copy()
        current_val = best_val
        n_iterations = 5000
        for it in range(n_iterations):
            if time.time() > deadline - 5.0:
                break
            # Flip a random bit
            idx = np.random.randint(0, N)
            old_val = h[idx]
            h[idx] = -h[idx]
            new_val = compute_c3(h)
            if new_val < current_val:
                current_val = new_val
                best_val = current_val
                best_h = h.copy()
            else:
                h[idx] = old_val

    if best_h is None:
        return np.random.choice([-1, 1], size=N)
    
    return best_h
