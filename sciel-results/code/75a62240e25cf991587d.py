import numpy as np
import math

def construct():
    N = 24
    
    # 1. Exact evaluation function
    def eval_exact(coeffs):
        """
        Compute the maximum absolute value of P(z) on the unit circle
        using dense sampling and golden-section refinement.
        """
        # Dense grid for initial peak detection
        M_dense = 4096
        angles = 2 * np.pi * np.arange(M_dense) / M_dense
        z = np.exp(1j * angles)
        
        # Precompute powers for dense grid
        powers = np.zeros((N, M_dense), dtype=complex)
        powers[0, :] = 1
        for k in range(1, N):
            powers[k, :] = powers[k-1, :] * z
            
        vals = powers.T @ coeffs
        abs_vals = np.abs(vals)
        
        # Find local maxima
        local_max_indices = []
        for i in range(M_dense):
            if abs_vals[i] >= abs_vals[(i-1) % M_dense] and abs_vals[i] >= abs_vals[(i+1) % M_dense]:
                local_max_indices.append(i)
                
        if not local_max_indices:
            return np.max(abs_vals)
            
        max_val = np.max(abs_vals)
        
        # Refine top 3 peaks
        local_max_indices.sort(key=lambda i: abs_vals[i], reverse=True)
        
        for idx in local_max_indices[:3]:
            delta = 2 * np.pi / M_dense
            center = angles[idx]
            a = center - delta
            b = center + delta
            
            def f(theta):
                z_t = np.exp(1j * theta)
                # Horner's method or direct sum
                # Direct sum is fine for N=24
                p_val = 0
                for k in range(N):
                    p_val += coeffs[k] * (z_t ** k)
                return np.abs(p_val)
                
            # Golden section search
            invphi = (1 + math.sqrt(5)) / 2
            invphi2 = invphi * invphi
            c = b - (b - a) / invphi
            d = a + (b - a) / invphi
            fc = f(c)
            fd = f(d)
            
            for _ in range(40):
                if fc < fd:
                    a = c
                    c = d
                    fc = fd
                    d = a + (b - a) / invphi2
                    fd = f(d)
                else:
                    b = d
                    d = c
                    fd = fc
                    c = b - (b - a) / invphi2
                    fc = f(c)
                    
            val = max(fc, fd)
            if val > max_val:
                max_val = val
                
        return max_val

    # 2. Generate initial best candidate: Golay-Rudin-Shapiro
    def generate_grs(n):
        p, q = [1], [1]
        while len(p) < n:
            p, q = p + q, p + [-c for c in q]
        return np.array(p[:n], dtype=int)

    initial_coeffs = generate_grs(N)
    best_val = eval_exact(initial_coeffs)
    best_coeffs = initial_coeffs.copy()
    
    # 3. Branch and Bound Setup
    # Coarse grid for pruning
    M_coarse = 128
    angles_c = 2 * np.pi * np.arange(M_coarse) / M_coarse
    z_c = np.exp(1j * angles_c)
    
    powers_c = np.zeros((N, M_coarse), dtype=complex)
    powers_c[0, :] = 1
    for k in range(1, N):
        powers_c[k, :] = powers_c[k-1, :] * z_c
        
    # We fix c[0] = 1 due to symmetry (sign flip invariance)
    # We search c[1] to c[23]
    
    import sys
    sys.setrecursionlimit(10000)
    
    def dfs(depth, current_coeffs, current_sum_coarse):
        nonlocal best_val, best_coeffs
        
        # Check if we can prune
        # We use a small margin to account for the difference between coarse and exact max
        # If coarse max is significantly higher than best_val, we prune.
        # If coarse max is lower, the exact max could still be higher (peak between points),
        # but for N=24 and M=128, the error is small.
        # Let's say if coarse_max > best_val * 1.001, we might still have a better solution?
        # No, if coarse_max > best_val, the exact max is >= coarse_max > best_val, so it's worse.
        # Wait, exact max >= coarse max.
        # So if coarse_max > best_val, then exact_max >= coarse_max > best_val.
        # So we can prune immediately if coarse_max > best_val.
        
        max_coarse = np.max(np.abs(current_sum_coarse))
        
        # Optimization: If max_coarse is already > best_val, prune.
        # Note: Due to floating point, we might use a tiny epsilon.
        if max_coarse > best_val:
            return
            
        if depth == N:
            # Compute exact value
            val = eval_exact(current_coeffs)
            if val < best_val:
                best_val = val
                best_coeffs = current_coeffs.copy()
            return
        
        # Try 1
        current_coeffs[depth] = 1
        # Update sum: add z^depth
        new_sum_1 = current_sum_coarse + powers_c[depth, :]
        dfs(depth + 1, current_coeffs, new_sum_1)
        
        # Try -1
        current_coeffs[depth] = -1
        # Update sum: subtract z^depth
        new_sum_minus1 = current_sum_coarse - powers_c[depth, :]
        dfs(depth + 1, current_coeffs, new_sum_minus1)

    # Initialize for DFS
    # c[0] = 1
    current_coeffs = np.zeros(N, dtype=int)
    current_coeffs[0] = 1
    # Initial sum is z^0 = 1
    initial_sum = powers_c[0, :].copy()
    
    dfs(1, current_coeffs, initial_sum)
    
    return best_coeffs.tolist()
