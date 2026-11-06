"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm

def _build_hamiltonians():
    """Build A (odd bonds + half field) and B (even bonds + half field) for 6-site XXZ chain."""
    # 6 sites, XXZ with anisotropy 0.8, field 0.3
    # Open boundary conditions
    # H = sum_j [0.5*J*(Sx_j Sx_{j+1} + Sy_j Sy_{j+1} + Jz Sz_j Sz_{j+1})] + h*sum_j Sz_j
    # Split: A = odd bonds (j=1,3,5 in 0-indexed: bonds 0-1, 2-3, 4-5) + half field
    #        B = even bonds (j=2,4,6 in 0-indexed: bonds 1-2, 3-4, 5-6) + half field
    # Wait, 6 sites means bonds 0-1, 1-2, 2-3, 3-4, 4-5 (5 bonds total)
    # Odd bonds: 0-1, 2-3, 4-5 (bonds at even indices)
    # Even bonds: 1-2, 3-4 (bonds at odd indices)
    
    J = 1.0  # coupling
    Jz = 0.8 * J  # anisotropy
    h = 0.3  # field
    
    # Single-site operators
    Sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    Sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    Sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)
    
    n = 6  # number of sites
    dim = 2 ** n
    
    def kron_all(ops):
        result = np.array([[1.0+0j]])
        for op in ops:
            result = np.kron(result, op)
        return result
    
    # Build A: odd bonds (0-1, 2-3, 4-5) + half field
    A = np.zeros((dim, dim), dtype=complex)
    # Bond 0-1
    ops = [Sx, Sx] + [I]*(n-2)
    A += J * 0.5 * kron_all(ops)
    ops = [Sy, Sy] + [I]*(n-2)
    A += J * 0.5 * kron_all(ops)
    ops = [Sz, Sz] + [I]*(n-2)
    A += J * 0.5 * Jz * kron_all(ops)
    # Bond 2-3
    ops = [I, I, Sx, Sx] + [I]*(n-4)
    A += J * 0.5 * kron_all(ops)
    ops = [I, I, Sy, Sy] + [I]*(n-4)
    A += J * 0.5 * kron_all(ops)
    ops = [I, I, Sz, Sz] + [I]*(n-4)
    A += J * 0.5 * Jz * kron_all(ops)
    # Bond 4-5
    ops = [I, I, I, I, Sx, Sx]
    A += J * 0.5 * kron_all(ops)
    ops = [I, I, I, I, Sy, Sy]
    A += J * 0.5 * kron_all(ops)
    ops = [I, I, I, I, Sz, Sz]
    A += J * 0.5 * Jz * kron_all(ops)
    # Half field
    for j in range(n):
        ops = [I]*j + [Sz] + [I]*(n-1-j)
        A += 0.5 * h * kron_all(ops)
    
    # Build B: even bonds (1-2, 3-4) + half field
    B = np.zeros((dim, dim), dtype=complex)
    # Bond 1-2
    ops = [I, Sx, Sx] + [I]*(n-3)
    B += J * 0.5 * kron_all(ops)
    ops = [I, Sy, Sy] + [I]*(n-3)
    B += J * 0.5 * kron_all(ops)
    ops = [I, Sz, Sz] + [I]*(n-3)
    B += J * 0.5 * Jz * kron_all(ops)
    # Bond 3-4
    ops = [I, I, I, Sx, Sx] + [I]*(n-5)
    B += J * 0.5 * kron_all(ops)
    ops = [I, I, I, Sy, Sy] + [I]*(n-5)
    B += J * 0.5 * kron_all(ops)
    ops = [I, I, I, Sz, Sz] + [I]*(n-5)
    B += J * 0.5 * Jz * kron_all(ops)
    # Half field
    for j in range(n):
        ops = [I]*j + [Sz] + [I]*(n-1-j)
        B += 0.5 * h * kron_all(ops)
    
    return A, B

def _compute_error(a, b, A, B, exact_prop, times):
    """Compute mean spectral norm error for given coefficients."""
    errors = []
    for t in times:
        U = np.eye(A.shape[0], dtype=complex)
        for k in range(len(a)):
            # exp(-i b_k t B) exp(-i a_k t A)
            U = expm(-1j * b[k] * t * B) @ expm(-1j * a[k] * t * A) @ U
        diff = U - exact_prop[t]
        # Spectral norm (largest singular value)
        s = np.linalg.svd(diff, compute_uv=False)
        errors.append(s[0])
    return np.mean(errors)

def coefficients(stages):
    """Optimize product formula coefficients using Nelder-Mead."""
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    A, B = _build_hamiltonians()
    times = [0.25, 0.5, 1.0]
    exact_prop = {}
    for t in times:
        exact_prop[t] = expm(-1j * t * (A + B))
    
    n = stages  # 12
    
    # Initial guess: symmetric 4th-order Suzuki-Trotter-like coefficients
    # For 12 stages, use a scaled version of known high-order coefficients
    # Simple initialization: uniform with small perturbations
    a0 = np.ones(n) / n
    b0 = np.ones(n) / n
    
    # Nelder-Mead optimization
    def objective(params):
        elapsed = time.time() - start_time
        if elapsed > deadline:
            return 1e10  # Timeout penalty
        
        a = params[:n]
        b = params[n:]
        
        # Enforce constraints: sum(a) = 1, sum(b) = 1
        # Use soft penalty for constraint violation
        a_sum = np.sum(a)
        b_sum = np.sum(b)
        constraint_penalty = 100.0 * (a_sum - 1.0)**2 + 100.0 * (b_sum - 1.0)**2
        
        # Clip to valid range
        a = np.clip(a, -50, 50)
        b = np.clip(b, -50, 50)
        
        err = _compute_error(a, b, A, B, exact_prop, times)
        return err + constraint_penalty
    
    # Nelder-Mead implementation
    def nelder_mead(f, x0, max_iter=200, tol=1e-6, time_limit=130):
        n = len(x0)
        # Initial simplex
        simplex = [np.array(x0, dtype=float)]
        for i in range(n):
            xi = np.array(x0, dtype=float)
            step = 0.1 if x0[i] == 0 else 0.05 * abs(x0[i])
            xi[i] += step
            simplex.append(xi)
        
        fvals = [f(x) for x in simplex]
        
        for iteration in range(max_iter):
            if time.time() - start_time > time_limit:
                break
            
            # Sort simplex
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            # Check convergence
            if abs(fvals[-1] - fvals[0]) < tol:
                break
            
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            alpha = 1.0
            xr = centroid + alpha * (centroid - simplex[-1])
            fr = f(xr)
            
            if fr < fvals[0]:
                # Expansion
                gamma = 2.0
                xe = centroid + gamma * (centroid - simplex[-1])
                fe = f(xe)
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            elif fr < fvals[-2]:
                simplex[-1] = xr
                fvals[-1] = fr
            else:
                # Contraction
                rho = 0.5
                if fr < fvals[-1]:
                    xc = centroid + rho * (xr - centroid)
                    fc = f(xc)
                    if fc <= fr:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            fvals[i] = f(simplex[i])
                else:
                    xc = centroid - rho * (centroid - simplex[-1])
                    fc = f(xc)
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            fvals[i] = f(simplex[i])
        
        # Return best point
        best_idx = np.argmin(fvals)
        return simplex[best_idx], fvals[best_idx]
    
    x0 = np.concatenate([a0, b0])
    x_opt, f_opt = nelder_mead(objective, x0, max_iter=100, tol=1e-5, time_limit=130)
    
    a_opt = np.clip(x_opt[:n], -50, 50)
    b_opt = np.clip(x_opt[n:], -50, 50)
    
    # Normalize to ensure sum = 1
    a_opt = a_opt / np.sum(a_opt) if np.sum(a_opt) != 0 else np.ones(n) / n
    b_opt = b_opt / np.sum(b_opt) if np.sum(b_opt) != 0 else np.ones(n) / n
    
    return list(a_opt), list(b_opt)
# EVOLVE-BLOCK-END
