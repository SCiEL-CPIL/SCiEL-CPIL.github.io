"""Optimized product formula for Heisenberg-chain time evolution."""
import numpy as np
import time
from scipy.linalg import expm

def _build_hamiltonians():
    """Build A (odd bonds + half field) and B (even bonds + half field) for 6-site open XXZ chain."""
    n = 6
    aniso = 0.8
    field = 0.3
    dim = 2 ** n
    
    # XXZ Hamiltonian: sum_j [ (1/2)(S_j^+ S_{j+1}^- + S_j^- S_{j+1}^+) + aniso * S_j^z S_{j+1}^z ] + field * sum_j S_j^z
    # S^+ = |0><1|, S^- = |1><0|, S^z = diag(1/2, -1/2)
    # In computational basis |0> = spin up (z=+1/2), |1> = spin down (z=-1/2)
    
    # Build full Hamiltonian first to verify
    H = np.zeros((dim, dim), dtype=complex)
    
    for j in range(n - 1):
        # S_j^+ S_{j+1}^- : flip j from 1 to 0, flip j+1 from 0 to 1
        # S_j^- S_{j+1}^+ : flip j from 0 to 1, flip j+1 from 1 to 0
        # S_j^z S_{j+1}^z
        for state in range(dim):
            bits = [(state >> (n - 1 - k)) & 1 for k in range(n)]
            # S^+ S^- term: bits[j]=1, bits[j+1]=0 -> flip both
            if bits[j] == 1 and bits[j+1] == 0:
                new_state = state ^ (1 << (n - 1 - j)) ^ (1 << (n - 1 - (j+1)))
                H[state, new_state] += 0.5
            if bits[j] == 0 and bits[j+1] == 1:
                new_state = state ^ (1 << (n - 1 - j)) ^ (1 << (n - 1 - (j+1)))
                H[state, new_state] += 0.5
            # S^z S^z term
            sz_j = 0.5 if bits[j] == 0 else -0.5
            sz_j1 = 0.5 if bits[j+1] == 0 else -0.5
            H[state, state] += aniso * sz_j * sz_j1
    
    # Field term: field * sum_j S_j^z
    for j in range(n):
        for state in range(dim):
            bits = [(state >> (n - 1 - k)) & 1 for k in range(n)]
            sz_j = 0.5 if bits[j] == 0 else -0.5
            H[state, state] += field * sz_j
    
    # Now split: A = odd bonds (j=0,2,4) + half field, B = even bonds (j=1,3,5) + half field
    # Wait, for 6 sites, bonds are j=0,1,2,3,4 (5 bonds)
    # Odd bonds: j=0,2,4 (bonds 0-1, 2-3, 4-5)
    # Even bonds: j=1,3 (bonds 1-2, 3-4)
    
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    for j in range(n - 1):
        for state in range(dim):
            bits = [(state >> (n - 1 - k)) & 1 for k in range(n)]
            if bits[j] == 1 and bits[j+1] == 0:
                new_state = state ^ (1 << (n - 1 - j)) ^ (1 << (n - 1 - (j+1)))
                val = 0.5
                if j % 2 == 0:
                    A[state, new_state] += val
                else:
                    B[state, new_state] += val
            if bits[j] == 0 and bits[j+1] == 1:
                new_state = state ^ (1 << (n - 1 - j)) ^ (1 << (n - 1 - (j+1)))
                val = 0.5
                if j % 2 == 0:
                    A[state, new_state] += val
                else:
                    B[state, new_state] += val
            sz_j = 0.5 if bits[j] == 0 else -0.5
            sz_j1 = 0.5 if bits[j+1] == 0 else -0.5
            val = aniso * sz_j * sz_j1
            if j % 2 == 0:
                A[state, state] += val
            else:
                B[state, state] += val
    
    # Half field in each
    half_field = field / 2.0
    for j in range(n):
        for state in range(dim):
            bits = [(state >> (n - 1 - k)) & 1 for k in range(n)]
            sz_j = 0.5 if bits[j] == 0 else -0.5
            val = half_field * sz_j
            A[state, state] += val
            B[state, state] += val
    
    return A, B


def _spectral_norm(M):
    """Compute spectral norm (largest singular value) of matrix M."""
    return np.linalg.norm(M, 2)


def _objective(a, b, A, B, times):
    """Compute mean spectral norm error."""
    dim = A.shape[0]
    H = A + B
    errors = []
    for t in times:
        # Exact propagator
        U_exact = expm(-1j * t * H)
        # Product formula
        U_approx = np.eye(dim, dtype=complex)
        for k in range(len(a)):
            U_approx = U_approx @ expm(-1j * b[k] * t * B) @ expm(-1j * a[k] * t * A)
        err = _spectral_norm(U_approx - U_exact)
        errors.append(err)
    return np.mean(errors)


def _nelder_mead(f, x0, bounds=None, max_iter=2000, tol=1e-8, time_limit=None, start_time=None):
    """Simple Nelder-Mead optimization with time guard."""
    n = len(x0)
    # Initialize simplex
    simplex = [list(x0)]
    for i in range(n):
        xi = list(x0)
        step = 0.1 if abs(x0[i]) < 1e-6 else 0.1 * abs(x0[i])
        xi[i] += step
        simplex.append(xi)
    
    # Evaluate initial simplex
    values = []
    for i in range(n + 1):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        values.append(f(simplex[i]))
    
    # Sort
    order = np.argsort(values)
    simplex = [simplex[i] for i in order]
    values = [values[i] for i in order]
    
    best_x = simplex[0]
    best_f = values[0]
    
    for iteration in range(max_iter):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        alpha = 1.0
        xr = centroid + alpha * (centroid - np.array(simplex[-1]))
        fr = f(xr)
        
        if fr < values[0]:
            # Expansion
            gamma = 2.0
            xe = centroid + gamma * (centroid - np.array(simplex[-1]))
            fe = f(xe)
            if fe < fr:
                simplex[-1] = xe
                values[-1] = fe
            else:
                simplex[-1] = xr
                values[-1] = fr
        elif fr < values[-2]:
            simplex[-1] = xr
            values[-1] = fr
        else:
            # Contraction
            rho = 0.5
            if fr < values[-1]:
                xc = centroid + rho * (np.array(simplex[-1]) - centroid)
            else:
                xc = centroid - rho * (np.array(simplex[-1]) - centroid)
            fc = f(xc)
            if fc < values[-1]:
                simplex[-1] = xc
                values[-1] = fc
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (np.array(simplex[i]) - np.array(simplex[0]))
                    if time_limit is not None and start_time is not None:
                        if time.time() - start_time > time_limit:
                            break
                    values[i] = f(simplex[i])
                if time_limit is not None and start_time is not None:
                    if time.time() - start_time > time_limit:
                        break
        
        # Re-sort
        order = np.argsort(values)
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]
        
        if values[0] < best_f:
            best_f = values[0]
            best_x = simplex[0]
        
        # Check convergence
        if max(values) - min(values) < tol:
            break
    
    return best_x, best_f


def coefficients(stages):
    """Find optimal product formula coefficients via numerical optimization."""
    start_time = time.time()
    time_budget = 140.0  # Leave 10s margin before 150s kill
    
    # Build Hamiltonians
    A, B = _build_hamiltonians()
    dim = A.shape[0]
    times = [0.25, 0.5, 1.0]
    
    # Precompute expm for efficiency - actually we need to compute these each time
    # since a_k and b_k change
    
    # Initial guess: Strang-like
    a_init = [0.5 / (stages // 2)] * (2 * (stages // 2))
    b_init = []
    substeps = stages // 2
    for _ in range(substeps):
        b_init += [1.0 / substeps, 0.0]
    while len(a_init) < stages:
        a_init.append(0.0)
        b_init.append(0.0)
    
    # Combine into single parameter vector
    x0 = a_init + b_init
    
    # Constraint: sum(a) = 1, sum(b) = 1
    # We'll use a penalty approach
    
    def penalized_objective(x):
        a = list(x[:stages])
        b = list(x[stages:2*stages])
        # Penalty for constraint violation
        penalty = 100.0 * (sum(a) - 1.0)**2 + 100.0 * (sum(b) - 1.0)**2
        base = _objective(a, b, A, B, times)
        return base + penalty
    
    # Optimize
    best_x, best_val = _nelder_mead(
        penalized_objective, x0,
        max_iter=3000, tol=1e-10,
        time_limit=time_budget - (time.time() - start_time),
        start_time=start_time
    )
    
    # Extract and enforce constraints exactly
    a = list(best_x[:stages])
    b = list(best_x[stages:2*stages])
    
    # Normalize to ensure sum = 1
    sum_a = sum(a)
    sum_b = sum(b)
    if abs(sum_a) > 1e-12:
        a = [ai / sum_a for ai in a]
    else:
        a = [1.0 / stages] * stages
    if abs(sum_b) > 1e-12:
        b = [bi / sum_b for bi in b]
    else:
        b = [1.0 / stages] * stages
    
    # Clamp to [-50, 50]
    a = [max(-50.0, min(50.0, ai)) for ai in a]
    b = [max(-50.0, min(50.0, bi)) for bi in b]
    
    # Final check on time
    if time.time() - start_time > time_budget - 5.0:
        # Return the normalized result
        pass
    
    return a, b
