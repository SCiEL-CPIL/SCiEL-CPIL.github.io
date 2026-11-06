import numpy as np
from scipy.optimize import minimize
import time

def _build_heisenberg_matrices():
    """Build A (odd bonds + half field) and B (even bonds + half field) for 6-site open XXZ chain."""
    N = 6
    dim = 2 ** N
    anisotropy = 0.8
    field = 0.3
    
    # XXZ Hamiltonian: sum_j (Sx_j Sx_{j+1} + Sy_j Sy_{j+1} + anisotropy * Sz_j Sz_{j+1}) - field * sum_j Sz_j
    # Using spin-1/2 operators
    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    identity = np.eye(2, dtype=complex)
    
    def tensor_product(ops):
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result
    
    # Build bond operators
    def bond_operator(site1, site2, op1, op2):
        ops = [identity] * N
        ops[site1] = op1
        ops[site2] = op2
        return tensor_product(ops)
    
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    # Bonds: (0,1), (1,2), (2,3), (3,4), (4,5)
    # Odd bonds: (0,1), (2,3), (4,5) -> A
    # Even bonds: (1,2), (3,4) -> B
    
    for j in range(N - 1):
        sx_sx = bond_operator(j, j+1, sx, sx)
        sy_sy = bond_operator(j, j+1, sy, sy)
        sz_sz = bond_operator(j, j+1, sz, sz)
        bond = sx_sx + sy_sy + anisotropy * sz_sz
        
        if j % 2 == 0:  # even index bond -> A (odd bond in 1-indexed: bonds 1,3,5)
            A += bond
        else:  # odd index bond -> B (even bond in 1-indexed: bonds 2,4)
            B += bond
    
    # Uniform field: -field * sum_j Sz_j, split half to A and half to B
    field_op = np.zeros((dim, dim), dtype=complex)
    for j in range(N):
        ops = [identity] * N
        ops[j] = sz
        field_op += tensor_product(ops)
    
    A += -0.5 * field * field_op
    B += -0.5 * field * field_op
    
    return A, B

def coefficients(stages):
    """Optimize 12-stage product formula coefficients for Heisenberg chain evolution."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    if stages != 12:
        # Fallback: simple Strang-like
        substeps = max(1, stages // 2)
        a = [0.5 / substeps] * (2 * substeps)
        b = []
        for _ in range(substeps):
            b += [1.0 / substeps, 0.0]
        while len(a) < stages:
            a.append(0.0)
            b.append(0.0)
        return a, b
    
    # Build matrices
    A, B = _build_heisenberg_matrices()
    dim = A.shape[0]
    
    # Precompute exp(-i t A) and exp(-i t B) for the three time points
    times = [0.25, 0.5, 1.0]
    
    from scipy.linalg import expm
    
    # Exact propagator: exp(-i t (A+B))
    AB = A + B
    exact = {t: expm(-1j * t * AB) for t in times}
    
    def compute_error(coeffs):
        """Compute mean spectral norm error for given coefficients."""
        a = coeffs[:12]
        b = coeffs[12:]
        
        total_error = 0.0
        for t in times:
            U = np.eye(dim, dtype=complex)
            for k in range(12):
                # exp(-i b_k t B) @ exp(-i a_k t A)
                U = expm(-1j * b[k] * t * B) @ expm(-1j * a[k] * t * A) @ U
            diff = U - exact[t]
            # Spectral norm = largest singular value
            total_error += np.linalg.norm(diff, 2)
        
        return total_error / 3.0
    
    # Initial guess: symmetric Strang-like
    # For 12 stages, use a symmetric pattern
    # a: [1/12, 1/12, ..., 1/12] with some structure
    # Let's start with uniform distribution
    x0 = np.zeros(24)
    for k in range(12):
        x0[k] = 1.0 / 12.0  # a_k
        x0[12 + k] = 1.0 / 12.0  # b_k
    
    # Try a few initial points and pick the best
    best_x = x0.copy()
    best_val = compute_error(x0)
    
    # Alternative initial guesses
    # Symmetric pattern: a = [1/2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1/2] style
    alt1 = np.zeros(24)
    alt1[0] = 0.5
    alt1[11] = 0.5
    for k in range(12):
        alt1[12+k] = 1.0/12.0
    val1 = compute_error(alt1)
    if val1 < best_val:
        best_val = val1
        best_x = alt1.copy()
    
    # Another: more concentrated
    alt2 = np.zeros(24)
    alt2[3] = 0.25
    alt2[4] = 0.25
    alt2[7] = 0.25
    alt2[8] = 0.25
    for k in range(12):
        alt2[12+k] = 1.0/12.0
    val2 = compute_error(alt2)
    if val2 < best_val:
        best_val = val2
        best_x = alt2.copy()
    
    # Use L-BFGS-B with constraints
    # Constraints: sum(a) = 1, sum(b) = 1
    # We'll handle constraints by parameterizing: a_i = x_i for i<11, a_11 = 1 - sum(x_0..x_10)
    # Similarly for b
    
    def constrained_objective(params):
        """params: 22 values, we reconstruct a and b with sum constraint."""
        a = np.zeros(12)
        b = np.zeros(12)
        for i in range(11):
            a[i] = params[i]
            b[i] = params[11 + i]
        a[11] = 1.0 - np.sum(a[:11])
        b[11] = 1.0 - np.sum(b[:11])
        
        # Clip to [-50, 50] for safety
        a = np.clip(a, -50, 50)
        b = np.clip(b, -50, 50)
        
        coeffs = np.concatenate([a, b])
        return compute_error(coeffs)
    
    # Convert best_x to constrained params
    def to_constrained(x):
        a = x[:12]
        b = x[12:]
        params = np.zeros(22)
        for i in range(11):
            params[i] = a[i]
            params[11+i] = b[i]
        return params
    
    p0 = to_constrained(best_x)
    
    # Run optimization with time check
    try:
        result = minimize(
            constrained_objective,
            p0,
            method='L-BFGS-B',
            bounds=[(-50, 50)] * 22,
            options={'maxiter': 200, 'maxfun': 500, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if result.fun < best_val:
            best_val = result.fun
            p_best = result.x
        else:
            p_best = p0
    except Exception:
        p_best = p0
    
    # Check time
    if time.time() > deadline - 5.0:
        # Not enough time for more iterations
        pass
    else:
        # Try Nelder-Mead on the full 24-dim space with penalty for constraint violation
        def penalized_objective(x):
            a = x[:12]
            b = x[12:]
            penalty = 100.0 * (abs(np.sum(a) - 1.0) + abs(np.sum(b) - 1.0))
            # Also penalty for out of bounds
            penalty += 100.0 * (np.sum(np.maximum(0, a - 50)) + np.sum(np.maximum(0, -50 - a)) +
                               np.sum(np.maximum(0, b - 50)) + np.sum(np.maximum(0, -50 - b)))
            return compute_error(x) + penalty
        
        try:
            result2 = minimize(
                penalized_objective,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            if result2.fun < best_val:
                # Extract a and b, project to satisfy constraints
                a = result2.x[:12]
                b = result2.x[12:]
                # Normalize to sum to 1
                if abs(np.sum(a)) > 1e-10:
                    a = a / np.sum(a)
                if abs(np.sum(b)) > 1e-10:
                    b = b / np.sum(b)
                # Clip
                a = np.clip(a, -50, 50)
                b = np.clip(b, -50, 50)
                # Re-normalize after clipping
                if abs(np.sum(a)) > 1e-10:
                    a = a / np.sum(a)
                if abs(np.sum(b)) > 1e-10:
                    b = b / np.sum(b)
                best_x = np.concatenate([a, b])
                best_val = compute_error(best_x)
        except Exception:
            pass
    
    # Final extraction
    a = best_x[:12]
    b = best_x[12:]
    
    # Ensure constraints
    a_sum = np.sum(a)
    b_sum = np.sum(b)
    if abs(a_sum) > 1e-10:
        a = a / a_sum
    else:
        a = np.ones(12) / 12.0
    if abs(b_sum) > 1e-10:
        b = b / b_sum
    else:
        b = np.ones(12) / 12.0
    
    # Clip and re-normalize
    a = np.clip(a, -50, 50)
    b = np.clip(b, -50, 50)
    a_sum = np.sum(a)
    b_sum = np.sum(b)
    if abs(a_sum) > 1e-10:
        a = a / a_sum
    if abs(b_sum) > 1e-10:
        b = b / b_sum
    
    return list(a), list(b)
