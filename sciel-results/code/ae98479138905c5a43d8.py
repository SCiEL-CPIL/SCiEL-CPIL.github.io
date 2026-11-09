"""Optimized product-formula coefficients for 6-site Heisenberg chain time evolution."""
import numpy as np
import time
import sys

def _build_hamiltonians():
    """Build A (odd bonds + half field) and B (even bonds + half field) for 6-site XXZ chain.
    Anisotropy = 0.8, field = 0.3, open boundary conditions.
    """
    N = 6
    dim = 2 ** N
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    
    anisotropy = 0.8
    field = 0.3
    
    def kron_all(ops):
        result = np.array([[1.0]], dtype=complex)
        for op in ops:
            result = np.kron(result, op)
        return result
    
    # Odd bonds: (0,1), (2,3), (4,5)
    # Even bonds: (1,2), (3,4)
    # Each bond term: sx⊗sx + sy⊗sy + anisotropy * sz⊗sz
    # Half field on each: field/2 * sum_i sz_i
    
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    for i in range(N):
        # Build sz on site i
        ops = [I2] * N
        ops[i] = sz
        A += (field / 2) * kron_all(ops)
        B += (field / 2) * kron_all(ops)
    
    # Odd bonds: (0,1), (2,3), (4,5)
    for (i, j) in [(0, 1), (2, 3), (4, 5)]:
        for sx_or_sy, s_op in [(sx, sx), (sy, sy)]:
            ops = [I2] * N
            ops[i] = s_op
            ops[j] = sx_or_sy
            A += kron_all(ops)
        ops = [I2] * N
        ops[i] = sz
        ops[j] = sz
        A += anisotropy * kron_all(ops)
    
    # Even bonds: (1,2), (3,4)
    for (i, j) in [(1, 2), (3, 4)]:
        for sx_or_sy, s_op in [(sx, sx), (sy, sy)]:
            ops = [I2] * N
            ops[i] = s_op
            ops[j] = sx_or_sy
            B += kron_all(ops)
        ops = [I2] * N
        ops[i] = sz
        ops[j] = sz
        B += anisotropy * kron_all(ops)
    
    return A, B

def _spectral_norm_error(a, b, A, B, times):
    """Compute mean spectral norm error over the given times."""
    dim = A.shape[0]
    H = A + B
    errors = []
    for t in times:
        # Exact propagator
        U_exact = np.linalg.expm(-1j * t * H)
        
        # Product formula
        U_pf = np.eye(dim, dtype=complex)
        for k in range(len(a)):
            U_pf = np.linalg.expm(-1j * b[k] * t * B) @ U_pf
            U_pf = np.linalg.expm(-1j * a[k] * t * A) @ U_pf
        
        err = np.linalg.norm(U_pf - U_exact, 2)
        errors.append(err)
    
    return np.mean(errors)

def coefficients(stages):
    """Return optimized coefficients for the 12-stage product formula."""
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for 150s limit
    
    A, B = _build_hamiltonians()
    times = [0.25, 0.5, 1.0]
    
    # Precompute matrix exponentials for various coefficients to speed up evaluation
    # We'll use a grid-based search followed by local refinement
    
    # Strategy: Use a structured parameterization that maintains sum(a)=sum(b)=1
    # and try to find good coefficients via random search + local optimization
    
    n = stages  # 12
    
    # Precompute exp(-i * c * t * A) and exp(-i * c * t * B) for a range of c values
    # to avoid recomputing expm repeatedly
    c_values = np.linspace(-5, 5, 21)
    exp_A_cache = {}
    exp_B_cache = {}
    for c in c_values:
        for t in times:
            key_A = (round(c, 4), round(t, 4))
            key_B = (round(c, 4), round(t, 4))
            if key_A not in exp_A_cache:
                exp_A_cache[key_A] = np.linalg.expm(-1j * c * t * A)
            if key_B not in exp_B_cache:
                exp_B_cache[key_B] = np.linalg.expm(-1j * c * t * B)
    
    def evaluate(a, b):
        """Evaluate error for given coefficient vectors."""
        errors = []
        for t in times:
            U_pf = np.eye(A.shape[0], dtype=complex)
            for k in range(n):
                # Find closest cached value or compute
                c_b = b[k]
                c_a = a[k]
                
                # Use cached if available, otherwise compute
                key_b = (round(c_b, 4), round(t, 4))
                key_a = (round(c_a, 4), round(t, 4))
                
                if key_b in exp_B_cache:
                    exp_b = exp_B_cache[key_b]
                else:
                    exp_b = np.linalg.expm(-1j * c_b * t * B)
                
                U_pf = exp_b @ U_pf
                
                if key_a in exp_A_cache:
                    exp_a = exp_A_cache[key_a]
                else:
                    exp_a = np.linalg.expm(-1j * c_a * t * A)
                
                U_pf = exp_a @ U_pf
            
            # Exact
            U_exact = np.linalg.expm(-1j * t * (A + B))
            err = np.linalg.norm(U_pf - U_exact, 2)
            errors.append(err)
        return np.mean(errors)
    
    # Initial guess: uniform splitting
    a_init = [1.0 / n] * n
    b_init = [1.0 / n] * n
    best_a = a_init[:]
    best_b = b_init[:]
    best_err = evaluate(best_a, best_b)
    
    # Random search with local refinement
    np.random.seed(42)
    
    # Phase 1: Random restarts with different structures
    for iteration in range(200):
        if time.time() > deadline - 20:
            break
        
        # Generate random coefficients with sum = 1
        # Use Dirichlet-like distribution
        raw_a = np.random.exponential(1.0, n)
        raw_a = raw_a / raw_a.sum()
        raw_b = np.random.exponential(1.0, n)
        raw_b = raw_b / raw_b.sum()
        
        # Clip to [-50, 50] and renormalize
        raw_a = np.clip(raw_a, -50, 50)
        raw_b = np.clip(raw_b, -50, 50)
        
        # Ensure sums are 1
        if abs(raw_a.sum()) > 1e-10:
            raw_a = raw_a / raw_a.sum()
        if abs(raw_b.sum()) > 1e-10:
            raw_b = raw_b / raw_b.sum()
        
        err = evaluate(list(raw_a), list(raw_b))
        if err < best_err:
            best_err = err
            best_a = list(raw_a)
            best_b = list(raw_b)
    
    # Phase 2: Local optimization using gradient-free method
    if time.time() < deadline - 10:
        # Simple coordinate descent
        for coord_iter in range(50):
            if time.time() > deadline - 10:
                break
            
            improved = False
            for k in range(n):
                for delta in [0.01, -0.01, 0.05, -0.05, 0.1, -0.1]:
                    new_a = best_a[:]
                    new_b = best_b[:]
                    new_a[k] += delta
                    new_a[(k + 1) % n] -= delta  # Keep sum = 1
                    new_a = [max(-50, min(50, x)) for x in new_a]
                    new_a_sum = sum(new_a)
                    if abs(new_a_sum) > 1e-10:
                        new_a = [x / new_a_sum for x in new_a]
                    
                    err_a = evaluate(new_a, best_b)
                    if err_a < best_err:
                        best_err = err_a
                        best_a = new_a
                        improved = True
                    
                    new_a = best_a[:]
                    new_b[k] += delta
                    new_b[(k + 1) % n] -= delta
                    new_b = [max(-50, min(50, x)) for x in new_b]
                    new_b_sum = sum(new_b)
                    if abs(new_b_sum) > 1e-10:
                        new_b = [x / new_b_sum for x in new_b]
                    
                    err_b = evaluate(best_a, new_b)
                    if err_b < best_err:
                        best_err = err_b
                        best_b = new_b
                        improved = True
            
            if not improved:
                break
    
    # Final validation
    if time.time() > deadline - 5:
        # Return current best
        pass
    
    # Ensure constraints are met
    a_sum = sum(best_a)
    b_sum = sum(best_b)
    if abs(a_sum) > 1e-10:
        best_a = [x / a_sum for x in best_a]
    if abs(b_sum) > 1e-10:
        best_b = [x / b_sum for x in best_b]
    
    # Clip
    best_a = [max(-50, min(50, x)) for x in best_a]
    best_b = [max(-50, min(50, x)) for x in best_b]
    
    # Renormalize after clipping
    a_sum = sum(best_a)
    b_sum = sum(best_b)
    if abs(a_sum) > 1e-10:
        best_a = [x / a_sum for x in best_a]
    if abs(b_sum) > 1e-10:
        best_b = [x / b_sum for x in best_b]
    
    return best_a, best_b
