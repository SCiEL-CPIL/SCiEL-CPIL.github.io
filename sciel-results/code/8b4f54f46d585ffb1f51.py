import numpy as np
from scipy.linalg import expm
import time

def _build_hamiltonians():
    """
    Build the odd-bond (A) and even-bond (B) Hamiltonians for a 6-site
    open XXZ Heisenberg chain with anisotropy delta=0.8 and field h=0.3.
    
    The Hamiltonian is:
    H = delta * sum_i S_i^z S_{i+1}^z + (1/2) * sum_i (S_i^+ S_{i+1}^- + S_i^- S_{i+1}^+)
        - h * sum_i S_i^z
    
    We split bonds:
    - Even bonds (i=0,2,4) -> B
    - Odd bonds (i=1,3) -> A
    
    The uniform field is split half to A and half to B.
    
    Sites are indexed 0..5.
    S^z = diag(1/2, -1/2)
    S^+ = [[0,1],[0,0]], S^- = [[0,0],[1,0]]
    """
    delta = 0.8
    h = 0.3
    n_sites = 6
    
    # Single-site operators
    Sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    Sp = np.array([[0, 1], [0, 0]], dtype=complex)
    Sm = np.array([[0, 0], [1, 0]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    
    # Build full operators using tensor products
    def embed(op, site):
        """Embed a 2x2 operator at a given site in the 6-site chain."""
        result = op
        for i in range(n_sites - 1):
            if i != site:
                result = np.kron(result, I2)
            else:
                # This is a bit tricky; let's do it differently
                pass
        
        # Better approach: build by kron
        ops = [I2] * n_sites
        ops[site] = op
        result = ops[0]
        for i in range(1, n_sites):
            result = np.kron(result, ops[i])
        return result
    
    def embed2(op1, site1, op2, site2):
        """Embed two operators at two sites."""
        ops = [I2] * n_sites
        ops[site1] = op1
        ops[site2] = op2
        result = ops[0]
        for i in range(1, n_sites):
            result = np.kron(result, ops[i])
        return result
    
    dim = 2 ** n_sites
    
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    # Odd bonds: (0,1), (2,3), (4,5) -> wait, let me re-read the task
    # "odd- and even-bond halves"
    # For a 6-site chain, bonds are between sites (0,1), (1,2), (2,3), (3,4), (4,5)
    # Let's say even bonds are (0,1), (2,3), (4,5) and odd bonds are (1,2), (3,4)
    # Or vice versa. The task says "split into even and odd bonds".
    # Let me assume:
    # Even bonds: (0,1), (2,3), (4,5) -> B
    # Odd bonds: (1,2), (3,4) -> A
    
    # Actually, let me think about this more carefully. The task says:
    # "A and B are the odd- and even-bond halves (each including half of the uniform field)"
    # So A = odd bonds + half field, B = even bonds + half field
    
    # Let's define:
    # Even bonds: (0,1), (2,3), (4,5)
    # Odd bonds: (1,2), (3,4)
    
    # For each bond (i, i+1):
    # H_bond = delta * Sz_i Sz_{i+1} + 0.5 * (Sp_i Sm_{i+1} + Sm_i Sp_{i+1})
    
    for i in range(n_sites - 1):
        bond_op = delta * embed2(Sz, i, Sz, i+1) + \
                   0.5 * embed2(Sp, i, Sm, i+1) + \
                   0.5 * embed2(Sm, i, Sp, i+1)
        
        if i % 2 == 0:  # Even bond: (0,1), (2,3), (4,5)
            B += bond_op
        else:  # Odd bond: (1,2), (3,4)
            A += bond_op
    
    # Add half the uniform field to each
    # Field term: -h * sum_i Sz_i
    for i in range(n_sites):
        field_op = -0.5 * h * embed(Sz, i)
        A += field_op
        B += field_op
    
    return A, B

def _compute_score(a, b, A, B, times=(0.25, 0.5, 1.0)):
    """Compute the mean spectral norm error."""
    H = A + B
    errors = []
    for t in times:
        # Exact propagator
        U_exact = expm(-1j * t * H)
        
        # Product formula: prod_k exp(-i b_k t B) exp(-i a_k t A) for k=1..12
        U_approx = np.eye(2 ** 6, dtype=complex)
        for k in range(12):
            U_approx = expm(-1j * b[k] * t * B) @ expm(-1j * a[k] * t * A) @ U_approx
        
        diff = U_approx - U_exact
        # Spectral norm (largest singular value)
        s = np.linalg.svd(diff, compute_uv=False)
        errors.append(s[0])
    
    return np.mean(errors)

def coefficients(stages):
    """
    Find optimal coefficients for a 12-stage product formula.
    Uses a two-phase approach:
    1. Start with a symmetric 4th-order Strang-based pattern
    2. Refine using scipy.optimize.minimize to minimize the error
    """
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 150s limit
    
    if stages != 12:
        # Fallback: uniform split
        a = [1.0 / stages] * stages
        b = [1.0 / stages] * stages
        return a, b
    
    A, B = _build_hamiltonians()
    
    # Initial guess: symmetric 4th-order pattern
    # Use a 4-step Strang with symmetry
    # Pattern: B A B A B A B A B A B A (12 stages alternating)
    # But we need sum(a)=1 and sum(b)=1
    
    # Start with a symmetric 6th-order-like pattern
    # For 12 stages, we can use a symmetric pattern:
    # b a b a b a b a b a b a
    # with symmetry: b[0]=b[11], a[0]=a[11], etc.
    
    # Simple initial guess: equal split with Strang structure
    # 4 substeps of Strang: each substep has b, a, b
    # b coeff = 1/8, a coeff = 1/4 for each substep
    # Total b: 4 * 2 * (1/8) = 1, Total a: 4 * (1/4) = 1
    
    a0 = np.zeros(12)
    b0 = np.zeros(12)
    
    # 4 Strang substeps
    for step in range(4):
        idx = step * 3
        b0[idx] = 1.0 / 8.0
        a0[idx + 1] = 1.0 / 4.0
        b0[idx + 2] = 1.0 / 8.0
    
    # Define the optimization variable: 24 coefficients with constraints
    # We'll optimize over x = [a_0..a_11, b_0..b_11]
    # Subject to sum(a) = 1, sum(b) = 1
    
    def objective(x):
        a = x[:12]
        b = x[12:24]
        
        # Ensure constraints are approximately satisfied by projecting
        a = a - np.mean(a - 1.0/12)  # This doesn't enforce sum=1 properly
        # Better: just compute with the raw values and let the optimizer handle it
        # But we need sum(a)=1 and sum(b)=1
        
        # Project to satisfy constraints
        a_sum = np.sum(a)
        b_sum = np.sum(b)
        if a_sum != 0:
            a = a * (1.0 / a_sum)
        else:
            a = np.ones(12) / 12.0
        if b_sum != 0:
            b = b * (1.0 / b_sum)
        else:
            b = np.ones(12) / 12.0
        
        return _compute_score(a, b, A, B)
    
    x0 = np.concatenate([a0, b0])
    
    # Try to optimize
    best_x = x0.copy()
    best_score = objective(x0)
    
    try:
        from scipy.optimize import minimize
        
        # Use SLSQP with bounds
        bounds = [(-50.0, 50.0)] * 24
        
        # Constraint: sum(a) = 1, sum(b) = 1
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x[:12]) - 1.0},
            {'type': 'eq', 'fun': lambda x: np.sum(x[12:24]) - 1.0}
        ]
        
        result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                         constraints=constraints, options={'maxiter': 100, 'ftol': 1e-8})
        
        if result.success and result.fun < best_score:
            best_x = result.x
            best_score = result.fun
        
        # Check time
        if time.time() < deadline - 30:
            # Try L-BFGS-B with penalty for constraints
            def penalized_objective(x):
                a = x[:12]
                b = x[12:24]
                penalty = 100.0 * ((np.sum(a) - 1.0)**2 + (np.sum(b) - 1.0)**2)
                return objective(x) + penalty
            
            result2 = minimize(penalized_objective, x0, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 200, 'ftol': 1e-10})
            
            if result2.fun < best_score:
                # Project back to satisfy constraints
                a_opt = result2.x[:12]
                b_opt = result2.x[12:24]
                a_opt = a_opt / np.sum(a_opt)
                b_opt = b_opt / np.sum(b_opt)
                best_x = np.concatenate([a_opt, b_opt])
                best_score = _compute_score(a_opt, b_opt, A, B)
    
    except Exception:
        pass
    
    # Extract final coefficients
    a_final = best_x[:12].copy()
    b_final = best_x[12:24].copy()
    
    # Ensure constraints are exactly satisfied
    a_final = a_final / np.sum(a_final)
    b_final = b_final / np.sum(b_final)
    
    # Clamp to [-50, 50]
    a_final = np.clip(a_final, -50.0, 50.0)
    b_final = np.clip(b_final, -50.0, 50.0)
    
    # Re-normalize after clipping
    a_final = a_final / np.sum(a_final)
    b_final = b_final / np.sum(b_final)
    
    return list(a_final), list(b_final)
