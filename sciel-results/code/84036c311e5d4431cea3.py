import numpy as np
from scipy.linalg import expm
import time

def _build_hamiltonians():
    """
    Builds the A and B Hamiltonians for a 6-site open XXZ Heisenberg chain.
    Anisotropy delta = 0.8, field h = 0.3.
    A: Odd bonds (1-2, 3-4, 5-6) + half field
    B: Even bonds (2-3, 4-5) + half field
    Note: Open boundary means no bond between 6 and 1.
    Sites are indexed 0 to 5.
    Bonds:
    0-1 (Odd)
    1-2 (Even)
    2-3 (Odd)
    3-4 (Even)
    4-5 (Odd)
    
    XXZ Hamiltonian term for bond (i, i+1):
    H_bond = 0.5 * (S_i^x S_{i+1}^x + S_i^y S_{i+1}^y) + delta * S_i^z S_{i+1}^z
    Field term: h * S_i^z for each site i.
    The field is split equally between A and B? 
    "A and B are the odd- and even-bond halves (each including half of the uniform field)"
    So A = sum_{odd bonds} H_bond + 0.5 * h * sum_{all sites} S_i^z
       B = sum_{even bonds} H_bond + 0.5 * h * sum_{all sites} S_i^z
    """
    N = 6
    delta = 0.8
    h = 0.3
    
    # Pauli matrices
    Sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    Sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    Sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    
    def kron_chain(ops):
        """ops is a list of operators for each site"""
        res = ops[0]
        for op in ops[1:]:
            res = np.kron(res, op)
        return res

    A = np.zeros((2**N, 2**N), dtype=complex)
    B = np.zeros((2**N, 2**N), dtype=complex)
    
    # Field term (split equally)
    field_term = 0.0
    for i in range(N):
        ops = [I2] * N
        ops[i] = Sz
        field_term += h * 0.5 * kron_chain(ops)
        
    A += field_term
    B += field_term
    
    # Bonds
    # Odd bonds: (0,1), (2,3), (4,5)
    # Even bonds: (1,2), (3,4)
    
    def add_bond(H, i, j):
        # Add XXZ term for sites i and j
        # Sx_i Sx_j + Sy_i Sy_j + delta Sz_i Sz_j
        # We need to construct these terms
        for op1, op2 in [(Sx, Sx), (Sy, Sy), (Sz, Sz)]:
            coeff = 1.0
            if op1 is Sz:
                coeff = delta
            ops = [I2] * N
            ops[i] = op1
            ops[j] = op2
            H += coeff * kron_chain(ops)

    # Odd bonds to A
    add_bond(A, 0, 1)
    add_bond(A, 2, 3)
    add_bond(A, 4, 5)
    
    # Even bonds to B
    add_bond(B, 1, 2)
    add_bond(B, 3, 4)
    
    return A, B

def _compute_error(coeffs, A, B, exact_propagators):
    """
    coeffs: (a, b) lists of 12 floats
    A, B: Hamiltonians
    exact_propagators: dict t -> expm(-i t (A+B))
    """
    a, b = coeffs
    errors = []
    for t, exact in exact_propagators.items():
        U = np.eye(2**6, dtype=complex)
        # Product formula: prod_{k=1}^{12} exp(-i b_k t B) exp(-i a_k t A)
        # Order k=1..12
        for k in range(12):
            # exp(-i a_k t A)
            U = U @ expm(-1j * a[k] * t * A)
            # exp(-i b_k t B)
            U = U @ expm(-1j * b[k] * t * B)
        err = np.linalg.norm(U - exact, 2)
        errors.append(err)
    return np.mean(errors)

def coefficients(stages):
    """
    Optimizes 12-stage product formula coefficients.
    """
    start_time = time.time()
    deadline = start_time + 50.0 # Safety margin
    
    A, B = _build_hamiltonians()
    H = A + B
    
    t_values = [0.25, 0.5, 1.0]
    exact_propagators = {t: expm(-1j * t * H) for t in t_values}
    
    # Initial guess: 6th order symmetric splitting coefficients
    # Reference: Kahan, 1993; or standard tables.
    # 6th order symmetric coefficients for 10 stages usually, but we need 12.
    # Let's use a generic symmetric high-order set or just optimize from a reasonable start.
    # A simple 4th order Strang-like with more steps might be a good start.
    # Or use known 6th order coefficients scaled/adjusted.
    # Let's start with a 4th order symmetric splitting repeated/adjusted.
    # Actually, let's just use a flat distribution as a safe start, or better, 
    # use the coefficients from a known 8th order splitting if available.
    # Since I don't have them memorized perfectly, I will use a heuristic start:
    # Symmetric distribution with weights that sum to 1.
    
    # Symmetric 12-stage: a1=a12, a2=a11, ..., a6=a7
    # b1=b12, b2=b11, ..., b6=b7
    
    # Start with 4th order symmetric coefficients (if we had 4 stages) and map to 12?
    # Better: Start with a "good" symmetric distribution.
    # For high order, coefficients often have alternating signs.
    # Let's initialize with small random perturbations around a symmetric base.
    
    # Base: Uniform distribution is 2nd order.
    # Let's try to initialize with coefficients that satisfy sum=1 and are symmetric.
    # We will optimize 6 independent a's and 6 independent b's.
    # x = [a1, a2, a3, a4, a5, a6, b1, b2, b3, b4, b5, b6]
    
    # Initial guess: 
    # a: [0.1, 0.15, 0.2, 0.2, 0.15, 0.1] -> sum = 1.0? 0.1+0.15+0.2+0.2+0.15+0.1 = 0.9. 
    # Let's just use uniform 1/12 as a safe start, or better, 
    # use coefficients from a 6th order splitting found online/literature.
    # 6th order symmetric (10 stages) coefficients:
    # a1 = 1/12, a2 = 1/12, ... no, that's not right.
    
    # Let's use a simple local optimization starting from a symmetric 4th-order-like guess.
    # 4th order symmetric (4 stages): a=[1/4, 1/4, 1/4, 1/4], b=[1/2, 1/2, 1/2, 1/2] (Strang is a=[1/2,0,0,1/2], b=[1,0,0,1]... wait)
    # Strang: exp(-i t/2 A) exp(-i t B) exp(-i t/2 A).
    # For 12 stages, let's just start with uniform and let the optimizer do the work.
    # Uniform: a_k = 1/12, b_k = 1/12.
    
    x0 = np.array([1.0/12.0] * 12)
    
    def objective(x):
        # x is 12 independent variables: a1..a6, b1..b6
        a_ind, b_ind = x[:6], x[6:]
        # Expand to 12 with symmetry
        a = np.array([a_ind[0], a_ind[1], a_ind[2], a_ind[3], a_ind[4], a_ind[5],
                      a_ind[5], a_ind[4], a_ind[3], a_ind[2], a_ind[1], a_ind[0]])
        b = np.array([b_ind[0], b_ind[1], b_ind[2], b_ind[3], b_ind[4], b_ind[5],
                      b_ind[5], b_ind[4], b_ind[3], b_ind[2], b_ind[1], b_ind[0]])
        
        # Check constraints? The optimizer might drift.
        # We should enforce sum(a)=1, sum(b)=1.
        # Since we are using Nelder-Mead or L-BFGS-B, we can use bounds or penalty.
        # Penalty is easier.
        err_a = np.sum(a) - 1.0
        err_b = np.sum(b) - 1.0
        penalty = 1e6 * (err_a**2 + err_b**2)
        
        if time.time() > deadline:
            return 1e10
            
        # Clip to avoid huge expm errors if optimizer goes wild
        a_clip = np.clip(a, -50, 50)
        b_clip = np.clip(b, -50, 50)
        
        base_err = _compute_error((a_clip, b_clip), A, B, exact_propagators)
        return base_err + penalty

    from scipy.optimize import minimize
    
    # Use L-BFGS-B with bounds
    bounds = [(-50, 50)] * 12
    
    try:
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 50, 'maxfun': 200, 'disp': False})
        
        x_opt = res.x
        a_ind, b_ind = x_opt[:6], x_opt[6:]
        a = np.array([a_ind[0], a_ind[1], a_ind[2], a_ind[3], a_ind[4], a_ind[5],
                      a_ind[5], a_ind[4], a_ind[3], a_ind[2], a_ind[1], a_ind[0]])
        b = np.array([b_ind[0], b_ind[1], b_ind[2], b_ind[3], b_ind[4], b_ind[5],
                      b_ind[5], b_ind[4], b_ind[3], b_ind[2], b_ind[1], b_ind[0]])
        
        # Normalize to ensure exact sum 1 (fixing any numerical drift)
        sum_a = np.sum(a)
        sum_b = np.sum(b)
        if abs(sum_a) > 1e-10:
            a = a / sum_a
        if abs(sum_b) > 1e-10:
            b = b / sum_b
            
        # Final check
        a = np.clip(a, -50, 50)
        b = np.clip(b, -50, 50)
        
        return list(a), list(b)
        
    except Exception:
        # Fallback to seed-like behavior or simple uniform
        a = [1.0/12.0] * 12
        b = [1.0/12.0] * 12
        return a, b

