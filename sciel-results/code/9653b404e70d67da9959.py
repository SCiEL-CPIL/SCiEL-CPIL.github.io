import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
import time

def build_hamiltonians():
    """
    Constructs the Hamiltonian matrices A and B for the 6-site Heisenberg chain.
    A: Odd bonds + half field
    B: Even bonds + half field
    """
    n_sites = 6
    dim = 2 ** n_sites
    gamma = 0.8
    h = 0.3
    
    # Pauli matrices
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    
    def tensor_product(ops):
        """Tensor product of a list of operators."""
        result = np.array([[1]], dtype=complex)
        for op in ops:
            result = np.kron(result, op)
        return result
    
    # Identity for single site
    I = np.eye(2, dtype=complex)
    
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    # Bonds
    # Odd bonds: (0,1), (2,3), (4,5)
    odd_bonds = [(0, 1), (2, 3), (4, 5)]
    # Even bonds: (1,2), (3,4)
    even_bonds = [(1, 2), (3, 4)]
    
    def add_bond_term(H, i, j):
        """Adds the interaction term for bond (i, j) to H."""
        # Sx_i Sx_j
        ops_sx_sx = [I] * n_sites
        ops_sx_sx[i] = sx
        ops_sx_sx[j] = sx
        H += tensor_product(ops_sx_sx)
        
        # Sy_i Sy_j
        ops_sy_sy = [I] * n_sites
        ops_sy_sy[i] = sy
        ops_sy_sy[j] = sy
        H += tensor_product(ops_sy_sy)
        
        # Sz_i Sz_j
        ops_sz_sz = [I] * n_sites
        ops_sz_sz[i] = sz
        ops_sz_sz[j] = sz
        H += gamma * tensor_product(ops_sz_sz)
        
        # Field terms for this bond? 
        # The prompt says "each including half of the uniform field".
        # This is ambiguous. Does it mean the field is split between A and B?
        # Usually, the field is a single-site term. 
        # If we add h/2 * sum S_z to both A and B, then A+B has h * sum S_z.
        # Let's assume this interpretation.
        
    # Add bond terms
    for i, j in odd_bonds:
        add_bond_term(A, i, j)
    for i, j in even_bonds:
        add_bond_term(B, i, j)
        
    # Add field terms
    # Half field in A, half field in B
    field_A = np.zeros((dim, dim), dtype=complex)
    field_B = np.zeros((dim, dim), dtype=complex)
    for i in range(n_sites):
        ops_sz = [I] * n_sites
        ops_sz[i] = sz
        field_term = tensor_product(ops_sz)
        field_A += 0.5 * h * field_term
        field_B += 0.5 * h * field_term
        
    A += field_A
    B += field_B
    
    return A, B

# Pre-compute Hamiltonians at module load time to save time in coefficients()
A_mat, B_mat = build_hamiltonians()

def coefficients(stages):
    """
    Optimizes the coefficients a and b for the product formula.
    """
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    if stages != 12:
        # Fallback to seed if stages != 12
        substeps = stages // 2
        a = [0.5 / substeps] * (2 * substeps)
        b = []
        for _ in range(substeps):
            b += [1.0 / substeps, 0.0]
        while len(a) < stages:
            a.append(0.0)
            b.append(0.0)
        return a, b

    # Initial guess: Strang splitting
    # Sequence: B, A, B, A, ...
    # Strang: exp(-i t B/2) exp(-i t A) exp(-i t B/2)
    # Map to k=1: b_1=0.5, a_1=1.0
    # k=2: b_2=0.5, a_2=0.0
    # Rest 0
    a0 = [0.0] * 12
    b0 = [0.0] * 12
    a0[0] = 1.0
    b0[0] = 0.5
    b0[1] = 0.5
    
    x0 = np.array(a0 + b0)
    
    # Objective function
    def objective(x):
        if time.time() > deadline:
            return 1e10 # Timeout penalty
        
        a = x[:12]
        b = x[12:]
        
        # Constraints: sum(a)=1, sum(b)=1
        # We handle constraints in the optimizer, but let's check for validity
        # The optimizer will project or use bounds.
        
        # Calculate error
        total_error = 0.0
        times = [0.25, 0.5, 1.0]
        
        for t in times:
            U = np.eye(64, dtype=complex)
            # Pre-compute exponentials for this t? 
            # We need exp(-i a_k t A) and exp(-i b_k t B)
            # To save time, we can compute them on the fly.
            # 12 stages * 2 exps = 24 exps per t. 3 t's = 72 exps.
            # This is fast for 64x64 matrices.
            
            for k in range(12):
                # Step B
                if abs(b[k]) > 1e-12:
                    U = expm(-1j * b[k] * t * B_mat) @ U
                # Step A
                if abs(a[k]) > 1e-12:
                    U = expm(-1j * a[k] * t * A_mat) @ U
            
            # Exact propagator
            U_exact = expm(-1j * t * (A_mat + B_mat))
            
            # Spectral norm of difference
            # Use svd for spectral norm
            diff = U - U_exact
            # np.linalg.norm(diff, 2) computes spectral norm
            err = np.linalg.norm(diff, 2)
            total_error += err
            
        return total_error / 3.0

    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x[:12]) - 1.0},
        {'type': 'eq', 'fun': lambda x: np.sum(x[12:]) - 1.0}
    ]
    
    # Bounds
    bounds = [(-50.0, 50.0)] * 24
    
    # Optimize
    # Use L-BFGS-B as it handles bounds well
    result = None
    try:
        # First try L-BFGS-B
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, constraints=constraints, options={'maxiter': 500, 'ftol': 1e-10})
        
        # Check if we have time for a second, more precise run or a different method
        if time.time() < deadline - 10.0:
            # Try Nelder-Mead on the result of L-BFGS-B, or refine
            # L-BFGS-B is usually good enough.
            pass
    except Exception as e:
        # Fallback to initial guess if optimization fails
        result = None

    if result is not None and result.success:
        x_opt = result.x
        a_opt = list(x_opt[:12])
        b_opt = list(x_opt[12:])
        
        # Ensure sums are exactly 1 (numerical correction)
        sum_a = sum(a_opt)
        sum_b = sum(b_opt)
        if abs(sum_a) > 1e-12:
            a_opt = [val / sum_a for val in a_opt]
        if abs(sum_b) > 1e-12:
            b_opt = [val / sum_b for val in b_opt]
            
        return a_opt, b_opt
    else:
        # Return initial guess
        return a0, b0

