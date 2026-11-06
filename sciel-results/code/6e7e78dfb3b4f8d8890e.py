import numpy as np
from scipy.optimize import minimize

def _build_hamiltonians():
    # Spin-1/2 matrices
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    
    N = 6
    delta = 0.8
    h_field = 0.3
    
    A = np.zeros((2**N, 2**N), dtype=complex)
    B = np.zeros((2**N, 2**N), dtype=complex)
    
    # Helper to add term to A or B
    def add_term(matrix, site1, site2, coeff):
        # site1, site2 are 0-indexed
        # Create operator on site1 and site2
        # This is a bit slow if done naively with kron, but N=6 is small
        ops = [np.eye(2, dtype=complex) for _ in range(N)]
        # Term: 0.5 Sx Sx + 0.5 Sy Sy + Delta Sz Sz
        # SxSx + SySy = 2 (S+ S- + S- S+)
        # Let's just use the matrices directly
        sx_sx = np.kron(sx, sx)
        sy_sy = np.kron(sy, sy)
        sz_sz = np.kron(sz, sz)
        
        # We need to place these at the correct sites
        # Efficient way: iterate over all basis states? No, use kron
        # For N=6, we can build the full operator
        pass

    # More efficient construction for small N
    # Build identity
    I = np.eye(2, dtype=complex)
    
    def kron_list(ops):
        res = ops[0]
        for op in ops[1:]:
            res = np.kron(res, op)
        return res

    for j in range(N - 1):
        # Bond between j and j+1
        # Determine if even or odd bond
        # Bond 0 (sites 0-1) is Even? 
        # Prompt: "even and odd bonds". 
        # Usually bond i is between site i and i+1.
        # If bond index is even (0, 2, 4), it's an even bond? 
        # Or if site indices are even?
        # "odd- and even-bond halves"
        # Standard convention: 
        # A (odd): bonds (1,2), (3,4)... in 1-based indexing -> (0,1), (2,3) in 0-based?
        # Let's assume bond index j (0-based).
        # If j is even (0, 2, 4), it connects site j to j+1.
        # Let's call j=0,2,4 "Even bonds" and j=1,3 "Odd bonds".
        # Or vice versa?
        # "A and B are the odd- and even-bond halves"
        # Usually, the split is A = odd bonds, B = even bonds.
        # Let's assume:
        # A: bonds (1,2), (3,4), (5,6) in 1-based -> (0,1), (2,3), (4,5) in 0-based. These are j=0, 2, 4.
        # B: bonds (2,3), (4,5) in 1-based -> (1,2), (3,4) in 0-based. These are j=1, 3.
        # Wait, 6 sites: bonds are 1-2, 2-3, 3-4, 4-5, 5-6.
        # Indices 0-1, 1-2, 2-3, 3-4, 4-5.
        # Even bonds (j=0,2,4): (0,1), (2,3), (4,5).
        # Odd bonds (j=1,3): (1,2), (3,4).
        
        # Let's assign:
        # A: Odd bonds (j=1, 3) + h/2 * Sum Sz
        # B: Even bonds (j=0, 2, 4) + h/2 * Sum Sz
        
        # Calculate interaction term
        # H_int = 0.5 SxSx + 0.5 SySy + Delta SzSz
        # We need to apply this to sites j and j+1
        
        ops_A = [I for _ in range(N)]
        ops_B = [I for _ in range(N)]
        
        # Interaction matrix on 2 sites
        H_2 = 0.5 * np.kron(sx, sx) + 0.5 * np.kron(sy, sy) + delta * np.kron(sz, sz)
        
        # We need to embed H_2 into the N-site space
        # This is tricky with simple kron if sites are not adjacent in the list order? They are adjacent.
        # We can build the full operator by iterating or using a helper.
        
        # Helper to embed 2-site operator at position j
        def embed_2site(op_2, j, N):
            # op_2 is 4x4
            # We need to place it at sites j and j+1
            # This is equivalent to:
            # I^{j} \otimes op_2 \otimes I^{N-j-2}
            # But we must be careful with the order of sites in the tensor product.
            # Standard: Site 0 is leftmost (most significant).
            # So I^{j} means j identities, then op_2, then remaining identities.
            left = np.eye(2**j, dtype=complex)
            right = np.eye(2**(N-j-2), dtype=complex)
            return np.kron(np.kron(left, op_2), right)
            
        H_j = embed_2site(H_2, j, N)
        
        if j % 2 == 1: # Odd bond (1, 3)
            A += H_j
        else: # Even bond (0, 2, 4)
            B += H_j

    # Add field
    # H_field = h * Sum S_z
    # Split: h/2 in A, h/2 in B
    H_field_total = np.zeros((2**N, 2**N), dtype=complex)
    for j in range(N):
        ops = [I for _ in range(N)]
        ops[j] = sz
        H_field_total += kron_list(ops)
        
    A += 0.5 * h_field * H_field_total
    B += 0.5 * h_field * H_field_total
    
    return A, B

_A, _B = None, None

def _get_hamiltonians():
    global _A, _B
    if _A is None:
        _A, _B = _build_hamiltonians()
    return _A, _B

def coefficients(stages):
    A, B = _get_hamiltonians()
    N_dim = A.shape[0]
    
    # Exact propagator components
    # We need exp(-i t (A+B))
    
    def objective(x):
        # x is 24 floats: a1..a12, b1..b12
        a = x[:12]
        b = x[12:]
        
        # Penalty for sum constraints
        sum_a = np.sum(a)
        sum_b = np.sum(b)
        penalty = 100.0 * ((sum_a - 1.0)**2 + (sum_b - 1.0)**2)
        
        if penalty > 1e-4:
            return penalty + 1e10 # Early exit if far from constraint
            
        total_err = 0.0
        times = [0.25, 0.5, 1.0]
        
        # Precompute exps? No, t and coeffs vary.
        # But we can precompute A and B powers? No, they are matrices.
        
        for t in times:
            # Exact
            H_total = A + B
            U_exact = np.linalg.matrix_exp(-1j * t * H_total)
            
            # Approx
            U_approx = np.eye(N_dim, dtype=complex)
            for k in range(12):
                # Order: exp(-i b_k t B) exp(-i a_k t A)
                # k=0 to 11
                # U = ... * ExpB * ExpA
                # The prompt says: prod_k exp(-i b_k t B) exp(-i a_k t A) in order k=1..12
                # This means U = (ExpB_12 ExpA_12) ... (ExpB_1 ExpA_1)
                # So we apply from right to left?
                # U(t) = P_12 ... P_1
                # P_k = ExpB_k ExpA_k
                # So U = P_12 @ ... @ P_1
                
                # In code:
                # U_approx = ExpB_k @ ExpA_k @ U_approx
                # Start with Identity.
                # k=0: U = ExpB_0 @ ExpA_0 @ I
                # k=1: U = ExpB_1 @ ExpA_1 @ U
                # ...
                
                U_approx = np.linalg.matrix_exp(-1j * b[k] * t * B) @ np.linalg.matrix_exp(-1j * a[k] * t * A) @ U_approx
                
            err = np.linalg.norm(U_approx - U_exact, 2)
            total_err += err
            
        return total_err / 3.0 + penalty

    # Initial guess: Suzuki 4th order
    c = 2.0 ** (1.0 / 3.0)
    alpha = 1.0 / (2.0 * (1.0 - c))
    beta = -alpha / c
    gamma = 1.0 - 2.0 * beta
    
    a_init = [alpha/2, alpha/2, 0, 0, alpha/2, alpha/2, alpha/2, alpha/2, 0, 0, alpha/2, alpha/2]
    b_init = [beta/2, beta/2, 0, 0, gamma/2, gamma/2, 0, 0, beta/2, beta/2, 0, 0]
    
    # Normalize init
    sa = sum(a_init)
    sb = sum(b_init)
    a_init = [x/sa for x in a_init]
    b_init = [x/sb for x in b_init]
    
    x0 = np.array(a_init + b_init)
    
    # Bounds
    bounds = [(-50, 50)] * 24
    
    # Optimize
    # Use L-BFGS-B for speed and bound handling
    # maxiter limited to ensure finish
    res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                   options={'maxiter': 50, 'ftol': 1e-6, 'gtol': 1e-4})
    
    x_opt = res.x
    a_opt = x_opt[:12]
    b_opt = x_opt[12:]
    
    # Final normalization to be safe
    sa = np.sum(a_opt)
    sb = np.sum(b_opt)
    if sa != 0: a_opt = a_opt / sa
    if sb != 0: b_opt = b_opt / sb
    
    return list(a_opt), list(b_opt)
