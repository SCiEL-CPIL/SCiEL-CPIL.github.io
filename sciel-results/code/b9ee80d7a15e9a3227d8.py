import numpy as np
import time
import scipy.linalg as la
import scipy.optimize as opt

def coefficients(stages):
    start_time = time.time()
    time_limit = 140.0  # Leave 10s margin for safety
    
    # Build the Hamiltonian
    # 6-site open XXZ Heisenberg chain
    # H = sum_{i=1}^{N-1} (S_i^x S_{i+1}^x + S_i^y S_{i+1}^y + delta S_i^z S_{i+1}^z) - h sum S_i^z
    # Anisotropy delta = 0.8, Field h = 0.3
    # Split into A (odd bonds) and B (even bonds)
    # Each includes half the field.
    
    N = 6
    dim = 2 ** N
    delta = 0.8
    h = 0.3
    
    # Pauli matrices
    sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    
    # Identity
    I2 = np.eye(2, dtype=complex)
    
    # Helper to apply operator on site i
    def op_on_site(op, i):
        ops = [I2] * N
        ops[i] = op
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result
    
    # Build A and B
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    # Field term: -h sum S_i^z
    # Split half to A, half to B
    field_term = -h * 0.5
    for i in range(N):
        field_op = field_term * op_on_site(sz, i)
        A += field_op
        B += field_op
        
    # Interaction terms
    # Bonds: (0,1), (1,2), (2,3), (3,4), (4,5)
    # Odd bonds: (0,1), (2,3), (4,5) -> A
    # Even bonds: (1,2), (3,4) -> B
    
    def add_interaction(mat, i, j):
        # S_i^x S_j^x + S_i^y S_j^y + delta S_i^z S_j^z
        # Note: S_i^x S_j^x + S_i^y S_j^y = 0.5 * (S_i^+ S_j^- + S_i^- S_j^+)
        # Let's just use the Pauli product form for simplicity, it's small enough.
        term_xx = np.kron(op_on_site(sx, i), op_on_site(sx, j)) # This is wrong, kron is for independent spaces
        # Correct way:
        # H_ij = S_i^x S_j^x + S_i^y S_j^y + delta S_i^z S_j^z
        # We can construct this by iterating over basis states or using kron carefully.
        pass

    # More efficient construction for 6 sites:
    # We can use the fact that we only need 64x64 matrices.
    # Let's build them explicitly.
    
    def get_op_on_site(site, op):
        res = op
        for i in range(N):
            if i == site:
                res = np.kron(res, op)
            else:
                res = np.kron(res, I2)
        # This puts op at the LAST position if we start from left.
        # We want op at position 'site' (0 is leftmost or rightmost?).
        # Standard: kron order. If we do kron(op0, op1, ..., op5), op0 is leftmost.
        # Let's stick to a consistent order.
        pass

    # Let's use a simpler loop to build A and B
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    # Field
    for i in range(N):
        sz_i = op_on_site(sz, i)
        A += -0.5 * h * sz_i
        B += -0.5 * h * sz_i
        
    # Interactions
    # Bond (i, i+1)
    def bond_op(i, j):
        # Returns matrix for S_i^x S_j^x + S_i^y S_j^y + delta S_i^z S_j^z
        # This is tricky with kron.
        # Let's just compute it element-wise or use a helper.
        # Actually, for N=6, we can just build the full Hamiltonian and split.
        pass

    # Let's build the full H first, then split.
    H = np.zeros((dim, dim), dtype=complex)
    for i in range(N):
        sz_i = op_on_site(sz, i)
        H += -h * sz_i
        
    for i in range(N-1):
        # Term on bond i, i+1
        # S_i^x S_{i+1}^x + S_i^y S_{i+1}^y + delta S_i^z S_{i+1}^z
        # We can construct this by applying operators.
        # Since N is small, we can iterate over basis states.
        # |b_0 b_1 ... b_5>
        # This might be slow in Python if not vectorized, but 64 states is tiny.
        pass

    # Vectorized construction of H
    # Basis states 0 to 63
    # b_k = (state >> (N-1-k)) & 1
    # S^z |b> = (-1)^b |b> * 0.5
    # S^x |b> = |b flipped at k>
    # S^y |b> = (-i)^{...} |b flipped at k>
    
    # Let's just use the matrix multiplication approach with kron, it's safe.
    # Re-implement op_on_site correctly.
    def kron_all(ops):
        res = ops[0]
        for op in ops[1:]:
            res = np.kron(res, op)
        return res

    def op_at(site, op):
        ops = [I2] * N
        ops[site] = op
        return kron_all(ops)

    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    for i in range(N):
        sz_i = op_at(i, sz)
        A += -0.5 * h * sz_i
        B += -0.5 * h * sz_i
        
    for i in range(N-1):
        # Bond i, i+1
        # S_i^x S_{i+1}^x
        term_xx = op_at(i, sx) @ op_at(i+1, sx)
        term_yy = op_at(i, sy) @ op_at(i+1, sy)
        term_zz = delta * op_at(i, sz) @ op_at(i+1, sz)
        
        bond_h = term_xx + term_yy + term_zz
        
        if i % 2 == 0: # Bond (0,1), (2,3), (4,5) are odd bonds? 
            # Task says "even and odd bonds".
            # Usually bond 1 is between 0-1.
            # If bond index is 1-based: 1, 2, 3, 4, 5.
            # Even bonds: 2, 4. Odd bonds: 1, 3, 5.
            # My loop i is 0-based: 0, 1, 2, 3, 4.
            # i=0 (bond 1): Odd -> A
            # i=1 (bond 2): Even -> B
            # i=2 (bond 3): Odd -> A
            # i=3 (bond 4): Even -> B
            # i=4 (bond 5): Odd -> A
            A += bond_h
        else:
            B += bond_h
            
    # Exact propagator
    # We need e^{-i t H} for t in [0.25, 0.5, 1.0]
    # H = A + B
    
    def calc_error(coeffs_a, coeffs_b):
        # coeffs_a, coeffs_b are lists of 12
        err = 0.0
        for t in [0.25, 0.5, 1.0]:
            U_approx = np.eye(dim, dtype=complex)
            for k in range(12):
                # exp(-i b_k t B) exp(-i a_k t A)
                # Order: k=0 to 11
                # U = U @ exp(-i b_k t B) @ exp(-i a_k t A)
                # Wait, the product is \prod_{k=1}^{12} ...
                # Usually this means U_1 U_2 ... U_12.
                # U_k = exp(-i b_k t B) exp(-i a_k t A)
                # So U_total = U_1 U_2 ... U_12
                # In matrix mult, if we apply U_1 first to state |psi>, it's U_1 |psi>.
                # Then U_2 (U_1 |psi>) = U_2 U_1 |psi>.
                # So the total matrix is U_12 ... U_2 U_1.
                # Let's check the convention.
                # "U(t) = prod_k ... in order k=1..12"
                # Standard math notation \prod_{k=1}^N X_k = X_1 X_2 ... X_N.
                # If this acts on a vector from the right: (X_1 ... X_N) v.
                # Then the first operation is X_N? No, X_1 is the leftmost.
                # However, in time evolution, we usually write U(t) = T exp(-i \int H dt).
                # For discrete steps, if we split into steps 1..N, the total operator is U_N ... U_1.
                # Let's assume the standard matrix product order: U_total = U_1 @ U_2 @ ... @ U_12.
                # This corresponds to applying U_12 first? No.
                # (U_1 U_2) v = U_1 (U_2 v). U_2 is applied first.
                # So if the formula is written as a product $P = M_1 M_2 ... M_N$, the operator $M_N$ acts first on the state.
                # Does the problem imply a specific time-ordering?
                # "approximates the time evolution ... U(t) = prod ... in order k=1..12"
                # I will assume U_total = M_1 M_2 ... M_12.
                
                exp_B = la.expm(-1j * coeffs_b[k] * t * B)
                exp_A = la.expm(-1j * coeffs_a[k] * t * A)
                M_k = exp_B @ exp_A
                U_approx = U_approx @ M_k
                
            H_exact = A + B
            U_exact = la.expm(-1j * t * H_exact)
            
            err += np.linalg.norm(U_approx - U_exact, 2)
        return err / 3.0

    # Initial guess: 4th order symmetric splitting
    # Symmetric: a_k = a_{13-k}, b_k = b_{13-k}
    # We optimize 6 a's and 6 b's.
    # x = [a1..a6, b1..b6]
    # a = [a1..a6, a6..a1]
    # b = [b1..b6, b6..b1]
    # Constraints: sum(a1..a6) = 0.5, sum(b1..b6) = 0.5
    
    # Start with uniform
    x0 = np.array([1/12.0]*6 + [1/12.0]*6)
    # Or use known 4th order coefficients?
    # Let's just start with uniform and let Nelder-Mead work.
    
    def objective(x):
        a_half = x[:6]
        b_half = x[6:]
        
        # Enforce sum = 0.5
        # This parameterization doesn't enforce it automatically.
        # We should parameterize 5 variables for a, 5 for b.
        # a6 = 0.5 - sum(a1..a5)
        # b6 = 0.5 - sum(b1..b5)
        
        a1 = x[0]
        a2 = x[1]
        a3 = x[2]
        a4 = x[3]
        a5 = x[4]
        a6 = 0.5 - (a1 + a2 + a3 + a4 + a5)
        
        b1 = x[5]
        b2 = x[6]
        b3 = x[7]
        b4 = x[8]
        b5 = x[9]
        b6 = 0.5 - (b1 + b2 + b3 + b4 + b5)
        
        a_full = [a1, a2, a3, a4, a5, a6, a6, a5, a4, a3, a2, a1]
        b_full = [b1, b2, b3, b4, b5, b6, b6, b5, b4, b3, b2, b1]
        
        # Check bounds? [-50, 50]
        # Nelder-Mead might go out of bounds.
        # We can add a penalty.
        
        return calc_error(a_full, b_full)
        
    # Optimize
    res = opt.minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6})
    
    x_opt = res.x
    a1, a2, a3, a4, a5 = x_opt[0:5]
    a6 = 0.5 - (a1 + a2 + a3 + a4 + a5)
    b1, b2, b3, b4, b5 = x_opt[5:10]
    b6 = 0.5 - (b1 + b2 + b3 + b4 + b5)
    
    a_full = [a1, a2, a3, a4, a5, a6, a6, a5, a4, a3, a2, a1]
    b_full = [b1, b2, b3, b4, b5, b6, b6, b5, b4, b3, b2, b1]
    
    # Clip to [-50, 50]
    a_full = [max(-50, min(50, val)) for val in a_full]
    b_full = [max(-50, min(50, val)) for val in b_full]
    
    # Re-normalize if clipping broke the sum?
    # Sum a should be 1.
    sum_a = sum(a_full)
    if sum_a != 0:
        a_full = [val / sum_a for val in a_full]
    sum_b = sum(b_full)
    if sum_b != 0:
        b_full = [val / sum_b for val in b_full]
        
    return a_full, b_full
