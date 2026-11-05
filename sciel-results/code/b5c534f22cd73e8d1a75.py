import numpy as np
import time
import sys

# Constants for the problem
N_SITES = 6
DIM = 4 ** N_SITES
DELTA = 0.8
FIELD = 0.3
TIMES = [0.25, 0.5, 1.0]

def build_hamiltonians():
    # Single site operators
    sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    
    # Build full space operators
    # Using kron products
    # This is expensive to build every time, so we do it once.
    ops = []
    for _ in range(N_SITES):
        ops.append([np.eye(4), sx, sy, sz])
        
    # We need to construct A and B
    # A: Odd bonds (0-1, 2-3, 4-5) + Field/2
    # B: Even bonds (1-2, 3-4) + Field/2
    
    def kron_all(ops_list):
        res = ops_list[0]
        for op in ops_list[1:]:
            res = np.kron(res, op)
        return res

    # Pre-compute identity
    I = np.eye(4)
    
    # Construct A
    A = np.zeros((DIM, DIM), dtype=complex)
    # Bonds: 0-1, 2-3, 4-5
    for bond in [(0, 1), (2, 3), (4, 5)]:
        i, j = bond
        # S_i^x S_j^x + S_i^y S_j^y + Delta S_i^z S_j^z
        term_xx = [I]*N_SITES
        term_xx[i] = sx; term_xx[j] = sx
        A += DELTA * 0 + np.kron(np.kron(term_xx[:2], term_xx[2:]), 1) # Placeholder, need proper kron
        
    # Proper kron construction
    def get_term(i, op_i, j, op_j):
        ops = [I] * N_SITES
        ops[i] = op_i
        ops[j] = op_j
        return kron_all(ops)

    for bond in [(0, 1), (2, 3), (4, 5)]:
        i, j = bond
        A += get_term(i, sx, j, sx)
        A += get_term(i, sy, j, sy)
        A += DELTA * get_term(i, sz, j, sz)
        
    # Field
    for i in range(N_SITES):
        A -= (FIELD / 2.0) * get_term(i, sz, i, I) # get_term with i==j is tricky, just use kron
        
    # Re-do field properly
    A_field = np.zeros((DIM, DIM), dtype=complex)
    for i in range(N_SITES):
        ops = [I] * N_SITES
        ops[i] = sz
        A_field += np.kron_all(ops)
    A -= (FIELD / 2.0) * A_field

    # Construct B
    B = np.zeros((DIM, DIM), dtype=complex)
    for bond in [(1, 2), (3, 4)]:
        i, j = bond
        B += get_term(i, sx, j, sx)
        B += get_term(i, sy, j, sy)
        B += DELTA * get_term(i, sz, j, sz)
        
    B -= (FIELD / 2.0) * A_field

    return A, B

# Pre-compute eigendecompositions for speed
A, B = build_hamiltonians()
eigvals_A, eigvecs_A = np.linalg.eigh(A)
eigvals_B, eigvecs_B = np.linalg.eigh(B)
# Change of basis
M_BA = eigvecs_B.conj().T @ eigvecs_A
M_AB = eigvecs_A.conj().T @ eigvecs_B

def compute_error_fast(params):
    # params: 22 floats (11 a, 11 b)
    # a_12 = 1 - sum(a_1..11)
    # b_12 = 1 - sum(b_1..11)
    a = list(params[:11])
    b = list(params[11:22])
    a.append(1.0 - sum(a))
    b.append(1.0 - sum(b))
    
    total_error = 0.0
    for t in TIMES:
        # Compute U_approx in B basis
        # U = V_B K_total V_B^dagger
        # K_total = K_B(b1) K_A(a1) ... K_B(b12) K_A(a12)
        # K_B(bk) = diag(exp(-i b_k t eigvals_B))
        # K_A(ak) = M_BA diag(exp(-i a_k t eigvals_A)) M_BA^dagger
        
        K = np.eye(DIM, dtype=complex)
        
        # To save time, we can pre-compute the diagonal matrices? No, they depend on t and coefficients.
        # But eigvals are fixed.
        
        for k in range(12):
            # Multiply by K_B(b_k)
            # K = K @ D_B
            # D_B is diagonal
            dB = np.exp(-1j * b[k] * t * eigvals_B)
            K = K * dB # Element-wise multiplication by diagonal
            
            # Multiply by K_A(a_k)
            # K = K @ M_BA @ D_A @ M_BA^dagger
            # This is 3 matrix multiplications?
            # K @ M_BA is O(N^2) if K is sparse? No, K is dense.
            # So K @ M_BA is O(N^3).
            # This is the bottleneck.
            
            # Optimization: Combine M_BA and D_A?
            # K_A = M_BA @ diag(exp(-i a_k t eigvals_A)) @ M_BA^dagger
            # We can pre-compute K_A for each unique a_k? No, a_k varies.
            
            # We must do the multiplication.
            # K = K @ M_BA
            K = K @ M_BA
            # K = K @ D_A
            dA = np.exp(-1j * a[k] * t * eigvals_A)
            K = K * dA
            # K = K @ M_BA^dagger
            K = K @ M_BA.conj().T
        
        # U_exact in B basis
        # U_exact = exp(-i t (A+B))
        # A+B = A_field + A_odd + B_even
        # Since A_field commutes, U_exact = exp(-i t A_field) exp(-i t (A_odd + B_even))
        # We can compute eigendecomposition of A+B once?
        # H = A + B
        # eigvals_H, eigvecs_H = np.linalg.eigh(H)
        # M_HB = eigvecs_B.conj().T @ eigvecs_H
        # K_exact = M_HB @ diag(exp(-i t eigvals_H)) @ M_HB^dagger
        
        # We didn't precompute H. Let's do it outside.
        
        # Error = || K - K_exact ||_F
        # K_exact is precomputed for each t?
        # Yes, K_exact depends only on t.
        
        K_exact = K_EXACTS[t]
        diff = K - K_exact
        err = np.linalg.norm(diff, 'fro')
        total_error += err
        
    return total_error / len(TIMES)

# Precompute K_EXACTS
H = A + B
eigvals_H, eigvecs_H = np.linalg.eigh(H)
M_HB = eigvecs_B.conj().T @ eigvecs_H
K_EXACTS = {}
for t in TIMES:
    dH = np.exp(-1j * t * eigvals_H)
    K_exact = M_HB @ (dH[:, None] * eigvecs_H.conj().T) # Wait, M_HB is V_B^dagger V_H
    # K_exact = V_B^dagger V_H diag(dH) V_H^dagger V_B
    # M_HB = V_B^dagger V_H
    # K_exact = M_HB @ diag(dH) @ M_HB^dagger
    K_exact = M_HB @ np.diag(dH) @ M_HB.conj().T
    K_EXACTS[t] = K_exact

def coefficients(stages):
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    # Initial guess: Strang splitting
    # a = [0.5/6]*12? No, Strang is a=0.5, b=1, a=0.5...
    # For 12 stages, we can use 6 substeps of Strang.
    # a_k = 1/12, b_k = 1/6?
    # Let's use uniform 1/12 for all.
    init_params = [1.0/12] * 22
    
    best_params = init_params
    best_error = compute_error_fast(init_params)
    
    # Nelder-Mead
    # Simple implementation
    n = 22
    simplex = [np.array(init_params)]
    for i in range(n):
        p = np.array(init_params)
        p[i] += 0.05
        simplex.append(p)
    
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    
    max_iter = 100
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        # Evaluate simplex
        values = [compute_error_fast(p) for p in simplex]
        indices = np.argsort(values)
        simplex = [simplex[i] for i in indices]
        values = [values[i] for i in indices]
        
        if values[0] < best_error:
            best_error = values[0]
            best_params = simplex[0]
            
        # Convergence check
        if abs(values[0] - values[-1]) < 1e-8:
            break
            
        # Centroid
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        x_r = centroid + alpha * (centroid - simplex[-1])
        f_r = compute_error_fast(x_r)
        
        if f_r < values[0]:
            # Expansion
            x_e = centroid + gamma * (x_r - centroid)
            f_e = compute_error_fast(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
            else:
                simplex[-1] = x_r
        elif f_r < values[-2]:
            simplex[-1] = x_r
        else:
            # Contraction
            x_c = centroid + rho * (simplex[-1] - centroid)
            f_c = compute_error_fast(x_c)
            if f_c < values[-1]:
                simplex[-1] = x_c
            else:
                # Shrink
                for i in range(1, n+1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
    
    # Reconstruct a, b from best_params
    a = list(best_params[:11])
    b = list(best_params[11:22])
    a.append(1.0 - sum(a))
    b.append(1.0 - sum(b))
    
    return a, b
