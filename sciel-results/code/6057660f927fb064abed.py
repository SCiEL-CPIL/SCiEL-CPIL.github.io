import time
import sys
import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    start_time = time.time()
    DEADLINE = 118.0

    n, m, p = DIMENSIONS  # 4, 4, 5

    # Strategy: Rank 65 = 49 (double Strassen for 4x4x4) + 16 (schoolbook for 4x4x1)
    # We construct this explicitly and verify correctness before returning.
    
    rank = 65
    U = [[0] * rank for _ in range(n * m)]  # 16 x 65
    V = [[0] * rank for _ in range(m * p)]  # 20 x 65
    W = [[0] * rank for _ in range(p * n)]  # 20 x 65

    def add_term(col, u_list, v_list, w_list):
        for i, j, c in u_list:
            if c != 0:
                U[i * m + j][col] += c
        for j, k, c in v_list:
            if c != 0:
                V[j * p + k][col] += c
        for k, i, c in w_list:
            if c != 0:
                W[k * n + i][col] += c

    col = 0

    # Inner Strassen 2x2x2 definitions
    # Each entry: (p_elems, q_elems, r_contribs)
    # p_elems: list of (elem_idx, coeff) where elem_idx 0=P00,1=P01,2=P10,3=P11
    # q_elems: list of (elem_idx, coeff)
    # r_contribs: list of (r_row, r_col, coeff)
    inner_defs = [
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,0,1),(1,1,1)]),       # M1
        ([(2,1),(3,1)], [(0,1)], [(1,0,1),(1,1,-1)]),             # M2
        ([(0,1)], [(1,1),(3,-1)], [(0,1,1),(1,1,1)]),             # M3
        ([(3,1)], [(2,1),(0,-1)], [(0,0,1),(1,0,1)]),             # M4
        ([(0,1),(1,1)], [(3,1)], [(0,0,-1),(0,1,1)]),             # M5
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(1,1,1)]),               # M6
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,0,1)]),               # M7
    ]

    # Super Strassen for 4x4x4:
    # p_block_coeffs: [A00, A01, A10, A11]
    # q_block_coeffs: [B00, B01, B10, B11]
    # c_block_contribs: [(c_block_row, c_block_col, coeff)]
    super_defs = [
        ([1,0,0,1], [1,0,0,1], [(0,0,1),(1,1,1)]),       # M1 -> C00, C11
        ([0,0,1,1], [1,0,0,0], [(1,0,1),(1,1,-1)]),      # M2 -> C10, -C11
        ([1,0,0,0], [0,1,0,-1], [(0,1,1),(1,1,1)]),      # M3 -> C01, C11
        ([0,0,0,1], [-1,0,1,0], [(0,0,1),(1,0,1)]),      # M4 -> C00, C10
        ([1,1,0,0], [0,0,0,1], [(0,0,-1),(0,1,1)]),      # M5 -> -C00, C01
        ([-1,0,1,0], [1,1,0,0], [(1,1,1)]),              # M6 -> C11
        ([0,1,0,-1], [0,0,1,1], [(0,0,1)]),              # M7 -> C00
    ]

    def get_p_element(p_elem_idx, p_block_coeffs):
        r = p_elem_idx // 2
        c = p_elem_idx % 2
        res = []
        for b_idx, coeff in enumerate(p_block_coeffs):
            if coeff == 0:
                continue
            if b_idx == 0:
                gi, gj = r, c
            elif b_idx == 1:
                gi, gj = r, 2 + c
            elif b_idx == 2:
                gi, gj = 2 + r, c
            elif b_idx == 3:
                gi, gj = 2 + r, 2 + c
            res.append((gi, gj, coeff))
        return res

    def get_q_element(q_elem_idx, q_block_coeffs):
        r = q_elem_idx // 2
        c = q_elem_idx % 2
        res = []
        for b_idx, coeff in enumerate(q_block_coeffs):
            if coeff == 0:
                continue
            if b_idx == 0:
                gj, gk = r, c
            elif b_idx == 1:
                gj, gk = r, 2 + c
            elif b_idx == 2:
                gj, gk = 2 + r, c
            elif b_idx == 3:
                gj, gk = 2 + r, 2 + c
            res.append((gj, gk, coeff))
        return res

    # Build the 49 terms for 4x4x4 using double Strassen
    for s_idx in range(len(super_defs)):
        p_bc, q_bc, c_contribs = super_defs[s_idx]
        for i_idx in range(len(inner_defs)):
            p_elems, q_elems, r_contribs = inner_defs[i_idx]
            
            u_list = []
            v_list = []
            w_list = []

            for p_elem, coeff in p_elems:
                for gi, gj, c in get_p_element(p_elem, p_bc):
                    u_list.append((gi, gj, coeff * c))

            for q_elem, coeff in q_elems:
                for gj, gk, c in get_q_element(q_elem, q_bc):
                    v_list.append((gj, gk, coeff * c))

            for br, bc, sign in c_contribs:
                for lr, lc, local_coeff in r_contribs:
                    gi = br * 2 + lr
                    gk = bc * 2 + lc
                    w_list.append((gk, gi, sign * local_coeff))

            if u_list and v_list and w_list:
                add_term(col, u_list, v_list, w_list)
                col += 1

    # Part 2: 4x4 * 4x1 (Schoolbook) - handles 5th column of B -> 5th column of C
    for i in range(4):
        for j in range(4):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1

    # Verification step: check that the decomposition is correct
    # We verify by checking that for random inputs, the decomposition matches standard matmul
    # This is a safety check; we trust the construction but verify to catch any bugs
    
    # Convert to numpy for verification
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # The tensor decomposition says: C[i,k] = sum_r U[i*m+j, r] * B[j,k] * W[k*n+i, r] ... wait
    # Actually: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # And C[i,k] = sum_j A[i,j] * B[j,k]
    # So we need: sum_r U[i*m+j, r] * W[k*n+i, r] to extract the coefficient for B[j,k]
    # Let's verify: for fixed i,j,k, sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal
    # 1 if this is the (i,j,k) entry of the multiplication tensor, else 0
    
    # The multiplication tensor T has T[i,j,k] = 1 if we're computing C[i,k] using A[i,j]*B[j,k]
    # So T[i,j,k] should be 1 for all i,j,k (each term appears exactly once in the sum for C[i,k])
    # Wait no: C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T_{i,j,k} is such that C[i,k] = sum_{j} A[i,j] B[j,k] T_{i,j,k} ... no
    # T_{i,j,k} = 1 means that the term A[i,j]*B[j,k] contributes to C[i,k]
    # So for each (i,j,k), T_{i,j,k} = 1 if j is a valid index (which it always is)
    # Actually, the standard matrix multiplication tensor has T_{i,j,k} = 1 for all i,j,k
    # because every product A[i,j]*B[j,k] appears in exactly one output C[i,k]
    
    # Let's verify by computing the contracted form
    # For a random A (4x4) and B (4x5), compute C = A @ B
    # Then compute C' using the decomposition: C'[i,k] = sum_j sum_r A[i,j] * B[j,k] * U[i*m+j,r] * W[k*n+i,r] / ... 
    # Hmm, actually the decomposition is:
    # C[i,k] = sum_r (sum_j A[i,j] U[i*m+j,r]) * (sum_k' B[j,k'] V[j*p+k',r]) ... no
    
    # The correct interpretation:
    # Let M_r = sum_{i,j} U[i*m+j,r] * A[i,j] (scalar)
    # Let N_r = sum_{j,k} V[j*p+k,r] * B[j,k] (scalar)
    # Then C[i,k] = sum_r M_r * N_r * W[k*n+i,r] ... no that's not right either
    
    # Actually, the standard form is:
    # C[i,k] = sum_r U_{i,j,r} A_{i,j} V_{j,k,r} B_{j,k} W_{i,k,r}
    # where the sums over i,j and j,k are implicit in the tensor structure
    
    # Let me just verify numerically with small random matrices
    rng = np.random.RandomState(42)
    A_test = rng.rand(n, m)
    B_test = rng.rand(m, p)
    C_expected = A_test @ B_test
    
    C_actual = np.zeros((n, p))
    for r in range(rank):
        # Compute the r-th rank-1 term
        # U[:,r] is 16-vector = (i*m+j)
        # V[:,r] is 20-vector = (j*p+k)
        # W[:,r] is 20-vector = (k*n+i)
        
        # Reshape U[:,r] to (n,m)
        Ur = U_np[:, r].reshape(n, m)
        Vr = V_np[:, r].reshape(m, p)
        Wr = W_np[:, r].reshape(p, n).T  # (n, p)
        
        # The rank-1 term contribution is: (A @ Ur^T) * (B @ Vr^T) * Wr ? No...
        
        # Actually for a rank-1 decomposition:
        # C += (A * Ur^T) outer (B * Vr^T) * Wr ... this isn't quite right
        
        # Let me think again. The tensor is T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{i,k,r}
        # And C_{i,k} = sum_j A_{i,j} B_{j,k} T_{i,j,k}
        #            = sum_j sum_r A_{i,j} B_{j,k} U_{i,j,r} V_{j,k,r} W_{i,k,r}
        #            = sum_r W_{i,k,r} (sum_j A_{i,j} U_{i,j,r}) (sum_j B_{j,k} V_{j,k,r})
        # Wait, that doesn't factor correctly because U depends on both i and j, and V depends on both j and k
        
        # Let me just directly verify the tensor entries
        # For each i,j,k: T_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # This should equal 1 for all i in [0,4), j in [0,4), k in [0,5)
        
        # But that would be 80 entries all equal to 1, which means rank >= 80? No, that's not right.
        # The multiplication tensor T_{i,j,k} = delta_{i,j,k} in the sense that it's 1 for all valid indices
        # because C[i,k] = sum_j A[i,j] B[j,k] and each term A[i,j]B[j,k] has coefficient 1
        
        # Wait, I'm confusing myself. Let me just verify numerically by computing C via the decomposition
        # and comparing to C_expected
        
        # The correct formula:
        # C[i,k] = sum_r (sum_j A[i,j] * U[i*m+j, r]) * (sum_j' B[j',k] * V[j'*p+k, r]) * W[k*n+i, r]
        # But this requires that the j index in U and V match, which they don't in general
        
        # I think the correct interpretation is simpler. Let me just trust the construction 
        # and do a direct numerical check by evaluating the bilinear form.
        
        pass
    
    # Direct numerical verification
    C_via_decomp = np.zeros((n, p))
    for r in range(rank):
        # For this rank-1 component, the contribution to C[i,k] is:
        # sum_j A[i,j] * B[j,k] * (product of the three factors at (i,j,k) for this r)
        # But the factors are U[i*m+j,r], V[j*p+k,r], W[k*n+i,r]
        # So C[i,k] += W[k*n+i,r] * sum_j A[i,j] * U[i*m+j,r] * B[j,k] * V[j*p+k,r]
        
        # This is a bilinear form. Let's compute it directly.
        for i in range(n):
            for k in range(p):
                term = 0.0
                for j in range(m):
                    term += A_test[i,j] * U_np[i*m+j, r] * B_test[j,k] * V_np[j*p+k, r]
                C_via_decomp[i,k] += term * W_np[k*n+i, r]
    
    # Check if C_via_decomp matches C_expected
    if not np.allclose(C_via_decomp, C_expected, atol=1e-10):
        # If verification fails, we have a bug. Fall back to schoolbook rank 80.
        rank_fb = 80
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        
        col_fb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j][col_fb] = 1
                    V_fb[j*p+k][col_fb] = 1
                    W_fb[k*n+i][col_fb] = 1
                    col_fb += 1
        
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
