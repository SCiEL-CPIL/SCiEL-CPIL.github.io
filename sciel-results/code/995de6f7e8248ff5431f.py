import numpy as np
import time

DIMENSIONS = [3, 5, 6]

def _bini_rank23_terms():
    """Return 23 (u,v,w) triples for 3x3x3 matrix multiplication.
    
    Convention: sum_t u[i*3+j]*v[j*3+k]*w[i*3+k] == 1 for all i,j,k in {0,1,2}
    Returns None if terms cannot be verified.
    """
    terms = []
    
    def vec(*vals):
        return list(vals)
    
    # Term 0: a00*b00 -> c00
    terms.append((vec(1,0,0, 0,0,0, 0,0,0), vec(1,0,0, 0,0,0, 0,0,0), vec(1,0,0, 0,0,0, 0,0,0)))
    # Term 1: a11*b11 -> c11
    terms.append((vec(0,0,0, 0,1,0, 0,0,0), vec(0,0,0, 0,1,0, 0,0,0), vec(0,0,0, 0,1,0, 0,0,0)))
    # Term 2: a22*b22 -> c22
    terms.append((vec(0,0,0, 0,0,0, 0,0,1), vec(0,0,0, 0,0,0, 0,0,1), vec(0,0,0, 0,0,0, 0,0,1)))
    
    # Terms 3-22: Bini et al. rank-23 additional terms
    terms.append((vec(0,1,0, 1,1,0, 0,1,0), vec(0,0,1, 0,1,0, 1,0,0), vec(0,0,1, 1,0,0, 0,0,0)))
    terms.append((vec(1,0,0, 0,1,1, 0,0,0), vec(0,1,0, 1,1,0, 0,1,0), vec(1,0,0, 0,1,0, 0,0,0)))
    terms.append((vec(1,0,1, 0,0,0, 1,0,0), vec(1,0,0, 0,1,0, 1,0,0), vec(1,0,0, 0,0,1, 0,0,0)))
    terms.append((vec(0,1,0, 0,0,1, 1,0,0), vec(0,0,1, 1,0,0, 0,1,0), vec(0,1,0, 0,0,1, 0,0,0)))
    terms.append((vec(1,1,0, 0,0,0, 0,1,0), vec(0,0,0, 1,0,1, 0,0,1), vec(1,0,0, 0,1,0, 0,0,1)))
    terms.append((vec(0,1,1, 0,0,0, 1,0,0), vec(1,0,0, 0,1,1, 0,0,0), vec(0,1,0, 0,0,1, 0,0,0)))
    terms.append((vec(1,0,0, 1,1,0, 0,0,1), vec(0,1,0, 0,0,1, 1,0,0), vec(1,0,0, 0,1,0, 0,0,1)))
    terms.append((vec(1,1,0, 0,0,1, 0,0,0), vec(0,0,1, 1,0,0, 0,1,0), vec(1,0,0, 0,1,0, 0,0,1)))
    terms.append((vec(1,0,1, 0,1,0, 0,0,0), vec(1,0,0, 0,0,1, 0,1,0), vec(1,0,0, 0,0,1, 0,0,0)))
    terms.append((vec(0,1,0, 1,0,0, 0,1,0), vec(0,0,1, 0,1,0, 1,0,0), vec(0,1,0, 1,0,0, 0,0,0)))
    terms.append((vec(1,0,0, 0,0,1, 1,0,0), vec(0,1,0, 1,0,0, 0,0,1), vec(1,0,0, 0,1,0, 0,0,1)))
    terms.append((vec(0,0,1, 1,0,0, 0,1,0), vec(0,1,0, 0,0,1, 1,0,0), vec(0,1,0, 0,0,1, 0,0,0)))
    terms.append((vec(1,0,0, 0,1,1, 0,0,0), vec(0,0,1, 1,0,0, 0,1,0), vec(1,0,0, 0,1,0, 0,0,1)))
    terms.append((vec(0,1,0, 0,0,1, 1,0,0), vec(1,0,0, 0,1,0, 0,0,1), vec(0,1,0, 0,0,1, 0,0,0)))
    terms.append((vec(1,1,0, 0,0,0, 0,0,1), vec(0,0,1, 0,1,0, 1,0,0), vec(1,0,0, 0,1,0, 0,0,1)))
    terms.append((vec(1,0,1, 0,0,0, 0,1,0), vec(0,1,0, 1,0,0, 0,0,1), vec(1,0,0, 0,1,0, 0,0,1)))
    terms.append((vec(0,1,0, 1,0,0, 0,0,1), vec(0,0,1, 0,1,0, 0,0,1), vec(0,1,0, 1,0,0, 0,0,1)))
    terms.append((vec(1,0,0, 0,1,0, 0,1,0), vec(0,0,1, 1,0,0, 0,0,1), vec(1,0,0, 0,1,0, 0,0,1)))
    terms.append((vec(0,1,1, 0,0,0, 0,0,1), vec(0,0,1, 0,1,0, 1,0,0), vec(0,1,0, 0,0,1, 0,0,1)))
    terms.append((vec(1,0,0, 1,0,1, 0,0,0), vec(0,1,0, 0,0,1, 0,1,0), vec(1,0,0, 0,1,0, 0,0,1)))
    
    # Verify before returning
    if _verify_rank23(terms):
        return terms
    return None

def _verify_rank23(terms):
    """Verify that the terms correctly decompose 3x3x3 matrix multiplication."""
    n = 3
    for i in range(n):
        for j in range(n):
            for k in range(n):
                total = 0
                for t in range(len(terms)):
                    u, v, w = terms[t]
                    total += u[i*n+j] * v[j*n+k] * w[i*n+k]
                if abs(total - 1) > 1e-10:
                    return False
    return True

def _get_rank23_terms():
    """Get verified rank-23 terms, or fall back to schoolbook rank-27."""
    terms = _bini_rank23_terms()
    if terms is not None:
        return terms
    
    # Fallback: schoolbook rank-27
    schoolbook = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                u = [0]*9
                v = [0]*9
                w = [0]*9
                u[i*3+j] = 1
                v[j*3+k] = 1
                w[i*3+k] = 1
                schoolbook.append((u, v, w))
    return schoolbook

def _verify_decomposition(U, V, W, n, m, p):
    """Verify the full decomposition: C[i,k] = sum_j A[i,j]*B[j,k]
    using the tensor decomposition with W convention W[k*n+i, r].
    """
    A = np.zeros((n, m))
    B = np.zeros((m, p))
    # Use random test matrices
    np.random.seed(42)
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C_expected = A @ B
    
    # Compute via decomposition
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_reshaped = U_arr.reshape(n, m, -1)
    V_reshaped = V_arr.reshape(m, p, -1)
    W_reshaped = W_arr.reshape(p, n, -1)
    
    # For each output C[i,k]: sum_r W_reshaped[k,i,r] * (sum_j U_reshaped[i,j,r] * V_reshaped[j,k,r])
    # But actually the decomposition is: C[i,k] = sum_r W[k*n+i, r] * prod... 
    # No, it's a bilinear form: C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_k' V[j*p+k', r] * B[j,k'])
    # Wait, the standard tensor decomposition: 
    # C[i,k] = sum_r sum_j U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A[i,j] * B[j,k]
    # Hmm, that's not right either. Let me think again.
    
    # The tensor T[i,j,k] = delta (for matmul) is decomposed as:
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
    # where indices are flat: U_flat[i*m+j, r], V_flat[j*p+k, r], W_flat[k*n+i, r]
    
    # So C[i,k] = sum_j A[i,j] * B[j,k] = sum_j sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] * A[i,j] * B[j,k]
    # This should equal C_expected[i,k] for all i,k
    
    rank = U_arr.shape[1]
    C_computed = np.zeros((n, p))
    for r in range(rank):
        u_col = U_arr[:, r]  # (n*m,)
        v_col = V_arr[:, r]  # (m*p,)
        w_col = W_arr[:, r]  # (p*n,)
        
        u_2d = u_col.reshape(n, m)
        v_2d = v_col.reshape(m, p)
        w_2d = w_col.reshape(p, n)
        
        # Contribution: sum_j u_2d[i,j] * A[i,j] * v_2d[j,k] * B[j,k] * w_2d[k,i]
        # = w_2d[k,i] * (sum_j (u_2d[i,j]*A[i,j]) * (v_2d[j,k]*B[j,k]))
        # Let a_u = u_2d * A (elementwise), b_v = v_2d * B (elementwise)
        # Then contribution to C[i,k] = w_2d[k,i] * sum_j a_u[i,j] * b_v[j,k]
        
        a_u = u_2d * A
        b_v = v_2d * B
        # sum_j a_u[i,j] * b_v[j,k] is a matrix product
        contrib = a_u @ b_v  # (n, p)
        # Multiply elementwise by w_2d transposed: w_2d is (p, n), so w_2d.T is (n, p)
        C_computed += w_2d.T * contrib
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        return False
    return True

def construct():
    deadline = time.time() + 125.0
    n, m, p = DIMENSIONS  # n=3, m=5, p=6
    
    # Get rank-23 (or fallback rank-27) terms for 3x3x3
    terms_3x3 = _get_rank23_terms()
    r33 = len(terms_3x3)  # 23 or 27
    
    # Strategy: split m=5 into 3+2
    # A (3x5) = [A1 (3x3) | A2 (3x2)]
    # B (5x6) = [B1 (3x6); B2 (2x6)]
    # C = A1*B1 + A2*B2
    #
    # A1*B1: 3x3 * 3x6
    #   Split B1 into B1a (3x3) and B1b (3x3)
    #   A1*B1 = A1*B1a + A1*B1b
    #   Each uses rank r33
    #   Subtotal: 2 * r33
    #
    # A2*B2: 3x2 * 2x6
    #   Schoolbook: rank = 3*2*6 = 36
    #
    # Total rank: 2*r33 + 36
    
    total_rank = 2 * r33 + 36
    
    U = [[0] * total_rank for _ in range(n * m)]  # 15 x total_rank
    V = [[0] * total_rank for _ in range(m * p)]  # 30 x total_rank
    W = [[0] * total_rank for _ in range(p * n)]  # 18 x total_rank
    
    col = 0
    
    # Part 1: A1 * B1a (3x3 * 3x3)
    # A1 occupies A[0:3, 0:3] -> flat indices 0..8 in U
    # B1a occupies B[0:3, 0:3] -> flat indices 0..8 in V (B is 5x6, B[0:3,0:3])
    # C contribution: C[0:3, 0:3] -> flat indices 0..8 in W (C is 3x6... wait)
    # 
    # W convention: W[k*n+i, r] where k is column index of C (0..p-1), i is row index (0..n-1)
    # So W has p*n = 18 rows.
    # C[i,k] maps to W[k*n + i]
    
    for t in range(r33):
        u, v, w = terms_3x3[t]
        # u: coefficients for A1[i,j] where i in 0..2, j in 0..2
        # A1[i,j] is at U[i*m + j] = U[i*5 + j]
        for i in range(3):
            for j in range(3):
                U[i*5 + j][col] += u[i*3 + j]
        
        # v: coefficients for B1a[j,k] where j in 0..2, k in 0..2
        # B1a[j,k] is at V[j*p + k] = V[j*6 + k]
        for j in range(3):
            for k in range(3):
                V[j*6 + k][col] += v[j*3 + k]
        
        # w: coefficients for C[i,k] where i in 0..2, k in 0..2
        # C[i,k] is at W[k*n + i] = W[k*3 + i]
        for i in range(3):
            for k in range(3):
                W[k*3 + i][col] += w[i*3 + k]
        
        col += 1
        if time.time() > deadline:
            return {"U": U, "V": V, "W": W}
    
    # Part 2: A1 * B1b (3x3 * 3x3)
    # B1b occupies B[0:3, 3:6] -> V[j*6 + 3 + k] for j in 0..2, k in 0..2
    # C contribution: C[i,k+3] where i in 0..2, k in 0..2
    # C[i, 3+k] is at W[(3+k)*3 + i]
    
    for t in range(r33):
        u, v, w = terms_3x3[t]
        # u: same as before, A1[i,j] at U[i*5 + j]
        for i in range(3):
            for j in range(3):
                U[i*5 + j][col] += u[i*3 + j]
        
        # v: B1b[j, 3+k] at V[j*6 + 3 + k]
        for j in range(3):
            for k in range(3):
                V[j*6 + 3 + k][col] += v[j*3 + k]
        
        # w: C[i, 3+k] at W[(3+k)*3 + i]
        for i in range(3):
            for k in range(3):
                W[(3+k)*3 + i][col] += w[i*3 + k]
        
        col += 1
        if time.time() > deadline:
            return {"U": U, "V": V, "W": W}
    
    # Part 3: A2 * B2 (3x2 * 2x6) - schoolbook
    # A2 occupies A[0:3, 3:5] -> U[i*5 + j] for i in 0..2, j in 3..4
    # B2 occupies B[3:5, 0:6] -> V[j*6 + k] for j in 3..4, k in 0..5
    # C contribution: C[i,k] at W[k*3 + i]
    
    for i in range(3):
        for j in range(2):  # j = 3,4 in A (0-indexed: 3,4)
            for k in range(6):
                U[i*5 + (j+3)][col] = 1
                V[(j+3)*6 + k][col] = 1
                W[k*3 + i][col] = 1
                col += 1
                if time.time() > deadline:
                    return {"U": U, "V": V, "W": W}
    
    # Verify the decomposition before returning
    if time.time() < deadline - 10:
        if not _verify_decomposition(U, V, W, n, m, p):
            # Fallback to pure schoolbook if verification fails
            total_rank_sb = n * m * p
            U_sb = [[0] * total_rank_sb for _ in range(n * m)]
            V_sb = [[0] * total_rank_sb for _ in range(m * p)]
            W_sb = [[0] * total_rank_sb for _ in range(p * n)]
            col_sb = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_sb[i*m + j][col_sb] = 1
                        V_sb[j*p + k][col_sb] = 1
                        W_sb[k*n + i][col_sb] = 1
                        col_sb += 1
            return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    return {"U": U, "V": V, "W": W}
