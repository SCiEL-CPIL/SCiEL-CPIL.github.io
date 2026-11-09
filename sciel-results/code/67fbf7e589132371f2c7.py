"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _schoolbook(n=4, m=4, p=5):
    """Standard schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _check_decomp(n, m, p, U, V, W, tol=1e-9):
    """
    Verify that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1 for all i,j,k.
    This corresponds to the matrix multiplication tensor where C[i,k] = sum_j A[i,j]B[j,k].
    The structure tensor T[i,j,k] is 1 if the term A[i,j]B[j,k] contributes to C[i,k],
    which is always true for valid indices in standard matrix multiplication.
    """
    n, m, p = int(n), int(m), int(p)
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]
    
    # Vectorized check using einsum
    # Reshape U, V, W to appropriate shapes
    # U: (n*m, rank) -> (n, m, rank)
    # V: (m*p, rank) -> (m, p, rank)
    # W: (p*n, rank) -> (p, n, rank)
    
    U_reshaped = Ua.reshape(n, m, rank)
    V_reshaped = Va.reshape(m, p, rank)
    W_reshaped = Wa.reshape(p, n, rank)
    
    # Compute T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # This is a bit tricky with standard einsum because of the index permutations.
    # Let's do it step-by-step or with a loop if vectorization is too complex/risky for speed.
    # Given small dimensions (4,4,5), a loop is fast enough and less error-prone.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_reshaped[i, j, :], V_reshaped[j, k, :]) * 0 # Placeholder logic, need triple product
                # Actually, it's element-wise product then sum.
                # val = np.sum(U_reshaped[i, j, :] * V_reshaped[j, k, :] * W_reshaped[k, i, :])
                val = np.sum(U_reshaped[i, j, :] * V_reshaped[j, k, :] * W_reshaped[k, i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _strassen_4x4x4():
    """
    Returns a rank-49 decomposition for 4x4x4 matrix multiplication.
    Uses Strassen's algorithm recursively.
    """
    n, m, p = 4, 4, 4
    # Strassen for 2x2 has rank 7.
    # For 4x4, we split into 2x2 blocks.
    # Rank = 7 * 7 = 49.
    
    # We need to construct U, V, W of shapes (16, 49), (16, 49), (16, 49).
    # Indices:
    # U index: i*m + j = 4*i + j
    # V index: j*p + k = 4*j + k
    # W index: k*n + i = 4*k + i
    
    # Strassen 2x2 decomposition:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A10-A11)(B10+B11)
    
    # Coefficients for A blocks (A00, A01, A10, A11) in each M
    A_coeffs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, -1, 1, 0] # M7 -- Wait, M7 = (A01-A11)(B10+B11)
    ]
    # Let's double check M7.
    # Standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    A_coeffs = [
        [1, 0, 0, 1], # M1: A00+A11
        [0, 0, 1, 1], # M2: A10+A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 1, 0, 0], # M5: A00+A01
        [-1, 0, 1, 0],# M6: A10-A00
        [0, 1, 0, -1] # M7: A01-A11
    ]
    
    B_coeffs = [
        [1, 0, 0, 1], # M1: B00+B11
        [1, 0, 0, 0], # M2: B00
        [0, 1, 0, -1],# M3: B01-B11
        [0, 0, 1, -1],# M4: B10-B00
        [0, 0, 0, 1], # M5: B11
        [1, 1, 0, 0], # M6: B00+B01
        [0, 0, 1, 1]  # M7: B10+B11
    ]
    
    C_coeffs = [
        [1, 0, 0, 1, -1, 0, 1], # C00 = M1 - M5 + M7 + M4?
        # Let's re-derive C from M.
        # C00 = M1 - M5 + M7 + M4
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        # Let's verify.
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 - M5 + M7 + M4
        # = (A00+A11)(B00+B11) - (A00+A01)B11 + (A01-A11)(B10+B11) + A11(B10-B00)
        # = A00B00 + A00B11 + A11B00 + A11B11 - A00B11 - A01B11 + A01B10 + A01B11 - A11B10 - A11B11 + A11B10 - A11B00
        # = A00B00 + A01B10
        # Correct.
        
        # C01 = M3 + M5
        # = A00(B01-B11) + (A00+A01)B11
        # = A00B01 - A00B11 + A00B11 + A01B11
        # = A00B01 + A01B11
        # Correct.
        
        # C10 = M2 + M4
        # = (A10+A11)B00 + A11(B10-B00)
        # = A10B00 + A11B00 + A11B10 - A11B00
        # = A10B00 + A11B10
        # Correct.
        
        # C11 = M1 - M2 + M3 + M6
        # = (A00+A11)(B00+B11) - (A10+A11)B00 + A00(B01-B11) + (A10-A00)(B00+B01)
        # = A00B00 + A00B11 + A11B00 + A11B11 - A10B00 - A11B00 + A00B01 - A00B11 + A10B00 + A10B01 - A00B00 - A00B01
        # = A11B11 + A10B01
        # Correct.
        
        # So C_coeffs (rows are C00, C01, C10, C11; cols are M1..M7):
        [1, -1, 0, 1, 1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ]
    
    rank = 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    # For each of the 7 outer Strassen terms (m_outer)
    # And each of the 7 inner Strassen terms (m_inner)
    # We have a rank-1 tensor.
    # The U vector is formed by the outer A coefficients and inner A coefficients.
    # The V vector is formed by the outer B coefficients and inner B coefficients.
    # The W vector is formed by the outer C coefficients and inner C coefficients.
    
    # Indices for 4x4 blocks:
    # Block (r,c) of 4x4 matrix consists of elements (2*r+i, 2*c+j) for i,j in 0..1.
    # U index: 4*i + j.
    # For block (r,c), i = 2*r+u_i, j = 2*c+u_j.
    # U[4*(2*r+u_i) + (2*c+u_j)]
    
    # Let's iterate.
    for m_outer in range(7):
        for m_inner in range(7):
            col = m_outer * 7 + m_inner
            
            # U coefficients
            # A_outer_block = sum_{b in 0..3} A_coeffs[m_outer][b] * A_block_b
            # A_inner_block = sum_{d in 0..3} A_coeffs[m_inner][d] * A_block_d (within the outer block)
            # Total A coefficient for element (i,j) is:
            # Let i = 2*r + u_i, j = 2*c + u_j.
            # Outer block index b = r*2 + c.
            # Inner block index d = u_i*2 + u_j.
            # U[4*i+j] = A_coeffs[m_outer][b] * A_coeffs[m_inner][d]
            
            for i in range(4):
                for j in range(4):
                    r = i // 2
                    c = j // 2
                    u_i = i % 2
                    u_j = j % 2
                    b = r * 2 + c
                    d = u_i * 2 + u_j
                    U[4*i+j][col] = A_coeffs[m_outer][b] * A_coeffs[m_inner][d]
            
            # V coefficients
            # Similarly for B.
            # V index: 4*j + k.
            # j = 2*c + v_j (c is outer col, v_j is inner col)
            # k = 2*d + v_k (d is outer row? No, B is 4x4. B[j,k].
            # Block (c, d) of B?
            # B is 4x4. j is row, k is col.
            # Outer block (c, d) where j=2*c+..., k=2*d+...
            # Wait, Strassen splits B into 2x2 blocks B00, B01, B10, B11.
            # B00 is top-left 2x2. j in 0,1; k in 0,1.
            # So outer block index for B is determined by j//2 and k//2.
            # Let j = 2*c + v_j, k = 2*d + v_k.
            # Outer block index b_B = c*2 + d.
            # Inner block index d_B = v_j*2 + v_k.
            # V[4*j+k] = B_coeffs[m_outer][b_B] * B_coeffs[m_inner][d_B]
            
            for j in range(4):
                for k in range(4):
                    c = j // 2
                    d = k // 2
                    v_j = j % 2
                    v_k = k % 2
                    b_B = c * 2 + d
                    d_B = v_j * 2 + v_k
                    V[4*j+k][col] = B_coeffs[m_outer][b_B] * B_coeffs[m_inner][d_B]
            
            # W coefficients
            # W index: 4*k + i.
            # C is 4x4. C[i,k].
            # Outer block (r, d) where i=2*r+..., k=2*d+...
            # Wait, C blocks correspond to A rows and B cols.
            # A rows are indexed by i. B cols by k.
            # Outer block (r, d) where i=2*r+w_i, k=2*d+w_k.
            # Outer block index b_C = r*2 + d.
            # Inner block index d_C = w_i*2 + w_k.
            # W[4*k+i] = C_coeffs[m_outer][b_C] * C_coeffs[m_inner][d_C]
            
            for k in range(4):
                for i in range(4):
                    r = i // 2
                    d = k // 2
                    w_i = i % 2
                    w_k = k % 2
                    b_C = r * 2 + d
                    d_C = w_i * 2 + w_k
                    W[4*k+i][col] = C_coeffs[m_outer][b_C] * C_coeffs[m_inner][d_C]
    
    return {"U": U, "V": V, "W": W}

def _extend_to_4x4x5():
    """
    Extends the 4x4x4 rank-49 decomposition to 4x4x5 by adding 4 rank-1 terms
    for the 5th column of B, which only contributes to the 5th column of C.
    Rank = 49 + 4 = 53.
    """
    n, m, p = 4, 4, 5
    base = _strassen_4x4x4()
    U_base = base["U"]
    V_base = base["V"]
    W_base = base["W"]
    
    rank_new = 53
    U = [row[:] for row in U_base] # 16 x 49
    V = [row[:] for row in V_base] # 16 x 49
    W = [row[:] for row in W_base] # 16 x 49
    
    # Add 4 columns
    for c in range(4):
        U.append([0]*rank_new) # Wait, U is list of lists.
        # Actually, U is 16 rows. We need to add 4 columns to each row.
        pass
    
    # Let's do it properly.
    U = [[0] * rank_new for _ in range(16)]
    V = [[0] * rank_new for _ in range(16)]
    W = [[0] * rank_new for _ in range(16)]
    
    # Copy base
    for i in range(16):
        for j in range(49):
            U[i][j] = U_base[i][j]
            V[i][j] = V_base[i][j]
            W[i][j] = W_base[i][j]
            
    # Add terms for k=4 (5th column of B)
    # C[i,4] = sum_j A[i,j] B[j,4]
    # This is a 4x4x1 multiplication.
    # Rank 4 decomposition:
    # For each i in 0..3:
    # Term i:
    # U[i*4+j, 49+i] = delta(i,j) ? No.
    # U index is i*m+j = 4*i+j.
    # We want sum_j A[4*i+j] * B[4*j+4] * W[4*4+i] = C[4*i+4]? No, W index is k*n+i = 4*4+i = 16+i.
    # Let's check indices.
    # U[4*i+j], V[4*j+4], W[4*4+i].
    # If we set U[4*i+j, c] = 1 if j==i? No.
    # We want A[i,j] * B[j,4].
    # So for a fixed i, we need to sum over j.
    # Term j for row i?
    # Let's use 4 terms, one for each i.
    # Term i (i=0..3):
    # U[4*i+j, 49+i] = 1 for all j? No, that would sum all A[i,j].
    # We need A[i,j] B[j,4].
    # So U[4*i+j, 49+i] should be 1 only if... wait.
    # If U[4*i+j, c] = 1, V[4*j+4, c] = 1, W[16+i, c] = 1.
    # Then sum_c ... = sum_j (if U[4*i+j,c]=1 and V[4*j+4,c]=1) * W[16+i,c].
    # If we set for term i:
    # U[4*i+j, 49+i] = 1 for all j.
    # V[4*j+4, 49+i] = 1 for all j.
    # W[16+i, 49+i] = 1.
    # Then sum_c U[4*i',j',c] V[4*j',k',c] W[4*k'+i',c]
    # If i'=i, k'=4:
    # Sum_j' U[4*i+j', 49+i] V[4*j'+4, 49+i] W[16+i, 49+i]
    # = Sum_j' 1 * 1 * 1 = 4.
    # We want 1. So divide by 4? But entries must be integer/half-integer.
    # Or, use 4 terms for each i? No, that increases rank.
    
    # Actually, the term A[i,j]B[j,4] is already handled if we just extend the schoolbook part?
    # No, Strassen doesn't handle the 5th column.
    # We can add 4 rank-1 terms for the 5th column.
    # For each i, we need to compute sum_j A[i,j]B[j,4].
    # This is an inner product.
    # We can decompose it into 4 rank-1 terms?
    # Term j for i?
    # Let's just use the schoolbook terms for the 5th column.
    # There are 4*4 = 16 terms for the 5th column in schoolbook.
    # Can we do better?
    # The sub-problem is 4x4x1. Rank is 4.
    # Rank 4 decomposition for 4x4x1:
    # C[i,4] = sum_j A[i,j]B[j,4].
    # U[4*i+j, c], V[4*j+4, c], W[16+i, c].
    # If we set c = 49 + j (for j=0..3).
    # U[4*i+j, 49+j] = 1 for all i?
    # If U[4*i+j, 49+j] = 1, V[4*j+4, 49+j] = 1, W[16+i, 49+j] = delta(i,j)?
    # Then sum_j' U[4*i+j', 49+j'] V[4*j'+4, 49+j'] W[16+i, 49+j']
    # = sum_j' 1 * 1 * delta(i,j') = 1 if j'=i.
    # So W[16+i, 49+j] = 1 if i==j else 0.
    # This works!
    
    for j in range(4):
        c = 49 + j
        for i in range(4):
            U[4*i+j][c] = 1
            V[4*j+4][c] = 1
            W[16+i][c] = 1 if i == j else 0
            
    return {"U": U, "V": V, "W": W}

def construct():
    deadline = time.time() + 100
    n, m, p = DIMENSIONS
    
    # Try rank 53 decomposition
    try:
        decomp = _extend_to_4x4x5()
        if _check_decomp(n, m, p, decomp["U"], decomp["V"], decomp["W"]):
            return decomp
    except Exception as e:
        pass
    
    # Fallback to schoolbook rank 80
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
