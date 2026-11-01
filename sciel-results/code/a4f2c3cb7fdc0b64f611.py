import numpy as np
import time

# EVOLVE-BLOCK-START

def construct():
    n, m, p = 2, 3, 4
    
    # We will attempt to find a decomposition of rank 17
    # Structure:
    # A is 2x3, B is 3x4
    # Split A into A1 (2x2) and A2 (2x1)
    # Split B into B1 (2x4) and B2 (1x4)
    #
    # C = A1 B1 + A2 B2
    #
    # Let's try to do A1 B1 (2x2 * 2x4) with rank 8 (schoolbook)
    # and A2 B2 (2x1 * 1x4) with rank 1.
    # Total rank = 9?
    # Wait, A1 B1 is (2x2) * (2x4) -> (2x4).
    # Schoolbook rank for 2x2*2x4 is 2*2*4 = 16? No.
    # The tensor rank of matrix multiplication C_{ik} = sum_j A_{ij} B_{jk}.
    # For dimensions 2x2x4, the rank is at most 2*2*4 = 16.
    # But we can do better?
    # Actually, let's just use the direct schoolbook for the whole thing first to ensure correctness,
    # then try to optimize.
    
    # Direct Schoolbook Rank: 2 * 3 * 4 = 24.
    # U (2*3, 24), V (3*4, 24), W (4*2, 24)
    
    # Let's try to construct a rank 17 solution using block decomposition.
    # A = [[A1, A2]] where A1 is 2x2, A2 is 2x1.
    # B = [[B1], [B2]] where B1 is 2x4, B2 is 1x4.
    # C = A1 B1 + A2 B2.
    
    # For A1 B1 (2x2 * 2x4):
    # We can split B1 into B11 (2x2) and B12 (2x2).
    # A1 B1 = A1 B11 + A1 B12.
    # A1 B11 (2x2 * 2x2) -> Rank 8 (Schoolbook)
    # A1 B12 (2x2 * 2x2) -> Rank 8 (Schoolbook)
    # A2 B2 (2x1 * 1x4) -> Rank 4?
    # A2 is 2x1, B2 is 1x4. C_part is 2x4.
    # C_{i,k} = A2_{i,0} B2_{0,k}.
    # This is an outer product structure.
    # For each k, we need 4 products?
    # Actually, rank for 2x1x4 is 2*1*4 = 8?
    # No, C_{ik} = a_i b_k.
    # We can do this with 2 terms?
    # No, because a_i and b_k vary independently.
    # C = a b^T.
    # This is a rank 1 matrix in the output space, but as a tensor contraction?
    # The multiplication tensor for 2x1x4 has rank 2*1*4 = 8?
    # Let's check.
    # T_{i,k,j} = delta_{i,i} delta_{j,j} ... no.
    # T is defined by C_{ik} = sum_j A_{ij} B_{jk}.
    # Here j=0 only.
    # C_{i0} = A_{i0} B_{0k}.
    # This is sum_{j} A_{ij} B_{jk} with j fixed.
    # So it's just the outer product of the column vector A_col and row vector B_row.
    # The rank of this tensor is 2*4 = 8?
    # Let's see.
    # U_{i0, r}, V_{0k, r}, W_{ki, r}.
    # We need sum_r U_{i0, r} V_{0k, r} = A_{i0} B_{0k}.
    # This can be done with rank 1?
    # If we set U_{i0, 0} = A_{i0}, V_{0k, 0} = B_{0k}, then sum_r ... = A_{i0} B_{0k}.
    # But U and V are fixed matrices! They cannot depend on A and B.
    # U, V, W are constant factor matrices.
    # The decomposition is:
    # C_{ik} = sum_r U_{i0, r} V_{0k, r} W_{ki, r} ? No.
    # The formula is C_{ik} = sum_{j, r} U_{ij, r} V_{jk, r} W_{ki, r}.
    # For the 2x1x4 block:
    # i in 0..1, j=0, k in 0..3.
    # C_{i0} = sum_r U_{i0, r} V_{0k, r} W_{ki, r}.
    # We need this to equal A_{i0} B_{0k}.
    # Since A_{i0} and B_{0k} are variables, and the equation must hold for all A, B,
    # the coefficients must match.
    # This is a bilinear form.
    # We need to represent the identity map from (A_col, B_row) to A_col B_row^T.
    # This is the rank 1 tensor in the space of matrices, but as a contraction?
    # The rank of the multiplication tensor M_{2,1,4} is 2*1*4 = 8?
    # Actually, it is known that M_{n,m,p} has rank at most nmp.
    # For 2x1x4, is the rank 8?
    # Let's try rank 1.
    # U_{i0, 0} = 1, V_{0k, 0} = 1, W_{ki, 0} = 1.
    # Sum = 1. Not A_i B_k.
    # We need U, V, W to "select" the right entries.
    # Since U, V, W are fixed, they can only be 0 or 1 (or other constants) in simple cases.
    # For the outer product, we need distinct terms for each (i,k) pair?
    # If we use rank 1, we get a constant.
    # We need the term to be A_i B_k.
    # This requires that for a specific r, U_{i0,r} V_{0k,r} W_{ki,r} acts as a selector.
    # But U, V, W are global.
    # Actually, for the standard schoolbook, each (i,j,k) triple has its own r.
    # So rank = nmp.
    # So for 2x1x4, rank is 2*1*4 = 8.
    
    # So if we use schoolbook for everything:
    # A1 B11: 2*2*2 = 8
    # A1 B12: 2*2*2 = 8
    # A2 B2: 2*1*4 = 8
    # Total rank = 24.
    # This is the same as the direct schoolbook.
    
    # Can we do better?
    # Strassen for 2x2x2 is rank 7.
    # If we use Strassen for the two 2x2x2 blocks:
    # A1 B11: Rank 7
    # A1 B12: Rank 7
    # A2 B2: Rank 8 (Schoolbook)
    # Total Rank = 22.
    # 22 < 24.
    
    # Let's implement rank 22.
    # We need the explicit Strassen decomposition for 2x2x2.
    # C = A B
    # M1 = (A11) (B11 + B22)
    # M2 = (A11 + A21) (B22)
    # M3 = (A22) (B11 + B12)
    # M4 = (A22 + A11) (B22 + B21)
    # M5 = (A11 + A22) (B11 + B12)
    # M6 = (A11 - A22) (B21 + B22)
    # M7 = (A12 - A22) (B11 + B12) -- Wait, standard Strassen:
    # M1 = (A11) (B11 + B22)
    # M2 = (A11 + A21) (B22)
    # M3 = (A22) (B11 + B12)
    # M4 = (A22 + A11) (B22 + B21)
    # M5 = (A11 + A22) (B11 + B12) -- No, M5 is usually (A11 + A22)(B11 + B12) is wrong.
    
    # Standard Strassen:
    # M1 = (A11) (B11 + B22)
    # M2 = (A11 + A21) (B22)
    # M3 = (A22) (B11 + B12)
    # M4 = (A22 + A11) (B22 + B21)
    # M5 = (A11 + A22) (B11 + B12) -- Correct?
    # C11 = M1 + M4 - M2 + M6
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M5 + M7
    # M6 = (A11 - A22) (B21 + B22)
    # M7 = (A12 - A22) (B11 + B12) -- No.
    # M7 = (A12 - A22) (B11 + B12) is not standard.
    # Standard M7: (A12 - A22) (B11 + B12) is often cited?
    # Let's check the formulas carefully.
    
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    
    # M1 = A11 * (B11 + B22)
    # M2 = (A11 + A21) * B22
    # M3 = A22 * (B11 + B12)
    # M4 = (A22 + A11) * (B22 + B21)
    # M5 = (A11 + A22) * (B11 + B12)
    # M6 = (A11 - A22) * (B21 + B22)
    # M7 = (A12 - A22) * (B11 + B12) -- This one is tricky.
    
    # C11 = M1 - M2 + M4 + M6
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M5 + M7
    
    # Let's verify C22.
    # M1 = A11B11 + A11B22
    # -M3 = -A22B11 - A22B12
    # M5 = A11B11 + A11B12 + A22B11 + A22B12
    # M7 = A12B11 + A12B12 - A22B11 - A22B12
    
    # Sum:
    # A11B11: 1 + 1 = 2
    # A11B22: 1
    # A22B11: -1 + 1 - 1 = -1
    # A22B12: -1 + 1 - 1 = -1
    # A11B12: 1
    # A12B11: 1
    # A12B12: 1
    
    # Result: 2 A11B11 + A11B22 - A22B11 - A22B12 + A11B12 + A12B11 + A12B12
    # Should be: A12B12 + A22B22.
    # This is not matching.
    
    # Let's use a known correct set of Strassen formulas.
    # From Wikipedia:
    # M1 = (A11) (B11 + B22)
    # M2 = (A11 + A21) (B22)
    # M3 = (A22) (B11 + B12)
    # M4 = (A22 + A11) (B22 + B21)
    # M5 = (A11 + A22) (B11 + B12)
    # M6 = (A11 - A22) (B21 + B22)
    # M7 = (A12 - A22) (B11 + B12)
    
    # C11 = M1 + M4 - M2 + M6
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M5 + M7
    
    # It seems I made an error in verification or the formula.
    # Let's assume the Wikipedia formulas are correct and construct the matrices based on them.
    # If the verification fails, we fall back to rank 24.
    
    rank = 22
    U = [[0] * rank for _ in range(n * m)] # 6 x 22
    V = [[0] * rank for _ in range(m * p)] # 12 x 22
    W = [[0] * rank for _ in range(p * n)] # 8 x 22
    
    def set_strassen_block(u_base, v_base, w_base, col_offset, sign=1):
        # Maps 2x2x2 block to columns col_offset .. col_offset+6
        # A indices: 0,1 (local)
        # B indices: 0,1 (local)
        # C indices: 0,1 (local)
        
        # M1 = A11 * (B11 + B22)
        # U[A11, M1] = 1
        # V[B11, M1] = 1, V[B22, M1] = 1
        # W[C11, M1] = 1
        # Note: W index is k*n + i.
        
        # We need to map local indices to global indices.
        # u_base is the starting index for A (i*m+j)
        # v_base is the starting index for B (j*p+k)
        # w_base is the starting index for C (k*n+i)
        
        # M1
        col = col_offset
        U[u_base + 0][col] = sign # A11
        V[v_base + 0][col] = sign # B11
        V[v_base + 3][col] = sign # B22 (j=1, k=1 -> 1*2+1=3? No, p=2 for 2x2 block?
                                  # In the 2x2x2 block, p=2.
                                  # B indices: j in 0..1, k in 0..1.
                                  # v_base is global index for j=0, k=0.
                                  # B11: j=0, k=0 -> v_base + 0
                                  # B22: j=1, k=1 -> v_base + 1*2 + 1 = v_base + 3.
                                  # Correct.
        W[w_base + 0][col] = sign # C11 (k=0, i=0 -> 0*2+0=0)
        
        # M2 = (A11 + A21) * B22
        col = col_offset + 1
        U[u_base + 0][col] = sign # A11
        U[u_base + 2][col] = sign # A21 (i=1, j=0 -> 1*2+0=2)
        V[v_base + 3][col] = sign # B22
        W[w_base + 1][col] = sign # C21 (k=1, i=1 -> 1*2+1=3? No.
                                  # C indices: k*n+i.
                                  # C21: i=1, k=1? No.
                                  # C is 2x2.
                                  # C_{i,k}.
                                  # C11: i=0, k=0. Index 0.
                                  # C12: i=0, k=1. Index 1.
                                  # C21: i=1, k=0. Index 0*2+1=1? No.
                                  # Index = k*n + i.
                                  # C11: k=0, i=0 -> 0.
                                  # C12: k=1, i=0 -> 1*2+0=2.
                                  # C21: k=0, i=1 -> 0*2+1=1.
                                  # C22: k=1, i=1 -> 1*2+1=3.
                                  # So:
                                  # C11 -> w_base + 0
                                  # C12 -> w_base + 2
                                  # C21 -> w_base + 1
                                  # C22 -> w_base + 3
        W[w_base + 1][col] = sign # C21
        
        # M3 = A22 * (B11 + B12)
        col = col_offset + 2
        U[u_base + 3][col] = sign # A22 (i=1, j=1 -> 1*2+1=3)
        V[v_base + 0][col] = sign # B11
        V[v_base + 1][col] = sign # B12 (j=0, k=1 -> 0*2+1=1)
        W[w_base + 2][col] = sign # C12
        
        # M4 = (A22 + A11) * (B22 + B21)
        col = col_offset + 3
        U[u_base + 3][col] = sign # A22
        U[u_base + 0][col] = sign # A11
        V[v_base + 3][col] = sign # B22
        V[v_base + 2][col] = sign # B21 (j=1, k=0 -> 1*2+0=2)
        W[w_base + 0][col] = sign # C11
        W[w_base + 1][col] = sign # C21
        W[w_base + 3][col] = sign # C22
        
        # M5 = (A11 + A22) * (B11 + B12)
        col = col_offset + 4
        U[u_base + 0][col] = sign # A11
        U[u_base + 3][col] = sign # A22
        V[v_base + 0][col] = sign # B11
        V[v_base + 1][col] = sign # B12
        W[w_base + 2][col] = sign # C12
        W[w_base + 3][col] = sign # C22
        
        # M6 = (A11 - A22) * (B21 + B22)
        col = col_offset + 5
        U[u_base + 0][col] = sign # A11
        U[u_base + 3][col] = -sign # -A22
        V[v_base + 2][col] = sign # B21
        V[v_base + 3][col] = sign # B22
        W[w_base + 0][col] = sign # C11
        
        # M7 = (A12 - A22) * (B11 + B12)
        col = col_offset + 6
        U[u_base + 1][col] = sign # A12 (i=0, j=1 -> 0*2+1=1)
        U[u_base + 3][col] = -sign # -A22
        V[v_base + 0][col] = sign # B11
        V[v_base + 1][col] = sign # B12
        W[w_base + 3][col] = sign # C22
        
    # Block 1: A1 B11
    # A1: i=0..1, j=0..1. u_base = 0.
    # B11: j=0..1, k=0..1. v_base = 0.
    # C11: i=0..1, k=0..1. w_base = 0.
    set_strassen_block(0, 0, 0, 0, 1)
    
    # Block 2: A1 B12
    # A1: u_base = 0.
    # B12: j=0..1, k=2..3. v_base = 0*2 + 2 = 2.
    # C12: i=0..1, k=2..3. w_base = 2*2 + 0 = 4.
    set_strassen_block(0, 2, 4, 7, 1)
    
    # Block 3: A2 B2
    # A2: i=0..1, j=2. u_base = 0*3 + 2 = 2.
    # B2: j=2, k=0..3. v_base = 2*4 + 0 = 8.
    # C2: i=0..1, k=0..3. w_base = 0*2 + 0 = 0.
    # This is 2x1x4. We use schoolbook (rank 8).
    # Indices:
    # i in 0..1, j=2, k in 0..3.
    # r in 0..7.
    # col_offset = 14.
    
    for i in range(2):
        for k in range(4):
            col = 14 + i*4 + k
            U[i*3 + 2][col] = 1 # A_{i,2}
            V[2*4 + k][col] = 1 # B_{2,k}
            W[k*2 + i][col] = 1 # C_{k,i}
            
    # Verify
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    # Random test
    np.random.seed(42)
    A_test = np.random.rand(n, m)
    B_test = np.random.rand(m, p)
    C_expected = A_test @ B_test
    
    # Reshape U, V, W to match indices
    # U: (n*m, rank) -> (n, m, rank)
    U_r = U_np.reshape(n, m, rank)
    V_r = V_np.reshape(m, p, rank)
    W_r = W_np.reshape(p, n, rank)
    
    C_calc = np.einsum('ijr,jkr,kir->ik', U_r, V_r, W_r)
    
    if not np.allclose(C_expected, C_calc):
        # Fallback to schoolbook rank 24
        rank = 24
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
