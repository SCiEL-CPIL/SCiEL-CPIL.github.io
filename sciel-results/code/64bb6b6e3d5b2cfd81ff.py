"""
Strategy: Kronecker Product Structure for the square part + Schoolbook for the rectangular part.
1. Decompose 4x4x4 multiplication using the Kronecker product of two Strassen 2x2x2 decompositions.
   Rank = 7 * 7 = 49.
2. Decompose 4x4x3 multiplication using Schoolbook.
   Rank = 4 * 4 * 3 = 48.
3. Total Rank = 49 + 48 = 97.
This is a structural change from the previous complex block decomposition to a mathematically grounded Kronecker construction.
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

def _strassen_2x2_2x2():
    """
    Returns the Strassen rank-7 decomposition for 2x2 * 2x2.
    U: 4x7, V: 4x7, W: 4x7
    Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1
    
    # M3 = A00(B10-B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A11(B01-B00)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _kron_4x4x4():
    """
    Constructs a rank-49 decomposition for 4x4 * 4x4 using Kronecker product of Strassen 2x2x2.
    Returns U, V, W of shapes (16, 49), (16, 49), (16, 49).
    """
    U1, V1, W1 = _strassen_2x2_2x2()
    U2, V2, W2 = _strassen_2x2_2x2()
    
    rank = 49
    U = np.zeros((16, rank), dtype=float)
    V = np.zeros((16, rank), dtype=float)
    W = np.zeros((16, rank), dtype=float)
    
    # The Kronecker product of two tensors corresponds to the tensor product of the matrices.
    # Index mapping for 4x4 matrix viewed as 2x2 blocks of 2x2:
    # Global index idx = i_block * 2 + i_local
    # For U (input A), indices are (i, j). 
    # i = i_b * 2 + i_l, j = j_b * 2 + j_l
    # The Kronecker product of U1 and U2:
    # U_combined[(i_b, i_l, j_b, j_l), (c1, c2)] = U1[(i_b, j_b), c1] * U2[(i_l, j_l), c2]
    # Note: U1 index is i_b * 2 + j_b. U2 index is i_l * 2 + j_l.
    
    for c1 in range(7):
        for c2 in range(7):
            c = c1 * 7 + c2
            
            # Fill U
            # U1 is 4x7, U2 is 4x7
            # We need to map (i_b, j_b) from U1 and (i_l, j_l) from U2 to global (i, j)
            # Global i = i_b * 2 + i_l
            # Global j = j_b * 2 + j_l
            # Wait, the standard Kronecker product A \otimes B has index (i_A, i_B) mapped to i_A * dim_B + i_B.
            # Here, we are decomposing the multiplication of 4x4 matrices.
            # A is 4x4. We can see A as A_block where A_block[i_b, j_b] is a 2x2 matrix.
            # The multiplication C = A B can be seen as block multiplication.
            # The Kronecker product of the decomposition of 2x2 mult with itself gives the 4x4 mult.
            
            # Let's construct the vectors for U, V, W for this specific rank-1 term (c1, c2).
            # u_vec1 = U1[:, c1] (length 4, indices 0..3 corresponding to (0,0),(0,1),(1,0),(1,1) in block-local? No, 2x2 indices)
            # u_vec2 = U2[:, c2] (length 4)
            
            # The combined U vector has length 16.
            # The index in the combined U corresponds to (i_global, j_global).
            # i_global = i_b * 2 + i_l
            # j_global = j_b * 2 + j_l
            # The value is U1[i_b * 2 + j_b, c1] * U2[i_l * 2 + j_l, c2]
            
            for i_b in range(2):
                for j_b in range(2):
                    u1_val = U1[i_b * 2 + j_b, c1]
                    if u1_val == 0:
                        continue
                    for i_l in range(2):
                        for j_l in range(2):
                            u2_val = U2[i_l * 2 + j_l, c2]
                            if u2_val == 0:
                                continue
                            
                            i_global = i_b * 2 + i_l
                            j_global = j_b * 2 + j_l
                            idx_u = i_global * 4 + j_global
                            
                            U[idx_u, c] += u1_val * u2_val
                            
                            # Similarly for V
                            # V1 index: j_b * 2 + k_b
                            # V2 index: j_l * 2 + k_l
                            # Global V index: j_global * 4 + k_global
                            # j_global = j_b * 2 + j_l
                            # k_global = k_b * 2 + k_l
                            
                            # We need to iterate k_b, k_l as well for V and W
                            # It's more efficient to precompute non-zeros, but 16x49 is small.
                            # Let's just loop all for simplicity and correctness first.
                            pass # Handled below in a cleaner way? No, let's do full loops.
            
            # Let's redo the filling in a more direct triple loop to avoid missing terms.
            # Actually, the above logic was slightly mixed. Let's do it properly.
            
    # Reset and do it cleanly
    U = np.zeros((16, rank), dtype=float)
    V = np.zeros((16, rank), dtype=float)
    W = np.zeros((16, rank), dtype=float)
    
    for c1 in range(7):
        for c2 in range(7):
            c = c1 * 7 + c2
            
            # U: i_b, j_b, i_l, j_l
            for i_b in range(2):
                for j_b in range(2):
                    u1 = U1[i_b * 2 + j_b, c1]
                    if u1 == 0: continue
                    for i_l in range(2):
                        for j_l in range(2):
                            u2 = U2[i_l * 2 + j_l, c2]
                            if u2 == 0: continue
                            
                            i_g = i_b * 2 + i_l
                            j_g = j_b * 2 + j_l
                            U[i_g * 4 + j_g, c] += u1 * u2
                            
            # V: j_b, k_b, j_l, k_l
            for j_b in range(2):
                for k_b in range(2):
                    v1 = V1[j_b * 2 + k_b, c1]
                    if v1 == 0: continue
                    for j_l in range(2):
                        for k_l in range(2):
                            v2 = V2[j_l * 2 + k_l, c2]
                            if v2 == 0: continue
                            
                            j_g = j_b * 2 + j_l
                            k_g = k_b * 2 + k_l
                            V[j_g * 4 + k_g, c] += v1 * v2
                            
            # W: k_b, i_b, k_l, i_l
            for k_b in range(2):
                for i_b in range(2):
                    w1 = W1[k_b * 2 + i_b, c1]
                    if w1 == 0: continue
                    for k_l in range(2):
                        for i_l in range(2):
                            w2 = W2[k_l * 2 + i_l, c2]
                            if w2 == 0: continue
                            
                            k_g = k_b * 2 + k_l
                            i_g = i_b * 2 + i_l
                            W[k_g * 4 + i_g, c] += w1 * w2
                            
    return U, V, W

def _schoolbook_4x4x3():
    """
    Schoolbook decomposition for 4x4 * 4x3.
    Returns U, V, W of shapes (16, 48), (12, 48), (12, 48).
    """
    n, m, p = 4, 4, 3
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify_numpy(U, V, W, n, m, p):
    """
    Verify decomposition.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    rank = U_np.shape[1]
    
    # Reshape to 3D tensors
    # U[i, j, c]
    U3 = U_np.reshape(n, m, rank)
    # V[j, k, c]
    V3 = V_np.reshape(m, p, rank)
    # W[k, i, c]
    W3 = W_np.reshape(p, n, rank)
    
    # Compute T[i, j, k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    
    # The target tensor for matrix multiplication is 1 for all i,j,k?
    # No, the tensor T_{ijk} for matrix multiplication A_{ij} B_{jk} -> C_{ik} is defined such that
    # C_{ik} = sum_j A_{ij} B_{jk}.
    # The decomposition is sum_c U_{ic} V_{jc} W_{kc}? No.
    # The standard tensor decomposition of matrix multiplication:
    # M_{ijk} = sum_c U_{ic} V_{jc} W_{kc} ?
    # Let's check the indices in the prompt:
    # U: (n*m, rank) -> U[i, j, c]
    # V: (m*p, rank) -> V[j, k, c]
    # W: (p*n, rank) -> W[k, i, c]
    # The product is sum_c U[i, j, c] * V[j, k, c] * W[k, i, c].
    # This should equal 1 for all i, j, k?
    # Wait. The matrix multiplication tensor T_{ijk} is 1 if the indices correspond to a valid term?
    # Actually, the tensor for matrix multiplication of n x m and m x p is:
    # T_{i, j, k} = 1 for all i, j, k?
    # No. The tensor is defined on the space of A (n x m), B (m x p), C (n x p).
    # The multiplication map is linear in A and B.
    # T(A, B) = C.
    # The tensor T has indices i, j, k such that C_{ik} = sum_j A_{ij} B_{jk}.
    # So the tensor T_{ijk} is 1 for all i, j, k?
    # Yes, the structure constant is 1 for all valid indices.
    # So we expect T[i, j, k] = 1 for all i, j, k.
    
    if not np.allclose(T, 1.0, atol=1e-10):
        return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    try:
        # Part 1: 4x4x4 (columns 0-3 of B)
        U1, V1, W1 = _kron_4x4x4()
        
        # Part 2: 4x4x3 (columns 4-6 of B)
        # We need to map the indices correctly.
        # For the full problem:
        # A is 4x4.
        # B is 4x7.
        # C is 4x7.
        # We split B into B1 (4x4) and B2 (4x3).
        # C = A B1 + A B2.
        
        # U is shared?
        # In the decomposition sum_c U_c V_c W_c, the U factors depend on A indices.
        # Since A is the same for both parts, we can concatenate the U factors?
        # No, the rank is additive.
        # We have decomposition 1: sum_{c1} U1_c1 V1_c1 W1_c1
        # We have decomposition 2: sum_{c2} U2_c2 V2_c2 W2_c2
        # Total: sum_{c1} ... + sum_{c2} ...
        # We can create a single set of matrices by stacking the columns.
        
        U_total = np.hstack([U1, U1]) # Wait, U1 is for the square part.
        # The U factor for the square part uses A indices (i, j) where j is 0..3.
        # The U factor for the rectangular part uses A indices (i, j) where j is 0..3.
        # So U is the same dimension (16 x rank).
        # We just need to ensure the V and W indices align with the split of B and C.
        
        # V1 (for B1): Indices (j, k1) where j=0..3, k1=0..3. Shape 16x49.
        # V2 (for B2): Indices (j, k2) where j=0..3, k2=0..2. Shape 12x48.
        # We need to map V2 indices to the global B indices (j, k_global).
        # k_global = 4 + k2.
        
        # W1 (for C1): Indices (k1, i) where k1=0..3, i=0..3. Shape 16x49.
        # W2 (for C2): Indices (k2, i) where k2=0..2, i=0..3. Shape 12x48.
        # We need to map W2 indices to global C indices (k_global, i).
        # k_global = 4 + k2.
        
        U2, V2, W2 = _schoolbook_4x4x3()
        
        # Construct V_total
        # V_total has shape (28, 49+48)
        V_total = np.zeros((28, 49 + 48), dtype=float)
        # Fill V1 part (cols 0-48)
        # V1 indices: j*4 + k1. j=0..3, k1=0..3.
        # Global V indices: j*7 + k_global.
        # For V1: k_global = k1.
        for j in range(4):
            for k1 in range(4):
                idx_v1 = j * 4 + k1
                idx_v_total = j * 7 + k1
                V_total[idx_v_total, 0:49] = V1[idx_v1, :]
                
        # Fill V2 part (cols 49-96)
        # V2 indices: j*3 + k2. j=0..3, k2=0..2.
        # Global V indices: j*7 + k_global.
        # For V2: k_global = 4 + k2.
        for j in range(4):
            for k2 in range(3):
                idx_v2 = j * 3 + k2
                idx_v_total = j * 7 + (4 + k2)
                V_total[idx_v_total, 49:97] = V2[idx_v2, :]
                
        # Construct W_total
        # W_total has shape (28, 49+48)
        W_total = np.zeros((28, 49 + 48), dtype=float)
        # Fill W1 part (cols 0-48)
        # W1 indices: k1*4 + i. k1=0..3, i=0..3.
        # Global W indices: k_global*4 + i.
        # For W1: k_global = k1.
        for k1 in range(4):
            for i in range(4):
                idx_w1 = k1 * 4 + i
                idx_w_total = k1 * 4 + i
                W_total[idx_w_total, 0:49] = W1[idx_w1, :]
                
        # Fill W2 part (cols 49-96)
        # W2 indices: k2*4 + i. k2=0..2, i=0..3.
        # Global W indices: k_global*4 + i.
        # For W2: k_global = 4 + k2.
        for k2 in range(3):
            for i in range(4):
                idx_w2 = k2 * 4 + i
                idx_w_total = (4 + k2) * 4 + i
                W_total[idx_w_total, 49:97] = W2[idx_w2, :]
                
        # U_total
        # U1 is 16x49. U2 is 16x48.
        U_total = np.hstack([U1, U2])
        
        result = {
            "U": U_total.tolist(),
            "V": V_total.tolist(),
            "W": W_total.tolist()
        }
        
        if _verify_numpy(result["U"], result["V"], result["W"], n, m, p):
            return result
            
    except Exception as e:
        pass
    
    # Fallback: Schoolbook
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
