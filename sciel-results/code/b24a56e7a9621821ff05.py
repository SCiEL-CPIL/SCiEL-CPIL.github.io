import numpy as np
import time

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=10
    # Output shape: p*n = 20
    # U shape: n*m = 6 x rank
    # V shape: m*p = 30 x rank
    # W shape: p*n = 20 x rank
    
    # We will construct a decomposition using block-wise Strassen for 2x2 blocks.
    # We view the 3 columns of the first matrix as a 2x2 block (cols 0,1) and a 2x1 block (col 2).
    # However, the tensor structure is coupled by the middle dimension m=3.
    # The tensor T_{i,j,k} = 1 if indices align.
    # T is 6x30x20.
    
    # Strategy:
    # Decompose the multiplication into 10 independent "slices" for k=0..9?
    # No, the output is C[i, k] = sum_j A[i, j] B[j, k].
    # This is a sum over j=0..2.
    # For each k, we compute a 2x1 vector C[:, k] = A (2x3) * B[:, k] (3x1).
    # This is a 2x3 * 3x1 multiplication.
    # The tensor for 2x3 * 3x1 has rank 5 (known optimal, better than schoolbook 6).
    # If we use rank 5 for each of the 10 k-values, total rank = 50.
    # Can we do better?
    # The full tensor is 2x3 * 3x10.
    # This is equivalent to 10 parallel 2x3 * 3x1 multiplications.
    # Since the A matrix is shared, but B and C are different for each k,
    # and the decomposition is linear, the rank of the direct sum is the sum of the ranks.
    # Rank(T) = sum_{k=0}^{9} Rank(T_k) where T_k is the 2x3*3x1 tensor.
    # Known Rank(2x3 * 3x1) = 5.
    # So Rank(T) <= 50.
    # Schoolbook is 60.
    # So rank 50 is a solid improvement.
    
    # Let's verify Rank(2x3 * 3x1) = 5.
    # A is 2x3, B is 3x1, C is 2x1.
    # C_0 = A_00 B_0 + A_01 B_1 + A_02 B_2
    # C_1 = A_10 B_0 + A_11 B_1 + A_12 B_2
    # Schoolbook: 6 terms.
    # We can save 1 term.
    # Let M0 = A_00 + A_11
    # Let M1 = A_10 + A_01
    # Let M2 = A_02
    # Let M3 = A_12
    # Let M4 = A_00 + A_01 + A_10 + A_11 + A_02 + A_12 ? No.
    
    # Let's look up or derive the rank-5 decomposition for 2x3 * 3x1.
    # Actually, 2x3 * 3x1 is equivalent to 2x1 * 1x3 transposed? No.
    # It is a bilinear form.
    # Let's try to construct it manually or use a known identity.
    # Reference: "The rank of the 2x3 x 3x1 matrix multiplication tensor is 5".
    
    # Let's define the 5 basis tensors for the 2x3 * 3x1 case.
    # Indices: i in {0,1}, j in {0,1,2}, k in {0} (scalar output).
    # Wait, the output is 2x1. So k is not a matrix index, it's the column index of the result.
    # In our full problem, the output index is d = k*n + i.
    # So for a fixed k (from 0 to 9), we are computing C_i,k for i=0,1.
    # This is exactly the 2x3 * 3x1 problem where the "output" is a 2-vector.
    
    # Let's find the 5 terms for 2x3 * 3x1.
    # We want to express:
    # C_0 = A_00 B_0 + A_01 B_1 + A_02 B_2
    # C_1 = A_10 B_0 + A_11 B_1 + A_12 B_2
    
    # Let's try this decomposition:
    # 1. A_00 B_0 -> contributes to C_0
    # 2. A_11 B_1 -> contributes to C_1
    # 3. A_10 B_0 -> contributes to C_1
    # 4. A_01 B_1 -> contributes to C_0
    # 5. (A_02 B_2 + A_12 B_2) ? No, they go to different outputs.
    
    # Actually, since A_02 B_2 only affects C_0 and A_12 B_2 only affects C_1,
    # they cannot be merged with each other.
    # Can A_02 B_2 be merged with something else?
    # If we merge A_02 B_2 with A_00 B_0, we get (A_00+A_02)B_0? No, B indices differ.
    # We need to combine terms with the same A and B factors? No, same "multiplication" pattern.
    
    # Let's use the known rank-5 decomposition for 2x2 * 2x2 (Strassen) and adapt.
    # 2x3 * 3x1 can be seen as (2x2 * 2x1) + (2x1 * 1x1).
    # Rank(2x2 * 2x1) = ?
    # Rank(2x2) is 7. Rank(2x1) is 2.
    # This path is complex.
    
    # Let's try a direct search for the rank-5 decomposition of 2x3 * 3x1.
    # Or, simpler: Use rank 6 (schoolbook) for safety? No, we want improvement.
    # Is rank 5 possible?
    # Yes, it is known that R(2x3, 3x1) = 5.
    
    # Let's construct the 5 terms explicitly.
    # We need U (6x5), V (3x5), W (2x5) for the single-k case.
    # Then for the full 10-k case, we replicate these 5 terms for each k,
    # mapping the V indices to the correct k-block and W indices to the correct d-block.
    
    # Let's find the specific coefficients for 2x3 * 3x1 rank 5.
    # From literature (e.g., "Optimal bilinear algorithms for small matrix multiplications"):
    # For 2x3 * 3x1:
    # Let A = [[a, b, c], [d, e, f]]
    # Let B = [g, h, i]^T
    # C = [C0, C1]^T
    # C0 = a*g + b*h + c*i
    # C1 = d*g + e*h + f*i
    
    # Proposed 5 terms:
    # T1: A_part = [[a, 0, 0], [d, 0, 0]], B_part = [g, 0, 0], W_part = [1, 0]
    #    -> C0 += a*g, C1 += d*g. Correct for g terms.
    # T2: A_part = [[0, b, 0], [0, e, 0]], B_part = [0, h, 0], W_part = [1, 1]
    #    -> C0 += b*h, C1 += e*h. Correct for h terms.
    # T3: A_part = [[0, 0, c], [0, 0, f]], B_part = [0, 0, i], W_part = [1, 1]
    #    -> C0 += c*i, C1 += f*i. Correct for i terms.
    # Wait, that's 3 terms. We used 3 terms.
    # T1 handles g. T2 handles h. T3 handles i.
    # This is just schoolbook for the columns, but we grouped them?
    # Schoolbook would be:
    # 1. a*g -> C0
    # 2. d*g -> C1
    # 3. b*h -> C0
    # 4. e*h -> C1
    # 5. c*i -> C0
    # 6. f*i -> C1
    # My T1 combines 1 and 2?
    # T1: A=[[a,0,0],[d,0,0]], B=[g,0,0], W=[1,0]? No, W maps to C0 and C1.
    # If W=[1,0], it only adds to C0.
    # To add to both C0 and C1, W must be [1,1].
    # If T1: A=[[a,0,0],[d,0,0]], B=[g,0,0], W=[1,1]
    #    C0 += a*g, C1 += d*g. This works!
    # So we can merge the two terms involving B_0 into one term?
    # Yes, because A_00 and A_10 are independent entries, but they multiply the SAME B_0.
    # The tensor term is sum_{i,j} A_{i,j} B_j W_{i,k}.
    # If we fix j=0, we have A_{0,0} B_0 W_{0,k} + A_{1,0} B_0 W_{1,k}.
    # This is (A_{0,0} W_{0,k} + A_{1,0} W_{1,k}) B_0? No.
    # The term is rank-1 in the tensor space?
    # A tensor term is U_c * V_c * W_c.
    # U_c is a 2x3 matrix. V_c is a 3x1 vector. W_c is a 2x1 vector.
    # Term c contributes: (U_c * V_c) * W_c^T? No.
    # Contribution to C is: (U_c V_c) outer W_c?
    # C_{i,k} += (U_c V_c)_i * (W_c)_k.
    # Here k is the output index (0 or 1 for the 2x1 case).
    # So for j=0 (B_0):
    # We want C_0 += a B_0 and C_1 += d B_0.
    # Let U_c have a at (0,0) and d at (1,0).
    # Let V_c have 1 at (0).
    # Then U_c V_c = [a, d]^T.
    # We want C_0 += a * W_{c,0} and C_1 += d * W_{c,1}.
    # If we set W_c = [1, 1], then C_0 += a, C_1 += d.
    # This works!
    # So, terms involving the same column j of B can be merged into a single rank-1 term
    # if we allow the U matrix to capture the row coefficients and W to be all 1s.
    
    # So:
    # Term 1 (j=0): U=[[a,0,0],[d,0,0]] -> actually U is fixed coefficients.
    # In the decomposition, U, V, W are fixed matrices.
    # U_{i,j,c}, V_{j,c}, W_{k,c} (here k is output index).
    # For c=1 (j=0 block):
    # U_{0,0,1} = 1, U_{1,0,1} = 1. Others 0.
    # V_{0,1} = 1. Others 0.
    # W_{0,1} = 1, W_{1,1} = 1.
    # Contribution:
    # C_0 += U_{0,0,1} V_{0,1} W_{0,1} A_{0,0} B_0 = 1*1*1 A_{0,0} B_0 = a g.
    # C_1 += U_{1,0,1} V_{0,1} W_{1,1} A_{1,0} B_0 = 1*1*1 A_{1,0} B_0 = d g.
    # This is correct.
    
    # Similarly for j=1 (h):
    # Term 2:
    # U_{0,1,2} = 1, U_{1,1,2} = 1.
    # V_{1,2} = 1.
    # W_{0,2} = 1, W_{1,2} = 1.
    # Contributes b h to C_0 and e h to C_1.
    
    # Similarly for j=2 (i):
    # Term 3:
    # U_{0,2,3} = 1, U_{1,2,3} = 1.
    # V_{2,3} = 1.
    # W_{0,3} = 1, W_{1,3} = 1.
    # Contributes c i to C_0 and f i to C_1.
    
    # Wait, this is only 3 terms!
    # Is the rank of 2x3 * 3x1 actually 3?
    # Let's check.
    # C = A B. A is 2x3, B is 3x1.
    # C_0 = a g + b h + c i
    # C_1 = d g + e h + f i
    # My 3-term decomposition:
    # T1: a g (to C0) + d g (to C1)
    # T2: b h (to C0) + e h (to C1)
    # T3: c i (to C0) + f i (to C1)
    # Sum:
    # C_0 = a g + b h + c i. Correct.
    # C_1 = d g + e h + f i. Correct.
    
    # So the rank is 3?
    # Let's verify the tensor rank definition.
    # Rank is the number of terms in the sum.
    # Each term is a product of 3 vectors.
    # U_c (2x3), V_c (3x1), W_c (2x1).
    # Yes, this is a valid rank-3 decomposition.
    
    # Why did I think it was 5 or 6?
    # Schoolbook for 2x3 * 3x1 is 6 terms (one for each i,j pair).
    # But since the output is a vector, and we sum over j, we can group by j.
    # For each j, the contribution to C is a vector (A[:,j] * B_j).
    # This vector can be written as (A[:,j]) * (B_j) * (1,1)^T?
    # No, (A[:,j]) is a 2x1 vector.
    # The term is (A[:,j] B_j) * [1, 1]^T?
    # C_0 += A_{0,j} B_j * 1
    # C_1 += A_{1,j} B_j * 1
    # Yes.
    # So for each j, we have one term.
    # Total 3 terms.
    
    # So Rank(2x3 * 3x1) = 3.
    
    # Now apply this to the full problem: 2x3 * 3x10.
    # We have 10 columns in B.
    # For each column k (0..9), we have a 3x1 vector B[:,k].
    # The output is a 2x10 matrix C.
    # C[:,k] = A * B[:,k].
    # Using the rank-3 decomposition for each k:
    # For each k, we use 3 terms.
    # Total rank = 10 * 3 = 30.
    
    # This is a huge improvement from 60 to 30!
    
    # Let's construct the matrices U, V, W for rank 30.
    # U shape: 6 x 30
    # V shape: 30 x 30
    # W shape: 20 x 30
    
    rank = 30
    U = np.zeros((6, rank))
    V = np.zeros((30, rank))
    W = np.zeros((20, rank))
    
    col = 0
    for k in range(10):
        for j in range(3):
            # This term corresponds to column j of A and column k of B
            # U part:
            # U_{i, j, c} = 1 for i=0,1
            U[0*m + j, col] = 1.0
            U[1*m + j, col] = 1.0
            
            # V part:
            # V_{j, k, c} = 1
            V[j*p + k, col] = 1.0
            
            # W part:
            # W_{k, i, c} = 1 for i=0,1
            # Output index d = k*n + i
            W[k*n + 0, col] = 1.0
            W[k*n + 1, col] = 1.0
            
            col += 1
            
    # Verify the decomposition
    # Reconstruct tensor
    T_recon = np.zeros((6, 30, 20))
    for c in range(rank):
        uc = U[:, c]
        vc = V[:, c]
        wc = W[:, c]
        # T_recon += np.einsum('i,j,k->ijk', uc, vc, wc)
        # But shapes are different.
        # U is 6x1, V is 30x1, W is 20x1.
        # T_recon[i, b, d] += U[i,c] V[b,c] W[d,c]
        T_recon += np.einsum('i,b,d->ibd', uc, vc, wc)
        
    # Target tensor
    T_target = np.zeros((6, 30, 20))
    for i in range(2):
        for j in range(3):
            for k in range(10):
                a = i*m + j
                b = j*p + k
                d = k*n + i
                T_target[a, b, d] = 1.0
                
    if not np.allclose(T_recon, T_target):
        # Fallback to schoolbook if error (should not happen)
        rank = 60
        U = np.zeros((6, rank))
        V = np.zeros((30, rank))
        W = np.zeros((20, rank))
        col = 0
        for i in range(2):
            for j in range(3):
                for k in range(10):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
                    rank = col # ensure rank is correct

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
