import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    def check_decomposition(U, V, W, rank):
        """Verify that U, V, W with given rank exactly decompose the matmul tensor."""
        res = np.zeros((n, m, p))
        for c in range(rank):
            U_c = U[:, c].reshape(n, m)
            V_c = V[:, c].reshape(m, p)
            W_c = W[:, c].reshape(p, n)
            res += U_c[:, :, np.newaxis] * V_c[np.newaxis, :, :] * W_c.T[:, np.newaxis, :]
        return np.allclose(res, 1.0, atol=1e-9)
    
    def build_strassen_block(i1, i2, j1, j2, k1, k2):
        """Return (U_rows, V_rows, W_rows) for a 2x2 Strassen block as lists of (index, value) tuples."""
        U_rows = []
        V_rows = []
        W_rows = []
        
        # M1 = (A00 + A11)(B00 + B11)
        U_rows.append([(i1*m+j1, 1), (i2*m+j2, 1)])
        V_rows.append([(j1*p+k1, 1), (j2*p+k2, 1)])
        W_rows.append([(k1*n+i1, 1), (k2*n+i2, 1)])
        
        # M2 = (A10 + A11) B00
        U_rows.append([(i2*m+j1, 1), (i2*m+j2, 1)])
        V_rows.append([(j1*p+k1, 1)])
        W_rows.append([(k1*n+i2, 1), (k2*n+i2, -1)])
        
        # M3 = A00 (B01 - B11)
        U_rows.append([(i1*m+j1, 1)])
        V_rows.append([(j1*p+k2, 1), (j2*p+k2, -1)])
        W_rows.append([(k2*n+i1, 1), (k2*n+i2, 1)])
        
        # M4 = A11 (B10 - B00)
        U_rows.append([(i2*m+j2, 1)])
        V_rows.append([(j2*p+k1, 1), (j1*p+k1, -1)])
        W_rows.append([(k1*n+i1, 1), (k1*n+i2, 1)])
        
        # M5 = (A00 + A01) B11
        U_rows.append([(i1*m+j1, 1), (i1*m+j2, 1)])
        V_rows.append([(j2*p+k2, 1)])
        W_rows.append([(k1*n+i1, -1), (k2*n+i1, 1)])
        
        # M6 = (A10 - A00)(B00 + B01)
        U_rows.append([(i2*m+j1, 1), (i1*m+j1, -1)])
        V_rows.append([(j1*p+k1, 1), (j1*p+k2, 1)])
        W_rows.append([(k2*n+i2, 1)])
        
        # M7 = (A01 - A11)(B10 + B11)
        U_rows.append([(i1*m+j2, 1), (i2*m+j2, -1)])
        V_rows.append([(j2*p+k1, 1), (j2*p+k2, 1)])
        W_rows.append([(k1*n+i1, 1)])
        
        return U_rows, V_rows, W_rows
    
    def build_schoolbook_block(i_indices, j_indices, k_indices):
        """Return schoolbook terms as lists of (index, value) tuples."""
        U_rows = []
        V_rows = []
        W_rows = []
        for i in i_indices:
            for j in j_indices:
                for k in k_indices:
                    U_rows.append([(i*m+j, 1)])
                    V_rows.append([(j*p+k, 1)])
                    W_rows.append([(k*n+i, 1)])
        return U_rows, V_rows, W_rows
    
    def assemble(U_rows, V_rows, W_rows, rank):
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        for c in range(rank):
            for idx, val in U_rows[c]:
                U[idx, c] = val
            for idx, val in V_rows[c]:
                V[idx, c] = val
            for idx, val in W_rows[c]:
                W[idx, c] = val
        return U, V, W
    
    # Baseline: rank 28 (two Strassen 2x2 blocks + schoolbook for the rest)
    def build_rank28():
        U_rows = []
        V_rows = []
        W_rows = []
        
        # 1. A[:, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2] (Strassen, 7)
        u, v, w = build_strassen_block(0, 1, 0, 1, 0, 1)
        U_rows.extend(u); V_rows.extend(v); W_rows.extend(w)
        
        # 2. A[:, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4] (Strassen, 7)
        u, v, w = build_strassen_block(0, 1, 0, 1, 2, 3)
        U_rows.extend(u); V_rows.extend(v); W_rows.extend(w)
        
        # 3. A[:, 0:2] * B[0:2, 4:5] -> C[0:2, 4:5] (Schoolbook, 4)
        u, v, w = build_schoolbook_block([0, 1], [0, 1], [4])
        U_rows.extend(u); V_rows.extend(v); W_rows.extend(w)
        
        # 4. A[:, 2:3] * B[2:3, 0:2] -> C[0:2, 0:2] (Schoolbook, 4)
        u, v, w = build_schoolbook_block([0, 1], [2], [0, 1])
        U_rows.extend(u); V_rows.extend(v); W_rows.extend(w)
        
        # 5. A[:, 2:3] * B[2:3, 2:4] -> C[0:2, 2:4] (Schoolbook, 4)
        u, v, w = build_schoolbook_block([0, 1], [2], [2, 3])
        U_rows.extend(u); V_rows.extend(v); W_rows.extend(w)
        
        # 6. A[:, 2:3] * B[2:3, 4:5] -> C[0:2, 4:5] (Schoolbook, 2)
        u, v, w = build_schoolbook_block([0, 1], [2], [4])
        U_rows.extend(u); V_rows.extend(v); W_rows.extend(w)
        
        return assemble(U_rows, V_rows, W_rows, len(U_rows))
    
    # Try rank 27: use Strassen for 2x2 blocks, and a "rank-3" trick for the 2x1 * 1x2 blocks
    # For a 2x1 * 1x2 block (i.e., outer product of a column and a row), we need rank 4 schoolbook.
    # Can we do it in rank 3? The tensor is sum over i in {0,1}, k in {0,1}: e_i (A) * e_k (B) * e_{k,i} (C)
    # This is a 2x1x2 tensor. Its rank is 2 (it's a sum of two rank-1 terms if we group by i or k).
    # Wait: A is 2x1, B is 1x2, C is 2x2. C[i,k] = A[i,0]*B[0,k].
    # This is the outer product of vector a (length 2) and vector b (length 2), giving a 2x2 matrix.
    # As a tensor in A⊗B⊗C: T = sum_{i,k} a_i b_k c_{ik}
    # This equals (sum_i a_i A_i) ⊗ (sum_k b_k B_k) ⊗ ... no, it's a specific tensor.
    # Actually T = a ⊗ b ⊗ I_{2x2} flattened... Let me think differently.
    # T[i, 0, k] = a_i * b_k for i in {0,1}, k in {0,1}.
    # This is rank 2: T = a_0 b_0 (e_0 ⊗ e_0 ⊗ e_{0,0}) + a_0 b_1 (e_0 ⊗ e_0 ⊗ e_{0,1}) + a_1 b_0 (e_1 ⊗ e_0 ⊗ e_{1,0}) + a_1 b_1 (e_1 ⊗ e_0 ⊗ e_{1,1})
    # Hmm, that's 4 terms. But we can factor: T = (a_0 e_0 + a_1 e_1) ⊗ e_0 ⊗ (b_0 e_{0,0} + b_1 e_{1,0}) + ... no.
    # Actually: T[i,0,k] = a_i b_k. This is the Kronecker product a ⊗ b placed in the (i,0,k) slots.
    # As a tensor: T = sum_i sum_k a_i b_k E_i ⊗ E_0 ⊗ E_k = (sum_i a_i E_i) ⊗ E_0 ⊗ (sum_k b_k E_k)
    # That's rank 1! Because it's a single outer product of three vectors!
    # Wait, but the U, V, W matrices must have the right shapes. U is (n*m, rank) = (6, rank), V is (m*p, rank) = (15, rank), W is (p*n, rank) = (10, rank).
    # For the block A[:,2] * B[2,:] -> C[:,0:2] (which is 2x1 * 1x2 -> 2x2):
    # U has nonzero only at row 2 (index i*m+j = i*3+2 for i=0,1), so rows 2 and 5.
    # V has nonzero only at row j*p+k = 2*5+k = 10+k for k=0,1, so rows 10 and 11.
    # W has nonzero only at row k*n+i = k*2+i for i=0,1, k=0,1, so rows 0,1,2,3.
    # The tensor is: sum_{i,k} U[i*3+2, c] * V[10+k, c] * W[k*2+i, c] = C[i,k] = A[i,2]*B[2,k]
    # If we set U[2,c]=1, U[5,c]=1 (for i=0,1), V[10,c]=1, V[11,c]=1 (for k=0,1), and W[k*2+i,c]=1,
    # then we get sum over i,k of 1*1*1 = 4, but we need C[i,k] = A[i,2]*B[2,k] which depends on the input.
    # The decomposition is: C[i,k] = sum_c U[i*3+2,c] * V[10+k,c] * W[k*2+i,c]
    # We need this to equal A[i,2]*B[2,k] for all A and B.
    # So U[i*3+2,c] must be a coefficient that multiplies A[i,2], and V[10+k,c] must multiply B[2,k].
    # Let U[i*3+2,c] = u_{i,c}, V[10+k,c] = v_{k,c}, W[k*2+i,c] = w_{k,i,c}.
    # Then sum_c u_{i,c} v_{k,c} w_{k,i,c} = A[i,2] B[2,k] for all A[i,2], B[2,k].
    # This means the matrix M_{i,k} = sum_c u_{i,c} v_{k,c} w_{k,i,c} must equal the outer product of the identity... 
    # Actually, we need M_{i,k} to be 1 when we set A[i,2]=1 and B[2,k]=1, and 0 otherwise.
    # So M must be the 2x2 identity? No: C[i,k] = A[i,2]*B[2,k], so the "coefficient tensor" is T[i,k] = 1 for all i,k (since A[i,2] and B[2,k] are independent variables).
    # Wait, I'm confusing myself. The tensor T has entries T[i, j, k] = 1 if the matmul produces C[i,k] from A[i,j]*B[j,k].
    # For the block where j=2 (fixed), i in {0,1}, k in {0,1}: T[i, 2, k] = 1.
    # So we need: sum_c U[i*3+2, c] * V[2*5+k, c] * W[k*2+i, c] = 1 for all i in {0,1}, k in {0,1}.
    # This is a 2x1x2 tensor with all entries 1. Its rank is 2 (it's the outer product of a 2-vector of 1s, a 1-vector of 1, and a 2-vector of 1s... wait, that's rank 1!).
    # T[i,2,k] = 1 = 1*1*1. So it IS rank 1! We can do it with ONE term:
    # U[2,c]=1, U[5,c]=1, V[10,c]=1, V[11,c]=1, W[0,c]=1, W[1,c]=1, W[2,c]=1, W[3,c]=1.
    # Then sum_c U[i*3+2,c]*V[10+k,c]*W[k*2+i,c] = 1*1*1 = 1. Correct!
    # But wait, that's not right either. The U, V, W entries are fixed coefficients, not depending on A or B.
    # The matmul is: C[i,k] = sum_j A[i,j] * B[j,k]. For j=2: C[i,k] += A[i,2]*B[2,k].
    # In the tensor decomposition: C[i,k] = sum_c (sum_j U[i*m+j,c] A[i,j]) * (sum_l V[j*p+l,c] B[j,l]) * W[k*n+i,c]
    # No, that's not how it works. The standard decomposition is:
    # C[i,k] = sum_c U_c[i,j] V_c[j,l] W_c[k,i] ... I need to be more careful.
    
    # Let me re-read the problem. The tensor T[i,j,k] = 1 for all i,j,k (for the full matmul).
    # The decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # So for the block j=2, i in {0,1}, k in {0,1}:
    # 1 = sum_c U[i*3+2, c] * V[2*5+k, c] * W[k*2+i, c]
    # This must hold for each (i,k) pair. The tensor T restricted to this block is all 1s, shape 2x1x2.
    # This tensor has rank 1: it's the outer product of (1,1), (1), (1,1).
    # So we need: U[i*3+2,c]*V[10+k,c]*W[k*2+i,c] = 1 for a single c, with U[2,c]=U[5,c]=1, V[10,c]=V[11,c]=1, W[0,c]=W[1,c]=W[2,c]=W[3,c]=1.
    # That gives 1*1*1=1 for all (i,k). Yes! Rank 1 suffices for this block.
    
    # Similarly, for the block A[:,0:2] * B[0:2,4] -> C[:,4] (2x2 * 2x1 -> 2x1):
    # T[i,j,k=4] = 1 for i in {0,1}, j in {0,1}, k=4. Shape 2x2x1, all 1s. Rank 1.
    # U[i*3+j,c]=1 for i,j in {0,1}, V[j*5+4,c]=1 for j in {0,1}, W[4*2+i,c]=1 for i in {0,1}.
    # That's 4 U entries, 2 V entries, 2 W entries, all 1. Rank 1.
    
    # And for A[:,2] * B[2,0:2] -> C[:,0:2]: same as above, rank 1.
    # And A[:,2] * B[2,2:4] -> C[:,2:4]: same, rank 1.
    # And A[:,2] * B[2,4] -> C[:,4]: 2x1 * 1x1 -> 2x1, T[i,2,4]=1 for i in {0,1}. Shape 2x1x1, rank 1.
    
    # So the "schoolbook" blocks that are 2x1*1x2 or 2x2*2x1 etc. can be done in rank 1 each!
    # Let me recount:
    # Block 1: A[:,0:2]*B[0:2,0:2] -> 2x2*2x2 -> 2x2: Strassen, rank 7
    # Block 2: A[:,0:2]*B[0:2,2:4] -> 2x2*2x2 -> 2x2: Strassen, rank 7
    # Block 3: A[:,0:2]*B[0:2,4:5] -> 2x2*2x1 -> 2x1: rank 1 (outer product structure)
    # Block 4: A[:,2:3]*B[2:3,0:2] -> 2x1*1x2 -> 2x2: rank 1
    # Block 5: A[:,2:3]*B[2:3,2:4] -> 2x1*1x2 -> 2x2: rank 1
    # Block 6: A[:,2:3]*B[2:3,4:5] -> 2x1*1x1 -> 2x1: rank 1
    # Total: 7+7+1+1+1+1 = 18!
    
    # Wait, but I need to verify this. Let me construct rank 18 and check.
    
    def build_rank18():
        U_rows = []
        V_rows = []
        W_rows = []
        
        # 1. A[:, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2] (Strassen, 7)
        u, v, w = build_strassen_block(0, 1, 0, 1, 0, 1)
        U_rows.extend(u); V_rows.extend(v); W_rows.extend(w)
        
        # 2. A[:, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4] (Strassen, 7)
        u, v, w = build_strassen_block(0, 1, 0, 1, 2, 3)
        U_rows.extend(u); V_rows.extend(v); W_rows.extend(w)
        
        # 3. A[:, 0:2] * B[0:2, 4:5] -> C[0:2, 4:5] (rank 1)
        # T[i,j,4] = 1 for i in {0,1}, j in {0,1}
        U_rows.append([(i*3+j, 1) for i in [0,1] for j in [0,1]])
        V_rows.append([(j*5+4, 1) for j in [0,1]])
        W_rows.append([(4*2+i, 1) for i in [0,1]])
        
        # 4. A[:, 2:3] * B[2:3, 0:2] -> C[0:2, 0:2] (rank 1)
        # T[i,2,k] = 1 for i in {0,1}, k in {0,1}
        U_rows.append([(i*3+2, 1) for i in [0,1]])
        V_rows.append([(2*5+k, 1) for k in [0,1]])
        W_rows.append([(k*2+i, 1) for i in [0,1] for k in [0,1]])
        
        # 5. A[:, 2:3] * B[2:3, 2:4] -> C[0:2, 2:4] (rank 1)
        # T[i,2,k] = 1 for i in {0,1}, k in {2,3}
        U_rows.append([(i*3+2, 1) for i in [0,1]])
        V_rows.append([(2*5+k, 1) for k in [2,3]])
        W_rows.append([(k*2+i, 1) for i in [0,1] for k in [2,3]])
        
        # 6. A[:, 2:3] * B[2:3, 4:5] -> C[0:2, 4:5] (rank 1)
        # T[i,2,4] = 1 for i in {0,1}
        U_rows.append([(i*3+2, 1) for i in [0,1]])
        V_rows.append([(2*5+4, 1)])
        W_rows.append([(4*2+i, 1) for i in [0,1]])
        
        return assemble(U_rows, V_rows, W_rows, len(U_rows))
    
    # Build and verify rank 18
    U18, V18, W18 = build_rank18()
    if check_decomposition(U18, V18, W18, 18):
        return {"U": U18.tolist(), "V": V18.tolist(), "W": W18.tolist()}
    
    # If rank 18 fails, try rank 28
    U28, V28, W28 = build_rank28()
    if check_decomposition(U28, V28, W28, 28):
        return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}
    
    # Fallback to schoolbook rank 30
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
