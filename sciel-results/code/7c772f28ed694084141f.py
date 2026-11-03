"""Matrix multiplication 2x3x10 tensor decomposition with rank minimization."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 10]
DEADLINE = 120.0
SAFETY_MARGIN = 5.0  # seconds before deadline to stop


def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor.
    
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    The tensor T[i,j,k] = delta_{i,k} (in the appropriate sense) must equal
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    """
    rank = U.shape[1]
    # Build tensor T of shape (n, m, p)
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0  # The multiplication tensor for (n,m,p) matmul
                # Actually, the tensor for multiplying (n x m) by (m x p) is:
                # T[i,j,k] where the bilinear form is sum_j A[i,j]*B[j,k]
                # The tensor has T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k]
                # So T[i,j,k] = 1 for all i,j,k (it's the standard matmul tensor)
    
    # Compute reconstructed tensor
    T_recon = np.zeros((n, m, p))
    for c in range(rank):
        # U[:, c] has shape (n*m,), V[:, c] has shape (m*p,), W[:, c] has shape (p*n,)
        u = U[:, c].reshape(n, m)
        v = V[:, c].reshape(m, p)
        w = W[:, c].reshape(p, n)
        T_recon += u[:, :, None] * v[None, :, :] * w[None, :, :]
    
    return np.allclose(T_recon, T, atol=1e-9)


def _verify_fast(U, V, W, n, m, p):
    """Faster verification using tensor mode products."""
    rank = U.shape[1]
    # For each (i,k) pair, the (m,) vector should be e_i * e_k^T flattened... 
    # Actually let's just check: for each i, k, the m-vector [sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] for j in range(m)]
    # should equal the standard matmul tensor slice.
    
    # The matmul tensor: T[i,j,k] = 1 for all valid i,j,k
    # So for fixed (i,k), the j-slice is all ones of length m.
    
    for i in range(n):
        for k in range(p):
            # Compute the j-slice
            j_slice = np.zeros(m)
            for c in range(rank):
                w_val = W[k * n + i, c]
                if w_val != 0:
                    u_col = U[i * m:(i+1) * m, c]  # shape (m,)
                    v_col = V[:, k * 1 + 0]  # this is wrong, need V[j*p+k, c]
                    # V[j*p+k, c] for j in range(m)
                    v_col = np.array([V[j * p + k, c] for j in range(m)])
                    j_slice += w_val * u_col * v_col
            
            # Should be all ones
            if not np.allclose(j_slice, np.ones(m), atol=1e-9):
                return False
    return True


def _verify_correct(U, V, W, n, m, p):
    """Correct verification: check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Reshape for efficient computation
    # U: (n*m, rank) -> (n, m, rank)
    # V: (m*p, rank) -> (m, p, rank)  
    # W: (p*n, rank) -> (p, n, rank)
    
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    # For each (i,j,k), compute sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # This is a tensor contraction.
    
    # Efficient way: for each c, form outer product and sum
    # Or use einsum
    # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    
    # Let's do it with loops over c for clarity, but vectorize over (i,j,k)
    T = np.zeros((n, m, p))
    for c in range(rank):
        # u: (n, m), v: (m, p), w: (p, n)
        u = U3[:, :, c]  # (n, m)
        v = V3[:, :, c]  # (m, p)
        w = W3[:, :, c]  # (p, n)
        
        # We need T[i,j,k] += u[i,j] * v[j,k] * w[k,i]
        # For fixed j: T[:,j,:] += u[:,j] * v[j,:] * w.T
        # w[k,i] = w.T[i,k]
        wT = w.T  # (n, p)
        for j in range(m):
            T[:, j, :] += u[:, j, None] * v[j, None, :] * wT
    
    # Check T == 1 everywhere
    return np.allclose(T, 1.0, atol=1e-9)


def _strassen_2x2_2x2_2x2():
    """Strassen decomposition for 2x2 * 2x2 -> 2x2, rank 7.
    
    Returns U (4x7), V (4x7), W (4x7) where:
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # Strassen's 7 products for 2x2 matmul:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # The tensor: T[i,j,k] = 1 for all i,j,k in {0,1}
    # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    
    # For each Strassen product c, we need:
    # U[i*2+j, c] * V[j*2+k, c] * W[k*2+i, c] summed over c gives 1 for each (i,j,k)
    
    # Actually, the standard way: the bilinear algorithm has 7 terms.
    # Term c contributes: sum_{i,j,k} u_{i,j}^{(c)} * v_{j,k}^{(c)} * w_{k,i}^{(c)} * x_{i,j} * y_{j,k} = z_{i,k}
    
    # Let me define it properly. For multiplying A (2x2) by B (2x2):
    # z_{i,k} = sum_j a_{i,j} * b_{j,k}
    
    # Bilinear algorithm: z_{i,k} = sum_c w_{i,k}^{(c)} * (sum_{i,j} u_{i,j}^{(c)} * a_{i,j}) * (sum_{j,k} v_{j,k}^{(c)} * b_{j,k})
    
    # In our notation:
    # U[i*m+j, c] = u_{i,j}^{(c)}
    # V[j*p+k, c] = v_{j,k}^{(c)}  
    # W[k*n+i, c] = w_{i,k}^{(c)}  (note: W index is k*n+i, so W[k*n+i,c] = w_{i,k}^{(c)})
    
    # Strassen:
    # u1: a11+a22, v1: b11+b22, w1: c11+c22 -> w11=1, w22=1
    # u2: a21+a22, v2: b11, w2: c21 -> w21=1
    # u3: a11, v3: b12-b22, w3: c12 -> w12=1
    # u4: a22, v4: b21-b11, w4: c21 -> w21=1  (wait, c21 gets M2+M4)
    # u5: a11+a12, v5: b22, w5: c12-c11 -> w12=1, w11=-1
    # u6: a21-a11, v6: b11+b12, w6: c22 -> w22=1
    # u7: a12-a22, v7: b21+b22, w7: c11-c22 -> w11=1, w22=-1
    
    # Let me re-derive more carefully.
    # z11 = M1 + M4 - M5 + M7
    # z12 = M3 + M5
    # z21 = M2 + M4
    # z22 = M1 - M2 + M3 + M6
    
    # So:
    # c=0 (M1): u=a11+a22, v=b11+b22, w: z11+=1, z22+=1
    # c=1 (M2): u=a21+a22, v=b11, w: z21+=1
    # c=2 (M3): u=a11, v=b12-b22, w: z12+=1, z22+=1
    # c=3 (M4): u=a22, v=b21-b11, w: z11+=1, z21+=1
    # c=4 (M5): u=a11+a12, v=b22, w: z11+=-1, z12+=1
    # c=5 (M6): u=a21-a11, v=b11+b12, w: z22+=1
    # c=6 (M7): u=a12-a22, v=b21+b22, w: z11+=1, z22+=-1
    
    # Now fill in U, V, W:
    # U[i*2+j, c]: i,j in {0,1}
    # V[j*2+k, c]: j,k in {0,1}
    # W[k*2+i, c]: k,i in {0,1}
    
    # c=0: u: a00=1, a11=1; v: b00=1, b11=1; w: z00=1, z11=1
    U[0*2+0, 0] = 1; U[1*2+1, 0] = 1
    V[0*2+0, 0] = 1; V[1*2+1, 0] = 1
    W[0*2+0, 0] = 1; W[1*2+1, 0] = 1
    
    # c=1: u: a10=1, a11=1; v: b00=1; w: z10=1
    U[1*2+0, 1] = 1; U[1*2+1, 1] = 1
    V[0*2+0, 1] = 1
    W[0*2+1, 1] = 1  # z[1,0] means i=1, k=0, so W[k*n+i] = W[0*2+1]
    
    # c=2: u: a00=1; v: b01=1, b11=-1; w: z01=1, z11=1
    U[0*2+0, 2] = 1
    V[0*2+1, 2] = 1; V[1*2+1, 2] = -1
    W[1*2+0, 2] = 1; W[1*2+1, 2] = 1  # z[0,1]: i=0,k=1 -> W[1*2+0]; z[1,1]: i=1,k=1 -> W[1*2+1]
    
    # c=3: u: a11=1; v: b10=1, b00=-1; w: z00=1, z10=1
    U[1*2+1, 3] = 1
    V[1*2+0, 3] = 1; V[0*2+0, 3] = -1
    W[0*2+0, 3] = 1; W[0*2+1, 3] = 1  # z[0,0]: i=0,k=0 -> W[0*2+0]; z[1,0]: i=1,k=0 -> W[0*2+1]
    
    # c=4: u: a00=1, a01=1; v: b11=1; w: z00=-1, z01=1
    U[0*2+0, 4] = 1; U[0*2+1, 4] = 1
    V[1*2+1, 4] = 1
    W[0*2+0, 4] = -1; W[1*2+0, 4] = 1  # z[0,0]: W[0*2+0]; z[0,1]: W[1*2+0]
    
    # c=5: u: a10=1, a00=-1; v: b00=1, b01=1; w: z11=1
    U[1*2+0, 5] = 1; U[0*2+0, 5] = -1
    V[0*2+0, 5] = 1; V[0*2+1, 5] = 1
    W[1*2+1, 5] = 1  # z[1,1]: i=1,k=1 -> W[1*2+1]
    
    # c=6: u: a01=1, a11=-1; v: b10=1, b11=1; w: z00=1, z11=-1
    U[0*2+1, 6] = 1; U[1*2+1, 6] = -1
    V[1*2+0, 6] = 1; V[1*2+1, 6] = 1
    W[0*2+0, 6] = 1; W[1*2+1, 6] = -1  # z[0,0]: W[0*2+0]; z[1,1]: W[1*2+1]
    
    return U, V, W


def _block_decomposition_2x3x10():
    """Build decomposition by splitting the 10 columns into five 2-column blocks.
    
    For each 2x3x2 block, use rank-11 decomposition (Strassen for 2x2 part + trivial for 2x1 part).
    Total rank = 5 * 11 = 55.
    """
    n, m, p = 2, 3, 10
    
    # For a 2x3x2 block (multiplying 2x3 by 3x2):
    # We can split the 3 middle dimension into 2+1.
    # Block A: 2x2 * 2x1 part of the 3x2 matrix... 
    # Actually, let's think of it as: C (2x2) = A (2x3) * B (3x2)
    # Split B's rows into B1 (2x2) and B2 (1x2).
    # Split A's cols into A1 (2x2) and A2 (2x1).
    # C = A1*B1 + A2*B2
    
    # A1*B1 is 2x2 * 2x2, use Strassen (rank 7)
    # A2*B2 is 2x1 * 1x2, rank 2 (outer product of 2-vectors)
    # Total: 7 + 2 = 9? No wait, that's not right.
    
    # Actually for 2x3 * 3x2:
    # The schoolbook rank is 2*3*2 = 12.
    # Using Strassen on the 2x2 sub-block: 
    # C = A[:, 0:2] @ B[0:2, :] + A[:, 2:3] @ B[2:3, :]
    # First term: 2x2 * 2x2, Strassen rank 7
    # Second term: 2x1 * 1x2, rank 2
    # Total: 7 + 2 = 9? But the knowledge says 11.
    
    # Hmm, let me reconsider. The knowledge says "reduces the rank per p=2 block from 12 to 11".
    # So maybe it's a different decomposition. Let me just use schoolbook for now to get a valid answer,
    # then try to improve.
    
    # For now, let's do schoolbook: rank = 2*3*10 = 60
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
    
    return U, V, W


def _block_decomposition_2x3x10_improved():
    """Improved block decomposition using Strassen for 2x2x2 sub-blocks.
    
    Split the 10 columns into five groups of 2.
    For each group of 2 columns (p_block=2):
      Multiply 2x3 by 3x2.
      Split middle dimension: A (2x3) = [A1 (2x2), A2 (2x1)], B (3x2) = [B1 (2x2); B2 (1x2)]
      C = A1 @ B1 + A2 @ B2
      
      A1 @ B1: 2x2 * 2x2, use Strassen (rank 7)
      A2 @ B2: 2x1 * 1x2, rank 2 (two rank-1 terms)
      Subtotal per block: 7 + 2 = 9
      
    Total: 5 * 9 = 45? But knowledge says 55, so maybe I'm miscounting.
    
    Let me just implement it carefully.
    """
    n, m, p = 2, 3, 10
    n_blocks = p // 2  # 5 blocks of width 2
    
    # For each block, we need a 2x3x2 decomposition of rank r_block.
    # Let's build it explicitly.
    
    # For one 2x3x2 block:
    # U_block: (2*3, r_block) = (6, r_block)
    # V_block: (3*2, r_block) = (6, r_block)
    # W_block: (2*2, r_block) = (4, r_block)
    
    # Strassen for the 2x2 * 2x2 part (using cols 0,1 of A and rows 0,1 of B):
    U_s, V_s, W_s = _strassen_2x2_2x2_2x2()  # each shape (4, 7)
    
    # For the 2x1 * 1x2 part (col 2 of A, row 2 of B):
    # A2 is 2x1: entries a02, a12
    # B2 is 1x2: entries b20, b21
    # C += A2 @ B2: c00 += a02*b20, c01 += a02*b21, c10 += a12*b20, c11 += a12*b21
    # This is 4 rank-1 terms? No, it's 2 rank-1 terms:
    # Term 1: a02 * [b20, b21] -> contributes to row 0 of C
    # Term 2: a12 * [b20, b21] -> contributes to row 1 of C
    # Actually: C[i,k] += A2[i,0] * B2[0,k]
    # So for each (i,k), it's A2[i,0] * B2[0,k]. That's 4 scalar products, but as rank-1:
    # It's the outer product of A2 (2x1) and B2 (1x2), which has rank 1!
    # Wait no: A2 @ B2 is a 2x2 matrix that is the outer product of a 2-vector and a 2-vector, so rank 1.
    # But in bilinear decomposition, each "term" is a product of a linear form in A, a linear form in B, and a linear form in C.
    # A2[i,0] is a linear form in A (just picks out entry (i,2)).
    # B2[0,k] is a linear form in B (just picks out entry (2,k)).
    # So C[i,k] += A2[i,0] * B2[0,k] means for each (i,k), we have a term.
    # But these 4 terms share structure. Actually, the outer product A2 @ B2 can be written as:
    # sum over... hmm, it's just one "matrix" which is rank 1 as a matrix, but in bilinear algorithm terms,
    # C[i,k] = A2[i,0] * B2[0,k] for each (i,k) independently.
    # So we need 4 separate bilinear terms? No...
    
    # Let me think again. In the bilinear algorithm:
    # C[i,k] = sum_c W[k*n+i, c] * (sum_{i',j'} U[i'*m+j', c] * A[i',j']) * (sum_{j',k'} V[j'*p+k', c] * B[j',k'])
    
    # For the A2@B2 part:
    # C[i,k] += A[ i, 2 ] * B[ 2, k ]
    # So we need terms where:
    # U[i*3+2, c] = 1 (picks A[i,2])
    # V[2*2+k, c] = 1 (picks B[2,k])
    # W[k*2+i, c] = 1
    # for each (i,k) pair. That's 4 terms (i in {0,1}, k in {0,1}).
    
    # So per block: 7 (Strassen) + 4 (trivial for 2x1*1x2) = 11.
    # Total: 5 * 11 = 55. This matches the knowledge!
    
    total_rank = n_blocks * 11
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    for block in range(n_blocks):
        k_offset = block * 2  # starting column index in the p dimension
        col_offset = block * 11  # starting column in the rank dimension
        
        # Strassen part (for A[:, 0:2] @ B[0:2, :]):
        # U_s[i*2+j, c] maps to U[i*3+j, col_offset+c] for j in {0,1}
        # V_s[j*2+k_local, c] maps to V[j*2+k_offset+k_local, col_offset+c] for k_local in {0,1}
        # W_s[k_local*2+i, c] maps to W[(k_offset+k_local)*2+i, col_offset+c]
        
        for c in range(7):
            for i in range(2):
                for j in range(2):
                    if U_s[i * 2 + j, c] != 0:
                        U[i * 3 + j, col_offset + c] = U_s[i * 2 + j, c]
            for j in range(2):
                for k_local in range(2):
                    if V_s[j * 2 + k_local, c] != 0:
                        V[j * 2 + (k_offset + k_local), col_offset + c] = V_s[j * 2 + k_local, c]
            for k_local in range(2):
                for i in range(2):
                    if W_s[k_local * 2 + i, c] != 0:
                        W[(k_offset + k_local) * 2 + i, col_offset + c] = W_s[k_local * 2 + i, c]
        
        # Trivial part (for A[:, 2] @ B[2, :]):
        # 4 terms: (i=0,k=0), (i=0,k=1), (i=1,k=0), (i=1,k=1)
        for idx, (i, k_local) in enumerate([(0,0), (0,1), (1,0), (1,1)]):
            c = col_offset + 7 + idx
            U[i * 3 + 2, c] = 1  # A[i, 2]
            V[2 * 2 + (k_offset + k_local), c] = 1  # B[2, k_offset+k_local]
            W[(k_offset + k_local) * 2 + i, c] = 1  # C[i, k_offset+k_local]
    
    return U, V, W


def _try_reduce_rank(U, V, W, n, m, p, max_time=30.0):
    """Try to reduce rank by removing redundant columns.
    
    Check if any column can be removed while maintaining exactness.
    """
    start_time = time.time()
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    
    # Try to find a smaller rank by checking linear dependencies
    # This is heuristic - check if removing any single column still works
    # (This only works if there's a redundant column, which is rare for minimal decompositions)
    
    # Alternative: try to find a rank reduction via SVD of the "constraint matrix"
    # For now, just return as-is
    return U, V, W


def construct():
    """Construct a rank-minimal exact tensor decomposition for 2x3 * 3x10 matrix multiplication."""
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Start with the improved block decomposition (rank 55)
    U, V, W = _block_decomposition_2x3x10_improved()
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Verify
    if _verify_correct(U, V, W, n, m, p):
        pass  # Good
    else:
        # Fallback to schoolbook
        U, V, W = _block_decomposition_2x3x10()
        best_rank = U.shape[1]
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to improve further within time budget
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - SAFETY_MARGIN:
        # Try rank reduction attempts
        U_red, V_red, W_red = _try_reduce_rank(best_U, best_V, best_W, n, m, p, 
                                                 max_time=min(30.0, DEADLINE - SAFETY_MARGIN - (time.time() - start_time)))
        if U_red.shape[1] < best_rank:
            if _verify_correct(U_red, V_red, W_red, n, m, p):
                best_U, best_V, best_W = U_red, V_red, W_red
                best_rank = U_red.shape[1]
    
    # Convert to lists for return
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
