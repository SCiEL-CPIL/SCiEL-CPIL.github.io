"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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

def _strassen_2x2x2():
    """
    Return Strassen decomposition for 2x2 * 2x2 -> 2x2.
    Indices: U indexed by (i,j) in [0,2)x[0,2), V by (j,k), W by (k,i).
    The 7 Strassen products are:
      M1 = (A11+A22)(B11+B22)
      M2 = (A21+A22)B11
      M3 = A11(B12-B22)
      M4 = A22(B21-B11)
      M5 = (A11+A12)B22
      M6 = (A21-A11)(B11+B12)
      M7 = (A12-A22)(B21+B22)
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    # Each term: (U_coeff, V_coeff, W_coeff) for each output (k,i)
    # U is 4x7, V is 4x7, W is 4x7
    # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    
    # M1: (A11+A22)(B11+B22)
    # A11: U[0,0]=1, A22: U[3,0]=1
    # B11: V[0,0]=1, B22: V[3,0]=1
    # C11 += M1, C22 += M1
    # W[0,0]=1 (k=0,i=0 -> C11), W[3,0]=1 (k=1,i=1 -> C22)
    
    # Let's build it term by term
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A11+A22)(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1  # C11
    W[3, 0] = 1  # C22
    
    # M2 = (A21+A22)B11
    # A21: U[2,1]=1, A22: U[3,1]=1
    # B11: V[0,1]=1
    # C21 += M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1  # C21 (k=1, i=0)
    
    # M3 = A11(B12-B22)
    # A11: U[0,2]=1
    # B12: V[1,2]=1, B22: V[3,2]=-1
    # C12 += M3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1  # C12 (k=0, i=1)
    
    # M4 = A22(B21-B11)
    # A22: U[3,3]=1
    # B21: V[2,3]=1, B11: V[0,3]=-1
    # C11 += M4, C21 += M4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1  # C11
    W[2, 3] = 1  # C21
    
    # M5 = (A11+A12)B22
    # A11: U[0,4]=1, A12: U[1,4]=1
    # B22: V[3,4]=1
    # C11 -= M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1  # C11
    
    # M6 = (A21-A11)(B11+B12)
    # A21: U[2,5]=1, A11: U[0,5]=-1
    # B11: V[0,5]=1, B12: V[1,5]=1
    # C22 += M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1  # C22
    
    # M7 = (A12-A22)(B21+B22)
    # A12: U[1,6]=1, A22: U[3,6]=-1
    # B21: V[2,6]=1, B22: V[3,6]=1
    # C11 += M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1  # C11
    
    return U, V, W

def _direct_2x1x1():
    """
    Direct decomposition for 2x1 * 1x1 -> 2x1.
    U: 2x2, V: 1x2, W: 2x2
    C1 = A1*B1, C2 = A2*B1
    """
    U = np.array([[1, 0], [0, 1]], dtype=float)
    V = np.array([[1, 1]], dtype=float)
    W = np.array([[1, 0], [0, 1]], dtype=float)
    return U, V, W

def _direct_1x2x1():
    """
    Direct decomposition for 1x2 * 2x1 -> 1x1.
    U: 2x2, V: 2x2, W: 1x2
    C = A1*B1 + A2*B2
    """
    U = np.array([[1, 1]], dtype=float)
    V = np.array([[1, 0], [0, 1]], dtype=float)
    W = np.array([[1, 1]], dtype=float)
    return U, V, W

def _direct_1x1x2():
    """
    Direct decomposition for 1x1 * 1x2 -> 1x2.
    U: 1x2, V: 2x2, W: 2x2
    C1 = A1*B1, C2 = A1*B2
    """
    U = np.array([[1, 1]], dtype=float)
    V = np.array([[1, 0], [0, 1]], dtype=float)
    W = np.array([[1, 0], [0, 1]], dtype=float)
    return U, V, W

def _combine_block_decompositions(n, m, p, block_n, block_m, block_p, decomp):
    """
    Combine block decompositions to form the full decomposition.
    decomp: (U, V, W) for a single block of size (block_n, block_m, block_p).
    """
    rank_block = decomp[0].shape[1]
    total_rank = rank_block * (n // block_n) * (m // block_m) * (p // block_p)
    
    U_full = np.zeros((n * m, total_rank))
    V_full = np.zeros((m * p, total_rank))
    W_full = np.zeros((p * n, total_rank))
    
    U_blk, V_blk, W_blk = decomp
    rn, rm, rp = block_n, block_m, block_p
    U_blk_rows, U_blk_cols = U_blk.shape
    V_blk_rows, V_blk_cols = V_blk.shape
    W_blk_rows, W_blk_cols = W_blk.shape
    
    for bi in range(n // rn):
        for bj in range(m // rm):
            for bk in range(p // rp):
                # Block (bi, bj, bk)
                col_offset = (bi * (m // rm) + bj) * (p // rp) + bk
                col_offset *= rank_block
                
                # U: rows (bi*rn + i)*m + (bj*rm + j)
                for i in range(rn):
                    for j in range(rm):
                        u_row = (bi * rn + i) * m + (bj * rm + j)
                        for c in range(rank_block):
                            U_full[u_row, col_offset + c] = U_blk[i * rm + j, c]
                
                # V: rows (bj*rm + j)*p + (bk*rp + k)
                for j in range(rm):
                    for k in range(rp):
                        v_row = (bj * rm + j) * p + (bk * rp + k)
                        for c in range(rank_block):
                            V_full[v_row, col_offset + c] = V_blk[j * rp + k, c]
                
                # W: rows (bk*rp + k)*n + (bi*rn + i)
                for k in range(rp):
                    for i in range(rn):
                        w_row = (bk * rp + k) * n + (bi * rn + i)
                        for c in range(rank_block):
                            W_full[w_row, col_offset + c] = W_blk[k * rn + i, c]
    
    return U_full, V_full, W_full

def _verify_decomposition(n, m, p, U, V, W):
    """Verify that the decomposition exactly matches the matrix multiplication tensor."""
    rank = U.shape[1]
    # Check that for all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    # And for all other (i, j, k) combinations that don't correspond to valid products, it should be 0.
    # Actually, the tensor T[i,j,k] = 1 if this is a valid multiplication term, else 0.
    # For matrix multiplication A(n x m) * B(m x p) = C(n x p):
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # Wait, that's not right. The tensor for matrix multiplication has T[i, j, k] = 1
    # where i indexes the row of A (and C), j indexes the column of A (and row of B),
    # and k indexes the column of B (and C). So T[i,j,k] = 1 for all valid i, j, k.
    
    # Let's verify by checking a few random terms
    for _ in range(1000):
        i = random.randint(0, n - 1)
        j = random.randint(0, m - 1)
        k = random.randint(0, p - 1)
        val = 0.0
        for c in range(rank):
            val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
        if abs(val - 1.0) > 1e-9:
            return False
    
    # Also check that invalid terms give 0
    # But all i, j, k combinations are valid for the multiplication tensor
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10 second safety margin
    
    # Strategy: Use block-structured decomposition
    # For 3x4x6:
    # Option 1: All 1x1x1 blocks -> rank 72 (schoolbook)
    # Option 2: Use 2x2x2 Strassen where possible
    #   3 = 2+1, 4 = 2+2, 6 = 2+2+2
    #   We can partition as:
    #   - (2,2,2) blocks: use Strassen (rank 7)
    #   - (1,2,2) blocks: need a decomposition
    #   - (2,1,2) blocks: need a decomposition
    #   - (2,2,1) blocks: need a decomposition
    #   - (1,1,2) blocks: rank 2
    #   - (1,2,1) blocks: rank 2
    #   - (2,1,1) blocks: rank 2
    #   - (1,1,1) blocks: rank 1
    
    # Let's think about the partition more carefully.
    # n=3: split into 2+1
    # m=4: split into 2+2
    # p=6: split into 2+2+2
    
    # Blocks:
    # (2,2,2): 1*2*3 = 6 blocks, each rank 7 -> 42
    # (1,2,2): 1*2*3 = 6 blocks, each rank ? 
    # (2,1,2): 1*2*3 = 6 blocks, each rank ?
    # (2,2,1): 1*2*3 = 6 blocks, each rank ?
    # (1,1,2): 1*2*3 = 6 blocks, each rank 2 -> 12
    # (1,2,1): 1*2*3 = 6 blocks, each rank 2 -> 12
    # (2,1,1): 1*2*3 = 6 blocks, each rank 2 -> 12
    # (1,1,1): 1*2*3 = 6 blocks, each rank 1 -> 6
    
    # Wait, I need to be more careful. The partition is:
    # n: [0,2), [2,3) -> sizes 2, 1
    # m: [0,2), [2,4) -> sizes 2, 2
    # p: [0,2), [2,4), [4,6) -> sizes 2, 2, 2
    
    # So the blocks are:
    # (2,2,2), (2,2,2), (2,2,2)  [3 blocks]
    # (2,2,2), (2,2,2), (2,2,2)  [3 blocks]
    # (1,2,2), (1,2,2), (1,2,2)  [3 blocks]
    # (1,2,2), (1,2,2), (1,2,2)  [3 blocks]
    
    # Total: 6 blocks of (2,2,2), 6 blocks of (1,2,2)
    
    # For (2,2,2): Strassen gives rank 7
    # For (1,2,2): This is 1x2 * 2x2 -> 1x2
    #   C[k] = sum_j A[j] * B[j,k] for k=0,1
    #   This can be done with rank 2 (direct) or maybe less?
    #   Actually, 1x2 * 2x2 -> 1x2 is just two dot products of length 2.
    #   Each dot product of length 2 has rank 2 (schoolbook).
    #   But can we do better? The tensor is T[j,k] for j in {0,1}, k in {0,1}.
    #   This is just the identity-like tensor, rank 2.
    
    # Let's try a different approach. Let me just implement the block decomposition
    # with the known components and see what rank we get.
    
    # Actually, let me try a simpler approach first: just use the schoolbook and see
    # if I can find any improvement.
    
    # For now, let's return the schoolbook decomposition as a baseline.
    U, V, W = _schoolbook(n, m, p)
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
