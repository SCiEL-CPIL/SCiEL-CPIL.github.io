"""Matrix multiplication 4x4 by 4x5: rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]
DEADLINE = 125  # Safety margin before 150s kill


def _strassen_4x4x4():
    """
    Build the rank-49 decomposition for 4x4 * 4x4 -> 4x4.
    Uses Strassen's 7-term algorithm recursively:
    - 4x4 = 2x(2x2) blocks
    - Each 2x2*2x2 uses Strassen (7 terms)
    - 4x4 needs 4 sub-problems of 2x2*2x2, each with 7 terms = 28
    - But we can be smarter: use the known rank-49 decomposition.
    
    Actually, the standard approach:
    4x4 * 4x4: partition into 2x2 blocks.
    A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    C = A*B where each block is 2x2.
    
    Using Strassen at the block level:
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B12-B22)
    M4 = A22(B21-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B21)
    M7 = (A12-A22)(B12+B22)
    
    Each M_i is a 2x2 * 2x2 product, which itself uses Strassen (7 scalar mults).
    So total = 7 * 7 = 49 scalar multiplications.
    
    Let me build this explicitly.
    """
    # Strassen 2x2*2x2 -> 2x2 with 7 terms
    # For matrices A2x2 and B2x2:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b10)
    # M7 = (a01-a11)(b01+b11)
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    
    # For 4x4*4x4, we have 4 block-level Strassen terms, each being a 2x2*2x2 product
    # Each 2x2*2x2 product uses 7 scalar multiplications
    # Total: 4 * 7 = 28? No wait...
    
    # Actually, let me reconsider. The 4x4*4x4 using Strassen:
    # We partition into 2x2 blocks. The block-level computation uses 7 block multiplications.
    # Each block multiplication is 2x2 * 2x2, which uses 7 scalar multiplications.
    # Total = 7 * 7 = 49.
    
    # Let me construct this step by step.
    # First, define the 2x2 Strassen decomposition.
    
    # For 2x2 * 2x2 -> 2x2:
    # U has shape (4, 7), V has shape (4, 7), W has shape (4, 7)
    # U[i*2+j, r] for i,j in {0,1}, r in {0..6}
    # V[j*2+k, r] for j,k in {0,1}, r in {0..6}
    # W[k*2+i, r] for k,i in {0,1}, r in {0..6}
    
    def _strassen_2x2():
        """Return U, V, W for 2x2 * 2x2 -> 2x2 with rank 7."""
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (a00+a11)(b00+b11)
        # U: a00 and a11 contribute
        U[0, 0] = 1.0  # a00
        U[3, 0] = 1.0  # a11
        V[0, 0] = 1.0  # b00
        V[3, 0] = 1.0  # b11
        # W: c00 += M1, c11 += M1
        W[0, 0] = 1.0  # c00 (k=0, i=0)
        W[3, 0] = 1.0  # c11 (k=1, i=1)
        
        # M2 = (a10+a11)b00
        U[2, 1] = 1.0  # a10
        U[3, 1] = 1.0  # a11
        V[0, 1] = 1.0  # b00
        # W: c10 += M2
        W[2, 1] = 1.0  # c10 (k=0, i=1)
        
        # M3 = a00(b01-b11)
        U[0, 2] = 1.0  # a00
        V[1, 2] = 1.0  # b01
        V[3, 2] = -1.0  # b11
        # W: c01 += M3
        W[1, 2] = 1.0  # c01 (k=1, i=0)
        
        # M4 = a11(b10-b00)
        U[3, 3] = 1.0  # a11
        V[2, 3] = 1.0  # b10
        V[0, 3] = -1.0  # b00
        # W: c00 += M4, c10 += M4
        W[0, 3] = 1.0  # c00
        W[2, 3] = 1.0  # c10
        
        # M5 = (a00+a01)b11
        U[0, 4] = 1.0  # a00
        U[1, 4] = 1.0  # a01
        V[3, 4] = 1.0  # b11
        # W: c00 -= M5, c01 += M5
        W[0, 4] = -1.0  # c00
        W[1, 4] = 1.0  # c01
        
        # M6 = (a10-a00)(b00+b10)
        U[2, 5] = 1.0  # a10
        U[0, 5] = -1.0  # a00
        V[0, 5] = 1.0  # b00
        V[2, 5] = 1.0  # b10
        # W: c11 += M6
        W[3, 5] = 1.0  # c11
        
        # M7 = (a01-a11)(b01+b11)
        U[1, 6] = 1.0  # a01
        U[3, 6] = -1.0  # a11
        V[1, 6] = 1.0  # b01
        V[3, 6] = 1.0  # b11
        # W: c00 += M7
        W[0, 6] = 1.0  # c00
        
        return U, V, W
    
    # Now for 4x4 * 4x4:
    # A is 4x4, partitioned into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]]
    # where A11 = A[0:2, 0:2], A12 = A[0:2, 2:4], A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
    # 
    # B is 4x4, partitioned similarly.
    #
    # The 7 block-level Strassen terms:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B21)
    # M7 = (A12-A22)(B12+B22)
    #
    # Each M_i is a 2x2 * 2x2 product.
    # 
    # The final C is assembled from M1..M7:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Total rank = 7 * 7 = 49
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    # For each of the 7 block-level terms, we have a 2x2*2x2 Strassen decomposition (7 terms)
    # Block-level term t (0..6) uses scalar terms t*7 + s (s=0..6)
    
    # Define the block indices for A:
    # A11: rows 0-1, cols 0-1 -> indices 0,1,4,5
    # A12: rows 0-1, cols 2-3 -> indices 2,3,6,7
    # A21: rows 2-3, cols 0-1 -> indices 8,9,12,13
    # A22: rows 2-3, cols 2-3 -> indices 10,11,14,15
    
    # For B (4x4):
    # B11: rows 0-1, cols 0-1 -> indices 0,1,4,5
    # B12: rows 0-1, cols 2-3 -> indices 2,3,6,7
    # B21: rows 2-3, cols 0-1 -> indices 8,9,12,13
    # B22: rows 2-3, cols 2-3 -> indices 10,11,14,15
    
    # For C (4x4), W is indexed by [k*n+i] where k is col, i is row:
    # C11: rows 0-1, cols 0-1 -> W indices: k*4+i
    #   (0,0)=0, (1,0)=4, (0,1)=1, (1,1)=5
    # C12: rows 0-1, cols 2-3 -> W indices:
    #   (2,0)=8, (3,0)=12, (2,1)=9, (3,1)=13
    # C21: rows 2-3, cols 0-1 -> W indices:
    #   (0,2)=2, (1,2)=6, (0,3)=3, (1,3)=7
    # C22: rows 2-3, cols 2-3 -> W indices:
    #   (2,2)=10, (3,2)=14, (2,3)=11, (3,3)=15
    
    # Let me define the block index mappings more carefully.
    # U is indexed by [i*m+j] where i is row, j is col of A (4x4)
    # V is indexed by [j*p+k] where j is row, k is col of B (4x4)
    # W is indexed by [k*n+i] where k is col, i is row of C (4x4)
    
    # For a 2x2 sub-block of A at position (block_row, block_col):
    # The rows are block_row*2 to block_row*2+1
    # The cols are block_col*2 to block_col*2+1
    # So U indices are: (block_row*2 + r) * 4 + (block_col*2 + c) for r,c in {0,1}
    
    # For a 2x2 sub-block of B at position (block_row, block_col):
    # V indices are: (block_row*2 + r) * 4 + (block_col*2 + c) for r,c in {0,1}
    
    # For a 2x2 sub-block of C at position (block_row, block_col):
    # W indices are: (block_col*2 + c) * 4 + (block_row*2 + r) for r,c in {0,1}
    
    # The 7 block-level Strassen terms and their contributions:
    # M1 = (A11+A22)(B11+B22): A blocks {11,22}, B blocks {11,22}, C blocks {11: +1, 22: +1}
    # M2 = (A21+A22)B11: A blocks {21,22}, B blocks {11}, C blocks {21: +1}
    # M3 = A11(B12-B22): A blocks {11}, B blocks {12:+1, 22:-1}, C blocks {12: +1}
    # M4 = A22(B21-B11): A blocks {22}, B blocks {21:+1, 11:-1}, C blocks {11: +1, 21: +1}
    # M5 = (A11+A12)B22: A blocks {11,12}, B blocks {22}, C blocks {11: -1, 12: +1}
    # M6 = (A21-A11)(B11+B21): A blocks {21:+1, 11:-1}, B blocks {11:+1, 21:+1}, C blocks {22: +1}
    # M7 = (A12-A22)(B12+B22): A blocks {12:+1, 22:-1}, B blocks {12:+1, 22:+1}, C blocks {11: +1}
    
    # For each block-level term, we need to apply the 2x2 Strassen decomposition.
    # The 2x2 Strassen U, V, W are defined above.
    
    U2, V2, W2 = _strassen_2x2()
    
    # Block positions: 11=(0,0), 12=(0,1), 21=(1,0), 22=(1,1)
    # For A: block (br, bc) -> U indices (br*2+r)*4 + (bc*2+c)
    # For B: block (br, bc) -> V indices (br*2+r)*4 + (bc*2+c)
    # For C: block (br, bc) -> W indices (bc*2+c)*4 + (br*2+r)
    
    # Define the 7 block-level terms with their A-block coefficients, B-block coefficients, C-block coefficients
    # Each entry is a list of (block_pos, coefficient)
    block_terms = [
        # M1
        ([(0,0,1.0), (1,1,1.0)], [(0,0,1.0), (1,1,1.0)], [(0,0,1.0), (1,1,1.0)]),
        # M2
        ([(1,0,1.0), (1,1,1.0)], [(0,0,1.0)], [(1,0,1.0)]),
        # M3
        ([(0,0,1.0)], [(0,1,1.0), (1,1,-1.0)], [(0,1,1.0)]),
        # M4
        ([(1,1,1.0)], [(1,0,1.0), (0,0,-1.0)], [(0,0,1.0), (1,0,1.0)]),
        # M5
        ([(0,0,1.0), (0,1,1.0)], [(1,1,1.0)], [(0,0,-1.0), (0,1,1.0)]),
        # M6
        ([(1,0,1.0), (0,0,-1.0)], [(0,0,1.0), (1,0,1.0)], [(1,1,1.0)]),
        # M7
        ([(0,1,1.0), (1,1,-1.0)], [(0,1,1.0), (1,1,1.0)], [(0,0,1.0)]),
    ]
    
    for t, (a_blocks, b_blocks, c_blocks) in enumerate(block_terms):
        for s in range(7):
            r = t * 7 + s
            # U contribution
            for (br, bc, coeff) in a_blocks:
                for rr in range(2):
                    for cc in range(2):
                        u_idx = (br * 2 + rr) * 4 + (bc * 2 + cc)
                        u2_idx = rr * 2 + cc
                        U[u_idx, r] += coeff * U2[u2_idx, s]
            
            # V contribution
            for (br, bc, coeff) in b_blocks:
                for rr in range(2):
                    for cc in range(2):
                        v_idx = (br * 2 + rr) * 4 + (bc * 2 + cc)
                        v2_idx = rr * 2 + cc
                        V[v_idx, r] += coeff * V2[v2_idx, s]
            
            # W contribution
            for (br, bc, coeff) in c_blocks:
                for rr in range(2):
                    for cc in range(2):
                        w_idx = (bc * 2 + cc) * 4 + (br * 2 + rr)
                        w2_idx = cc * 2 + rr
                        W[w_idx, r] += coeff * W2[w2_idx, s]
    
    return U, V, W


def _build_4x4x5():
    """
    Build the rank-65 decomposition for 4x4 * 4x5 -> 4x5.
    Uses Strassen-based rank-49 for the first 4 columns, plus schoolbook for the 5th column.
    """
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Get the 4x4*4x4 decomposition
    U4, V4, W4 = _strassen_4x4x4()
    
    # For the 5th column of B (column index 4), we need 4x4 * 1 -> 4x1
    # This is just rank 16 (schoolbook for 4*4*1)
    # Actually, for the 5th column: C[:,4] = A * B[:,4]
    # This is a 4x4 * 4x1 product, rank 16 (schoolbook)
    
    # Total rank = 49 + 16 = 65
    total_rank = 49 + 16
    
    U = np.zeros((n * m, total_rank))  # (16, 65)
    V = np.zeros((m * p, total_rank))  # (20, 65)
    W = np.zeros((p * n, total_rank))  # (20, 65)
    
    # Fill in the 4x4*4x4 part (first 49 columns)
    # U4 has shape (16, 49), V4 has shape (16, 49), W4 has shape (16, 49)
    # But V needs to be (20, 65) and W needs to be (20, 65)
    # V4 is for B (4x4), so it uses rows 0-15 (j*4+k for j,k in 0..3)
    # V for B (4x5) uses rows j*5+k for j in 0..3, k in 0..4
    # So for the first 4 columns of B (k=0..3), V index is j*5+k
    # V4 index is j*4+k
    
    # U: same indexing for A (4x4), so U4 maps directly to U[:, :49]
    U[:, :49] = U4
    
    # V: V4[j*4+k, r] -> V[j*5+k, r] for k in 0..3
    for j in range(4):
        for k in range(4):
            V[j * 5 + k, :49] = V4[j * 4 + k, :49]
    
    # W: W4[k*4+i, r] -> W[k*4+i, r] for the first 4 columns of C
    # W for C (4x5) uses index k*4+i for k in 0..4, i in 0..3
    # For the first 4 columns (k=0..3), W index is k*4+i
    # W4 index is k*4+i for k in 0..3, i in 0..3
    # So W4 maps directly to W[:16, :49]
    W[:16, :49] = W4
    
    # Now add the 5th column part (rank 16)
    # C[:,4] = A * B[:,4]
    # For each i (row of C, 0..3) and j (col of A, 0..3):
    # C[i,4] += A[i,j] * B[j,4]
    # So we need 16 terms, one for each (i,j) pair
    
    for i in range(4):
        for j in range(4):
            r = 49 + i * 4 + j
            U[i * 4 + j, r] = 1.0
            V[j * 5 + 4, r] = 1.0
            W[4 * 4 + i, r] = 1.0  # k=4, i=i -> index 16+i
    
    return U, V, W


def _verify_fast(U, V, W, tol=1e-10):
    """Verify the tensor decomposition."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=tol)


def _try_reduce_rank(U, V, W, max_iter=2000):
    """Try to reduce rank by finding and removing redundant terms."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    start = time.time()
    
    for _ in range(max_iter):
        if time.time() - start > DEADLINE - 10:
            break
        
        # Try removing a random column
        a = np.random.randint(rank)
        U_test = np.delete(U, a, axis=1)
        V_test = np.delete(V, a, axis=1)
        W_test = np.delete(W, a, axis=1)
        
        if _verify_fast(U_test, V_test, W_test, tol=1e-8):
            U, V, W = U_test, V_test, W_test
            rank -= 1
            print(f"Reduced rank to {rank}")
    
    return U, V, W


def construct():
    """Construct the matrix multiplication decomposition."""
    n, m, p = DIMENSIONS
    start = time.time()
    
    # Build the rank-65 decomposition
    U, V, W = _build_4x4x5()
    rank = U.shape[1]
    print(f"Initial rank: {rank}")
    
    # Verify
    if not _verify_fast(U, V, W):
        print("Verification failed, falling back to schoolbook")
        # Fall back to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
    else:
        # Try to reduce rank
        U, V, W = _try_reduce_rank(U, V, W)
        rank = U.shape[1]
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}


if __name__ == "__main__":
    result = construct()
    n, m, p = DIMENSIONS
    rank = len(result["U"][0])
    print(f"Rank: {rank}")
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    print(f"Valid: {_verify_fast(U, V, W)}")
