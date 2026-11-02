"""Strassen-based rank decomposition for 3x4 @ 4x8 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition (rank 7)."""
    U = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ], dtype=float)
    V = np.array([
        [1, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0],
        [1, -1, 0, 0, 0, 0, 0],
        [0, 0, 1, -1, 0, 0, 0],
    ], dtype=float)
    W = np.array([
        [0, 0, 1, 1, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, -1, 0, 0, 1],
    ], dtype=float)
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    r = U.shape[1]
    # Check: sum_s U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s] should be 1 if (i,j,k) is valid, 0 otherwise
    # Actually, the tensor T[i,j,k] = sum_s U[i*m+j,s] * V[j*p+k,s] * W[k*n+i,s]
    # For matrix multiplication, T[i,j,k] = 1 for all i,j,k (since C[i,k] = sum_j A[i,j]*B[j,k])
    # Wait, no. The tensor for matrix multiplication M_{n,m,p} has T[i,j,k] = 1 for all i in [n], j in [m], k in [p].
    # The decomposition is: T = sum_s u_s ⊗ v_s ⊗ w_s
    # where u_s[i*m+j] = U[i*m+j, s], v_s[j*p+k] = V[j*p+k, s], w_s[k*n+i] = W[k*n+i, s]
    # So we need: for all i,j,k: sum_s U[i*m+j,s] * V[j*p+k,s] * W[k*n+i,s] = 1
    
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    # Build the tensor
    tensor = np.zeros((n, m, p))
    for s in range(r):
        u = U[:, s].reshape(n, m)
        v = V[:, s].reshape(m, p)
        w = W[:, s].reshape(p, n)
        # u ⊗ v ⊗ w contributes u[i,j] * v[j,k] * w[k,i] to T[i,j,k]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    tensor[i,j,k] += u[i,j] * v[j,k] * w[k,i]
    
    # Check if tensor is all ones
    if not np.allclose(tensor, 1.0, atol=1e-10):
        return False
    return True

def _construct_strassen_decomposition(n, m, p):
    """
    Construct a Strassen-based decomposition for M_{3,4,8}.
    
    Strategy:
    - Split the 3x4 matrix A into:
      - A_top: 1x4 (row 0)
      - A_bottom: 2x4 (rows 1,2)
    - Split the 4x8 matrix B into 4 blocks of 4x2: B_0, B_1, B_2, B_3
    - The product C = A*B is 3x8
    
    For the 1x4 @ 4x8 part:
      This is just a dot product for each output, rank = 4*8 = 32
    
    For the 2x4 @ 4x8 part:
      Split B into two 4x4 blocks: B_left (cols 0-3), B_right (cols 4-7)
      A_bottom * B_left: 2x4 @ 4x4, use Strassen-like approach
      A_bottom * B_right: 2x4 @ 4x4, use Strassen-like approach
      
      For 2x4 @ 4x4:
        Split A_bottom into two 1x4 rows
        Split B into 4x2 blocks (2 blocks)
        Actually, let's use a different approach.
        
    Let me try a cleaner approach:
    - Use the fact that M_{2,4,8} can be decomposed using Strassen on 2x2 subproblems.
    
    Actually, let me just implement a valid decomposition and verify it.
    I'll use a block approach:
    
    For M_{3,4,8}:
    - Row 0 of A (1x4) times B (4x8): rank 32
    - Rows 1-2 of A (2x4) times B (4x8): use Strassen decomposition
    
    For M_{2,4,8}, I'll split B into 4x2 blocks (4 blocks):
    C_bottom = A_bottom * B
    For each 4x2 block B_j of B:
      A_bottom * B_j is 2x2, use Strassen (rank 7)
    Total for bottom: 4 * 7 = 28
    
    Total rank: 32 + 28 = 60
    """
    r_total = 60
    U = np.zeros((n*m, r_total))
    V = np.zeros((m*p, r_total))
    W = np.zeros((p*n, r_total))
    
    col = 0
    
    # Part 1: Row 0 of A (1x4) times B (4x8)
    # This is just schoolbook: for each j in [4], k in [8], one product
    for j in range(m):
        for k in range(p):
            U[0*m + j, col] = 1  # A[0, j]
            V[j*p + k, col] = 1  # B[j, k]
            W[k*n + 0, col] = 1  # C[0, k]
            col += 1
    
    # Part 2: Rows 1-2 of A (2x4) times B (4x8)
    # Split B into 4 blocks of 4x2
    U_str, V_str, W_str = _strassen_2x2x2()
    
    for block in range(4):
        # B_block is columns 2*block to 2*block+1 of B
        # A_bottom is rows 1-2 of A, all 4 columns
        # Product is 2x2: C[1:3, 2*block:2*block+2]
        
        # Strassen for 2x2 @ 2x2:
        # A_sub = A[1:3, :]  (2x4)
        # B_sub = B[:, 2*block:2*block+2]  (4x2)
        # We need to express this as a 2x2 @ 2x2 problem
        
        # Actually, A_bottom is 2x4 and B_block is 4x2.
        # We can split A_bottom into 2x2 blocks: A_bottom[:, 0:2] and A_bottom[:, 2:4]
        # And B_block is 4x2, which we can view as two 2x2 blocks stacked.
        
        # Let me use a different approach. For 2x4 @ 4x2:
        # C[i, k] = sum_j A[1+i, j] * B[j, 2*block+k]
        # This is a 2x2 matrix.
        
        # We can use the fact that 2x4 @ 4x2 can be decomposed using two 2x2 @ 2x2 Strassen decompositions.
        # A_bottom = [A1 | A2] where A1, A2 are 2x2
        # B_block = [B1; B2] where B1, B2 are 2x2
        # C = A1*B1 + A2*B2
        
        # Each A1*B1 and A2*B2 is 2x2 @ 2x2, rank 7
        # Total for this block: 14
        
        # But wait, that gives 4 blocks * 14 = 56 for the bottom part, total 32 + 56 = 88.
        # That's still better than 96!
        
        for sub_block in range(2):
            # A_sub = A_bottom[:, 2*sub_block:2*sub_block+2]  (2x2)
            # B_sub = B_block[2*sub_block:2*sub_block+2, :]  (2x2)
            
            for s in range(7):
                # U_str entries for A_sub
                for ai in range(2):
                    for aj in range(2):
                        coeff = U_str[ai*2 + aj, s]
                        if coeff != 0:
                            U[(1+ai)*m + (2*sub_block+aj), col] += coeff
                
                # V_str entries for B_sub
                for bi in range(2):
                    for bj in range(2):
                        coeff = V_str[bi*2 + bj, s]
                        if coeff != 0:
                            V[(2*sub_block+bi)*p + (2*block+bj), col] += coeff
                
                # W_str entries for C_sub
                for ci in range(2):
                    for cj in range(2):
                        coeff = W_str[ci*2 + cj, s]
                        if coeff != 0:
                            W[(2*block+cj)*n + (1+ci), col] += coeff
                
                col += 1
    
    return U.tolist(), V.tolist(), W.tolist(), col

def _construct_schoolbook(n, m, p):
    """Standard schoolbook decomposition, rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    return U.tolist(), V.tolist(), W.tolist(), rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 150.0
    
    # Try Strassen-based decomposition
    U, V, W, rank = _construct_strassen_decomposition(n, m, p)
    
    # Verify
    if _verify(U, V, W, n, m, p):
        if time.time() - start_time < deadline - 10:
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    U, V, W, rank = _construct_schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
