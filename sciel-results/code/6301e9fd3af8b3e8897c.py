import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 4
    
    # We'll use a block decomposition:
    # Split A (3x4) into blocks: top 2 rows (2x4) and bottom 1 row (1x4)
    # Split B (4x4) into blocks: left 2 cols (4x2) and right 2 cols (4x2)
    # 
    # For the 2x2 sub-block (top-left of A, top-left of B): use Strassen (rank 7)
    # For the 2x2 sub-block (top-left of A, top-right of B): use Strassen (rank 7)
    # For the 2x2 sub-block (bottom row of A, top-left of B): use trivial (rank 4)
    # For the 2x2 sub-block (bottom row of A, top-right of B): use trivial (rank 4)
    #
    # Total rank: 7 + 7 + 4 + 4 = 22... wait, that's not right.
    #
    # Let me reconsider. The matrix multiplication is C = A * B where:
    # A is 3x4, B is 4x4, C is 3x4
    #
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # Let's partition by the j index (4 columns of A, 4 rows of B):
    # We can group the 4 j-values into two groups of 2: {0,1} and {2,3}
    #
    # For j in {0,1}:
    #   C[i,k] += A[i,0]*B[0,k] + A[i,1]*B[1,k]
    #   This is a (3x2) * (2x4) multiplication, rank at most 3*2*4=24, but we can do better
    #   Actually, for fixed j-group, we have a bilinear map from A[:,0:2] (3x2) and B[0:2,:] (2x4)
    #   to C[:,0:4] (3x4). The tensor rank of this is at most 3*2*4 = 24, but we can use structure.
    #
    # Actually, let me think differently. Let's use the known result that a 3x4x4 matrix 
    # multiplication can be done with rank 44 using block methods.
    #
    # Here's a concrete approach:
    # Partition the 4 intermediate indices into two groups of 2.
    # For each group, we do a (3x2) x (2x4) multiplication.
    # The (3x2)x(2x4) multiplication has a known optimal rank. 
    # Actually, (m x n) x (n x p) has rank at most m*n*p, but can be less.
    # For (3x2)x(2x4): we can view this as computing 3*4=12 outputs, each a sum of 2 products.
    # 
    # Let me try a direct construction with rank 44:
    # 
    # Actually, let me just construct a valid decomposition. The simplest is schoolbook with rank 48.
    # But we want to do better. Let's try rank 44.
    #
    # Here's a strategy: use 2 Strassen blocks (each rank 7) for the 2x2x2 subproblems,
    # and handle the rest with trivial multiplications.
    #
    # Let me partition:
    # A rows: {0,1} (2 rows) and {2} (1 row)
    # B cols: {0,1} (2 cols) and {2,3} (2 cols)
    # A cols / B rows: {0,1} (2) and {2,3} (2)
    #
    # Block (A[0:2,:], B[:,0:2]) -> C[0:2,0:2]: 2x2x2, use Strassen rank 7
    # Block (A[0:2,:], B[:,2:4]) -> C[0:2,2:4]: 2x2x2, use Strassen rank 7  
    # Block (A[2:3,:], B[:,0:2]) -> C[2:3,0:2]: 1x2x2, rank at most 4
    # Block (A[2:3,:], B[:,2:4]) -> C[2:3,2:4]: 1x2x2, rank at most 4
    #
    # Total: 7+7+4+4 = 22? That seems too low. Let me verify.
    #
    # Wait, the issue is that the A and B blocks share the intermediate index j.
    # A[0:2, :] uses all 4 j's, not just 2. So I need to be more careful.
    #
    # Let me restart with a cleaner approach. I'll construct a rank-44 decomposition
    # by combining Strassen-like 2x2x2 blocks.
    #
    # Actually, let me just provide the schoolbook rank-48 solution which is guaranteed correct,
    # and then try to improve it.
    
    # Schoolbook rank 48
    rank = n * m * p  # 3*4*4 = 48
    U = np.zeros((n * m, rank))  # 12 x 48
    V = np.zeros((m * p, rank))  # 16 x 48
    W = np.zeros((p * n, rank))  # 12 x 48
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    # Now try to reduce the rank using a block decomposition
    # Strategy: Use Strassen's algorithm for 2x2x2 blocks
    #
    # Partition:
    # i (rows of A, cols of C): {0,1} and {2}
    # j (cols of A, rows of B): {0,1} and {2,3}
    # k (cols of B, cols of C): {0,1} and {2,3}
    #
    # This gives 3 blocks in i, 2 in j, 2 in k = 12 sub-problems
    # But many are trivial.
    #
    # For the 2x2x2 sub-problem (i in {0,1}, j in {0,1}, k in {0,1}):
    #   C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    #   This is exactly the 2x2x2 matrix multiplication, solvable with rank 7 (Strassen)
    #
    # For 2x2x2 (i in {0,1}, j in {2,3}, k in {0,1}):
    #   C[i,k] = A[i,2]*B[2,k] + A[i,3]*B[3,k]
    #   Also 2x2x2, rank 7
    #
    # For 2x2x2 (i in {0,1}, j in {0,1}, k in {2,3}):
    #   C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    #   Also 2x2x2, rank 7
    #
    # For 2x2x2 (i in {0,1}, j in {2,3}, k in {2,3}):
    #   C[i,k] = A[i,2]*B[2,k] + A[i,3]*B[3,k]
    #   Also 2x2x2, rank 7
    #
    # For 1x2x2 (i=2, j in {0,1}, k in {0,1}):
    #   C[2,k] = A[2,0]*B[0,k] + A[2,1]*B[1,k]
    #   Rank at most 4 (trivial)
    #
    # For 1x2x2 (i=2, j in {2,3}, k in {0,1}):
    #   C[2,k] = A[2,2]*B[2,k] + A[2,3]*B[3,k]
    #   Rank at most 4
    #
    # For 1x2x2 (i=2, j in {0,1}, k in {2,3}):
    #   C[2,k] = A[2,0]*B[0,k] + A[2,1]*B[1,k]
    #   Rank at most 4
    #
    # For 1x2x2 (i=2, j in {2,3}, k in {2,3}):
    #   C[2,k] = A[2,2]*B[2,k] + A[2,3]*B[3,k]
    #   Rank at most 4
    #
    # Total: 4*7 + 4*4 = 28 + 16 = 44
    
    # Let's construct this rank-44 decomposition
    rank44 = 44
    U44 = np.zeros((n * m, rank44))  # 12 x 44
    V44 = np.zeros((m * p, rank44))  # 16 x 44
    W44 = np.zeros((p * n, rank44))  # 12 x 44
    
    col = 0
    
    # Helper: add a Strassen decomposition for a 2x2x2 block
    # i_offset, j_offset, k_offset: starting indices
    # The block computes C[i_offset+i, k_offset+k] = sum_{j=0}^{1} A[i_offset+i, j_offset+j] * B[j_offset+j, k_offset+k]
    def add_strassen(i_off, j_off, k_off):
        nonlocal col
        # Strassen's 7 products for 2x2x2:
        # M1 = (A[i,0]+A[i,1]) * (B[0,k]+B[1,k])  -- but this isn't quite right
        # 
        # Standard Strassen for C = A*B where A,B,C are 2x2:
        # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
        # M2 = (A[1,0]+A[1,1])*B[0,0]
        # M3 = A[0,0]*(B[0,1]-B[1,1])
        # M4 = A[1,1]*(B[1,0]-B[0,0])
        # M5 = (A[0,0]+A[0,1])*B[1,1]
        # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
        # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
        # 
        # C[0,0] = M1 - M4 + M6 - M7
        # C[0,1] = M3 + M5 + M7
        # C[1,0] = M2 + M4 - M6
        # C[1,1] = M1 - M2 + M3 + M5
        
        # M1 = (A[i_off, j_off] + A[i_off+1, j_off+1]) * (B[j_off, k_off] + B[j_off+1, k_off+1])
        U44[i_off * m + j_off, col] = 1.0
        U44[(i_off+1) * m + (j_off+1), col] = 1.0
        V44[j_off * p + k_off, col] = 1.0
        V44[(j_off+1) * p + (k_off+1), col] = 1.0
        W44[k_off * n + i_off, col] = 1.0
        W44[(k_off+1) * n + (i_off+1), col] = 1.0
        col += 1
        
        # M2 = (A[i_off+1, j_off] + A[i_off+1, j_off+1]) * B[j_off, k_off]
        U44[(i_off+1) * m + j_off, col] = 1.0
        U44[(i_off+1) * m + (j_off+1), col] = 1.0
        V44[j_off * p + k_off, col] = 1.0
        W44[k_off * n + (i_off+1), col] = 1.0
        col += 1
        
        # M3 = A[i_off, j_off] * (B[j_off, k_off+1] - B[j_off+1, k_off+1])
        U44[i_off * m + j_off, col] = 1.0
        V44[j_off * p + (k_off+1), col] = 1.0
        V44[(j_off+1) * p + (k_off+1), col] = -1.0
        W44[(k_off+1) * n + i_off, col] = 1.0
        col += 1
        
        # M4 = A[i_off+1, j_off+1] * (B[j_off+1, k_off] - B[j_off, k_off])
        U44[(i_off+1) * m + (j_off+1), col] = 1.0
        V44[(j_off+1) * p + k_off, col] = 1.0
        V44[j_off * p + k_off, col] = -1.0
        W44[k_off * n + (i_off+1), col] = 1.0
        col += 1
        
        # M5 = (A[i_off, j_off] + A[i_off, j_off+1]) * B[j_off+1, k_off+1]
        U44[i_off * m + j_off, col] = 1.0
        U44[i_off * m + (j_off+1), col] = 1.0
        V44[(j_off+1) * p + (k_off+1), col] = 1.0
        W44[(k_off+1) * n + i_off, col] = 1.0
        col += 1
        
        # M6 = (A[i_off+1, j_off] - A[i_off, j_off]) * (B[j_off, k_off] + B[j_off, k_off+1])
        U44[(i_off+1) * m + j_off, col] = 1.0
        U44[i_off * m + j_off, col] = -1.0
        V44[j_off * p + k_off, col] = 1.0
        V44[j_off * p + (k_off+1), col] = 1.0
        W44[k_off * n + (i_off+1), col] = 1.0
        W44[(k_off+1) * n + (i_off+1), col] = 1.0
        col += 1
        
        # M7 = (A[i_off, j_off+1] - A[i_off+1, j_off+1]) * (B[j_off+1, k_off] + B[j_off+1, k_off+1])
        U44[i_off * m + (j_off+1), col] = 1.0
        U44[(i_off+1) * m + (j_off+1), col] = -1.0
        V44[(j_off+1) * p + k_off, col] = 1.0
        V44[(j_off+1) * p + (k_off+1), col] = 1.0
        W44[k_off * n + i_off, col] = 1.0
        W44[(k_off+1) * n + i_off, col] = 1.0
        col += 1
    
    # Add trivial 1x2x2 blocks
    # C[i_off, k_off+k] = A[i_off, j_off] * B[j_off, k_off+k] + A[i_off, j_off+1] * B[j_off+1, k_off+k]
    def add_trivial_1x2x2(i_idx, j_off, k_off):
        nonlocal col
        # Term 1: A[i_idx, j_off] * B[j_off, k_off]
        U44[i_idx * m + j_off, col] = 1.0
        V44[j_off * p + k_off, col] = 1.0
        W44[k_off * n + i_idx, col] = 1.0
        col += 1
        
        # Term 2: A[i_idx, j_off] * B[j_off, k_off+1]
        U44[i_idx * m + j_off, col] = 1.0
        V44[j_off * p + (k_off+1), col] = 1.0
        W44[(k_off+1) * n + i_idx, col] = 1.0
        col += 1
        
        # Term 3: A[i_idx, j_off+1] * B[j_off+1, k_off]
        U44[i_idx * m + (j_off+1), col] = 1.0
        V44[(j_off+1) * p + k_off, col] = 1.0
        W44[k_off * n + i_idx, col] = 1.0
        col += 1
        
        # Term 4: A[i_idx, j_off+1] * B[j_off+1, k_off+1]
        U44[i_idx * m + (j_off+1), col] = 1.0
        V44[(j_off+1) * p + (k_off+1), col] = 1.0
        W44[(k_off+1) * n + i_idx, col] = 1.0
        col += 1
    
    # Block (i=0,1; j=0,1; k=0,1): Strassen
    add_strassen(0, 0, 0)
    # Block (i=0,1; j=2,3; k=0,1): Strassen
    add_strassen(0, 2, 0)
    # Block (i=0,1; j=0,1; k=2,3): Strassen
    add_strassen(0, 0, 2)
    # Block (i=0,1; j=2,3; k=2,3): Strassen
    add_strassen(0, 2, 2)
    
    # Block (i=2; j=0,1; k=0,1): Trivial
    add_trivial_1x2x2(2, 0, 0)
    # Block (i=2; j=2,3; k=0,1): Trivial
    add_trivial_1x2x2(2, 2, 0)
    # Block (i=2; j=0,1; k=2,3): Trivial
    add_trivial_1x2x2(2, 0, 2)
    # Block (i=2; j=2,3; k=2,3): Trivial
    add_trivial_1x2x2(2, 2, 2)
    
    # Verify col == 44
    assert col == 44, f"Expected 44 columns, got {col}"
    
    # Verify the decomposition
    triplets = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    u_indices = np.array([i * m + j for i, j, k in triplets])
    v_indices = np.array([j * p + k for i, j, k in triplets])
    w_indices = np.array([k * n + i for i, j, k in triplets])
    
    U_flat = U44[u_indices]
    V_flat = V44[v_indices]
    W_flat = W44[w_indices]
    result = np.sum(U_flat * V_flat * W_flat, axis=1)
    
    if np.allclose(result, 1.0, atol=1e-10):
        return {
            "U": U44.tolist(),
            "V": V44.tolist(),
            "W": W44.tolist()
        }
    else:
        # Fallback to schoolbook rank 48
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
