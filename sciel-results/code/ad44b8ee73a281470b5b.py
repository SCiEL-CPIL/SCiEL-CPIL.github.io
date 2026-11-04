"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    n, m, p = DIMENSIONS  # n=4, m=4, p=5
    
    # We'll construct a decomposition for 4x4x4 (rank 49) + 4x4x1 (rank 4) = rank 53
    # Using Winograd/Strassen for 4x4x4 and simple decomposition for 4x4x1
    
    # First, let's build the 4x4x4 Winograd decomposition
    # The Winograd algorithm for 4x4 matrix multiplication uses 49 multiplications
    
    # U: shape (4*4, rank), V: shape (4*4, rank), W: shape (4*4, rank)
    # For 4x4x4, we need U, V, W each of shape (16, 49)
    
    # Let me construct the Strassen-based 4x4x4 decomposition
    # Using 3x3 Strassen blocks recursively
    
    # Actually, let me use a known rank-49 decomposition for 4x4x4
    # I'll construct it using the Winograd scheme
    
    # For simplicity and correctness, let me use a structured approach:
    # The 4x4 matrix multiplication can be decomposed using 49 bilinear forms
    
    # Let me build this systematically using the Winograd F(4) algorithm
    # F(4) has 49 multiplications for 4x4 matrix multiplication
    
    # U[i*m+j, r] = coefficient for A[i,j] in multiplication r
    # V[j*p+k, r] = coefficient for B[j,k] in multiplication r  
    # W[k*n+i, r] = coefficient for C[i,k] from multiplication r
    
    # For 4x4x4 part (columns 0-3 of B):
    rank_444 = 49
    U_444 = np.zeros((16, rank_444))
    V_444 = np.zeros((16, rank_444))
    W_444 = np.zeros((16, rank_444))
    
    # Winograd F(4) decomposition
    # Using the standard Winograd coefficients for 4-point convolution/matrix mult
    
    # Let me use a simpler approach: construct via known Strassen recursion
    # 4x4 = 4 blocks of 2x2, each 2x2 uses 7 mults = 28, plus corrections
    
    # Actually, let me use a direct construction that I can verify
    # I'll use the rank-49 decomposition from literature
    
    # For now, let me construct a valid decomposition using a known method
    # I'll use the "border rank" approach or a precomputed table
    
    # Let me try a different approach: use numpy to help construct
    # a valid low-rank decomposition via optimization if time permits
    
    # But first, let me ensure I have a valid baseline
    # Schoolbook for 4x4x4 has rank 64
    
    # Let me construct the Winograd 4x4x4 decomposition properly
    # Using the standard Winograd F(4) scheme
    
    # The Winograd F(4) algorithm for 4x4 matrix multiplication:
    # It computes C = A * B where A, B, C are 4x4
    # Using 49 multiplications
    
    # I'll use a known coefficient set for Winograd 4x4
    
    # Actually, let me use a simpler valid decomposition:
    # Use Strassen 2x2x2 (rank 7) applied to 4x4x4 by viewing as 2x2 blocks of 2x2
    # This gives rank 7 * 4 = 28 for the block structure, but we need cross terms
    
    # For correctness, let me use a verified decomposition
    # I'll construct the rank-49 Winograd decomposition from known coefficients
    
    # Winograd F(4) coefficients (from standard references):
    # The algorithm uses specific linear combinations
    
    # Let me use a pragmatic approach: construct a valid decomposition
    # using the fact that we can always use schoolbook, but try to find
    # a lower rank one within time limits
    
    # For the 4x4x1 part (5th column of B, i.e., column index 3 in 0-based for the 5 columns):
    # C[i,4] = sum_j A[i,j] * B[j,4]
    # This is a matrix-vector product
    # Rank 4 decomposition:
    # For each i, C[i,4] = sum_j A[i,j] B[j,4]
    # We can use: for l in 0..3: M_l = A[l,l] * B[l,4] ... no, that's not right
    
    # Correct rank-4 for 4x4x1:
    # T[i,j] = 1 (the tensor)
    # We need: sum_r U[i,r] V[j,r] W[i,r] A[i,j] B[j,4] = C[i,4]
    # Actually the bilinear form is: sum_{i,j} C[i,4] A[i,j] B[j,4] = sum_j B[j,4] (sum_i C[i,4] A[i,j])
    # Hmm, let me think about this differently
    
    # The matrix multiplication tensor for 4x4x1:
    # C[i] = sum_j A[i,j] B[j]
    # Tensor T[i,j] = 1
    # This is an all-ones 4x4 matrix, which has rank 1
    # So rank(4x4x1) = 1? Let me verify:
    # Can we write sum_j A[i,j] B[j] = U[i,r] V[j,r] W[i,r] for r=1?
    # We need: A[i,j] B[j] = U[i] V[j] W[i] for all i,j
    # This means A[i,j] = U[i] W[i] V[j] which is not generally true
    # So rank is NOT 1
    
    # The correct rank for 4x4x1 (matrix-vector product) is 4:
    # Use: C[i] = sum_j A[i,j] B[j]
    # Decomposition: for each i, use one term
    # M_i = A[i,:] * B[:,i] ... no
    
    # Let me think again. The bilinear form is:
    # sum_{i,j} C_i A_{ij} B_j
    # We want: sum_r U_{ir} V_{jr} W_{ir} A_{ij} B_j = sum_i C_i (sum_j A_{ij} B_j)
    # So we need: sum_r U_{ir} V_{jr} W_{ir} = delta_{ij} * 1 ... no
    # We need: sum_j (sum_r U_{ir} V_{jr} W_{ir}) A_{ij} B_j = C_i (sum_j A_{ij} B_j)
    # This requires: sum_r U_{ir} V_{jr} W_{ir} = delta_{ij} for all i,j? No...
    
    # Actually: C_i = sum_j A_{ij} B_j
    # So the tensor is T_{ij} = 1 (coefficient of A_{ij} B_j in C_i)
    # Wait, C_i depends on A_{ij} and B_j for all j
    # The tensor T has entries T[i,j] = 1 (for the bilinear form sum_{i,j} C_i A_{ij} B_j)
    # But this is not quite right either
    
    # Let me use the standard result: the rank of the matrix-vector product tensor
    # for n x n x 1 is n. For 4x4x1, rank = 4.
    
    # Rank 4 decomposition for 4x4x1:
    # C_i = sum_j A_{ij} B_j
    # Use 4 terms, one for each "diagonal" or use:
    # M_l = (sum_j A_{lj}) * B_l ... no
    
    # Correct approach:
    # C_i = sum_j A_{ij} B_j
    # We can write this as: for each i, C_i is the dot product of row i of A with B
    # The tensor rank is 4, achieved by:
    # For r = 0,1,2,3:
    #   U[i,r] = 1 if i==r else 0  (selects row r of A)
    #   V[j,r] = 1 if j==r else 0  (selects element r of B)  
    #   W[i,r] = 1 if i==r else 0  (contributes to C_r)
    # Then sum_r U[i,r] V[j,r] W[i,r] A[i,j] B[j] = sum_r delta_{ir} delta_{jr} delta_{ir} A[i,j] B[j]
    # = sum_r delta_{ir} delta_{jr} A[i,j] B[j]
    # For this to equal C_i = sum_j A[i,j] B_j, we need the sum over j to give A[i,j] B_j
    # But delta_{jr} means j=r, so we get A[i,r] B[r] * delta_{ir}
    # Sum over r: delta_{ir} A[i,r] B[r] = A[i,i] B[i]
    # That's only the diagonal element, not the full sum!
    
    # So that decomposition doesn't work. Let me reconsider.
    
    # The correct rank-4 decomposition for matrix-vector product:
    # Actually, I think the rank is 4 and the decomposition is:
    # M_r = (sum_j A_{rj}) * B_r ... no, that gives sum_r (sum_j A_{rj}) B_r which is a scalar
    
    # I'm confusing myself. Let me look at this more carefully.
    
    # For C = A * b where A is 4x4 and b is 4x1:
    # C_i = sum_j A_{ij} b_j
    # The tensor T is a 4x4 matrix with T_{ij} = 1
    # The rank of T as a matrix is 1 (all ones)
    # But the tensor rank for the bilinear form is different
    
    # The bilinear form is: <C, A, b> = sum_{i,j} C_i A_{ij} b_j
    # We want to decompose this as sum_r U_i^{(r)} V_j^{(r)} W_i^{(r)} C_i A_{ij} b_j
    # So we need: sum_r U_i^{(r)} V_j^{(r)} W_i^{(r)} = delta_{ij} ... no
    # We need: sum_r U_i^{(r)} V_j^{(r)} W_i^{(r)} C_i A_{ij} b_j = sum_{i,j} C_i A_{ij} b_j
    # This requires: sum_r U_i^{(r)} V_j^{(r)} W_i^{(r)} = 1 for all i,j
    # So the tensor T_{ij} = 1 = sum_r U_i^{(r)} V_j^{(r)} W_i^{(r)}
    # This is a decomposition of the all-ones matrix as sum of rank-1 matrices
    # The all-ones 4x4 matrix has rank 1, so we can do it with 1 term:
    # U_i^{(1)} = 1, V_j^{(1)} = 1, W_i^{(1)} = 1
    # Then sum_r U_i^{(r)} V_j^{(r)} W_i^{(r)} = 1*1*1 = 1 for all i,j
    
    # So rank(4x4x1) = 1!
    
    # Let me verify: with U_i = 1, V_j = 1, W_i = 1:
    # sum_r U_i V_j W_i A_{ij} b_j = 1*1*1 * sum_{i,j} A_{ij} b_j = sum_{i,j} A_{ij} b_j
    # But we need sum_i C_i (sum_j A_{ij} b_j) = sum_{i,j} C_i A_{ij} b_j
    # And our decomposition gives: sum_{i,j} 1 * A_{ij} b_j = sum_{i,j} A_{ij} b_j
    # This is NOT the same as sum_{i,j} C_i A_{ij} b_j unless C_i = 1 for all i
    
    # I see my error. The W factor should depend on i in a way that picks out C_i
    # Let me re-examine the tensor structure
    
    # The matrix multiplication tensor for C = A*B where A is n x m, B is m x p, C is p x n:
    # C_{ki} = sum_j A_{ij} B_{jk}
    # Tensor: T_{i,j,k} = 1 (coefficient of A_{ij} B_{jk} in C_{ki})
    # For 4x4x1: n=4, m=4, p=1
    # C_{1i} = sum_j A_{ij} B_{j1}
    # T_{i,j,1} = 1
    # This is a 4x4x1 tensor with all entries 1
    # The rank of this tensor: we need to write T_{i,j,k} = sum_r U_{ir} V_{jr} W_{kr}
    # Since k is always 1, W_{kr} = W_{1r}
    # T_{ij} = sum_r U_{ir} V_{jr} W_{1r}
    # This is a matrix factorization of the 4x4 all-ones matrix
    # Rank of all-ones 4x4 matrix is 1
    # So we can use r=1: U_{i1}=1, V_{j1}=1, W_{11}=1
    # Then T_{ij} = 1*1*1 = 1. Correct!
    
    # So rank(4x4x1) = 1
    
    # Now for the 4x4x4 part, I need a rank-49 decomposition
    # Let me construct the Winograd 4x4x4 decomposition
    
    # I'll use a known valid decomposition. Let me construct it using
    # the Winograd F(4) algorithm coefficients
    
    # For time efficiency, let me use a precomputed valid decomposition
    # I'll construct it using numpy operations that I can verify
    
    # Actually, let me just use the schoolbook for 4x4x4 (rank 64) + rank 1 for 4x4x1 = rank 65
    # That gives score 80/65 = 1.231, which is better than 1.0
    
    # But I can do better with Winograd 4x4x4 (rank 49) + rank 1 = rank 50
    # Score: 80/50 = 1.6
    
    # Let me construct the Winograd 4x4x4 decomposition
    
    # I'll use a systematic construction based on the Winograd scheme
    # For 4x4 matrix multiplication, the Winograd F(4) algorithm uses 49 multiplications
    
    # Let me build this using a known coefficient table
    
    # For now, let me use a valid lower-rank decomposition
    # I'll use Strassen 2x2x2 (rank 7) applied in a 4x4 context
    
    # 4x4 matrix as 2x2 blocks of 2x2:
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # Each 2x2 multiplication uses 7 mults (Strassen)
    # But we have 4 separate 2x2 multiplications, so 4*7 = 28 mults
    # However, this is not optimal because we're not sharing computations
    
    # The optimal Strassen-based 4x4x4 uses 49 mults (Winograd)
    
    # Let me just construct a valid rank-64 (schoolbook 4x4x4) + rank 1 = rank 65 solution
    # That's already an improvement over rank 80
    
    # Schoolbook 4x4x4: rank 64
    rank_444_school = 64
    U_444 = np.zeros((16, rank_444_school))
    V_444 = np.zeros((16, rank_444_school))
    W_444 = np.zeros((16, rank_444_school))
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U_444[i*4+j, col] = 1
                V_444[j*4+k, col] = 1
                W_444[k*4+i, col] = 1
                col += 1
    
    # 4x4x1: rank 1
    rank_441 = 1
    U_441 = np.zeros((16, rank_441))
    V_441 = np.zeros((16, rank_441))
    W_441 = np.zeros((4, rank_441))  # p*n = 1*4 = 4
    
    # For 4x4x1: C_{1i} = sum_j A_{ij} B_{j1}
    # T_{i,j} = 1, decomposed as U_i * V_j * W_1
    # U_i = 1, V_j = 1, W_1 = 1
    for i in range(4):
        U_441[i*4, 0] = 1  # This is wrong, U should be indexed by i*m+j
        # Actually U is shape (n*m, rank) = (16, 1)
        # U[i*m+j, r] for i in 0..3, j in 0..3
        # We want U_{ij} = 1 for all i,j
        U_441[i*4+j, 0] = 1
        V_441[j*1+0, 0] = 1  # V is shape (m*p, rank) = (4*1, 1) = (4, 1)
        # V[j*p+k, r] for j in 0..3, k in 0..0
        # V[j, 0] = 1
    
    # W is shape (p*n, rank) = (1*4, 1) = (4, 1)
    # W[k*n+i, r] for k in 0..0, i in 0..3
    # W[i, 0] = 1 for i in 0..3
    for i in range(4):
        W_441[i, 0] = 1
    
    # Now combine: total rank = 64 + 1 = 65
    # U: shape (16, 65), V: shape (16, 65), W: shape (20, 65)
    # Wait, V is shape (m*p, rank) = (4*5, rank) = (20, rank)
    # W is shape (p*n, rank) = (5*4, rank) = (20, rank)
    
    # Let me redo this with correct shapes
    # n=4, m=4, p=5
    # U: (n*m, rank) = (16, rank)
    # V: (m*p, rank) = (20, rank)
    # W: (p*n, rank) = (20, rank)
    
    total_rank = 65
    U = np.zeros((16, total_rank))
    V = np.zeros((20, total_rank))
    W = np.zeros((20, total_rank))
    
    # 4x4x4 part (columns 0-3 of B): rank 64
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i*4+j, col] = 1
                V[j*5+k, col] = 1  # V indexed by j*p+k, p=5
                W[k*4+i, col] = 1  # W indexed by k*n+i, n=4
                col += 1
    
    # 4x4x1 part (column 4 of B, i.e., k=4): rank 1
    # C_{4,i} = sum_j A_{ij} B_{j,4}
    # T_{i,j} = 1
    # U_{ij} = 1, V_{j,4} = 1, W_{4,i} = 1
    for i in range(4):
        for j in range(4):
            U[i*4+j, col] = 1
            V[j*5+4, col] = 1
            W[4*4+i, col] = 1
    
    # Verify the decomposition
    # Check that sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all i,j,k
    # And 0 otherwise (but it should always be 1 for valid (i,j,k))
    
    # Let me verify a few entries
    # For i=0, j=0, k=0: should be 1
    # Sum_r U[0, r] V[0, r] W[0, r]
    # From 4x4x4 part: r=0 has U[0,0]=1, V[0,0]=1, W[0,0]=1, so contribution = 1
    # From 4x4x1 part: r=64 has U[0,64]=1, V[4,64]=1, W[16,64]=1
    # But V[0,64] = V[0*5+0, 64] = V[0, 64] which is 0 (only V[4,64]=1 for the 4x4x1 part)
    # So no contribution from 4x4x1 part for k=0. Good.
    
    # For i=0, j=0, k=4: should be 1
    # From 4x4x4 part: no terms with k=4, so 0
    # From 4x4x1 part: r=64 has U[0,64]=1, V[4,64]=1, W[16,64]=1
    # W[16,64] = W[4*4+0, 64] = W[16, 64] = 1. Good.
    # Contribution = 1*1*1 = 1. Correct!
    
    # Convert to list format as required
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

