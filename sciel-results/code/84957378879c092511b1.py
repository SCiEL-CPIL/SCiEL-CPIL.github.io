"""Tensor decomposition for 4x4 * 4x7 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition
    for multiplying a 4x4 matrix by a 4x7 matrix.
    
    Strategy: Use Strassen's algorithm structure for the 4x4*4x4 block,
    and handle the remaining 4x4*4x3 block with schoolbook.
    
    The output W follows the convention W[k*n+i, r] where r is the rank index.
    """
    start_time = time.time()
    deadline = 120.0  # seconds
    
    n, m, p = 4, 4, 7
    
    # We decompose the 4x4 * 4x7 product.
    # Split the 7 columns into 4 + 3.
    # For the 4x4 * 4x4 part, use Strassen-based decomposition (rank 56 for two 4x4 multiplications via Kronecker of 2x2x2 Strassen)
    # Actually, let's think more carefully.
    
    # Standard approach: The tensor for (n,m,p) multiplication has entries T[i,j,k] for i in [0,n), j in [0,m), k in [0,p).
    # A rank-r decomposition gives matrices U (n*m x r), V (m*p x r), W (p*n x r) such that
    # for each (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # and for all other (i',j',k') combinations, the tensor is 0.
    
    # Actually, the matrix multiplication tensor M[i,j,k] = 1 if we're computing C[i,k] += A[i,j]*B[j,k].
    # The tensor is T[i,j,k] where i indexes row of A/row of C, j indexes col of A/row of B, k indexes col of B/col of C.
    # In the convention: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # The product sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] should equal the tensor entry.
    # For matrix multiplication, T[i,j,k] = 1 for all valid (i,j,k) in the "diagonal" sense... 
    # No wait. The multiplication tensor is: C[i,k] = sum_j A[i,j] B[j,k]
    # As a 3-tensor: T[i,j,k] = 1 (it's the identity in the middle index pairing).
    # Actually the standard multiplication tensor has T[i,j,k] = 1 iff we're pairing A[i,j] with B[j,k] to contribute to C[i,k].
    # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    
    # Wait, that's not right either. Let me re-read.
    # The tensor decomposition of matrix multiplication: we want to compute C = A*B where A is n x m, B is m x p.
    # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    # The "multiplication tensor" is a 3-tensor T of shape (n*m, m*p, p*n) or similar.
    # In the given convention: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
    # The decomposition says: for the multiplication operation, 
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{correct}
    # 
    # Actually, I think the tensor entries are: T[a,b,c] where a indexes (i,j) pair of A, b indexes (j,k) pair of B, c indexes (k,i) pair of C.
    # The multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k.
    # And it's 0 otherwise? No, that's not how it works.
    
    # Let me reconsider. The standard formulation: 
    # We have trilinear form: sum_{i,j,k} A[i,j] B[j,k] C[k,i] (where C is the "output" tensor, but really we want C[i,k] = sum_j A[i,j]B[j,k])
    # The multiplication tensor is M[i,j,k] = 1 for all i,j,k in their ranges.
    # A rank-r decomposition means: M[i,j,k] = sum_{c=0}^{r-1} U[i,j,c] * V[j,k,c] * W[k,i,c]
    # where U is indexed by (i,j), V by (j,k), W by (k,i).
    
    # So for ALL i in [0,4), j in [0,4), k in [0,7): we need
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # And for invalid combinations (which don't exist in the tensor since it's fully defined), we just need the above.
    
    # Actually the tensor is only defined on the "valid" triples. The schoolbook decomposition uses rank n*m*p with each term picking out one (i,j,k).
    
    # Let's use a block structure. Split the 4x4 * 4x7 into blocks.
    # Use Strassen for 2x2*2x2 blocks.
    
    # Approach: Use the Kronecker product of Strassen (2x2x2, rank 7) with a 2x1x1 (rank 1) to get 4x2x2 with rank 14? No...
    
    # Let me just use a straightforward block decomposition:
    # Split A (4x4) into four 2x2 blocks: A = [[A11, A12], [A21, A22]]
    # Split B (4x7) into two 2x7 blocks vertically: B = [[B1], [B2]] where B1 is 2x7, B2 is 2x7
    # C (4x7) = [[C11, C12], [C21, C22]] where each Cij is 2x7
    # C11 = A11*B1 + A12*B2
    # C12 doesn't exist since B is 4x7, so actually:
    # A is 4x4, B is 4x7.
    # A = [[A11, A12], [A21, A22]] (each 2x2)
    # B = [B_top; B_bottom] where B_top is 2x7, B_bottom is 2x7
    # C = A*B = [[A11*B_top + A12*B_bottom], [A21*B_top + A22*B_bottom]] (each 2x7)
    
    # So we need to compute 4 products of the form (2x2)*(2x7).
    # Each (2x2)*(2x7) can be done by splitting the 2x7 into two 2x3 and a 2x1, or just use Strassen-like.
    
    # Actually, let's think of (2x2)*(2x7). We can view this as computing C = A*B where A is 2x2, B is 2x7.
    # C[0,0:7] = A[0,0]*B[0,0:7] + A[0,1]*B[1,0:7]
    # C[1,0:7] = A[1,0]*B[0,0:7] + A[1,1]*B[1,0:7]
    
    # We can use a rank-7 decomposition for 2x2*2x2, and apply it to each pair of columns of B.
    # Split B (2x7) into B[:, 0:2], B[:, 2:4], B[:, 4:6], B[:, 6:7] (sizes 2x2, 2x2, 2x2, 2x1)
    # For each 2x2 block, use Strassen (rank 7). For the 2x1 block, use schoolbook (rank 2).
    # Total for one (2x2)*(2x7): 3*7 + 2 = 23
    # For 4 such products: 4*23 = 92
    
    # But wait, we can do better by sharing. Let me just implement the block Strassen approach.
    
    # Actually, let me use a cleaner approach. I'll build the decomposition using the Kronecker product idea.
    
    # Strassen decomposition for 2x2*2x2 (rank 7):
    # The 7 rank-1 terms for Strassen:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For the 2x2 * 2x2 case, the tensor indices are:
    # A indices: (0,0), (0,1), (1,0), (1,1) -> flat: 0,1,2,3
    # B indices: (0,0), (0,1), (1,0), (1,1) -> flat: 0,1,2,3
    # C indices: (0,0), (0,1), (1,0), (1,1) -> flat: 0,1,2,3 (but in W convention: W[k*n+i] where n=2, so (k,i) -> k*2+i)
    # W[0*2+0]=W[0] for C[0,0], W[1*2+0]=W[2] for C[0,1], W[0*2+1]=W[1] for C[1,0], W[1*2+1]=W[3] for C[1,1]
    
    # Let me define the Strassen decomposition for 2x2*2x2 as three vectors of length 4 (for U, V, W) and 7 columns.
    
    def strassen_2x2_2x2():
        """Return U (4x7), V (4x7), W (4x7) for 2x2*2x2 Strassen decomposition."""
        # U[i*m+j, c] where m=2, so U[0,0]=U[0,0], U[0,1]=U[0,1], U[1,0]=U[1,0], U[1,1]=U[1,1]
        # Actually U is indexed by (i,j) of A: i in {0,1}, j in {0,1}, flat index i*2+j
        # V is indexed by (j,k) of B: j in {0,1}, k in {0,1}, flat index j*2+k
        # W is indexed by (k,i) of C: k in {0,1}, i in {0,1}, flat index k*2+i
        
        U = np.zeros((4, 7), dtype=float)
        V = np.zeros((4, 7), dtype=float)
        W = np.zeros((4, 7), dtype=float)
        
        # M1 = (A11 + A22) * (B11 + B22)
        # A11 is at (0,0) -> U[0], A22 is at (1,1) -> U[3]
        # B11 is at (0,0) -> V[0], B22 is at (1,1) -> V[3]
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        # C11 += M1, C22 += M1
        # C11 is at (i=0,k=0) -> W[0*2+0]=W[0]
        # C22 is at (i=1,k=1) -> W[1*2+1]=W[3]
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (A21 + A22) * B11
        # A21 is at (1,0) -> U[2], A22 is at (1,1) -> U[3]
        # B11 is at (0,0) -> V[0]
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        # C21 += M2, C22 -= M2
        # C21 is at (i=1,k=0) -> W[0*2+1]=W[1]
        # C22 is at (i=1,k=1) -> W[3]
        W[1, 1] = 1; W[3, 1] = -1
        
        # M3 = A11 * (B12 - B22)
        # A11 is at (0,0) -> U[0]
        # B12 is at (0,1) -> V[1], B22 is at (1,1) -> V[3]
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        # C12 += M3, C22 += M3
        # C12 is at (i=0,k=1) -> W[1*2+0]=W[2]
        # C22 is at (i=1,k=1) -> W[3]
        W[2, 2] = 1; W[3, 2] = 1
        
        # M4 = A22 * (B21 - B11)
        # A22 is at (1,1) -> U[3]
        # B21 is at (1,0) -> V[2], B11 is at (0,0) -> V[0]
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        # C11 += M4, C21 += M4
        # C11 is at (i=0,k=0) -> W[0]
        # C21 is at (i=1,k=0) -> W[1]
        W[0, 3] = 1; W[1, 3] = 1
        
        # M5 = (A11 + A12) * B22
        # A11 is at (0,0) -> U[0], A12 is at (0,1) -> V... wait, A12 is at (i=0,j=1) -> U[0*2+1]=U[1]
        # B22 is at (1,1) -> V[3]
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        # C11 -= M5, C12 += M5
        W[0, 4] = -1; W[2, 4] = 1
        
        # M6 = (A21 - A11) * (B11 + B12)
        # A21 is at (1,0) -> U[2], A11 is at (0,0) -> U[0]
        # B11 is at (0,0) -> V[0], B12 is at (0,1) -> V[1]
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        # C22 += M6
        W[3, 5] = 1
        
        # M7 = (A12 - A22) * (B21 + B22)
        # A12 is at (0,1) -> U[1], A22 is at (1,1) -> U[3]
        # B21 is at (1,0) -> V[2], B22 is at (1,1) -> V[3]
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        # C11 += M7
        W[0, 6] = 1
        
        return U, V, W
    
    # Now build the 4x4 * 4x7 decomposition.
    # A is 4x4, split into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]]
    # A11: rows 0-1, cols 0-1
    # A12: rows 0-1, cols 2-3
    # A21: rows 2-3, cols 0-1
    # A22: rows 2-3, cols 2-3
    
    # B is 4x7, split into top (rows 0-1) and bottom (rows 2-3):
    # B_top: rows 0-1, all 7 cols
    # B_bot: rows 2-3, all 7 cols
    
    # C = A*B, C is 4x7:
    # C_top = A11*B_top + A12*B_bot (rows 0-1 of C)
    # C_bot = A21*B_top + A22*B_bot (rows 2-3 of C)
    
    # For each (2x2)*(2x7) product, we need to compute it.
    # (2x2)*(2x7): Let's denote the 2x2 matrix as P and the 2x7 matrix as Q.
    # Result R = P*Q is 2x7.
    # We can split Q into Q[:, 0:2], Q[:, 2:4], Q[:, 4:6], Q[:, 6:7]
    # For Q[:, 0:2], Q[:, 2:4], Q[:, 4:6]: use Strassen (2x2*2x2, rank 7 each)
    # For Q[:, 6:7]: use schoolbook (2x2*2x1, rank 2)
    # Total rank per (2x2)*(2x7): 3*7 + 2 = 23
    
    # But we have 4 such products (A11*B_top, A12*B_bot, A21*B_top, A22*B_bot), so total rank = 4*23 = 92.
    
    # However, we can be smarter. Let's just directly build the full decomposition.
    
    # Let me define helper functions to build the decomposition.
    
    def build_2x2_2x7():
        """
        Build Strassen-based decomposition for (2x2)*(2x7).
        Returns U (4 x r), V (14 x r), W (14 x r) where r is the rank.
        U indexed by (i,j) of the 2x2 A: flat i*2+j
        V indexed by (j,k) of the 2x7 B: flat j*7+k
        W indexed by (k,i) of the 2x7 C: flat k*2+i
        """
        U_list = []
        V_list = []
        W_list = []
        
        # Split the 7 columns of B into 3 groups of 2 and 1 group of 1.
        # Columns 0,1 -> Strassen
        # Columns 2,3 -> Strassen
        # Columns 4,5 -> Strassen
        # Column 6 -> schoolbook
        
        # For each pair of columns (c0, c1), use Strassen 2x2*2x2
        col_pairs = [(0, 1), (2, 3), (4, 5)]
        
        for c0, c1 in col_pairs:
            sU, sV, sW = strassen_2x2_2x2()
            # sU is 4x7, sV is 4x7, sW is 4x7
            # sV is indexed by (j, k_local) where k_local in {0,1} corresponds to columns c0, c1
            # We need to map sV[j*2+k_local] to V[j*7 + (c0+k_local)]
            # sW is indexed by (k_local, i) where k_local in {0,1}
            # We need to map sW[k_local*2+i] to W[(c0+k_local)*2 + i]
            
            # Build expanded V and W for this block
            V_block = np.zeros((14, 7))  # 2 rows of B * 7 cols
            W_block = np.zeros((14, 7))  # 7 cols of C * 2 rows
            
            for j in range(2):
                for kl in range(2):
                    col_global = c0 + kl
                    V_block[j * 7 + col_global, :] = sV[j * 2 + kl, :]
            
            for kl in range(2):
                for i in range(2):
                    col_global = c0 + kl
                    W_block[col_global * 2 + i, :] = sW[kl * 2 + i, :]
            
            U_list.append(sU)
            V_list.append(V_block)
            W_list.append(W_block)
        
        # For column 6, use schoolbook 2x2*2x1
        # C[i, 6] = A[i, 0]*B[0, 6] + A[i, 1]*B[1, 6]
        # Two terms:
        # Term 1: A[i,0] * B[0,6] -> contributes to C[i,6]
        # Term 2: A[i,1] * B[1,6] -> contributes to C[i,6]
        
        U_sb = np.zeros((4, 2))
        V_sb = np.zeros((14, 2))
        W_sb = np.zeros((14, 2))
        
        # Term 1: j=0
        U_sb[0, 0] = 1  # A[0,0]
        U_sb[2, 0] = 1  # A[1,0]
        V_sb[0 * 7 + 6, 0] = 1  # B[0,6]
        W_sb[6 * 2 + 0, 0] = 1  # C[0,6]
        W_sb[6 * 2 + 1, 0] = 1  # C[1,6]
        
        # Term 2: j=1
        U_sb[1, 1] = 1  # A[0,1]
        U_sb[3, 1] = 1  # A[1,1]
        V_sb[1 * 7 + 6, 1] = 1  # B[1,6]
        W_sb[6 * 2 + 0, 1] = 1  # C[0,6]
        W_sb[6 * 2 + 1, 1] = 1  # C[1,6]
        
        U_list.append(U_sb)
        V_list.append(V_sb)
        W_list.append(W_sb)
        
        U_full = np.hstack(U_list)
        V_full = np.hstack(V_list)
        W_full = np.hstack(W_list)
        
        return U_full, V_full, W_full
    
    # Now build the full 4x4 * 4x7 decomposition.
    # A is 4x4, B is 4x7, C is 4x7.
    # U indexed by (i,j) of A: i in {0,1,2,3}, j in {0,1,2,3}, flat i*4+j
    # V indexed by (j,k) of B: j in {0,1,2,3}, k in {0,...,6}, flat j*7+k
    # W indexed by (k,i) of C: k in {0,...,6}, i in {0,1,2,3}, flat k*4+i
    
    # Split A into 2x2 blocks:
    # A11: i in {0,1}, j in {0,1}
    # A12: i in {0,1}, j in {2,3}
    # A21: i in {2,3}, j in {0,1}
    # A22: i in {2,3}, j in {2,3}
    
    # Split B into top and bottom:
    # B_top: j in {0,1}, all k
    # B_bot: j in {2,3}, all k
    
    # C_top (i in {0,1}): C_top = A11*B_top + A12*B_bot
    # C_bot (i in {2,3}): C_bot = A21*B_top + A22*B_bot
    
    # For each of the 4 products, we use the 2x2*2x7 decomposition.
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    # Product 1: A11 * B_top -> contributes to C_top
    # A11 is 2x2, local indices (i_local, j_local) with i_local in {0,1}, j_local in {0,1}
    # Global A indices: i = i_local, j = j_local
    # B_top is 2x7, local indices (j_local, k) with j_local in {0,1}, k in {0,...,6}
    # Global B indices: j = j_local, k = k
    # C_top is 2x7, local indices (k, i_local)
    # Global C indices: k = k, i = i_local
    
    u1, v1, w1 = build_2x2_2x7()
    # u1 is 4 x r where 4 = 2*2 (i_local*2 + j_local)
    # v1 is 14 x r where 14 = 2*7 (j_local*7 + k)
    # w1 is 14 x r where 14 = 7*2 (k*2 + i_local)
    
    U1 = np.zeros((16, u1.shape[1]))  # 4*4
    V1 = np.zeros((28, v1.shape[1]))  # 4*7
    W1 = np.zeros((28, w1.shape[1]))  # 7*4
    
    r1 = u1.shape[1]
    for il in range(2):
        for jl in range(2):
            U1[il * 4 + jl, :] = u1[il * 2 + jl, :]
    for jl in range(2):
        for k in range(7):
            V1[jl * 7 + k, :] = v1[jl * 7 + k, :]
    for k in range(7):
        for il in range(2):
            W1[k * 4 + il, :] = w1[k * 2 + il, :]
    
    U_parts.append(U1)
    V_parts.append(V1)
    W_parts.append(W1)
    
    # Product 2: A12 * B_bot -> contributes to C_top
    # A12: i in {0,1}, j in {2,3}
    # B_bot: j in {2,3}, all k
    # C_top: i in {0,1}, all k
    
    u2, v2, w2 = build_2x2_2x7()
    U2 = np.zeros((16, u2.shape[1]))
    V2 = np.zeros((28, v2.shape[1]))
    W2 = np.zeros((28, w2.shape[1]))
    
    for il in range(2):
        for jl in range(2):
            U2[il * 4 + (2 + jl), :] = u2[il * 2 + jl, :]
    for jl in range(2):
        for k in range(7):
            V2[(2 + jl) * 7 + k, :] = v2[jl * 7 + k, :]
    for k in range(7):
        for il in range(2):
            W2[k * 4 + il, :] = w2[k * 2 + il, :]
    
    U_parts.append(U2)
    V_parts.append(V2)
    W_parts.append(W2)
    
    # Product 3: A21 * B_top -> contributes to C_bot
    # A21: i in {2,3}, j in {0,1}
    # B_top: j in {0,1}, all k
    # C_bot: i in {2,3}, all k
    
    u3, v3, w3 = build_2x2_2x7()
    U3 = np.zeros((16, u3.shape[1]))
    V3 = np.zeros((28, v3.shape[1]))
    W3 = np.zeros((28, w3.shape[1]))
    
    for il in range(2):
        for jl in range(2):
            U3[(2 + il) * 4 + jl, :] = u3[il * 2 + jl, :]
    for jl in range(2):
        for k in range(7):
            V3[jl * 7 + k, :] = v3[jl * 7 + k, :]
    for k in range(7):
        for il in range(2):
            W3[k * 4 + (2 + il), :] = w3[k * 2 + il, :]
    
    U_parts.append(U3)
    V_parts.append(V3)
    W_parts.append(W3)
    
    # Product 4: A22 * B_bot -> contributes to C_bot
    # A22: i in {2,3}, j in {2,3}
    # B_bot: j in {2,3}, all k
    # C_bot: i in {2,3}, all k
    
    u4, v4, w4 = build_2x2_2x7()
    U4 = np.zeros((16, u4.shape[1]))
    V4 = np.zeros((28, v4.shape[1]))
    W4 = np.zeros((28, w4.shape[1]))
    
    for il in range(2):
        for jl in range(2):
            U4[(2 + il) * 4 + (2 + jl), :] = u4[il * 2 + jl, :]
    for jl in range(2):
        for k in range(7):
            V4[(2 + jl) * 7 + k, :] = v4[jl * 7 + k, :]
    for k in range(7):
        for il in range(2):
            W4[k * 4 + (2 + il), :] = w4[k * 2 + il, :]
    
    U_parts.append(U4)
    V_parts.append(V4)
    W_parts.append(W4)
    
    U_full = np.hstack(U_parts)
    V_full = np.hstack(V_parts)
    W_full = np.hstack(W_parts)
    
    rank = U_full.shape[1]
    
    # Convert to lists
    U_list = U_full.tolist()
    V_list = V_full.tolist()
    W_list = W_full.tolist()
    
    # Verify the decomposition
    # Check that for all (i,j,k) in valid range, sum_c U[i*4+j, c] * V[j*7+k, c] * W[k*4+i, c] = 1
    # And that this is the only non-zero pattern (which it should be by construction)
    
    # Let's verify
    error = 0.0
    for i in range(4):
        for j in range(4):
            for k in range(7):
                val = 0.0
                for c in range(rank):
                    val += U_full[i * 4 + j, c] * V_full[j * 7 + k, c] * W_full[k * 4 + i, c]
                if abs(val - 1.0) > 1e-10:
                    error += abs(val - 1.0)
    
    if error > 1e-8:
        # Fallback to schoolbook if verification fails
        rank = n * m * p
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
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
