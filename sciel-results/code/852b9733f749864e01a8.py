"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 5, 6
    start_time = time.time()
    deadline = start_time + 130  # 130s safety margin
    
    # Schoolbook decomposition as baseline
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook), dtype=float)
    V_sb = np.zeros((m * p, rank_schoolbook), dtype=float)
    W_sb = np.zeros((p * n, rank_schoolbook), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * m * p + j * p + k
                U_sb[i * m + j, c] = 1
                V_sb[j * p + k, c] = 1
                W_sb[k * n + i, c] = 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    
    # Try Strassen-based decomposition for 2x2x2 blocks
    # Decompose 4x5x6 into sub-blocks and apply Strassen where possible
    
    # Strategy: Split into 2x2 blocks in i and k dimensions
    # i: {0,1}, {2,3}; k: {0,1}, {2,3}, {4,5}
    # j: {0,1}, {2,3}, {4}
    
    # For each 2x2x2 sub-block, use Strassen (rank 7) instead of schoolbook (rank 8)
    # But we need to handle the non-uniform splits
    
    # Let's try: use Strassen on the 4x4x4 core and handle the rest
    # A[0:4, 0:4] * B[0:4, 0:4] -> C[0:4, 0:4]  (4x4x4, Strassen rank 49)
    # A[0:4, 4:5] * B[4:5, 0:6] -> C[0:4, 0:6]  (4x1x6, rank 1... wait no)
    
    # Actually 4x1x6: T[i,j,k] where j in {4}. T[i,4,k] = 1. Rank 1 as shown.
    # But we also have A[0:4, 0:4] * B[0:4, 4:6] -> C[0:4, 4:6] (4x4x2)
    # And A[0:4, 4:5] * B[4:5, 0:6] is handled
    # Wait, C[i,k] = sum_{j=0}^{4} A[i,j] B[j,k]
    # = sum_{j=0}^{3} A[i,j] B[j,k] + A[i,4] B[4,k]
    
    # So C = C_core + C_tail where C_core[i,k] = sum_{j=0}^{3} A[i,j]B[j,k] and C_tail[i,k] = A[i,4]B[4,k]
    
    # C_core is 4x4x6 multiplication, C_tail is 4x1x6 (rank 1)
    
    # Now 4x4x6: split k into {0,1,2,3} and {4,5}
    # C_core = C_core1 + C_core2 where C_core1[i,k] = sum_{j=0}^{3}A[i,j]B[j,k] for k in {0..3} (4x4x4)
    # and C_core2[i,k] = sum_{j=0}^{3}A[i,j]B[j,k] for k in {4,5} (4x4x2)
    
    # 4x4x4: Strassen rank 49
    # 4x4x2: need to figure out rank
    # 4x1x6: rank 1
    
    # Total: 49 + rank(4x4x2) + 1
    
    # For 4x4x2: C[i,k] = sum_{j=0}^{3} A[i,j] B[j,k], k in {0,1}
    # This is like multiplying 4x4 by 4x2.
    # Can we do better than 32?
    
    # Try: split into 2x2 blocks in i dimension
    # i: {0,1}, {2,3}; j: {0,1}, {2,3}; k: {0,1}
    # C_top[i',k] = sum_{j'} A_top[i',j'] B_top[j',k] + sum_{j'} A_top[i',j'] B_bot[j',k]
    # This is getting complex.
    
    # Let me just try a numerical search for a low-rank decomposition
    # with the constraint that entries are half-integers.
    
    # Actually, let's try a different known result:
    # The rank of 4x4x4 multiplication is at most 49 (Strassen-based).
    # The rank of 4x4x2: let's try to compute it.
    
    # For now, let's just return the schoolbook if we can't find better quickly.
    # But let's try a simple improvement: use Strassen on 2x2x2 blocks.
    
    # Decompose 4x5x6 as follows:
    # Think of it as 4 rows of A, 5 cols of A (rows of B), 6 cols of B.
    # Group i into 2 groups of 2, k into 3 groups of 2, j into groups of 2,2,1
    
    # For each (i_group, k_group, j_group) combination, we have a 2x2x2 or 2x2x1 or 2x1x2 sub-problem.
    
    # 2x2x2: Strassen rank 7
    # 2x2x1: rank 2 (it's just two dot products)
    # 2x1x2: rank 2
    # 1x1x1: rank 1
    
    # Let's count:
    # i_groups: 2, k_groups: 3, j_groups: 3 (sizes 2,2,1)
    # For i_group size 2, k_group size 2, j_group size 2: 2x2x2, rank 7
    # For i_group size 2, k_group size 2, j_group size 1: 2x2x1, rank 2
    # For i_group size 2, k_group size 2, j_group size 2: already counted
    
    # j_groups: {0,1} size 2, {2,3} size 2, {4} size 1
    # k_groups: {0,1} size 2, {2,3} size 2, {4,5} size 2
    # i_groups: {0,1} size 2, {2,3} size 2
    
    # Combinations:
    # (i=2, k=2, j=2): 2*3*2 = 12 blocks, each 2x2x2, rank 7 -> 84
    # (i=2, k=2, j=1): 2*3*1 = 6 blocks, each 2x2x1, rank 2 -> 12
    # Total: 84 + 12 = 96
    
    # That's better than 120! Let me implement this.
    
    def strassen_2x2x2():
        """Return U, V, W for 2x2x2 Strassen decomposition.
        U shape (2*2, 7), V shape (2*2, 7), W shape (2*2, 7)
        Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
        """
        # Strassen: 7 products
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A10)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B11+B01)
        # C00 = M1+M4-M5+M7
        # C01 = M3+M5
        # C10 = M2+M4
        # C11 = M1-M2+M3+M6
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # U[i*2+j, c] for i in {0,1}, j in {0,1}
        # V[j*2+k, c] for j in {0,1}, k in {0,1}
        # W[k*2+i, c] for k in {0,1}, i in {0,1}
        
        # M1 = (A00+A11)(B00+B11)
        # U: A00=U[0,0]=1, A11=U[3,0]=1
        # V: B00=V[0,0]=1, B11=V[3,0]=1
        # W: C00+=1, C11+=1 -> W[0,0]=1 (k=0,i=0), W[3,0]=1 (k=1,i=1)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (A10+A11)B00
        # U: A10=U[2,1]=1, A11=U[3,1]=1
        # V: B00=V[0,1]=1
        # W: C10+=1 -> W[2,1]=1 (k=1,i=0)
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1
        
        # M3 = A00(B01-B11)
        # U: A00=U[0,2]=1
        # V: B01=V[1,2]=1, B11=V[3,2]=-1
        # W: C01+=1 -> W[1,2]=1 (k=1,i=0)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1
        
        # M4 = A11(B10-B00)
        # U: A11=U[3,3]=1
        # V: B10=V[2,3]=1, B00=V[0,3]=-1
        # W: C00+=1, C11+=1 -> W[0,3]=1, W[3,3]=1
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[3, 3] = 1
        
        # M5 = (A00+A10)B11
        # U: A00=U[0,4]=1, A10=U[2,4]=1
        # V: B11=V[3,4]=1
        # W: C00-=1, C01+=1 -> W[0,4]=-1, W[1,4]=1
        U[0, 4] = 1; U[2, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6 = (A10-A00)(B00+B10)
        # U: A10=U[2,5]=1, A00=U[0,5]=-1
        # V: B00=V[0,5]=1, B10=V[2,5]=1
        # W: C11+=1 -> W[3,5]=1
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[2, 5] = 1
        W[3, 5] = 1
        
        # M7 = (A01-A11)(B11+B01)
        # U: A01=U[1,6]=1, A11=U[3,6]=-1
        # V: B11=V[3,6]=1, B01=V[1,6]=1
        # W: C00+=1, C11-=1 -> W[0,6]=1, W[3,6]=-1
        U[1, 6] = 1; U[3, 6] = -1
        V[3, 6] = 1; V[1, 6] = 1
        W[0, 6] = 1; W[3, 6] = -1
        
        return U, V, W
    
    # Build the full decomposition using block Strassen
    # i_groups: [{0,1}, {2,3}], k_groups: [{0,1}, {2,3}, {4,5}], j_groups: [{0,1}, {2,3}, {4}]
    
    i_groups = [(0, 1), (2, 3)]
    k_groups = [(0, 1), (2, 3), (4, 5)]
    j_groups = [(0, 1), (2, 3), (4, 4)]  # Last group is just {4}
    
    U_list = []
    V_list = []
    W_list = []
    
    for i_start, i_end in i_groups:
        for k_start, k_end in k_groups:
            for j_start, j_end in j_groups:
                i_size = i_end - i_start + 1
                k_size = k_end - k_start + 1
                j_size = j_end - j_start + 1
                
                if i_size == 2 and k_size == 2 and j_size == 2:
                    # 2x2x2 Strassen
                    U_sub, V_sub, W_sub = strassen_2x2x2()
                    # Map to global indices
                    U_g = np.zeros((20, 7))  # n*m = 4*5 = 20
                    V_g = np.zeros((30, 7))  # m*p = 5*6 = 30
                    W_g = np.zeros((24, 7))  # p*n = 6*4 = 24
                    
                    for ii in range(2):
                        for jj in range(2):
                            U_g[(i_start + ii) * 5 + (j_start + jj), :] = U_sub[ii * 2 + jj, :]
                    for jj in range(2):
                        for kk in range(2):
                            V_g[(j_start + jj) * 6 + (k_start + kk), :] = V_sub[jj * 2 + kk, :]
                    for kk in range(2):
                        for ii in range(2):
                            W_g[(k_start + kk) * 4 + (i_start + ii), :] = W_sub[kk * 2 + ii, :]
                    
                    U_list.append(U_g)
                    V_list.append(V_g)
                    W_list.append(W_g)
                    
                elif i_size == 2 and k_size == 2 and j_size == 1:
                    # 2x2x1: C[i,k] = A[i,j]*B[j,k], rank 2
                    U_g = np.zeros((20, 2))
                    V_g = np.zeros((30, 2))
                    W_g = np.zeros((24, 2))
                    for ii in range(2):
                        for kk in range(2):
                            c = ii * 2 + kk
                            U_g[(i_start + ii) * 5 + j_start, c] = 1
                            V_g[j_start * 6 + (k_start + kk), c] = 1
                            W_g[(k_start + kk) * 4 + (i_start + ii), c] = 1
                    U_list.append(U_g)
                    V_list.append(V_g)
                    W_list.append(W_g)
    
    U_total = np.hstack(U_list)
    V_total = np.hstack(V_list)
    W_total = np.hstack(W_list)
    
    total_rank = U_total.shape[1]
    
    if total_rank < best_rank:
        best_rank = total_rank
        best_U, best_V, best_W = U_total, V_total, W_total
    
    # Convert to list format with integer/half-integer entries
    def to_list(arr):
        # Round to nearest half-integer
        arr_rounded = np.round(arr * 2) / 2
        return arr_rounded.tolist()
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END

