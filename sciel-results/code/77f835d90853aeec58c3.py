import time
import sys
import numpy as np

N, M, P = 2, 4, 7
# Output ordering: W[k*n + i, r]  => W index = k*N + i
# U index = i*M + j   (A[i,j])
# V index = j*P + k   (B[j,k])

def _strassen_2x2():
    """Return U, V, W factors for 2x2x2 Strassen decomposition (rank 7).
    U: (4, 7) indexing [i*2+j]
    V: (4, 7) indexing [j*2+k]
    W: (4, 7) indexing [k*2+i]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Indices: a=0, b=1, c=2, d=3 (for U: i*2+j)
    #           e=0, f=1, g=2, h=3 (for V: j*2+k)
    #           c00=0, c01=1, c10=2, c11=3 (for W: k*2+i)
    
    # M1 = (a+d)(e+h) -> C00 += M1, C11 += M1
    U[0,0] += 1; U[3,0] += 1
    V[0,0] += 1; V[3,0] += 1
    W[0,0] += 1;  W[3,0] += 1
    
    # M2 = (c+d)e -> C10 += M2, C11 -= M2
    U[2,1] += 1; U[3,1] += 1
    V[0,1] += 1
    W[2,1] += 1;  W[3,1] -= 1
    
    # M3 = a(f-h) -> C01 += M3, C11 += M3
    U[0,2] += 1
    V[1,2] += 1; V[3,2] -= 1
    W[1,2] += 1;  W[3,2] += 1
    
    # M4 = d(g-e) -> C00 += M4, C10 += M4
    U[3,3] += 1
    V[2,3] += 1; V[0,3] -= 1
    W[0,3] += 1;  W[2,3] += 1
    
    # M5 = (a+b)h -> C00 -= M5, C01 += M5
    U[0,4] += 1; U[1,4] += 1
    V[3,4] += 1
    W[0,4] -= 1;  W[1,4] += 1
    
    # M6 = (c-a)(e+f) -> C11 += M6
    U[2,5] += 1; U[0,5] -= 1
    V[0,5] += 1; V[1,5] += 1
    W[3,5] += 1
    
    # M7 = (b-d)(g+h) -> C00 += M7
    U[1,6] += 1; U[3,6] -= 1
    V[2,6] += 1; V[3,6] += 1
    W[0,6] += 1
    
    return U, V, W

def _strassen_2x2x1():
    """Return U, V, W factors for 2x2x1 schoolbook decomposition (rank 4).
    U: (4, 4) indexing [i*2+j]
    V: (4, 4) indexing [j*2+k] where k is fixed (last column)
    W: (2, 4) indexing [k*2+i] where k is fixed
    """
    U = np.zeros((4, 4))
    V = np.zeros((4, 4))
    W = np.zeros((2, 4))
    
    # P1 = a*e -> C0 += P1
    U[0,0] = 1; V[0,0] = 1; W[0,0] = 1
    # P2 = b*g -> C0 += P2
    U[1,1] = 1; V[2,1] = 1; W[0,1] = 1
    # P3 = c*e -> C1 += P3
    U[2,2] = 1; V[0,2] = 1; W[1,2] = 1
    # P4 = d*g -> C1 += P4
    U[3,3] = 1; V[2,3] = 1; W[1,3] = 1
    
    return U, V, W

def construct():
    t0 = time.time()
    DEADLINE = t0 + 100  # safety margin under 120s
    
    # Strategy: Decompose 2x4x7 as:
    # - Two 2x2x2 Strassen blocks for j-pairs (0,1) and (2,3) with k-pairs (0,1),(2,3),(4,5)
    #   Each 2x2x2 block has rank 7, and we have 3 such blocks per j-pair = 21 per j-pair
    #   But we can use Kronecker structure to be more efficient.
    
    # Alternative approach: Use the structure A(2x4) * B(4x7)
    # Split B into B1(4x6) and B2(4x1)
    # A * B1: 2x4x6 = can use 2x2x2 Strassen for each of 3 k-pairs per j-pair
    #   j-pair (0,1): 3 Strassen blocks * 7 = 21
    #   j-pair (2,3): 3 Strassen blocks * 7 = 21
    # A * B2: 2x4x1 = schoolbook, 2 j-pairs * 4 = 8
    # Total: 21 + 21 + 8 = 50 (same as before)
    
    # Better: Use Kronecker product structure
    # The 2x4x7 tensor can be seen as:
    # For each j-pair (j0, j0+1) where j0 in {0, 2}:
    #   A[:, j0:j0+2] (2x2) * B[j0:j0+2, :] (2x7)
    # This is a 2x2x7 multiplication
    
    # 2x2x7 can be decomposed as:
    # - Three 2x2x2 Strassen blocks (for k-pairs 0-1, 2-3, 4-5): rank 3*7 = 21
    # - One 2x2x1 schoolbook block (for k=6): rank 4
    # Total per j-pair: 25
    # Two j-pairs: 50
    
    # To improve, we need a smarter decomposition of 2x2x7
    # Known: rank(2x2x7) <= 3*rank(2x2x2) + rank(2x2x1) = 21+4 = 25
    # But there might be better ways...
    
    # Let's try: Use Strassen for 2x2x2 and combine with the last column
    # Actually, let's try a different approach:
    # View 2x2x7 as 2x2x(6+1) where 6 = 3*2
    # Use 3 Strassen blocks + 1 schoolbook = 25 per j-pair
    
    # For now, let's implement the Kronecker-based approach cleanly
    # and see if we can find redundancies
    
    U, V, W = _strassen_2x2()
    U2, V2, W2 = _strassen_2x2x1()
    
    # Build full decomposition using Kronecker-like structure
    # Total rank estimate: 2 j-pairs * (3*7 + 4) = 2*25 = 50
    
    total_rank = 50
    U_full = np.zeros((N*M, total_rank))
    V_full = np.zeros((M*P, total_rank))
    W_full = np.zeros((P*N, total_rank))
    
    col = 0
    for j0 in (0, 2):
        # For each of the 3 k-pairs (0,1), (2,3), (4,5): use Strassen 2x2x2
        for k0 in (0, 2, 4):
            # Map Strassen U indices [i*2+j] to full U index [i*M+j]
            # j in Strassen corresponds to j0+j
            for i in range(2):
                for j_local in range(2):
                    j_full = j0 + j_local
                    u_idx = i * M + j_full
                    u_val = U[i*2+j_local, col - (col - 0)]  # This won't work directly
            
            # Let's do it properly:
            # Strassen U has shape (4, 7), indices [i*2+j_local]
            # We need to map j_local to j_full = j0 + j_local
            for i in range(2):
                for j_local in range(2):
                    j_full = j0 + j_local
                    u_idx = i * M + j_full
                    u_val = U[i*2 + j_local, 0]  # We need to add all 7 columns
            
            # This is getting complex. Let me rebuild from scratch with explicit indexing.
            pass
        
        # Let me restart with a cleaner approach
        break
    
    # Clean rebuild
    U_full = np.zeros((N*M, total_rank))
    V_full = np.zeros((M*P, total_rank))
    W_full = np.zeros((P*N, total_rank))
    
    col = 0
    for j0 in (0, 2):
        for k0 in (0, 2, 4):
            # Add Strassen 2x2x2 block
            for r in range(7):
                # U: map [i*2+j_local] -> [i*M + (j0+j_local)]
                for i in range(2):
                    for j_local in range(2):
                        j_full = j0 + j_local
                        u_idx = i * M + j_full
                        U_full[u_idx, col] += U[i*2 + j_local, r]
                
                # V: map [j_local*2 + k_local] -> [(j0+j_local)*P + (k0+k_local)]
                for j_local in range(2):
                    for k_local in range(2):
                        j_full = j0 + j_local
                        k_full = k0 + k_local
                        v_idx = j_full * P + k_full
                        V_full[v_idx, col] += V[j_local*2 + k_local, r]
                
                # W: map [k_local*2 + i] -> [(k0+k_local)*N + i]
                for k_local in range(2):
                    for i in range(2):
                        k_full = k0 + k_local
                        w_idx = k_full * N + i
                        W_full[w_idx, col] += W[k_local*2 + i, r]
                
                col += 1
        
        # Add schoolbook 2x2x1 block for k=6
        for r in range(4):
            # U: map [i*2+j_local] -> [i*M + (j0+j_local)]
            for i in range(2):
                for j_local in range(2):
                    j_full = j0 + j_local
                    u_idx = i * M + j_full
                    U_full[u_idx, col] += U2[i*2 + j_local, r]
            
            # V: map [j_local*2 + k_local] -> [(j0+j_local)*P + 6] where k_local=0
            for j_local in range(2):
                j_full = j0 + j_local
                v_idx = j_full * P + 6
                V_full[v_idx, col] += V2[j_local*2 + 0, r]
            
            # W: map [i] -> [6*N + i]
            for i in range(2):
                w_idx = 6 * N + i
                W_full[w_idx, col] += W2[i, r]
            
            col += 1
    
    # Convert to lists
    U_list = U_full.tolist()
    V_list = V_full.tolist()
    W_list = W_full.tolist()
    
    # Remove zero columns
    rank = len(U_list[0])
    keep = []
    for r in range(rank):
        col_u = [U_list[i][r] for i in range(len(U_list))]
        col_v = [V_list[i][r] for i in range(len(V_list))]
        col_w = [W_list[i][r] for i in range(len(W_list))]
        if any(x != 0 for x in col_u) and any(x != 0 for x in col_v) and any(x != 0 for x in col_w):
            keep.append(r)
    
    if len(keep) < rank:
        U_list = [[U_list[i][r] for r in keep] for i in range(len(U_list))]
        V_list = [[V_list[i][r] for r in keep] for i in range(len(V_list))]
        W_list = [[W_list[i][r] for r in keep] for i in range(len(W_list))]
    
    return {"U": U_list, "V": V_list, "W": W_list}
