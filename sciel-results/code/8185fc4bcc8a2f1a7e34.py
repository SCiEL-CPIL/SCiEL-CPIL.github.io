<<<<<<< SEARCH
tgt[(i * m + j, j * p + k, k * n + i)] = 1
=======
import time
import numpy as np

def construct():
    start_time = time.time()
    timeout = 110.0  # Safety margin before 120s limit
    
    # Dimensions
    n, m, p = 3, 4, 6
    
    # Target tensor T[i, j, k] = delta_{j,k} for C = A*B where A is n x m, B is m x p
    # Actually, the tensor for matrix multiplication C_{ik} = sum_j A_{ij} B_{jk}
    # The tensor T has indices i (0..n-1), j (0..m-1), k (0..p-1)
    # T[i, j, k] = 1 if j == k? No.
    # The multiplication tensor T is defined such that T(A,B) = C.
    # In the standard basis, T_{ijk} = 1 if the term A_{ij} B_{jk} contributes to C_{ik}.
    # So T[i, j, k] = 1 for all i, j, k? No.
    # Let's look at the trace convention mentioned: W[k*n+i, r].
    # The tensor is T_{i, j, k} such that C_{i,k} = sum_j T_{i,j,k} A_{i,j} B_{j,k}.
    # Thus T_{i,j,k} = 1 for all i, j, k.
    
    # We need to find rank-minimal decomposition.
    # The strategy suggests:
    # Partition A (3x4) into two 3x2 blocks: A1 (cols 0,1), A2 (cols 2,3)
    # Partition B (4x6) into two 2x6 blocks: B1 (rows 0,1), B2 (rows 2,3)
    # C = A1 B1 + A2 B2
    # Each term is (3x2) * (2x6).
    # We can use Strassen for 2x2, but here we have 3x2 and 2x6.
    # Wait, the strategy says "applying the standard rank-12 Strassen-based algorithm for 2x2 matrix multiplication to each of the two resulting 3x2x6 sub-problems".
    # This implies decomposing the 3x2 * 2x6 product.
    # A 3x2 * 2x6 product can be seen as summing over 2 middle indices.
    # C_sub = A_sub B_sub.
    # We can decompose this using rank 3*2*2 = 12? No, naive is 3*2*2=12.
    # Strassen for 2x2 is rank 7.
    # How to apply Strassen to 3x2 * 2x6?
    # We can view the 3x2 matrix as 3 rows of length 2.
    # We can view the 2x6 matrix as 2 rows of length 6.
    # This is not a square matrix multiplication.
    # However, we can embed 3x2 * 2x6 into a larger square multiplication or use a specific decomposition.
    # Alternatively, we can just use the naive decomposition for the sub-problems.
    # Naive rank for 3x2 * 2x6 is 3*2*2 = 12.
    # Total rank would be 12 + 12 = 24.
    # The task asks for rank-minimal. Is 24 minimal?
    # The total naive rank is 3*4*6 = 72.
    # 24 is much better.
    # Can we do better?
    # The strategy explicitly says "Construct a rank-24 decomposition".
    # So we will implement the rank-24 decomposition.
    
    # U shape: (n*m, rank) = (12, 24)
    # V shape: (m*p, rank) = (24, 24)
    # W shape: (p*n, rank) = (18, 24)
    
    rank = 24
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Block 1: A1 (rows 0..2, cols 0..1) * B1 (rows 0..1, cols 0..5)
    # Indices in U: i*m + j. For A1, i in 0..2, j in 0..1.
    # Indices in V: j*p + k. For B1, j in 0..1, k in 0..5.
    # Indices in W: k*n + i. For C, k in 0..5, i in 0..2.
    
    # Naive decomposition for A1 B1:
    # For each i in 0..2, j in 0..1, k in 0..5:
    # Term: A1[i,j] * B1[j,k] -> C[i,k]
    # This corresponds to a rank-1 tensor.
    # U[i*m + j, r] = 1
    # V[j*p + k, r] = 1
    # W[k*n + i, r] = 1
    
    # We need to map these to the global rank indices.
    # Let's assign ranks 0..11 to the first block and 12..23 to the second block.
    
    r_offset = 0
    for i in range(3):
        for j in range(2):
            for k in range(6):
                u_idx = i * m + j  # i*4 + j
                v_idx = j * p + k  # j*6 + k
                w_idx = k * n + i  # k*3 + i
                
                U[u_idx, r_offset] = 1.0
                V[v_idx, r_offset] = 1.0
                W[w_idx, r_offset] = 1.0
                r_offset += 1
    
    # Block 2: A2 (rows 0..2, cols 2..3) * B2 (rows 2..3, cols 0..5)
    # A2 corresponds to A[:, 2:4]
    # B2 corresponds to B[2:4, :]
    # In global indices:
    # A2[i, j_local] where j_local in 0..1 corresponds to global j = 2 + j_local.
    # B2[j_local, k] where j_local in 0..1 corresponds to global j = 2 + j_local.
    
    r_offset = 12
    for i in range(3):
        for j_local in range(2):
            j_global = 2 + j_local
            for k in range(6):
                u_idx = i * m + j_global  # i*4 + (2+j_local)
                v_idx = j_global * p + k  # (2+j_local)*6 + k
                w_idx = k * n + i  # k*3 + i
                
                U[u_idx, r_offset] = 1.0
                V[v_idx, r_offset] = 1.0
                W[w_idx, r_offset] = 1.0
                r_offset += 1
    
    # Check time
    if time.time() - start_time > timeout:
        # If we somehow ran out of time (unlikely for this static construction), return what we have
        pass
        
    return {'U': U, 'V': V, 'W': W}
>>>>>>> REPLACE
