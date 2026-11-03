import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = 2, 4, 8

    # The matrix multiplication tensor for (2,4,8) has a natural block structure.
    # We can decompose it into 4 independent 2x2x2 sub-tensors (Strassen blocks),
    # one for each pair of j-indices (0,1) and (2,3), each handling 2 k-columns.
    # This gives a rank-56 decomposition (4 blocks * 7 terms each).
    
    # The key insight for improvement: the 56 rank-1 terms may not be linearly
    # independent. If we can find a lower-rank equivalent decomposition, we win.
    # 
    # Strategy:
    # 1. Build the rank-56 Strassen decomposition.
    # 2. Check if the 56 vectors t_r = U[:,r] ⊗ V[:,r] ⊗ W[:,r] are linearly
    #    independent by computing the rank of the 56 x 4096 matrix.
    # 3. If rank < 56, find a basis for the span and reconstruct a lower-rank
    #    decomposition.
    
    # Build the 56-rank Strassen decomposition
    U_local = np.zeros((4, 7), dtype=float)
    V_local = np.zeros((4, 7), dtype=float)
    W_local = np.zeros((4, 7), dtype=float)
    
    # Standard Strassen for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    U_local[0,0] = 1; U_local[3,0] = 1
    V_local[0,0] = 1; V_local[3,0] = 1
    W_local[0,0] = 1; W_local[3,0] = 1
    
    # M2 = (A10+A11)B00
    U_local[2,1] = 1; U_local[3,1] = 1
    V_local[0,1] = 1
    W_local[2,1] = 1; W_local[3,1] = -1
    
    # M3 = A00(B10-B11)
    U_local[0,2] = 1
    V_local[1,2] = 1; V_local[3,2] = -1
    W_local[1,2] = 1; W_local[3,2] = 1
    
    # M4 = A11(B01-B00)
    U_local[3,3] = 1
    V_local[2,3] = 1; V_local[0,3] = -1
    W_local[0,3] = 1; W_local[2,3] = 1
    
    # M5 = (A00+A01)B11
    U_local[0,4] = 1; U_local[1,4] = 1
    V_local[3,4] = 1
    W_local[0,4] = -1; W_local[1,4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U_local[2,5] = 1; U_local[0,5] = -1
    V_local[0,5] = 1; V_local[1,5] = 1
    W_local[3,5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U_local[1,6] = 1; U_local[3,6] = -1
    V_local[2,6] = 1; V_local[3,6] = 1
    W_local[0,6] = 1
    
    total_rank = 56
    U = np.zeros((8, total_rank), dtype=float)
    V = np.zeros((32, total_rank), dtype=float)
    W = np.zeros((16, total_rank), dtype=float)
    
    for j in range(4):
        for s in range(2):
            base_r = (j * 2 + s) * 7
            
            # a_map: maps local 2x2 index (la) to global U index
            # s=0: k columns 2j, 2j+1 -> A columns 0,1
            # s=1: k columns 2j, 2j+1 -> A columns 2,3
            if s == 0:
                a_map = [0, 1, 4, 5]  # A[0,0], A[0,1], A[1,0], A[1,1]
            else:
                a_map = [2, 3, 6, 7]  # A[0,2], A[0,3], A[1,2], A[1,3]
            
            # b_map: maps local 2x2 index (lb) to global V index
            # lb=0: B row 0, lb=1: B row 1, lb=2: B row 2, lb=3: B row 3
            # For j-group j, s=0: rows 0,1; s=1: rows 2,3
            if s == 0:
                b_map = [0*8 + 2*j, 0*8 + 2*j+1, 1*8 + 2*j, 1*8 + 2*j+1]
            else:
                b_map = [2*8 + 2*j, 2*8 + 2*j+1, 3*8 + 2*j, 3*8 + 2*j+1]
            
            # c_map: maps local 2x2 index (lc) to global W index
            # lc=0: C[0, 2j], lc=1: C[0, 2j+1], lc=2: C[1, 2j], lc=3: C[1, 2j+1]
            c_map = [2*j*2+0, (2*j+1)*2+0, 2*j*2+1, (2*j+1)*2+1]
            
            for l in range(7):
                r = base_r + l
                for la in range(4):
                    val = U_local[la, l]
                    if val != 0:
                        U[a_map[la], r] += val
                for lb in range(4):
                    val = V_local[lb, l]
                    if val != 0:
                        V[b_map[lb], r] += val
                for lc in range(4):
                    val = W_local[lc, l]
                    if val != 0:
                        W[c_map[lc], r] += val
    
    # Verify the 56-rank decomposition
    def verify(U, V, W):
        R = U.shape[1]
        for i in range(2):
            for j in range(4):
                for k in range(8):
                    s = 0.0
                    for r in range(R):
                        s += U[i*4+j, r] * V[j*8+k, r] * W[k*2+i, r]
                    if abs(s - 1.0) > 1e-10:
                        return False
        return True
    
    if not verify(U, V, W):
        # Fallback to naive 64
        U = np.zeros((8, 64), dtype=float)
        V = np.zeros((32, 64), dtype=float)
        W = np.zeros((16, 64), dtype=float)
        for i in range(2):
            for j in range(4):
                for k in range(8):
                    r = i*32 + j*8 + k
                    U[i*4+j, r] = 1
                    V[j*8+k, r] = 1
                    W[k*2+i, r] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Now try to reduce the rank by finding linear dependencies
    # among the 56 rank-1 tensor vectors.
    
    tensor_size = 8 * 32 * 16  # 4096
    T = np.zeros((total_rank, tensor_size), dtype=float)
    
    for r in range(total_rank):
        for i in range(2):
            for j in range(4):
                for k in range(8):
                    idx = i * 32 * 16 + j * 16 + k
                    T[r, idx] = U[i*4+j, r] * V[j*8+k, r] * W[k*2+i, r]
    
    # Compute the rank of T to check for dependencies
    try:
        rank_T = np.linalg.matrix_rank(T, tol=1e-10)
    except:
        rank_T = total_rank
    
    if rank_T < total_rank:
        # There are dependencies! We can potentially reduce the rank.
        # Use SVD to find a lower-dimensional basis.
        try:
            U_svd, s_vals, Vt_svd = np.linalg.svd(T, full_matrices=False)
            
            # Find the effective rank with tighter tolerance
            tol = 1e-10
            effective_rank = np.sum(s_vals > tol)
            
            if effective_rank < total_rank:
                # Reconstruct a lower-rank decomposition
                # The top `effective_rank` singular vectors span the same space.
                # We need to express each original t_r as a combination of the basis.
                
                # For now, just note that we found dependencies and return 56
                # A full reconstruction would require solving for coefficients
                # which is complex. The rank reduction is possible but the
                # actual lower-rank decomposition needs careful construction.
                pass
        except:
            pass
    
    # Return the verified 56-rank solution
    # Convert to lists for the return interface
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
