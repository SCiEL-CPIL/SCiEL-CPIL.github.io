import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140  # seconds, safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # n=2, m=4, p=8
    
    # Strassen decomposition for 2x2 matrix multiplication (rank 7)
    # Local indices:
    # A: 2x2 -> flat i*2+j
    # B: 2x2 -> flat j*2+k
    # C: 2x2 -> flat i*2+k (local), mapped to W convention k*n+i globally
    
    # Strassen components:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    U_local = [[0]*7 for _ in range(4)]
    V_local = [[0]*7 for _ in range(4)]
    W_local = [[0]*7 for _ in range(4)]
    
    # r=0 (M1): (a00+a11)(b00+b11)
    U_local[0][0] = 1; U_local[3][0] = 1
    V_local[0][0] = 1; V_local[3][0] = 1
    W_local[0][0] = 1  # C00
    W_local[3][0] = 1  # C11
    
    # r=1 (M2): (a10+a11)b00
    U_local[2][1] = 1; U_local[3][1] = 1
    V_local[0][1] = 1
    W_local[2][1] = 1   # C10
    W_local[3][1] = -1  # C11
    
    # r=2 (M3): a00(b01-b11)
    U_local[0][2] = 1
    V_local[1][2] = 1; V_local[3][2] = -1
    W_local[1][2] = 1   # C01
    W_local[3][2] = 1   # C11
    
    # r=3 (M4): a11(b10-b00)
    U_local[3][3] = 1
    V_local[2][3] = 1; V_local[0][3] = -1
    W_local[0][3] = 1   # C00
    W_local[2][3] = 1   # C10
    
    # r=4 (M5): (a00+a01)b11
    U_local[0][4] = 1; U_local[1][4] = 1
    V_local[3][4] = 1
    W_local[0][4] = -1  # C00
    W_local[1][4] = 1   # C01
    
    # r=5 (M6): (a10-a00)(b00+b01)
    U_local[2][5] = 1; U_local[0][5] = -1
    V_local[0][5] = 1; V_local[1][5] = 1
    W_local[3][5] = 1   # C11
    
    # r=6 (M7): (a01-a11)(b10+b11)
    U_local[1][6] = 1; U_local[3][6] = -1
    V_local[2][6] = 1; V_local[3][6] = 1
    W_local[0][6] = 1   # C00

    total_rank = 56
    
    U = [[0] * total_rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * total_rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * total_rank for _ in range(p * n)]  # 16 x 56
    
    # Decompose 2x4 x 4x8 into 8 blocks of 2x2 x 2x2
    for j in range(4):
        for s in range(2):
            base_r = (j * 2 + s) * 7
            
            a_map = [
                0 * 4 + (2 * s + 0),
                0 * 4 + (2 * s + 1),
                1 * 4 + (2 * s + 0),
                1 * 4 + (2 * s + 1),
            ]
            
            b_map = [
                (2 * s + 0) * 8 + (2 * j + 0),
                (2 * s + 0) * 8 + (2 * j + 1),
                (2 * s + 1) * 8 + (2 * j + 0),
                (2 * s + 1) * 8 + (2 * j + 1),
            ]
            
            c_map = [
                (2 * j + 0) * 2 + 0,
                (2 * j + 1) * 2 + 0,
                (2 * j + 0) * 2 + 1,
                (2 * j + 1) * 2 + 1,
            ]
            
            for l in range(7):
                r = base_r + l
                for la in range(4):
                    val = U_local[la][l]
                    if val != 0:
                        U[a_map[la]][r] += val
                for lb in range(4):
                    val = V_local[lb][l]
                    if val != 0:
                        V[b_map[lb]][r] += val
                for lc in range(4):
                    val = W_local[lc][l]
                    if val != 0:
                        W[c_map[lc]][r] += val
                        
    # Convert to numpy
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Build tensor matrix: M[i,j,k, r] = U[i,r] * V[j,r] * W[k,r]
    # Shape: (n*m*p) x total_rank = 64 x 56
    M = np.zeros((n * m * p, total_rank))
    for r in range(total_rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    M[i * m * p + j * p + k, r] = U_np[i, r] * V_np[j, r] * W_np[k, r]
    
    # Find null space to reduce rank
    try:
        U_svd, s_vals, Vt_svd = np.linalg.svd(M, full_matrices=True)
        tol = 1e-10
        rank_m = int(np.sum(s_vals > tol))
        
        if rank_m < total_rank:
            # Null space vectors give linear dependencies among rank-1 terms
            # null_space: 56 x (56 - rank_m)
            null_space = Vt_svd[rank_m:, :].T
            
            # Greedy elimination: for each null vector, eliminate the term with max |c_r|
            # Keep track of which terms are still active
            active = list(range(total_rank))
            eliminated = set()
            
            for nc in range(null_space.shape[1]):
                c = null_space[:, nc]
                # Find active index with largest |c_r|
                best_r = -1
                best_val = 0
                for r in active:
                    if r not in eliminated:
                        val = abs(c[r])
                        if val > best_val:
                            best_val = val
                            best_r = r
                if best_r >= 0 and best_val > tol:
                    eliminated.add(best_r)
            
            # Build new factor matrices using only non-eliminated terms
            keep = [r for r in range(total_rank) if r not in eliminated]
            new_rank = len(keep)
            
            U_new = [[0] * new_rank for _ in range(n * m)]
            V_new = [[0] * new_rank for _ in range(m * p)]
            W_new = [[0] * new_rank for _ in range(p * n)]
            
            for new_idx, old_r in enumerate(keep):
                for i in range(n * m):
                    U_new[i][new_idx] = U[i][old_r]
                for j in range(m * p):
                    V_new[j][new_idx] = V[j][old_r]
                for k in range(p * n):
                    W_new[k][new_idx] = W[k][old_r]
            
            # Verify correctness
            U_v = np.array(U_new, dtype=float)
            V_v = np.array(V_new, dtype=float)
            W_v = np.array(W_new, dtype=float)
            
            M_verify = np.zeros((n * m * p, new_rank))
            for r in range(new_rank):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            M_verify[i * m * p + j * p + k, r] = U_v[i, r] * V_v[j, r] * W_v[k, r]
            
            # Check that the tensor is still exactly represented
            # The original tensor should be all ones for the matrix mult pattern
            # Verify: sum_r U[i,r]*V[j,r]*W[k,r] should equal 1 for all i,j,k
            tensor_recon = np.zeros((n, m, p))
            for r in range(new_rank):
                tensor_recon += np.outer(np.outer(U_v[:, r], V_v[:, r]), W_v[:, r])
            
            # For matrix multiplication, T[i,j,k] = 1 for all valid i,j,k
            max_err = np.max(np.abs(tensor_recon - 1.0))
            
            if max_err < 1e-8:
                U, V, W = U_new, V_new, W_new
    except Exception:
        pass
    
    if time.time() - start_time > deadline:
        pass
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
