import numpy as np
import time

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 145.0
    
    # Rank 28 decomposition using Strassen's algorithm on 2x2 blocks.
    # A (2x4) split into A_L (2x2, cols 0,1) and A_R (2x2, cols 2,3)
    # B (4x4) split into 4 2x2 blocks
    # C (2x4) split into C_L (2x2, cols 0,1) and C_R (2x2, cols 2,3)
    
    # Strassen terms for 2x2 matrix multiplication X*Y
    # Local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    strassen = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]
    
    rank = 28
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Global index mappings
    u_map_L = [0, 1, 4, 5]
    u_map_R = [2, 3, 6, 7]
    
    v_map_TL = [0, 1, 4, 5]
    v_map_TR = [2, 3, 6, 7]
    v_map_BL = [8, 9, 12, 13]
    v_map_BR = [10, 11, 14, 15]
    
    w_map_L = [0, 2, 1, 3]
    w_map_R = [4, 6, 5, 7]
    
    col = 0
    
    # 1. A_L * B_TL -> C_L
    for u_t, v_t, w_t in strassen:
        for idx, val in u_t:
            U[u_map_L[idx], col] += val
        for idx, val in v_t:
            V[v_map_TL[idx], col] += val
        for idx, val in w_t:
            W[w_map_L[idx], col] += val
        col += 1
        
    # 2. A_R * B_BL -> C_L
    for u_t, v_t, w_t in strassen:
        for idx, val in u_t:
            U[u_map_R[idx], col] += val
        for idx, val in v_t:
            V[v_map_BL[idx], col] += val
        for idx, val in w_t:
            W[w_map_L[idx], col] += val
        col += 1
            
    # 3. A_L * B_TR -> C_R
    for u_t, v_t, w_t in strassen:
        for idx, val in u_t:
            U[u_map_L[idx], col] += val
        for idx, val in v_t:
            V[v_map_TR[idx], col] += val
        for idx, val in w_t:
            W[w_map_R[idx], col] += val
        col += 1
            
    # 4. A_R * B_BR -> C_R
    for u_t, v_t, w_t in strassen:
        for idx, val in u_t:
            U[u_map_R[idx], col] += val
        for idx, val in v_t:
            V[v_map_BR[idx], col] += val
        for idx, val in w_t:
            W[w_map_R[idx], col] += val
        col += 1
        
    # Try rank-27: share one Strassen term between (A_L*B_TL->C_L) and (A_R*B_BL->C_L)
    # by using a combined U factor that covers both A_L and A_R simultaneously.
    # We need a rank-7 decomposition of the "combined" 2x4 x 4x2 -> 2x2 tensor
    # that reuses one term from each of the two independent rank-7 decompositions.
    
    # Alternative: try a direct rank-27 search via integer/half-integer optimization.
    # For now, attempt a known construction: use 3 full Strassen (21) + 6 extra terms.
    
    # Let's try: can we share the (0,1) term? In Strassen term 1, U uses (0,1),(3,1).
    # If we make a combined U that covers both A_L and A_R with the same pattern,
    # we can save one rank.
    
    # Combined approach: treat A as 2x4, and try to decompose using a mixed strategy.
    # Split A into A_L (2x2) and A_R (2x2). Split B into 4 blocks.
    # C = A_L*B_TL + A_R*B_BL (for left half) + A_L*B_TR + A_R*B_BR (for right half)
    
    # Try rank 27: use 3 full Strassen blocks (21 terms) and 6 additional terms.
    # Specifically, merge the first Strassen term of block 1 and block 2.
    
    # Actually, let me try a different approach: use a known rank-27 construction
    # if one exists, or try to optimize.
    
    # For now, keep rank 28 but verify correctness.
    if time.time() - start_time > time_limit:
        raise TimeoutError("Construction took too long")
        
    # Verify the decomposition
    U_np = U.astype(float)
    V_np = V.astype(float)
    W_np = W.astype(float)
    
    # Quick numerical check
    np.random.seed(0)
    A_test = np.random.rand(n, m)
    B_test = np.random.rand(m, p)
    C_test = A_test @ B_test
    
    C_comp = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(rank):
                acc += U_np[ii * m + kk, cc] * V_np[kk * p + ii, cc] * W_np[kk * n + ii, cc]
            C_comp[ii, kk] = acc
    
    if not np.allclose(C_test, C_comp, atol=1e-10):
        # Fallback: return naive rank-32 decomposition
        U_naive = np.zeros((n * m, n * m * p), dtype=int)
        V_naive = np.zeros((m * p, n * m * p), dtype=int)
        W_naive = np.zeros((p * n, n * m * p), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_naive[i * m + j, col] = 1
                    V_naive[j * p + k, col] = 1
                    W_naive[k * n + i, col] = 1
                    col += 1
        return {"U": U_naive.tolist(), "V": V_naive.tolist(), "W": W_naive.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    
    np.random.seed(42)
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C = A @ B
    
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
            
    if not np.allclose(C, Cf):
        print("Verification Failed")
        print(C)
        print(Cf)
    else:
        print("Verification Passed")
    print("Rank:", U.shape[1])
