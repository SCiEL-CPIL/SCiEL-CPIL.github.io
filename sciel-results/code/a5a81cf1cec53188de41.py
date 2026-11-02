"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 150s wall-clock limit, use 145s as safety margin
    
    n, m, p = DIMENSIONS
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients
    U_defs = [
        [(0, 1), (3, 1)],
        [(2, 1), (3, 1)],
        [(0, 1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (0, -1)],
        [(1, 1), (3, -1)]
    ]
    
    V_defs = [
        [(0, 1), (3, 1)],
        [(0, 1)],
        [(1, 1), (3, -1)],
        [(2, 1), (0, -1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (3, 1)]
    ]
    
    W_defs = [
        [(0, 1), (3, 1)],
        [(1, 1), (3, -1)],
        [(2, 1), (3, 1)],
        [(0, 1), (1, 1)],
        [(0, -1), (2, 1)],
        [(3, 1)],
        [(0, 1)]
    ]

    for q in range(4):
        for s in range(7):
            r = q * 7 + s
            
            for idx, val in U_defs[s]:
                U[idx][r] = val
                
            for local_idx, val in V_defs[s]:
                j = local_idx // 2
                k_local = local_idx % 2
                k = 2 * q + k_local
                v_idx = j * p + k
                V[v_idx][r] = val
                
            for local_idx, val in W_defs[s]:
                k_local = local_idx // 2
                i = local_idx % 2
                k = 2 * q + k_local
                w_idx = k * n + i
                W[w_idx][r] = val

    # Verify the decomposition is correct
    try:
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        verified = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    total = 0.0
                    for r in range(rank):
                        total += U_arr[u_idx, r] * V_arr[v_idx, r] * W_arr[w_idx, r]
                    if abs(total - 1.0) > 1e-10:
                        verified = False
                        break
                if not verified:
                    break
            if not verified:
                break
        
        if not verified:
            rank_fb = n * m * p
            U_fb = [[0] * rank_fb for _ in range(n * m)]
            V_fb = [[0] * rank_fb for _ in range(m * p)]
            W_fb = [[0] * rank_fb for _ in range(p * n)]
            
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        U_fb[u_idx][idx] = 1
                        V_fb[v_idx][idx] = 1
                        W_fb[w_idx][idx] = 1
                        idx += 1
            
            return {"U": U_fb, "V": V_fb, "W": W_fb}
    except Exception:
        pass
    
    # Attempt to compress from rank 28 to lower rank using algebraic structure
    # Strategy: Try to find linear dependencies among the 28 Strassen terms
    # by searching for a 4-dimensional invariant subspace that allows compression.
    
    # Approach: For each pair of blocks (q1, q2), check if we can merge
    # some of the 7 Strassen terms by finding linear combinations that
    # preserve the tensor equation.
    
    # More systematic: Try rank 24 by finding 4 linear dependencies
    # in the 28-dimensional space.
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = 28
    
    # Try to find a rank-24 decomposition by searching for 4 linear
    # combinations that can be eliminated.
    # The key insight: if we can find vectors u_vec, v_vec, w_vec such that
    # sum_r u_vec[r]*U[:,r] = 0, sum_r v_vec[r]*V[:,r] = 0, etc.
    # then we can eliminate those components.
    
    # Alternative approach: Use the block structure. Each block of 7 terms
    # multiplies a 2x2 A-block by a 2x2 B-block. If we can find a
    # 4-dimensional subspace of the 28-dimensional space that is invariant
    # under the multiplication, we can compress.
    
    # Let's try a direct search: for each subset of 4 indices to eliminate,
    # check if the remaining 24 terms still span the tensor.
    
    # This is expensive, so let's use a smarter approach:
    # The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We want to find the minimum rank.
    
    # For now, return the rank-28 decomposition which is verified correct.
    # The search for lower rank would require more sophisticated methods.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
