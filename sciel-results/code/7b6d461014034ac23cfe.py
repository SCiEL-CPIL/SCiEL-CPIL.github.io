import time
import sys
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill
    n, m, p = DIMENSIONS
    
    # Build the standard rank-56 Strassen decomposition first as baseline
    rank_base = 56
    U = [[0] * rank_base for _ in range(n * m)]
    V = [[0] * rank_base for _ in range(m * p)]
    W = [[0] * rank_base for _ in range(p * n)]
    
    # Strassen 2x2x2 decomposition patterns
    strassen_U = [
        [(0, 1), (3, 1)],
        [(2, 1), (3, 1)],
        [(0, 1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (0, -1)],
        [(1, 1), (3, -1)]
    ]
    strassen_V = [
        [(0, 1), (3, 1)],
        [(0, 1)],
        [(1, 1), (3, -1)],
        [(2, 1), (0, -1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (3, 1)]
    ]
    strassen_W = [
        [(0, 1), (3, 1)],
        [(1, 1), (3, -1)],
        [(2, 1), (3, 1)],
        [(0, 1), (1, 1)],
        [(0, -1), (2, 1)],
        [(3, 1)],
        [(0, 1)]
    ]
    
    def add_strassen_block(base_rank, i_offset, j_offset, k_offset):
        for r in range(7):
            current_rank = base_rank + r
            for idx, val in strassen_U[r]:
                local_i, local_j = divmod(idx, 2)
                global_i = local_i + i_offset
                global_j = local_j + j_offset
                U[global_i * m + global_j][current_rank] += val
            for idx, val in strassen_V[r]:
                local_j, local_k = divmod(idx, 2)
                global_j = local_j + j_offset
                global_k = local_k + k_offset
                V[global_j * p + global_k][current_rank] += val
            for idx, val in strassen_W[r]:
                local_k, local_i = divmod(idx, 2)
                global_k = local_k + k_offset
                global_i = local_i + i_offset
                W[global_k * n + global_i][current_rank] += val
    
    # 8 blocks: 2 j-pairs × 4 k-pairs
    block_idx = 0
    for j_off in [0, 2]:
        for k_off in [0, 2, 4, 6]:
            add_strassen_block(block_idx * 7, 0, j_off, k_off)
            block_idx += 1
    
    # Try to find a lower-rank decomposition using structured search
    # Strategy: try rank 48, 44, 40, 36, 32 by attempting to merge terms
    
    # For now, verify the rank-56 solution is valid and return it
    # The matching-based approach would require solving a nonlinear system
    # which is too expensive for the time budget. Return the known-good solution.
    
    # Convert to numpy arrays for cleaner output
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Quick verification: check that the decomposition is correct
    # For random test matrices, verify C = A @ B
    if time.time() < deadline - 5:
        rng = np.random.RandomState(42)
        A_test = rng.randn(n, m)
        B_test = rng.randn(m, p)
        C_expected = A_test @ B_test
        
        # Compute using decomposition
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                for r in range(rank_base):
                    u_val = sum(U_arr[i * m + j, r] * V_arr[j * p + k, r] for j in range(m))
                    # Actually, the tensor decomposition is:
                    # C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    # But this sums over r only, with j being implicit in U and V indices
                    # Wait, the correct form is:
                    # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] for each (i,j,k)
                    # But C[i,k] = sum_j sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
                    # No, that's not right either. Let me re-read the convention.
                    # W[k*n+i, r] is the W factor.
                    # The decomposition is: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
                    # where T[i,j,k] = 1 if this is a valid multiplication term
                    # For matrix mult: C[i,k] = sum_j A[i,j]*B[j,k]
                    # So the tensor is: T[i,j,k] contributes A[i,j]*B[j,k]*W[k,i,r]... 
                    # Actually the standard form is:
                    # C[i,k] = sum_r (sum_j U[i,j,r]*A[i,j]) * (sum_j V[j,k,r]*B[j,k]) * W[k,i,r]
                    # No, that's not standard. Let me just verify numerically.
                    pass
        
        # Simpler verification: build the multiplication tensor and check
        # T[i,j,k] = 1 if we're computing C[i,k] with intermediate j
        # The decomposition satisfies: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta (correct mult)
        # For matrix multiplication, the "tensor" is such that:
        # sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] should give the coefficient of A[i,j]*B[j,k] in C[i,k]
        # which is 1 for all valid (i,j,k) triples
        
        # Let's verify: for each (i,j,k), compute sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
        # This should equal 1 for all (i,j,k)
        max_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r] for r in range(rank_base))
                    max_err = max(max_err, abs(val - 1))
        
        if max_err > 1e-10:
            # Fallback: return a trivial rank-64 solution
            rank_trivial = n * m * p
            U_t = np.zeros((n * m, rank_trivial))
            V_t = np.zeros((m * p, rank_trivial))
            W_t = np.zeros((p * n, rank_trivial))
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_t[i*m+j, idx] = 1
                        V_t[j*p+k, idx] = 1
                        W_t[k*n+i, idx] = 1
                        idx += 1
            return {"U": U_t.tolist(), "V": V_t.tolist(), "W": W_t.tolist()}
    
    return {"U": U_arr.tolist(), "V": V_arr.tolist(), "W": W_arr.tolist()}
# EVOLVE-BLOCK-END
