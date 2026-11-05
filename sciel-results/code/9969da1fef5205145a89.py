import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START

def _strassen_factors():
    """
    Return list of (U_vec, V_vec, W_vec) for Strassen 2x2x2.
    """
    factors = []
    factors.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
    factors.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]))
    factors.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]))
    factors.append(([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]))
    factors.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]))
    factors.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]))
    factors.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]))
    return factors

def _verify(n, m, p, U, V, W, tol=1e-9):
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    rank = U_arr.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U_arr[i * m + j]
                v_row = V_arr[j * p + k]
                w_row = W_arr[k * n + i]
                val = np.dot(u_row, v_row * w_row)
                if abs(val - 1.0) > tol:
                    return False
    return True

def _construct_4x4_strassen_factors():
    """
    Generate rank-49 factors for 4x4x4 matrix multiplication using nested Strassen.
    Returns U, V, W as lists of lists (16x49).
    """
    strassen_2x2 = _strassen_factors()
    U = np.zeros((16, 49), dtype=np.float64)
    V = np.zeros((16, 49), dtype=np.float64)
    W = np.zeros((16, 49), dtype=np.float64)
    
    # Mapping of Strassen steps to local 2x2 matrix compositions
    # Each step k defines A'_k and B'_k as linear combos of A and B blocks.
    # We define the mapping from local 2x2 index (l) to global 4x4 indices and coefficients.
    
    # Helper to add a factor
    col = 0
    for k in range(7):
        # Define A'_k and B'_k mappings
        # local_idx -> list of (global_idx, coeff)
        # Indices for A (16 entries: i*4+j)
        # Indices for B (16 entries: j*4+k)
        
        if k == 0: # M1 = (A00+A11)(B00+B11)
            A_map = {
                0: [(0, 1), (10, 1)], # A00 + A11 (0,0) -> A[0,0] + A[2,2]
                1: [(1, 1), (11, 1)], # A[0,1] + A[2,3]
                2: [(4, 1), (14, 1)], # A[1,0] + A[3,2]
                3: [(5, 1), (15, 1)]  # A[1,1] + A[3,3]
            }
            B_map = {
                0: [(0, 1), (10, 1)], # B00 + B11
                1: [(1, 1), (11, 1)],
                2: [(4, 1), (14, 1)],
                3: [(5, 1), (15, 1)]
            }
        elif k == 1: # M2 = (A10+A11)B00
            A_map = {
                0: [(8, 1), (10, 1)], # A10 + A11 (2,0) + (2,2)
                1: [(9, 1), (11, 1)],
                2: [(12, 1), (14, 1)],
                3: [(13, 1), (15, 1)]
            }
            B_map = {
                0: [(0, 1)], # B00
                1: [(1, 1)],
                2: [(4, 1)],
                3: [(5, 1)]
            }
        elif k == 2: # M3 = A00(B01-B11)
            A_map = {
                0: [(0, 1)], # A00
                1: [(1, 1)],
                2: [(4, 1)],
                3: [(5, 1)]
            }
            B_map = {
                0: [(2, 1), (10, -1)], # B01 - B11 (0,2) - (2,2)
                1: [(3, 1), (11, -1)],
                2: [(6, 1), (14, -1)],
                3: [(7, 1), (15, -1)]
            }
        elif k == 3: # M4 = A11(B10-B00)
            A_map = {
                0: [(10, 1)], # A11
                1: [(11, 1)],
                2: [(14, 1)],
                3: [(15, 1)]
            }
            B_map = {
                0: [(8, 1), (0, -1)], # B10 - B00 (2,0) - (0,0)
                1: [(9, 1), (1, -1)],
                2: [(12, 1), (4, -1)],
                3: [(13, 1), (5, -1)]
            }
        elif k == 4: # M5 = (A00+A01)B11
            A_map = {
                0: [(0, 1)], # A00 + A01
                1: [(1, 1), (3, 1)],
                2: [(4, 1)],
                3: [(5, 1), (7, 1)]
            }
            B_map = {
                0: [(10, 1)], # B11
                1: [(11, 1)],
                2: [(14, 1)],
                3: [(15, 1)]
            }
        elif k == 5: # M6 = (A10-A00)(B00+B01)
            A_map = {
                0: [(8, 1), (0, -1)], # A10 - A00
                1: [(9, 1), (1, -1)],
                2: [(12, 1), (4, -1)],
                3: [(13, 1), (5, -1)]
            }
            B_map = {
                0: [(0, 1), (2, 1)], # B00 + B01
                1: [(1, 1), (3, 1)],
                2: [(4, 1), (6, 1)],
                3: [(5, 1), (7, 1)]
            }
        else: # M7 = (A01-A11)(B10+B11)
            A_map = {
                0: [(2, 1), (10, -1)], # A01 - A11
                1: [(3, 1), (11, -1)],
                2: [(6, 1), (14, -1)],
                3: [(7, 1), (15, -1)]
            }
            B_map = {
                0: [(8, 1), (10, 1)], # B10 + B11
                1: [(9, 1), (11, 1)],
                2: [(12, 1), (14, 1)],
                3: [(13, 1), (15, 1)]
            }

        for (uf, vf, wf) in strassen_2x2:
            # uf, vf, wf are length 4
            for l in range(4):
                if uf[l] != 0:
                    for g_idx, coeff in A_map[l]:
                        U[g_idx, col] += uf[l] * coeff
                if vf[l] != 0:
                    for g_idx, coeff in B_map[l]:
                        V[g_idx, col] += vf[l] * coeff
                if wf[l] != 0:
                    # W mapping is same as B mapping for indices?
                    # W is indexed by k*n+i.
                    # In 4x4, W index is k*4+i.
                    # The local index l in W corresponds to (k_local, i_local).
                    # The global indices for W are determined by the same block structure.
                    # For W, the "row" is k and "col" is i.
                    # So W_map is identical to B_map in terms of index calculation?
                    # B is j*4+k. W is k*4+i.
                    # The structure of the linear combination is the same.
                    for g_idx, coeff in B_map[l]:
                        W[g_idx, col] += wf[l] * coeff
            col += 1
            
    return U.tolist(), V.tolist(), W.tolist()

def _construct_5x5x5():
    n, m, p = 5, 5, 5
    # Partition 5 = 4 + 1
    block_sizes = [4, 1]
    offsets = [0, 4]
    
    total_rank = 0
    # Block (0,0,0): 4x4x4, Rank 49
    total_rank += 49
    # Block (0,0,1): 4x4x1, Rank 16
    total_rank += 16
    # Block (0,1,0): 4x1x4, Rank 16
    total_rank += 16
    # Block (0,1,1): 4x1x1, Rank 4
    total_rank += 4
    # Block (1,0,0): 1x4x4, Rank 16
    total_rank += 16
    # Block (1,0,1): 1x4x1, Rank 4
    total_rank += 4
    # Block (1,1,0): 1x1x4, Rank 4
    total_rank += 4
    # Block (1,1,1): 1x1x1, Rank 1
    total_rank += 1
    
    U = np.zeros((n * m, total_rank), dtype=np.float64)
    V = np.zeros((m * p, total_rank), dtype=np.float64)
    W = np.zeros((p * n, total_rank), dtype=np.float64)
    
    col = 0
    
    # Block (0,0,0): 4x4x4
    U4, V4, W4 = _construct_4x4_strassen_factors()
    for c in range(49):
        for i in range(16):
            U[i, col] = U4[i][c]
            V[i, col] = V4[i][c]
            W[i, col] = W4[i][c]
        col += 1
    
    # Helper to add schoolbook block
    def add_schoolblock(i_off, j_off, k_off, ni, mj, pk):
        nonlocal col
        for li in range(ni):
            for lj in range(mj):
                for lk in range(pk):
                    U_idx = (i_off + li) * m + (j_off + lj)
                    V_idx = (j_off + lj) * p + (k_off + lk)
                    W_idx = (k_off + lk) * n + (i_off + li)
                    U[U_idx, col] = 1.0
                    V[V_idx, col] = 1.0
                    W[W_idx, col] = 1.0
                    col += 1

    # Block (0,0,1): 4x4x1
    add_schoolblock(0, 0, 4, 4, 4, 1)
    # Block (0,1,0): 4x1x4
    add_schoolblock(0, 4, 0, 4, 1, 4)
    # Block (0,1,1): 4x1x1
    add_schoolblock(0, 4, 4, 4, 1, 1)
    # Block (1,0,0): 1x4x4
    add_schoolblock(4, 0, 0, 1, 4, 4)
    # Block (1,0,1): 1x4x1
    add_schoolblock(4, 0, 4, 1, 4, 1)
    # Block (1,1,0): 1x1x4
    add_schoolblock(4, 4, 0, 1, 1, 4)
    # Block (1,1,1): 1x1x1
    add_schoolblock(4, 4, 4, 1, 1, 1)
    
    assert col == total_rank, f"Column mismatch: {col} != {total_rank}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}, total_rank

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    result, rank = _construct_5x5x5()
    
    if _verify(n, m, p, result["U"], result["V"], result["W"]):
        return result
    
    # Fallback
    U = np.zeros((n * m, 125), dtype=np.float64)
    V = np.zeros((m * p, 125), dtype=np.float64)
    W = np.zeros((p * n, 125), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
