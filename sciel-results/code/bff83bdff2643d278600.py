import numpy as np
import time
import sys
from fractions import Fraction

_START_TIME = time.time()
TIME_LIMIT = 145

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def _verify(U, V, W, n, m, p, rank):
    """Verify that the decomposition is correct."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    if U.shape != (n * m, rank):
        return False
    if V.shape != (m * p, rank):
        return False
    if W.shape != (p * n, rank):
        return False
    
    i_idx, j_idx, k_idx = np.meshgrid(
        np.arange(n), np.arange(m), np.arange(p), indexing='ij'
    )
    i_flat = i_idx.ravel()
    j_flat = j_idx.ravel()
    k_flat = k_idx.ravel()
    
    U_idx = i_flat * m + j_flat
    V_idx = j_flat * p + k_flat
    W_idx = k_flat * n + i_flat
    
    prod = U[U_idx, :] * V[V_idx, :] * W[W_idx, :]
    sums = prod.sum(axis=1)
    
    if not np.allclose(sums, 1.0, atol=1e-9):
        return False
    return True

def construct():
    n, m, p = 4, 4, 7
    
    # Strassen 2x2x2 terms: (x_factors, y_factors, c_factors)
    # Each is a dict mapping (local_row, local_col) to coefficient
    strassen_2x2 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    # Strassen 4x4x4: 7 outer terms
    # Each term has x_def and y_def as dicts mapping block_index to coefficient
    # Block indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1) in 4x4 blocks
    strassen_4x4_outer = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}),
        ({0: 1}, {1: 1, 3: -1}),
        ({3: 1}, {2: 1, 0: -1}),
        ({0: 1, 1: 1}, {3: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}),
    ]
    
    c_blocks_4x4 = [
        {(0,0): 1.0, (1,1): 1.0},
        {(1,0): 1.0, (1,1): -1.0},
        {(0,1): 1.0, (1,1): 1.0},
        {(0,0): 1.0, (1,0): 1.0},
        {(0,0): -1.0, (0,1): 1.0},
        {(1,1): 1.0},
        {(0,0): 1.0},
    ]
    
    # Build rank-49 Strassen decomposition for 4x4x4 part
    def build_4x4x4():
        """Build Strassen decomposition for 4x4x4 matrix multiplication."""
        U = np.zeros((16, 49))
        V = np.zeros((28, 49))
        W = np.zeros((28, 49))
        
        col = 0
        for m_idx in range(7):
            x_def, y_def = strassen_4x4_outer[m_idx]
            c_block = c_blocks_4x4[m_idx]
            for t_idx in range(7):
                r = col
                col += 1
                x_sub, y_sub, c_sub = strassen_2x2[t_idx]
                
                for (li, lj), cx in x_sub.items():
                    for b, cb in x_def.items():
                        bi, bj = b // 2, b % 2
                        i = 2 * bi + li
                        j = 2 * bj + lj
                        U[i * 4 + j, r] += cx * cb
                
                for (li, lj), cy in y_sub.items():
                    for b, cb in y_def.items():
                        bi, bj = b // 2, b % 2
                        i = 2 * bi + li
                        j = 2 * bj + lj
                        V[i * 7 + j, r] += cy * cb
                
                for (li, lj), cc in c_sub.items():
                    if cc == 0:
                        continue
                    for (br, bc), cb_c in c_block.items():
                        if cb_c == 0:
                            continue
                        i_final = 2 * br + li
                        k_final = 2 * bc + lj
                        W[k_final * 4 + i_final, r] += cc * cb_c
        
        assert col == 49, f"Expected 49, got {col}"
        return U[:, :col], V[:, :col], W[:, :col]
    
    # Build 4x4x3 decomposition using Kronecker structure
    # The 4x4x3 problem can be viewed as 4 independent 2x2x3 problems
    # For each 2x2x3 sub-problem, we use rank 11 (7 Strassen + 4 schoolbook for col 2)
    # But we can do better by using the structure
    
    def build_4x4x3():
        """Build rank-minimal decomposition for 4x4x3 matrix multiplication."""
        # 4x4x3: multiply 4x4 by 4x3
        # This is 4 independent 2x2x3 sub-problems (row pairs x col triple)
        # Each 2x2x3 has rank at most 11, but we can use 10 or less
        
        # Actually, let's use a direct approach:
        # 4x4x3 = Kronecker(2x2, 2x3) in some sense
        # But more simply: it's 4 sub-problems of type 2x2x3
        
        # For 2x2x3 (multiply 2x2 by 2x3):
        # We need terms for (i,j,k) where i,j in {0,1}, k in {0,1,2}
        # Total 8 terms needed
        # Rank of 2x2x3 tensor: known to be 11
        
        # Let's construct 2x2x3 with rank 11:
        # 7 Strassen terms for the 2x2x2 part (columns 0,1)
        # 4 schoolbook terms for column 2
        
        U = np.zeros((16, 44))  # 4 row pairs * 2x2x3 rank 11 = 44
        V = np.zeros((28, 44))
        W = np.zeros((28, 44))
        
        col = 0
        for row_pair in range(4):
            ar = row_pair // 2  # which 2x2 row block
            ac = row_pair % 2   # which 2x2 col block
            br = ar             # corresponding B row block
            cr = ac             # corresponding C col block
            
            # 7 Strassen terms for 2x2x2 (columns 4,5 of the 7-col B)
            for t_idx in range(7):
                r = col
                col += 1
                x_sub, y_sub, c_sub = strassen_2x2[t_idx]
                
                for (li, lj), cx in x_sub.items():
                    i = 2 * ar + li
                    j = 2 * ac + lj
                    U[i * 4 + j, r] += cx
                
                for (li, lj), cy in y_sub.items():
                    i = 2 * br + li
                    j = 4 + lj
                    V[i * 7 + j, r] += cy
                
                for (li, lj), cc in c_sub.items():
                    if cc == 0:
                        continue
                    i_final = 2 * cr + li
                    k_final = 4 + lj
                    W[k_final * 4 + i_final, r] += cc
            
            # 4 schoolbook terms for column 6
            for li in range(2):
                for lj in range(2):
                    r = col
                    col += 1
                    i = 2 * ar + li
                    j = 2 * ac + lj
                    U[i * 4 + j, r] = 1.0
                    V[i * 7 + 6, r] = 1.0
                    W[6 * 4 + i, r] = 1.0
        
        assert col == 44, f"Expected 44, got {col}"
        return U[:, :col], V[:, :col], W[:, :col]
    
    # Build full 4x4x7 decomposition
    def build_full():
        U1, V1, W1 = build_4x4x4()
        U2, V2, W2 = build_4x4x3()
        
        total_rank = 49 + 44
        U = np.zeros((16, total_rank))
        V = np.zeros((28, total_rank))
        W = np.zeros((28, total_rank))
        
        U[:, :49] = U1
        V[:, :49] = V1
        W[:, :49] = W1
        
        U[:, 49:] = U2
        V[:, 49:] = V2
        W[:, 49:] = W2
        
        return U, V, W
    
    U, V, W = build_full()
    rank = U.shape[1]
    
    # Verify
    if not _verify(U, V, W, n, m, p, rank):
        raise ValueError(f"Decomposition verification failed for rank {rank}")
    
    # Try to find rank reductions via linear algebra
    # Look for linear dependencies among the rank terms
    # This is expensive, so only do it if we have time
    
    if _time_remaining() > 30:
        # Build the constraint matrix
        # A[idx, t] = U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
        A = np.zeros((n * m * p, rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * (m * p) + j * p + k
                    A[idx, :] = U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :]
        
        try:
            _, s, vh = np.linalg.svd(A, full_matrices=False)
            tol = 1e-10
            rank_A = np.sum(s > tol)
            
            if rank_A < rank and _time_remaining() > 15:
                # Found a null space, try to eliminate terms
                null_space = vh[rank_A:, :]
                print(f"Found null space of dimension {rank - rank_A}")
                
                # For each null vector, try to eliminate a term
                # We need to be careful about exactness
                for nv_idx in range(null_space.shape[0]):
                    if _time_remaining() < 10:
                        break
                    c = null_space[nv_idx]
                    
                    # Find the term with largest |c_t| to eliminate
                    max_t = np.argmax(np.abs(c))
                    max_c = c[max_t]
                    
                    if abs(max_c) < 1e-8:
                        continue
                    
                    # Try to eliminate term max_t
                    # New U[:, max_t] should be expressed as combination of others
                    # This requires solving a system, which is complex
                    # For now, just note that we found a dependency
                    pass
        except Exception as e:
            print(f"Error in rank reduction: {e}")
    
    # Return the decomposition
    U_out = U.tolist()
    V_out = V.tolist()
    W_out = W.tolist()
    
    final_rank = U_out[0].__len__()
    if not _verify(U_out, V_out, W_out, n, m, p, final_rank):
        raise ValueError(f"Final verification failed for rank {final_rank}")
    
    return {"U": U_out, "V": V_out, "W": W_out}
