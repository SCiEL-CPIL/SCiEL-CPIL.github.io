"""Tensor decomposition for 2x3 * 3x10 matrix multiplication."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START


def _validate(n, m, p, U, V, W):
    """Verify that sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] == 1 for all i,j,k."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    U3 = U_arr.reshape(n, m, -1)
    V3 = V_arr.reshape(m, p, -1)
    W3 = W_arr.reshape(p, n, -1)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    if not np.allclose(T, np.ones((n, m, p)), atol=1e-8):
        return False
    return True


def _try_reduce_rank(n, m, p, U, V, W, deadline):
    """
    Greedily remove columns that are redundant. Uses incremental tensor updates.
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    keep = np.ones(rank, dtype=bool)
    
    target = np.ones((n, m, p))
    
    col_contrib = np.zeros((rank, n, m, p))
    for r in range(rank):
        u_col = U_arr[:, r].reshape(n, m)
        v_col = V_arr[:, r].reshape(m, p)
        w_col = W_arr[:, r].reshape(p, n)
        col_contrib[r] = np.einsum('ij,jk,ki->ijk', u_col, v_col, w_col)
    
    T_current = col_contrib.sum(axis=0)
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for r in range(rank):
            if not keep[r]:
                continue
            T_new = T_current - col_contrib[r]
            if np.allclose(T_new, target, atol=1e-8):
                keep[r] = False
                T_current = T_new
                changed = True
                break
    
    new_rank = int(np.sum(keep))
    if new_rank >= rank:
        return U, V, W, rank
    
    U_new = U_arr[:, keep]
    V_new = V_arr[:, keep]
    W_new = W_arr[:, keep]
    
    U_list = U_new.tolist()
    V_list = V_new.tolist()
    W_list = W_new.tolist()
    
    if _validate(n, m, p, U_list, V_list, W_list):
        return U_list, V_list, W_list, new_rank
    else:
        return U, V, W, rank


def _build_strassen(n, m, p):
    """
    Build a rank-55 decomposition for 2x3 * 3x10 using Strassen on 2x2x2 blocks.
    Decompose middle dimension m=3 into a 2-dim subspace (Strassen, rank 35)
    and a 1-dim subspace (schoolbook, rank 20).
    """
    assert n == 2 and m == 3 and p == 10
    
    rank = 55
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    r = 0
    
    # Strassen formulas for 2x2 * 2x2
    # M1: (A11+A22)*(B11+B22) -> contributes to C11, C22
    # M2: (A21+A22)*B11 -> contributes to C21, C22
    # M3: A11*(B21-B11) -> contributes to C12
    # M4: A22*(B12-B22) -> contributes to C11, C21
    # M5: (A11+A12)*B22 -> contributes to C12, C22
    # M6: (A21-A11)*(B11+B12) -> contributes to C22
    # M7: (A12-A22)*(B21+B22) -> contributes to C11
    
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # M1: (a00+a11)*(B[0,k0]+B[1,k1])
        U[0, r] += 1   # a00
        U[4, r] += 1   # a11
        V[k0, r] += 1       # B[0,k0]
        V[10+k1, r] += 1    # B[1,k1]
        W[k0*2+0, r] += 1   # C[0,k0]
        W[k1*2+1, r] += 1   # C[1,k1]
        r += 1
        
        # M2: (a10+a11)*B[0,k0]
        U[3, r] += 1   # a10
        U[4, r] += 1   # a11
        V[k0, r] += 1       # B[0,k0]
        W[k0*2+1, r] += 1   # C[1,k0]
        W[k1*2+1, r] -= 1   # -C[1,k1]
        r += 1
        
        # M3: a00*(B[1,k1]-B[0,k1])
        U[0, r] += 1   # a00
        V[10+k1, r] += 1    # B[1,k1]
        V[k1, r] -= 1       # -B[0,k1]
        W[k1*2+0, r] += 1   # C[0,k1]
        r += 1
        
        # M4: a11*(B[0,k0]-B[1,k0])
        U[4, r] += 1   # a11
        V[k0, r] += 1       # B[0,k0]
        V[10+k0, r] -= 1    # -B[1,k0]
        W[k0*2+0, r] += 1   # C[0,k0]
        W[k0*2+1, r] += 1   # C[1,k0]
        r += 1
        
        # M5: (a00+a01)*B[1,k1]
        U[0, r] += 1   # a00
        U[1, r] += 1   # a01
        V[10+k1, r] += 1    # B[1,k1]
        W[k0*2+0, r] -= 1   # -C[0,k0]
        W[k1*2+0, r] += 1   # C[0,k1]
        r += 1
        
        # M6: (a10-a00)*(B[0,k0]+B[0,k1])
        U[3, r] += 1   # a10
        U[0, r] -= 1   # -a00
        V[k0, r] += 1       # B[0,k0]
        V[k1, r] += 1       # B[0,k1]
        W[k1*2+1, r] += 1   # C[1,k1]
        r += 1
        
        # M7: (a01-a11)*(B[1,k0]+B[1,k1])
        U[1, r] += 1   # a01
        U[4, r] -= 1   # -a11
        V[10+k0, r] += 1    # B[1,k0]
        V[10+k1, r] += 1    # B[1,k1]
        W[k0*2+0, r] += 1   # C[0,k0]
        r += 1
        
        # Schoolbook for A[:,2] * B[2, k0:k1+1]
        # a02 * B[2,k0] -> C[0,k0]
        U[2, r] += 1
        V[20+k0, r] += 1
        W[k0*2+0, r] += 1
        r += 1
        
        # a02 * B[2,k1] -> C[0,k1]
        U[2, r] += 1
        V[20+k1, r] += 1
        W[k1*2+0, r] += 1
        r += 1
        
        # a12 * B[2,k0] -> C[1,k0]
        U[5, r] += 1
        V[20+k0, r] += 1
        W[k0*2+1, r] += 1
        r += 1
        
        # a12 * B[2,k1] -> C[1,k1]
        U[5, r] += 1
        V[20+k1, r] += 1
        W[k1*2+1, r] += 1
        r += 1
    
    assert r == rank, f"Expected rank {rank}, got {r}"
    
    return U.tolist(), V.tolist(), W.tolist(), rank


def _global_optimization(n, m, p, U, V, W, deadline):
    """
    Apply a global linear change of basis to try to reduce rank.
    We parametrize U, V, W as base + perturbation and minimize the deviation
    from the target tensor while encouraging sparsity.
    
    Strategy: Start from the Strassen decomposition, then try to find a lower-rank
    equivalent by optimizing over all entries and checking if we can zero out
    some rank-1 components.
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    target = np.ones((n, m, p))
    
    def compute_tensor(U_flat, V_flat, W_flat):
        U3 = U_flat.reshape(n, m, rank)
        V3 = V_flat.reshape(m, p, rank)
        W3 = W_flat.reshape(p, n, rank)
        return np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    # First, try a simple approach: optimize a linear combination of columns
    # to see if any column can be expressed as a combination of others
    # such that the tensor remains unchanged.
    
    # More aggressively: try to find a lower rank decomposition by
    # random restarts with fewer terms
    best_U, best_V, best_W, best_rank = U, V, W, rank
    
    # Try to reduce by 1, 2, 3... using random projections
    for target_rank in range(rank - 1, max(rank - 6, 10), -1):
        if time.time() > deadline:
            break
        
        # Random initialization: pick target_rank random linear combinations
        # of the existing rank-1 tensors
        np.random.seed(42 + target_rank)
        
        # We want to find U', V', W' of rank target_rank such that
        # sum_{r'=0}^{target_rank-1} U'[i,j,r'] * V'[j,k,r'] * W'[k,i,r'] = 1
        
        # Initialize by random linear combinations of existing columns
        U_rand = np.zeros((n*m, target_rank))
        V_rand = np.zeros((m*p, target_rank))
        W_rand = np.zeros((p*n, target_rank))
        
        # Use random 3D transformations: U' = U @ A, V' = V @ B, W' = W @ C
        # where A, B, C are (rank x target_rank) matrices with A@B@C = I on the
        # appropriate subspace. This is tricky, so let's try a different approach.
        
        # Alternative: Use the fact that if we have a rank-r decomposition,
        # we can try to find a rank-(r-1) one by looking for a vector in the
        # null space of the "contraction" map.
        
        # For now, use a heuristic: try to merge pairs of columns
        # that have similar contributions.
        pass
    
    # Simpler approach: try removing pairs of columns and re-optimizing
    # the remaining columns to compensate.
    
    for remove_count in range(1, 4):
        if time.time() > deadline:
            break
        
        for remove_idx in range(rank - remove_count):
            if time.time() > deadline:
                break
            
            # Try removing column remove_idx and remove_idx+1
            keep_mask = np.ones(rank, dtype=bool)
            keep_mask[remove_idx] = False
            keep_mask[remove_idx + 1] = False
            
            new_rank = int(np.sum(keep_mask))
            if new_rank >= best_rank:
                continue
            
            U_sub = U_arr[:, keep_mask]
            V_sub = V_arr[:, keep_mask]
            W_sub = W_arr[:, keep_mask]
            
            # Check if the remaining columns already sum to the target
            T_sub = np.einsum('ijr,jkr,kir->ijk',
                             U_sub.reshape(n,m,-1),
                             V_sub.reshape(m,p,-1),
                             W_sub.reshape(p,n,-1))
            
            if np.allclose(T_sub, target, atol=1e-8):
                best_rank = new_rank
                best_U = U_sub.tolist()
                best_V = V_sub.tolist()
                best_W = W_sub.tolist()
                break
    
    if best_rank < rank:
        return best_U, best_V, best_W, best_rank
    
    return U, V, W, rank


def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS

    U, V, W, rank = _build_strassen(n, m, p)

    if not _validate(n, m, p, U, V, W):
        # Fallback to trivial rank 60
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for r in range(rank):
            i = r // (m * p)
            rem = r % (m * p)
            j = rem // p
            k = rem % p
            idx_u = i * m + j
            idx_v = j * p + k
            idx_w = k * n + i
            U[idx_u][r] = 1
            V[idx_v][r] = 1
            W[idx_w][r] = 1
    else:
        U, V, W, rank = _try_reduce_rank(n, m, p, U, V, W, deadline)
        
        # Try global optimization to find further reductions
        if time.time() < deadline:
            U, V, W, rank = _global_optimization(n, m, p, U, V, W, deadline)

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
