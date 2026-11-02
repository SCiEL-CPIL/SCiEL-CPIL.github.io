"""Find rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 8]

def _build_subtensor(n, m, p):
    """Build the (n,m,p) matrix multiplication tensor with shapes (n*m, m*p, p*n)."""
    nm, mp, pn = n*m, m*p, p*n
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _find_rank_decomp(T, nm, mp, pn, target_rank, deadline, n_trials=50):
    """Try to find a rank-target_rank exact decomposition of tensor T.
    Returns (U, V, W) or None."""
    T_flat = T.reshape(nm * mp, pn)
    for trial in range(n_trials):
        if time.time() > deadline - 5:
            return None
        np.random.seed(12345 + target_rank * 7919 + trial * 31)
        U = np.zeros((nm, target_rank))
        V = np.zeros((mp, target_rank))
        for c in range(target_rank):
            # Sparse random vectors with half-integer entries
            u_vals = np.random.choice([-1.0, -0.5, 0.5, 1.0], size=nm)
            v_vals = np.random.choice([-1.0, -0.5, 0.5, 1.0], size=mp)
            u_mask = np.random.random(nm) < 0.4
            v_mask = np.random.random(mp) < 0.4
            U[:, c] = np.where(u_mask, u_vals, 0.0)
            V[:, c] = np.where(v_mask, v_vals, 0.0)
        K = np.zeros((nm * mp, target_rank))
        for c in range(target_rank):
            K[:, c] = np.outer(U[:, c], V[:, c]).flatten()
        try:
            W_sol, _, _, _ = np.linalg.lstsq(K, T_flat, rcond=1e-12)
        except np.linalg.LinAlgError:
            continue
        if np.abs(K @ W_sol - T_flat).max() < 1e-9:
            Wc = W_sol * 2
            if np.allclose(Wc, np.round(Wc), atol=1e-6):
                W = np.round(W_sol.T * 2) / 2
                recon = np.zeros((nm, mp, pn))
                for c in range(target_rank):
                    recon += np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])
                if np.abs(recon - T).max() < 1e-8:
                    return (U, V, W)
    return None

def _find_3x3x4_decomposition(deadline):
    """Find a rank-19 exact decomposition for 3x3x4 matrix multiplication.
    Returns (U4, V4, W4) with shapes (9,19), (12,19), (12,19) or None."""
    n, m, p = 3, 3, 4
    nm, mp, pn = n*m, m*p, p*n  # 9, 12, 12
    T4 = _build_subtensor(n, m, p)
    # Try ranks from 19 downward; 19 is the known optimal for 3x3x4
    for target_rank in [19, 20, 21]:
        result = _find_rank_decomp(T4, nm, mp, pn, target_rank, deadline, n_trials=30)
        if result is not None:
            return result
    return None

def construct():
    """Find a low-rank exact tensor decomposition for (3,3,8) matrix multiplication.
    
    C[i,k] = sum_j A[i,j] * B[j,k]
    U shape: (n*m, rank) indexed by i*m+j
    V shape: (m*p, rank) indexed by j*p+k
    W shape: (p*n, rank) indexed by k*n+i
    """
    n, m, p = DIMENSIONS
    nm = n * m  # 9
    mp = m * p  # 24
    pn = p * n  # 24
    
    # Build target tensor
    T = _build_subtensor(n, m, p)
    
    schoolbook_rank = n * m * p  # 72
    
    # Schoolbook decomposition as baseline
    U_sb = np.zeros((nm, schoolbook_rank))
    V_sb = np.zeros((mp, schoolbook_rank))
    W_sb = np.zeros((pn, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1.0
                V_sb[j*p+k, col] = 1.0
                W_sb[k*n+i, col] = 1.0
                col += 1
    best_decomp = (U_sb, V_sb, W_sb)
    best_rank = schoolbook_rank
    
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # BLOCK-PARALLEL STRATEGY:
    # Decompose 3x3x8 into two independent 3x3x4 sub-problems.
    # Sub-problem 1: output columns k=0..3 (first half)
    # Sub-problem 2: output columns k=4..7 (second half)
    # For each sub-problem, find a rank-19 decomposition of 3x3x4.
    # Total rank = 19 + 19 = 38.
    
    # Build the two 3x3x4 sub-tensors
    n_sub, m_sub, p_sub = 3, 3, 4
    nm_sub, mp_sub, pn_sub = n_sub*m_sub, m_sub*p_sub, p_sub*n_sub  # 9, 12, 12
    T_sub = _build_subtensor(n_sub, m_sub, p_sub)  # (9, 12, 12)
    
    # Find rank-19 decomposition for 3x3x4
    sub_decomp = _find_3x3x4_decomposition(deadline)
    
    if sub_decomp is not None and time.time() < deadline - 10:
        U4, V4, W4 = sub_decomp
        r4 = U4.shape[1]  # should be 19
        
        # Construct the full 3x3x8 decomposition by stacking two copies
        # Sub-problem 1 handles k=0..3, Sub-problem 2 handles k=4..7
        total_rank = 2 * r4
        U_full = np.zeros((nm, total_rank))
        V_full = np.zeros((mp, total_rank))
        W_full = np.zeros((pn, total_rank))
        
        # Copy 1: for k=0..3
        # U is the same for both copies (input A is shared)
        U_full[:, 0:r4] = U4
        # V for sub-problem 1: B rows j*4+k for k=0..3, which is B[j,0..3]
        # In the full problem, B is indexed by j*p+k = j*8+k
        # For sub-problem 1, B indices are j*4+k where j=0..2, k=0..3
        # In full B: index = j*8 + k (for k=0..3)
        for j in range(m_sub):
            for k in range(p_sub):
                v4_idx = j * p_sub + k  # index in V4
                v_full_idx = j * p + k  # index in full V (k=0..3)
                V_full[v_full_idx, 0:r4] = V4[v4_idx, :]
        
        # W for sub-problem 1: output C[i, k] for k=0..3
        # In full W: index = k*n + i (for k=0..3)
        for k in range(p_sub):
            for i in range(n_sub):
                w4_idx = k * n_sub + i  # index in W4
                w_full_idx = k * n + i  # index in full W (k=0..3)
                W_full[w_full_idx, 0:r4] = W4[w4_idx, :]
        
        # Copy 2: for k=4..7
        U_full[:, r4:2*r4] = U4
        # V for sub-problem 2: B rows j*4+k for k=0..3, but these correspond to B[j, 4+k] in full
        # In full B: index = j*8 + (4+k)
        for j in range(m_sub):
            for k in range(p_sub):
                v4_idx = j * p_sub + k  # index in V4
                v_full_idx = j * p + (4 + k)  # index in full V (k=4..7)
                V_full[v_full_idx, r4:2*r4] = V4[v4_idx, :]
        
        # W for sub-problem 2: output C[i, 4+k] for k=0..3
        # In full W: index = (4+k)*n + i
        for k in range(p_sub):
            for i in range(n_sub):
                w4_idx = k * n_sub + i  # index in W4
                w_full_idx = (4 + k) * n + i  # index in full W (k=4..7)
                W_full[w_full_idx, r4:2*r4] = W4[w4_idx, :]
        
        # Verify the full decomposition
        recon = np.zeros((nm, mp, pn))
        for c in range(total_rank):
            recon += np.einsum('a,b,c->abc', U_full[:,c], V_full[:,c], W_full[:,c])
        if np.abs(recon - T).max() < 1e-8:
            best_rank = total_rank
            best_decomp = (U_full, V_full, W_full)
    
    # If block-parallel didn't work, fall back to direct search
    if best_rank == schoolbook_rank and time.time() < deadline - 15:
        T_flat = T.reshape(nm * mp, pn)
        for target_rank in [40, 38, 36, 34, 32]:
            if time.time() > deadline - 10:
                break
            result = _find_rank_decomp(T, nm, mp, pn, target_rank, deadline, n_trials=15)
            if result is not None:
                best_rank = target_rank
                best_decomp = result
                break
    
    U_out, V_out, W_out = best_decomp
    U_list = [[float(x) for x in row] for row in U_out]
    V_list = [[float(x) for x in row] for row in V_out]
    W_list = [[float(x) for x in row] for row in W_out]
    
    return {"U": U_list, "V": V_list, "W": W_list}
