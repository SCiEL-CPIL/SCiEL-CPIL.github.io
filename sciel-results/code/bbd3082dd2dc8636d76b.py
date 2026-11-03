"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START

def _get_strassen_222():
    n, m, p = 2, 2, 2
    r = 7
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    return U, V, W

def _get_school_221():
    n, m, p = 2, 2, 1
    r = 4
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    for c in range(r):
        i = c // m
        j = c % m
        k = 0
        U[i * m + j, c] = 1.0
        V[j * p + k, c] = 1.0
        W[k * n + i, c] = 1.0
    return U, V, W

def _verify(U, V, W, n, m, p):
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    s = np.sum(W[k*n+i, :] * U[i*m+j, :] * V[jp*p+k, :])
                    expected = 1.0 if j == jp else 0.0
                    if abs(s - expected) > 1e-9:
                        return False
    return True

def _build_block_decomp(n, m, p, deadline=None):
    """
    Builds a decomposition by splitting m into 2s and p into 2s/1s.
    Returns U, V, W, rank.
    """
    if deadline and time.time() > deadline:
        return None
    
    m_blocks = []
    for j_start in range(0, m, 2):
        m_blocks.append((j_start, min(j_start + 2, m)))
        
    p_blocks = []
    for k_start in range(0, p, 2):
        if k_start + 2 <= p:
            p_blocks.append((k_start, k_start + 2, 'strassen'))
        else:
            p_blocks.append((k_start, k_start + 1, 'school'))

    total_rank = 0
    for j_start, j_end in m_blocks:
        for k_start, k_end, p_type in p_blocks:
            total_rank += 7 if p_type == 'strassen' else 4

    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    col_offset = 0
    
    U_s, V_s, W_s = _get_strassen_222()
    U_sc, V_sc, W_sc = _get_school_221()

    for j_start, j_end in m_blocks:
        m_sub = j_end - j_start
        for k_start, k_end, p_type in p_blocks:
            p_sub = k_end - k_start
            if p_type == 'strassen':
                U_sub, V_sub, W_sub = U_s, V_s, W_s
                r_sub = 7
            else:
                U_sub, V_sub, W_sub = U_sc, V_sc, W_sc
                r_sub = 4
            
            for i in range(n):
                for j_sub in range(m_sub):
                    j = j_start + j_sub
                    U[i * m + j, col_offset:col_offset + r_sub] = U_sub[i * 2 + j_sub, :]
            for j_sub in range(m_sub):
                j = j_start + j_sub
                for k_sub in range(p_sub):
                    k = k_start + k_sub
                    V[j * p + k, col_offset:col_offset + r_sub] = V_sub[j_sub * p_sub + k_sub, :]
            for k_sub in range(p_sub):
                k = k_start + k_sub
                for i in range(n):
                    W[k * n + i, col_offset:col_offset + r_sub] = W_sub[k_sub * 2 + i, :]
            col_offset += r_sub
            
    return U, V, W, total_rank

def _tensor_to_vec(U, V, W, n, m, p):
    """Flatten tensor T[i,j,jp,k] = sum_c U[i*m+j,c]*V[jp*p+k,c]*W[k*n+i,c]"""
    T = np.zeros((n, m, m, p))
    r = U.shape[1]
    for c in range(r):
        uc = U[:, c]
        vc = V[:, c]
        wc = W[:, c]
        for i in range(n):
            u_i = uc[i*m:(i+1)*m]
            for k in range(p):
                w_ki = wc[k*n+i]
                if w_ki != 0:
                    v_k = vc[k*m:(k+1)*m]
                    T[i, :, :, k] += np.outer(u_i, v_k) * w_ki
    return T

def _vec_to_tensor(T, n, m, p):
    return T

def _compute_residual(U, V, W, n, m, p):
    T = _tensor_to_vec(U, V, W, n, m, p)
    R = np.zeros((n, m, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                R[i, j, j, k] = 1.0
    R -= T
    return R

def _score_term(u, v, w, R, n, m, p):
    """Score = <u⊗v⊗w, R> = sum u[i,j] v[jp,k] w[k,i] R[i,j,jp,k]"""
    score = 0.0
    for i in range(n):
        for j in range(m):
            uj = u[i*m+j]
            if uj == 0: continue
            for jp in range(m):
                for k in range(p):
                    r_val = R[i, j, jp, k]
                    if r_val == 0: continue
                    vk = v[jp*p+k]
                    if vk == 0: continue
                    wk = w[k*n+i]
                    if wk == 0: continue
                    score += uj * vk * wk * r_val
    return score

def _update_residual(R, u, v, w, n, m, p):
    """R -= u⊗v⊗w"""
    for i in range(n):
        for j in range(m):
            uj = u[i*m+j]
            if uj == 0: continue
            for jp in range(m):
                for k in range(p):
                    R[i, j, jp, k] -= uj * v[jp*p+k] * w[k*n+i]

def _greedy_search(n, m, p, target_rank, deadline, seed=42, max_candidates=500):
    rng = np.random.RandomState(seed)
    R = np.zeros((n, m, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                R[i, j, j, k] = 1.0
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    for step in range(target_rank):
        if time.time() > deadline:
            return None
        
        best_score = -np.inf
        best_u = None
        best_v = None
        best_w = None
        
        for _ in range(max_candidates):
            u = np.zeros(n * m)
            v = np.zeros(m * p)
            w = np.zeros(p * n)
            
            # Sparse random
            for idx in range(n * m):
                if rng.random() < 0.2:
                    u[idx] = rng.choice([-1, 1])
            for idx in range(m * p):
                if rng.random() < 0.2:
                    v[idx] = rng.choice([-1, 1])
            for idx in range(p * n):
                if rng.random() < 0.2:
                    w[idx] = rng.choice([-1, 1])
            
            score = _score_term(u, v, w, R, n, m, p)
            if score > best_score:
                best_score = score
                best_u = u.copy()
                best_v = v.copy()
                best_w = w.copy()
        
        if best_u is None:
            return None
            
        _update_residual(R, best_u, best_v, best_w, n, m, p)
        U_cols.append(best_u)
        V_cols.append(best_v)
        W_cols.append(best_w)
        
        if np.max(np.abs(R)) < 1e-9:
            break
            
    if len(U_cols) == 0:
        return None
        
    U = np.column_stack(U_cols)
    V = np.column_stack(V_cols)
    W = np.column_stack(W_cols)
    
    if _verify(U, V, W, n, m, p):
        return U, V, W
        
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115
    
    # 1. Build baseline block decomposition
    U, V, W, known_rank = _build_block_decomp(n, m, p, deadline)
    if U is None:
        # Fallback to schoolbook full
        r = n * m * p
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        known_rank = r
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    if not _verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    best_rank = known_rank
    best_U, best_V, best_W = U, V, W
    
    # 2. Try greedy search for lower ranks
    # Theoretical lower bound is around 2 * R(2,4,7)? No, just try a range.
    # Block rank is 50.
    # Try to beat 50.
    
    # We can try to find a decomposition of rank < 50.
    # Greedy search is stochastic.
    
    # Let's try a few target ranks
    # We know R(2,2,2)=7. R(2,2,1)=4.
    # R(2,4,4) <= 2*14 = 28? No, R(2,2,4)=14, so R(2,4,4) <= 28.
    # R(2,4,7) <= R(2,4,4) + R(2,4,3) = 28 + (14+4)*? No.
    # R(2,4,3) <= R(2,2,3)*2 = 11*2 = 22.
    # Total 50.
    
    # Maybe we can do better with a global search?
    # Unlikely to find < 40 quickly.
    # But let's try.
    
    # Try ranks from 49 down to 30
    for target_rank in range(known_rank - 1, 29, -1):
        if time.time() > deadline - 5:
            break
        result = _greedy_search(n, m, p, target_rank, deadline - 5, seed=target_rank, max_candidates=200)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = target_rank
            break # Found a valid one, return it
            
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
