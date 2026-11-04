import time
import random
import math
import numpy as np

DIMENSIONS = [3, 4, 4]

_DEADLINE = time.time() + 150.0
_SAFE_MARGIN = 5.0

def _target_tensor_bilinear(n, m, p):
    """
    Returns the target tensor for the bilinear condition.
    K[i, j, l, k] = 1 if j == l else 0.
    """
    K = np.zeros((n, m, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                K[i, j, j, k] = 1.0
    return K

def _compute_K(U, V, W, n, m, p):
    """
    Computes K[i, j, l, k] = sum_r U[i, j, r] * V[l, k, r] * W[k, i, r]
    Note: U shape (n*m, r) -> (n, m, r)
          V shape (m*p, r) -> (m, p, r) -- WAIT.
          
          In the bilinear condition:
          C[i, k] = sum_r (sum_j U[i, j, r] A[i, j]) (sum_l V[l, k, r] B[l, k]) W[k, i, r]
          
          Wait, the standard form is:
          u_r(A) = sum_{i,j} U_{ijr} A_{ij}
          v_r(B) = sum_{j,k} V_{jkr} B_{jk}
          w_r(C) = sum_{i,k} W_{kir} C_{ik} -- No, W is the coefficient for C.
          
          The product is sum_r u_r(A) v_r(B) w_r.
          w_r is a linear functional on C? No, w_r is the vector added to C.
          So C = sum_r u_r(A) v_r(B) w_r.
          
          This implies C_{ik} = sum_r u_r(A) v_r(B) W_{kir}.
          
          Let's check the indices in the prompt's W convention: W[k*n+i, r].
          So W_{kir} is the coefficient of C_{ik} in term r.
          
          So the condition is:
          sum_r (sum_j U_{ijr} A_{ij}) (sum_l V_{lkr} B_{lk}) W_{kir} = sum_j A_{ij} B_{jk}
          
          Coefficient of A_{ij} B_{lk} in LHS:
          sum_r U_{ijr} V_{lkr} W_{kir}.
          
          This must equal delta_{jl}.
          
          So K[i, j, l, k] = sum_r U_{ijr} V_{lkr} W_{kir}.
          
          My previous derivation was correct.
    """
    r = U.shape[1]
    U2 = U.reshape(n, m, r)
    V2 = V.reshape(m, p, r) # V index is j*p+k. So V2[j, k, r]
    W2 = W.reshape(p, n, r) # W index is k*n+i. So W2[k, i, r]
    
    # K[i, j, l, k] = sum_r U2[i, j, r] * V2[l, k, r] * W2[k, i, r]
    # Note: V2 index is (l, k). W2 index is (k, i).
    K = np.einsum('ijr,lkr,kir->ijlk', U2, V2, W2)
    return K

def _objective_bilinear(U, V, W, n, m, p, t):
    K = _compute_K(U, V, W, n, m, p)
    T = _target_tensor_bilinear(n, m, p)
    resid = K - T
    # Smooth max of |resid|
    M = 50.0
    a = M * np.abs(resid)
    a_max = a.max()
    logsum = a_max + np.log(np.exp(a - a_max).sum())
    smooth_max = logsum / M
    return smooth_max + t

def _grad_bilinear(U, V, W, t, n, m, p):
    K = _compute_K(U, V, W, n, m, p)
    T = _target_tensor_bilinear(n, m, p)
    resid = K - T
    M = 50.0
    a = M * np.abs(resid)
    a_max = a.max()
    s = np.exp(a - a_max)
    Z = s.sum()
    dsig = M * np.sign(resid) * (s / Z)
    
    r = U.shape[1]
    U2 = U.reshape(n, m, r)
    V2 = V.reshape(m, p, r)
    W2 = W.reshape(p, n, r)
    
    # gU2[i, j, r] = sum_{l,k} dsig[i, j, l, k] * V2[l, k, r] * W2[k, i, r]
    gU2 = np.einsum('ijlk,lkr,kir->ijr', dsig, V2, W2)
    
    # gV2[l, k, r] = sum_{i,j} dsig[i, j, l, k] * U2[i, j, r] * W2[k, i, r]
    gV2 = np.einsum('ijlk,ijr,kir->lkr', dsig, U2, W2)
    
    # gW2[k, i, r] = sum_{j,l} dsig[i, j, l, k] * U2[i, j, r] * V2[l, k, r]
    gW2 = np.einsum('ijlk,ijr,lkr->kir', dsig, U2, V2)
    
    return gU2.reshape(n*m, r), gV2.reshape(m*p, r), gW2.reshape(p*n, r), 1.0

def _local_search_bilinear(U, V, W, t, n, m, p, max_iters, step, tol, deadline):
    best = _objective_bilinear(U, V, W, n, m, p, t)
    U = U.copy(); V = V.copy(); W = W.copy()
    for it in range(max_iters):
        if time.time() > deadline:
            break
        gU, gV, gW, gt = _grad_bilinear(U, V, W, t, n, m, p)
        step_size = step
        for _ in range(20):
            nU = U - step_size * gU
            nV = V - step_size * gV
            nW = W - step_size * gW
            nt = t - step_size * gt
            nv = _objective_bilinear(nU, nV, nW, n, m, p, nt)
            if nv < best:
                U, V, W, t = nU, nV, nW, nt
                best = nv
                break
            step_size *= 0.5
        if step_size < 1e-12:
            break
        if best < tol:
            break
    return U, V, W, t, best

def _verify_bilinear(U, V, W, n, m, p, tol=1e-7):
    K = _compute_K(U, V, W, n, m, p)
    T = _target_tensor_bilinear(n, m, p)
    return np.max(np.abs(K - T)) < tol

def _try_rank_bilinear(r, n, m, p, time_budget, seed):
    rng = random.Random(seed)
    deadline = time.time() + time_budget
    best_obj = float('inf')
    best = None
    n_restarts = max(1, int(time_budget // 2.0))
    for rs in range(n_restarts):
        if time.time() > deadline:
            break
        U = np.random.randn(n*m, r) * 0.5
        V = np.random.randn(m*p, r) * 0.5
        W = np.random.randn(p*n, r) * 0.5
        t = 1.0
        U2, V2, W2, t2, obj = _local_search_bilinear(U, V, W, t, n, m, p, max_iters=100, step=0.05, tol=1e-6, deadline=deadline)
        if obj < best_obj:
            best_obj = obj
            best = (U2, V2, W2, t2)
        if best_obj < 1e-6:
            break
    if best is None:
        return None
    U, V, W, t = best
    if not _verify_bilinear(U, V, W, n, m, p, tol=1e-5):
        return None
    # Polish
    U, V, W, t, obj = _local_search_bilinear(U, V, W, t, n, m, p, max_iters=100, step=0.01, tol=1e-9, deadline=deadline)
    if not _verify_bilinear(U, V, W, n, m, p, tol=1e-7):
        return None
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = _DEADLINE - _SAFE_MARGIN
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    # Try to find a low rank decomposition for 3x2x2
    # If found, combine 4 copies for 3x4x4
    n_sub, m_sub, p_sub = 3, 2, 2
    
    # Search for rank 6
    res_sub = _try_rank_bilinear(6, n_sub, m_sub, p_sub, time_budget=30.0, seed=123)
    
    if res_sub is not None:
        U_sub, V_sub, W_sub = res_sub
        # Construct 3x4x4 from 4 copies
        # A is 3x4, split into A0 (3x2), A1 (3x2)
        # B is 4x4, split into B00, B01, B10, B11 (2x2)
        # C is 3x4, split into C0 (3x2), C1 (3x2)
        # C0 = A0 B00 + A1 B10
        # C1 = A0 B01 + A1 B11
        
        # We need to map the indices.
        # U_sub[i, j, r] corresponds to A_sub[i, j]
        # V_sub[j, k, r] corresponds to B_sub[j, k]
        # W_sub[k, i, r] corresponds to C_sub[k, i] (Note: W index is k*n+i)
        
        # For the full problem:
        # U_full[i, j_full, r_full]
        # j_full = 2*j_sub + j_block (j_block 0 for A0, 1 for A1)
        
        # We have 4 sub-problems.
        # 1. A0 B00 -> contributes to C0
        # 2. A1 B10 -> contributes to C0
        # 3. A0 B01 -> contributes to C1
        # 4. A1 B11 -> contributes to C1
        
        # Total rank = 4 * 6 = 24.
        rank_total = 24
        U = np.zeros((n*m, rank_total))
        V = np.zeros((m*p, rank_total))
        W = np.zeros((p*n, rank_total))
        
        # Block 0: A0 (cols 0,1) * B00 (rows 0,1, cols 0,1) -> C0 (cols 0,1)
        # r_offset = 0
        for i in range(n_sub):
            for j in range(m_sub):
                U[i*m + j, 0:6] = U_sub[i*m_sub + j, :]
            for j in range(m_sub):
                for k in range(p_sub):
                    V[j*p + k, 0:6] = V_sub[j*p_sub + k, :]
            for k in range(p_sub):
                for i in range(n_sub):
                    W[k*n + i, 0:6] = W_sub[k*n_sub + i, :]
                    
        # Block 1: A1 (cols 2,3) * B10 (rows 2,3, cols 0,1) -> C0 (cols 0,1)
        # r_offset = 6
        for i in range(n_sub):
            for j in range(m_sub):
                # A1 corresponds to j_full = 2 + j
                U[i*m + (2+j), 6:12] = U_sub[i*m_sub + j, :]
            for j in range(m_sub):
                for k in range(p_sub):
                    # B10 corresponds to j_full = 2+j, k_full = k
                    V[(2+j)*p + k, 6:12] = V_sub[j*p_sub + k, :]
            for k in range(p_sub):
                for i in range(n_sub):
                    # C0 corresponds to k_full = k
                    W[k*n + i, 6:12] = W_sub[k*n_sub + i, :]

        # Block 2: A0 (cols 0,1) * B01 (rows 0,1, cols 2,3) -> C1 (cols 2,3)
        # r_offset = 12
        for i in range(n_sub):
            for j in range(m_sub):
                U[i*m + j, 12:18] = U_sub[i*m_sub + j, :]
            for j in range(m_sub):
                for k in range(p_sub):
                    # B01 corresponds to j_full = j, k_full = 2+k
                    V[j*p + (2+k), 12:18] = V_sub[j*p_sub + k, :]
            for k in range(p_sub):
                for i in range(n_sub):
                    # C1 corresponds to k_full = 2+k
                    W[(2+k)*n + i, 12:18] = W_sub[k*n_sub + i, :]

        # Block 3: A1 (cols 2,3) * B11 (rows 2,3, cols 2,3) -> C1 (cols 2,3)
        # r_offset = 18
        for i in range(n_sub):
            for j in range(m_sub):
                U[i*m + (2+j), 18:24] = U_sub[i*m_sub + j, :]
            for j in range(m_sub):
                for k in range(p_sub):
                    V[(2+j)*p + (2+k), 18:24] = V_sub[j*p_sub + k, :]
            for k in range(p_sub):
                for i in range(n_sub):
                    W[(2+k)*n + i, 18:24] = W_sub[k*n_sub + i, :]

        # Verify full
        if _verify_bilinear(U, V, W, n, m, p, tol=1e-5):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to schoolbook
    return schoolbook()
