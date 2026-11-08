import numpy as np
import time
import itertools
import random

DIMENSIONS = [2, 3, 3]

def _build_rank17():
    n, m, p = DIMENSIONS
    U = np.zeros((n * m, 17))
    V = np.zeros((m * p, 17))
    W = np.zeros((p * n, 17))
    
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1
    
    U[2, 7] = 1
    V[6, 7] = 1
    W[0, 7] = 1
    
    U[2, 8] = 1
    V[7, 8] = 1
    W[2, 8] = 1
    
    U[5, 9] = 1
    V[6, 9] = 1
    W[1, 9] = 1
    
    U[5, 10] = 1
    V[7, 10] = 1
    W[3, 10] = 1
    
    U[0, 11] = 1
    V[2, 11] = 1
    W[4, 11] = 1
    
    U[1, 12] = 1
    V[5, 12] = 1
    W[4, 12] = 1
    
    U[2, 13] = 1
    V[8, 13] = 1
    W[4, 13] = 1
    
    U[3, 14] = 1
    V[2, 14] = 1
    W[5, 14] = 1
    
    U[4, 15] = 1
    V[5, 15] = 1
    W[5, 15] = 1
    
    U[5, 16] = 1
    V[8, 16] = 1
    W[5, 16] = 1
    
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-9):
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Vectorized verification."""
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    result = np.zeros((n, m, p))
    for r in range(rank):
        Ur = U3[:, :, r]
        Vr = V3[:, :, r]
        Wr = W3[:, :, r]
        for i in range(n):
            for j in range(m):
                if Ur[i,j] != 0:
                    result[i,j,:] += Ur[i,j] * Vr[j,:] * Wr[:,i]
    return np.all(np.abs(result - 1.0) < tol)

def _try_remove_term(U, V, W, n, m, p, term_idx):
    rank = U.shape[1]
    U2 = np.delete(U, term_idx, axis=1)
    V2 = np.delete(V, term_idx, axis=1)
    W2 = np.delete(W, term_idx, axis=1)
    return _verify(U2, V2, W2, n, m, p)

def _build_strassen_block(U, V, W, row_pair, col_pair, out_pair, base_idx):
    n, m, p = DIMENSIONS
    r0, r1 = row_pair
    c0, c1 = col_pair
    o0, o1 = out_pair
    
    A00 = r0*m+c0; A01 = r0*m+c1
    A10 = r1*m+c0; A11 = r1*m+c1
    B00 = c0*p+o0; B01 = c0*p+o1
    B10 = c1*p+o0; B11 = c1*p+o1
    C00 = o0*n+r0; C01 = o0*n+r1
    C10 = o1*n+r0; C11 = o1*n+r1
    
    U[A00, base_idx] = 1; U[A11, base_idx] = 1
    V[B00, base_idx] = 1; V[B11, base_idx] = 1
    W[C00, base_idx] = 1; W[C11, base_idx] = 1
    base_idx += 1
    
    U[A10, base_idx] = 1; U[A11, base_idx] = 1
    V[B00, base_idx] = 1
    W[C10, base_idx] = 1; W[C11, base_idx] = -1
    base_idx += 1
    
    U[A00, base_idx] = 1
    V[B01, base_idx] = 1; V[B11, base_idx] = -1
    W[C01, base_idx] = 1; W[C11, base_idx] = 1
    base_idx += 1
    
    U[A11, base_idx] = 1
    V[B10, base_idx] = 1; V[B00, base_idx] = -1
    W[C00, base_idx] = 1; W[C10, base_idx] = 1
    base_idx += 1
    
    U[A00, base_idx] = 1; U[A01, base_idx] = 1
    V[B11, base_idx] = 1
    W[C00, base_idx] = -1; W[C01, base_idx] = 1
    base_idx += 1
    
    U[A10, base_idx] = 1; U[A00, base_idx] = -1
    V[B00, base_idx] = 1; V[B01, base_idx] = 1
    W[C11, base_idx] = 1
    base_idx += 1
    
    U[A01, base_idx] = 1; U[A11, base_idx] = -1
    V[B10, base_idx] = 1; V[B11, base_idx] = 1
    W[C00, base_idx] = 1
    base_idx += 1
    
    return base_idx

def _try_all_subblock_strassen(n, m, p, time_budget):
    start_time = time.time()
    best = None
    
    row_pairs = [(0, 1)]
    col_pairs = [(0,1), (0,2), (1,2)]
    out_pairs = [(0,1), (0,2), (1,2)]
    
    for row_pair in row_pairs:
        for col_pair in col_pairs:
            for out_pair in out_pairs:
                if time.time() - start_time > time_budget:
                    break
                
                U = np.zeros((n*m, 20))
                V = np.zeros((m*p, 20))
                W = np.zeros((p*n, 20))
                
                next_idx = _build_strassen_block(U, V, W, row_pair, col_pair, out_pair, 0)
                
                covered = set()
                r0, r1 = row_pair
                c0, c1 = col_pair
                o0, o1 = out_pair
                for i in (r0, r1):
                    for j in (c0, c1):
                        for k in (o0, o1):
                            covered.add((i, j, k))
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if (i, j, k) not in covered:
                                if next_idx >= 20:
                                    break
                                U[i*m+j, next_idx] = 1
                                V[j*p+k, next_idx] = 1
                                W[k*n+i, next_idx] = 1
                                next_idx += 1
                        if next_idx >= 20:
                            break
                    if next_idx >= 20:
                        break
                    if time.time() - start_time > time_budget:
                        break
                
                if next_idx >= 20:
                    continue
                
                U = U[:, :next_idx]
                V = V[:, :next_idx]
                W = W[:, :next_idx]
                
                if _verify(U, V, W, n, m, p):
                    if best is None or next_idx < best[0]:
                        best = (next_idx, U, V, W)
                
                if time.time() - start_time > time_budget:
                    break
            if time.time() - start_time > time_budget:
                break
        if time.time() - start_time > time_budget:
            break
    
    return best

def _optimize_rank16_from_rank17(U17, V17, W17, n, m, p, time_budget=30):
    start_time = time.time()
    rank17 = U17.shape[1]
    
    for a in range(rank17):
        for b in range(a+1, rank17):
            if time.time() - start_time > time_budget:
                break
            
            fixed_indices = [r for r in range(rank17) if r != a and r != b]
            U_fixed = U17[:, fixed_indices]
            V_fixed = V17[:, fixed_indices]
            W_fixed = W17[:, fixed_indices]
            
            residual = np.ones((n, m, p))
            for r in range(len(fixed_indices)):
                Ur = U_fixed[:, r]
                Vr = V_fixed[:, r]
                Wr = W_fixed[:, r]
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            residual[i,j,k] -= Ur[i*m+j] * Vr[j*p+k] * Wr[k*n+i]
            
            np.random.seed(a * 100 + b)
            c = np.ones(n*m)
            d = np.ones(m*p)
            e = np.ones(p*n)
            
            for iteration in range(100):
                if time.time() - start_time > time_budget:
                    break
                
                for i in range(n):
                    for j in range(m):
                        vals = []
                        for k in range(p):
                            denom = d[j*p+k] * e[k*n+i]
                            if abs(denom) > 1e-15:
                                vals.append(residual[i,j,k] / denom)
                        if vals:
                            c[i*m+j] = np.mean(vals)
                
                for j in range(m):
                    for k in range(p):
                        vals = []
                        for i in range(n):
                            denom = c[i*m+j] * e[k*n+i]
                            if abs(denom) > 1e-15:
                                vals.append(residual[i,j,k] / denom)
                        if vals:
                            d[j*p+k] = np.mean(vals)
                
                for k in range(p):
                    for i in range(n):
                        vals = []
                        for j in range(m):
                            denom = c[i*m+j] * d[j*p+k]
                            if abs(denom) > 1e-15:
                                vals.append(residual[i,j,k] / denom)
                        if vals:
                            e[k*n+i] = np.mean(vals)
            
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = c[i*m+j] * d[j*p+k] * e[k*n+i]
                        error += (val - residual[i,j,k])**2
                        if time.time() - start_time > time_budget:
                            break
                    if time.time() - start_time > time_budget:
                        break
                if time.time() - start_time > time_budget:
                    break
                if error > 1e-6:
                    break
            
            if error < 1e-6:
                U16 = np.zeros((n*m, 16))
                V16 = np.zeros((m*p, 16))
                W16 = np.zeros((p*n, 16))
                
                for new_idx, old_idx in enumerate(fixed_indices):
                    U16[:, new_idx] = U17[:, old_idx]
                    V16[:, new_idx] = V17[:, old_idx]
                    W16[:, new_idx] = W17[:, old_idx]
                
                U16[:, 15] = c
                V16[:, 15] = d
                W16[:, 15] = e
                
                if _verify(U16, V16, W16, n, m, p):
                    return U16, V16, W16
        
        if time.time() - start_time > time_budget:
            break
    
    return None

def _search_rank16_by_perturbation(U, V, W, n, m, p, max_iters=5000, time_budget=20):
    """Try to find a rank-16 solution by perturbing and projecting"""
    start_time = time.time()
    for _ in range(max_iters):
        if time.time() - start_time > time_budget:
            break
        
        U2 = U.copy()
        V2 = V.copy()
        W2 = W.copy()
        idx = random.randint(0, 16)
        U2[:, idx] = 0
        V2[:, idx] = 0
        W2[:, idx] = 0
        
        if _verify(U2, V2, W2, n, m, p):
            mask = np.ones(17, dtype=bool)
            mask[idx] = False
            U3 = U2[:, mask]
            V3 = V2[:, mask]
            W3 = W2[:, mask]
            if _verify(U3, V3, W3, n, m, p):
                return U3, V3, W3
    
    return None

def construct():
    start_time = time.time()
    deadline = 130.0
    
    n, m, p = DIMENSIONS
    
    U, V, W = _build_rank17()
    
    if not _verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_rank = 17
    best_U, best_V, best_W = U, V, W
    
    for idx in range(17):
        if time.time() > deadline:
            break
        if _try_remove_term(U, V, W, n, m, p, idx):
            mask = np.ones(17, dtype=bool)
            mask[idx] = False
            U16 = U[:, mask]
            V16 = V[:, mask]
            W16 = W[:, mask]
            if _verify(U16, V16, W16, n, m, p):
                best_rank = 16
                best_U, best_V, best_W = U16, V16, W16
                break
    
    if best_rank > 16:
        if time.time() < deadline - 10:
            result = _try_all_subblock_strassen(n, m, p, time_budget=20)
            if result is not None:
                r, U_new, V_new, W_new = result
                if r < best_rank:
                    best_rank = r
                    best_U, best_V, best_W = U_new, V_new, W_new
    
    if best_rank > 16:
        if time.time() < deadline - 15:
            result = _optimize_rank16_from_rank17(U, V, W, n, m, p, time_budget=30)
            if result is not None:
                best_rank = 16
                best_U, best_V, best_W = result
    
    if best_rank > 16:
        if time.time() < deadline - 10:
            result = _search_rank16_by_perturbation(U, V, W, n, m, p, max_iters=10000, time_budget=20)
            if result is not None:
                best_rank = 16
                best_U, best_V, best_W = result
    
    if best_rank > 15:
        if time.time() < deadline - 5:
            for idx1 in range(17):
                if time.time() > deadline:
                    break
                for idx2 in range(idx1 + 1, 17):
                    mask = np.ones(17, dtype=bool)
                    mask[idx1] = False
                    mask[idx2] = False
                    U15 = U[:, mask]
                    V15 = V[:, mask]
                    W15 = W[:, mask]
                    if _verify(U15, V15, W15, n, m, p):
                        best_rank = 15
                        best_U, best_V, best_W = U15, V15, W15
                        break
                if best_rank == 15:
                    break
    
    if not _verify(best_U, best_V, best_W, n, m, p):
        U, V, W = _build_rank17()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
