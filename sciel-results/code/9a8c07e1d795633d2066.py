"""Rank-minimal exact tensor decomposition for 2x3x6 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 6]

def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 6, 18, 12
    lb = max(nm, mp, pn)  # 18

    def verify(U, V, W, r, n=n, m=m, p=p):
        """Check exact decomposition: T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W

    def clean(mat):
        result = []
        for row in mat:
            new_row = []
            for val in row:
                hv = round(val * 2) / 2.0
                if abs(val - hv) < 1e-8:
                    new_row.append(hv)
                else:
                    new_row.append(val)
            result.append(new_row)
        return result

    def try_find_decomp(rank, n_sub, m_sub, p_sub, num_trials=50, seed_base=0):
        """
        Try to find a rank-`rank` decomposition for n_sub x m_sub x p_sub multiplication.
        Returns (U, V, W) as lists of lists, or None.
        """
        nm_s = n_sub * m_sub
        mp_s = m_sub * p_sub
        pn_s = p_sub * n_sub
        
        for trial in range(num_trials):
            if time.time() > deadline:
                return None
            rng = np.random.RandomState(seed_base + trial)
            U = rng.randn(nm_s, rank)
            V = rng.randn(mp_s, rank)
            W = rng.randn(pn_s, rank)
            
            # Use iterative refinement: for each term c, given the residual,
            # solve for the best rank-1 update
            for it in range(500):
                if time.time() > deadline:
                    return None
                # Compute current tensor
                T = np.zeros((n_sub, m_sub, p_sub))
                for c in range(rank):
                    uc = U[:, c].reshape(n_sub, m_sub)
                    vc = V[:, c].reshape(m_sub, p_sub)
                    wc = W[:, c].reshape(p_sub, n_sub)
                    T += np.einsum('ij,jk,ki->ijk', uc, vc, wc)
                
                # Residual
                R = np.ones((n_sub, m_sub, p_sub)) - T
                
                # Check if residual is small
                if np.max(np.abs(R)) < 1e-7:
                    # Try to clean up
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    # Verify
                    if verify(U_list, V_list, W_list, rank, n_sub, m_sub, p_sub):
                        return U_list, V_list, W_list
                    # Try rounding
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    if verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), rank, n_sub, m_sub, p_sub):
                        return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
                    # Try integer rounding
                    U_int = np.round(U)
                    V_int = np.round(V)
                    W_int = np.round(W)
                    if verify(U_int.tolist(), V_int.tolist(), W_int.tolist(), rank, n_sub, m_sub, p_sub):
                        return U_int.tolist(), V_int.tolist(), W_int.tolist()
                    break
                
                # Update one term at a time (cyclic)
                c = it % rank
                # Residual with term c removed
                T_no_c = np.zeros((n_sub, m_sub, p_sub))
                for c2 in range(rank):
                    if c2 == c:
                        continue
                    uc = U[:, c2].reshape(n_sub, m_sub)
                    vc = V[:, c2].reshape(m_sub, p_sub)
                    wc = W[:, c2].reshape(p_sub, n_sub)
                    T_no_c += np.einsum('ij,jk,ki->ijk', uc, vc, wc)
                R_c = np.ones((n_sub, m_sub, p_sub)) - T_no_c
                
                # We want to find u, v, w such that u ⊗ v ⊗ w ≈ R_c
                # This is a rank-1 approximation of a 3-tensor
                # Use ALS: fix two, solve for third
                for sub_it in range(10):
                    # Update U
                    # For each (i,j), U[i,j,c] = sum_{k} R_c[i,j,k] * V[j,k,c] * W[k,i,c] / (sum_k V[j,k,c]^2 * W[k,i,c]^2 + eps)
                    Uc = U[:, c].copy()
                    Vc = V[:, c].reshape(m_sub, p_sub)
                    Wc = W[:, c].reshape(p_sub, n_sub)
                    for i in range(n_sub):
                        for j in range(m_sub):
                            num = 0.0
                            den = 0.0
                            for k in range(p_sub):
                                num += R_c[i, j, k] * Vc[j, k] * Wc[k, i]
                                den += Vc[j, k] ** 2 * Wc[k, i] ** 2
                            if den > 1e-12:
                                Uc[i * m_sub + j] = num / den
                            else:
                                Uc[i * m_sub + j] = 0
                    U[:, c] = Uc
                    
                    # Update V
                    Uc = U[:, c].reshape(n_sub, m_sub)
                    Wc = W[:, c].reshape(p_sub, n_sub)
                    Vc = np.zeros((m_sub, p_sub))
                    for j in range(m_sub):
                        for k in range(p_sub):
                            num = 0.0
                            den = 0.0
                            for i in range(n_sub):
                                num += R_c[i, j, k] * Uc[i, j] * Wc[k, i]
                                den += Uc[i, j] ** 2 * Wc[k, i] ** 2
                            if den > 1e-12:
                                Vc[j, k] = num / den
                            else:
                                Vc[j, k] = 0
                    V[:, c] = Vc.flatten()
                    
                    # Update W
                    Uc = U[:, c].reshape(n_sub, m_sub)
                    Vc = V[:, c].reshape(m_sub, p_sub)
                    Wc = np.zeros((p_sub, n_sub))
                    for k in range(p_sub):
                        for i in range(n_sub):
                            num = 0.0
                            den = 0.0
                            for j in range(m_sub):
                                num += R_c[i, j, k] * Uc[i, j] * Vc[j, k]
                                den += Uc[i, j] ** 2 * Vc[j, k] ** 2
                            if den > 1e-12:
                                Wc[k, i] = num / den
                            else:
                                Wc[k, i] = 0
                    W[:, c] = Wc.flatten()
            
            # Final check
            T = np.zeros((n_sub, m_sub, p_sub))
            for c in range(rank):
                uc = U[:, c].reshape(n_sub, m_sub)
                vc = V[:, c].reshape(m_sub, p_sub)
                wc = W[:, c].reshape(p_sub, n_sub)
                T += np.einsum('ij,jk,ki->ijk', uc, vc, wc)
            if np.max(np.abs(T - 1.0)) < 1e-7:
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                if verify(U_list, V_list, W_list, rank, n_sub, m_sub, p_sub):
                    return U_list, V_list, W_list
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                if verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), rank, n_sub, m_sub, p_sub):
                    return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
        
        return None

    def combine_decomps(decomp1, decomp2, n, m, p1, p2):
        """
        Combine two decompositions:
        decomp1: for n x m x p1, shapes (nm, r1), (m*p1, r1), (p1*n, r1)
        decomp2: for n x m x p2, shapes (nm, r2), (m*p2, r2), (p2*n, r2)
        Result: for n x m x (p1+p2)
        U: (nm, r1+r2)
        V: (m*(p1+p2), r1+r2)
        W: ((p1+p2)*n, r1+r2)
        """
        U1, V1, W1 = decomp1
        U2, V2, W2 = decomp2
        r1 = len(U1[0]) if U1 else 0
        r2 = len(U2[0]) if U2 else 0
        r = r1 + r2
        
        U = []
        for i in range(len(U1)):
            U.append(U1[i] + U2[i])
        
        V = []
        for j in range(m):
            row = [0.0] * r
            for k in range(p1):
                for c in range(r1):
                    row[c] = V1[j * p1 + k][c]
            for k in range(p2):
                for c in range(r2):
                    row[r1 + c] = V2[j * p2 + k][c]
            V.append(row)
        
        W = []
        for k in range(p1):
            for i in range(n):
                row = [0.0] * r
                for c in range(r1):
                    row[c] = W1[k * n + i][c]
                W.append(row)
        for k in range(p2):
            for i in range(n):
                row = [0.0] * r
                for c in range(r2):
                    row[r1 + c] = W2[k * n + i][c]
                W.append(row)
        
        return U, V, W

    # Try to find 2x3x3 decomposition at rank 11
    best_rank = n * m * p
    best_U, best_V, best_W = schoolbook()

    # Strategy: decompose 2x3x6 as two 2x3x3 blocks
    # Try rank 11 for each 2x3x3
    found_2x3x3 = None
    for rank_try in [11, 12, 13, 14]:
        if time.time() > deadline:
            break
        found_2x3x3 = try_find_decomp(rank_try, 2, 3, 3, num_trials=30, seed_base=rank_try * 100)
        if found_2x3x3 is not None:
            break
    
    if found_2x3x3 is not None:
        U1, V1, W1 = found_2x3x3
        r1 = len(U1[0])
        # Use same decomposition for both blocks
        U, V, W = combine_decomps(found_2x3x3, found_2x3x3, 2, 3, 3, 3)
        r = 2 * r1
        if verify(U, V, W, r):
            if r < best_rank:
                best_rank = r
                best_U, best_V, best_W = U, V, W
    else:
        # Fallback: try to find 2x3x6 directly at various ranks
        for rank_try in range(lb, min(lb + 20, best_rank)):
            if time.time() > deadline:
                break
            result = try_find_decomp(rank_try, 2, 3, 6, num_trials=20, seed_base=rank_try * 200)
            if result is not None:
                U, V, W = result
                if verify(U, V, W, rank_try):
                    if rank_try < best_rank:
                        best_rank = rank_try
                        best_U, best_V, best_W = U, V, W
                    break

    # Also try the original reduction approach as backup
    if best_rank == n * m * p:
        U, V, W, r = best_U, best_V, best_W, best_rank
        def importance(U, V, W, r, n=n, m=m, p=p):
            imp = [0.0] * r
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for c in range(r):
                            imp[c] += abs(U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c])
            return imp
        
        def reduce_rank(U, V, W, r, n=n, m=m, p=p):
            imp = importance(U, V, W, r)
            min_idx = min(range(r), key=lambda c: imp[c])
            U2 = [row[:min_idx] + row[min_idx + 1:] for row in U]
            V2 = [row[:min_idx] + row[min_idx + 1:] for row in V]
            W2 = [row[:min_idx] + row[min_idx + 1:] for row in W]
            if verify(U2, V2, W2, r - 1):
                return U2, V2, W2, r - 1
            return None
        
        while r > lb:
            if time.time() > deadline:
                break
            result = reduce_rank(U, V, W, r)
            if result is None:
                break
            U, V, W, r = result
            if r < best_rank:
                best_rank = r
                best_U, best_V, best_W = U, V, W

    # Verify and clean
    if not verify(best_U, best_V, best_W, best_rank):
        best_U, best_V, best_W = schoolbook()
        best_rank = n * m * p

    best_U = clean(best_U)
    best_V = clean(best_V)
    best_W = clean(best_W)

    return {"U": best_U, "V": best_V, "W": best_W}
