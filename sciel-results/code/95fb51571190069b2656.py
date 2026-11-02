import numpy as np
import time
import itertools

DIMENSIONS = [2, 4, 4]

def _build_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n))
    for i_out in range(n):
        for j in range(m):
            for k_out in range(p):
                T[i_out*m + j, j*p + k_out, k_out*n + i_out] = 1.0
    return T

def _verify(U, V, W, T, tol=1e-10):
    R = U.shape[1]
    Tc = np.zeros_like(T)
    for r in range(R):
        Tc += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
    return np.allclose(Tc, T, atol=tol)

def _strassen_terms_2x2():
    terms = []
    u = {0: 1, 3: 1}
    v = {0: 1, 3: 1}
    w = {0: 1, 3: 1}
    terms.append((u, v, w))
    u = {2: 1, 3: 1}
    v = {0: 1}
    w = {2: 1, 3: -1}
    terms.append((u, v, w))
    u = {0: 1}
    v = {1: 1, 3: -1}
    w = {1: 1, 3: 1}
    terms.append((u, v, w))
    u = {3: 1}
    v = {2: 1, 0: -1}
    w = {0: 1, 2: 1}
    terms.append((u, v, w))
    u = {0: 1, 1: 1}
    v = {3: 1}
    w = {0: -1, 1: 1}
    terms.append((u, v, w))
    u = {2: 1, 0: -1}
    v = {0: 1, 1: 1}
    w = {3: 1}
    terms.append((u, v, w))
    u = {1: 1, 3: -1}
    v = {2: 1, 3: 1}
    w = {0: 1}
    terms.append((u, v, w))
    return terms

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    T = _build_tensor()
    T_flat = T.flatten()
    
    # Build the rank-28 Strassen construction
    R = 28
    U = np.zeros((n * m, R), dtype=float)
    V = np.zeros((m * p, R), dtype=float)
    W = np.zeros((p * n, R), dtype=float)

    S = _strassen_terms_2x2()

    def mA(li, off):
        r, c = li // 2, li % 2
        return r * m + (off * 2 + c)

    def mB(li, ro, co):
        r, c = li // 2, li % 2
        return (ro * 2 + r) * p + (co * 2 + c)

    def mC(li, co):
        r, c = li // 2, li % 2
        return (co * 2 + c) * n + r

    blocks = [
        (0, 0, 0, 0, 0),
        (1, 1, 0, 0, 7),
        (0, 0, 1, 1, 14),
        (1, 1, 1, 1, 21),
    ]

    for a_off, b_ro, b_co, c_co, base in blocks:
        if time.time() > deadline:
            break
        for j, (u_loc, v_loc, w_loc) in enumerate(S):
            ci = base + j
            for li, cf in u_loc.items():
                U[mA(li, a_off), ci] += cf
            for li, cf in v_loc.items():
                V[mB(li, b_ro, b_co), ci] += cf
            for li, cf in w_loc.items():
                W[mC(li, c_co), ci] += cf

    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = R

    # Verify the construction
    if not _verify(best_U, best_V, best_W, T):
        R = 32
        U = np.zeros((n*m, R), dtype=float)
        V = np.zeros((m*p, R), dtype=float)
        W = np.zeros((p*n, R), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i*m*p + j*p + k
                    U[i*m+j, r] = 1
                    V[j*p+k, r] = 1
                    W[k*n+i, r] = 1
        best_U, best_V, best_W = U, V, W
        best_rank = R

    # Pre-compute the "basis" tensors for each of the 28 terms
    basis_tensors = []
    for r in range(R):
        t = np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
        basis_tensors.append(t.flatten())
    
    # Try removing 1 term (Rank 27)
    found_27 = False
    if time.time() < deadline:
        for r_remove in range(R):
            keep = [r for r in range(R) if r != r_remove]
            if not keep:
                continue
            
            A = np.column_stack([basis_tensors[i] for i in keep])
            
            try:
                mu, residuals, rank, sv = np.linalg.lstsq(A, T_flat, rcond=None)
            except Exception:
                continue
            
            if residuals and residuals[0] < 1e-8:
                lambdas = []
                for m_val in mu:
                    if m_val < 0:
                        lam = -np.cbrt(-m_val)
                    else:
                        lam = np.cbrt(m_val)
                    lambdas.append(lam)
                    
                U_new = np.zeros((n*m, R-1))
                V_new = np.zeros((m*p, R-1))
                W_new = np.zeros((p*n, R-1))
                
                for idx, r_keep in enumerate(keep):
                    U_new[:, idx] = U[:, r_keep] * lambdas[idx]
                    V_new[:, idx] = V[:, r_keep] * lambdas[idx]
                    W_new[:, idx] = W[:, r_keep] * lambdas[idx]
                
                if _verify(U_new, V_new, W_new, T):
                    best_U, best_V, best_W = U_new, V_new, W_new
                    best_rank = R - 1
                    found_27 = True
                    break
    
    # Try removing 2 terms (Rank 26)
    if not found_27 and best_rank > 26 and time.time() < deadline:
        for r1 in range(R):
            for r2 in range(r1+1, R):
                if time.time() > deadline:
                    break
                keep = [r for r in range(R) if r != r1 and r != r2]
                if not keep:
                    continue
                
                A = np.column_stack([basis_tensors[i] for i in keep])
                try:
                    mu, residuals, rank, sv = np.linalg.lstsq(A, T_flat, rcond=None)
                except Exception:
                    continue
                
                if residuals and residuals[0] < 1e-8:
                    lambdas = []
                    for m_val in mu:
                        if m_val < 0:
                            lam = -np.cbrt(-m_val)
                        else:
                            lam = np.cbrt(m_val)
                        lambdas.append(lam)
                    
                    U_new = np.zeros((n*m, R-2))
                    V_new = np.zeros((m*p, R-2))
                    W_new = np.zeros((p*n, R-2))
                    
                    for idx, r_keep in enumerate(keep):
                        U_new[:, idx] = U[:, r_keep] * lambdas[idx]
                        V_new[:, idx] = V[:, r_keep] * lambdas[idx]
                        W_new[:, idx] = W[:, r_keep] * lambdas[idx]
                    
                    if _verify(U_new, V_new, W_new, T):
                        best_U, best_V, best_W = U_new, V_new, W_new
                        best_rank = R - 2
                        break
            if best_rank == R - 2:
                break

    if time.time() > deadline:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    A = np.arange(n * m).reshape(n, m) + 1.0
    B = np.arange(m * p).reshape(m, p) + 2.0
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", U.shape[1])
