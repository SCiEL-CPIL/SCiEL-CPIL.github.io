"""Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 7 matrix."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 7]

def _verify_decomposition(U, V, W, n, m, p, rank, tol=1e-6):
    """Check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _snap_to_rational(arr, tol=1e-6):
    """Snap each value independently to nearest integer or half-integer."""
    result = np.array(arr, dtype=float)
    lst = []
    for row in result:
        r = []
        for v in row:
            # Try integer
            iv = round(v)
            if abs(v - iv) < tol:
                r.append(int(iv))
            else:
                # Try half-integer
                hv = round(v * 2) / 2.0
                if abs(v - hv) < tol:
                    if abs(hv - round(hv)) < 1e-9:
                        r.append(int(round(hv)))
                    else:
                        r.append(hv)
                else:
                    # Fall back to nearest half-integer anyway
                    if abs(v - round(v)) < abs(v - round(v*2)/2.0):
                        r.append(int(round(v)))
                    else:
                        r.append(round(v * 2) / 2.0)
        lst.append(r)
    return lst

def _construct_rank5_2x3x3():
    """
    Construct a rank-5 decomposition for 2x3 * 3x3 = 2x3 multiplication.
    Uses ALS with structured initialization.
    """
    n, m, p = 2, 3, 3
    rank = 5
    nm, mp, pn = n*m, m*p, p*n
    
    best_err = float('inf')
    best_U, best_V, best_W = None, None, None
    
    for trial in range(30):
        np.random.seed(42 + trial)
        U = np.random.randn(nm, rank) * 0.5
        V = np.random.randn(mp, rank) * 0.5
        W = np.random.randn(pn, rank) * 0.5
        
        for it in range(800):
            # Update W
            U3 = U.reshape(n, m, rank)
            V3 = V.reshape(m, p, rank)
            M = np.einsum('ijc,jkc->kic', U3, V3)
            for k in range(p):
                for i in range(n):
                    vec = M[k, i, :]
                    norm_sq = np.dot(vec, vec)
                    if norm_sq > 1e-14:
                        W[k*n+i, :] = vec / norm_sq
            # Update U
            W3 = W.reshape(p, n, rank)
            N = np.einsum('jkc,kic->ijc', V3, W3)
            for i in range(n):
                for j in range(m):
                    vec = N[i, j, :]
                    norm_sq = np.dot(vec, vec)
                    if norm_sq > 1e-14:
                        U[i*m+j, :] = vec / norm_sq
            # Update V
            U3 = U.reshape(n, m, rank)
            P = np.einsum('ijc,kic->jkc', U3, W3)
            for j in range(m):
                for k in range(p):
                    vec = P[j, k, :]
                    norm_sq = np.dot(vec, vec)
                    if norm_sq > 1e-14:
                        V[j*p+k, :] = vec / norm_sq
        
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        recon = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        err = np.max(np.abs(recon - 1.0))
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if err < 1e-8:
                break
    
    if best_err < 1e-6:
        return best_U, best_V, best_W, True
    return best_U, best_V, best_W, False

def _construct_rank6_2x3x1():
    """
    Construct a rank-6 decomposition for 2x3 * 3x1 = 2x1 multiplication.
    This is essentially just the schoolbook: 6 independent products.
    C[0] = a1*b1 + a2*b2 + a3*b3
    C[1] = a4*b1 + a5*b2 + a6*b3
    """
    n, m, p = 2, 3, 1
    rank = 6
    nm, mp, pn = n*m, m*p, p*n
    
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    
    return U, V, W, True

def _compose_decompositions(U1, V1, W1, U2, V2, W2, n1, m1, p1, n2, m2, p2):
    """Placeholder - not used in current strategy."""
    pass

def construct():
    deadline = time.time() + 145
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Schoolbook baseline (rank 42)
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1.0
                V_sb[j*p+k, col] = 1.0
                W_sb[k*n+i, col] = 1.0
                col += 1
    
    best = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_sb
    
    # Strategy 1: Compose rank-5 2x3x3 and rank-6 2x3x1 -> rank 16
    if time.time() < deadline - 10:
        U5, V5, W5, ok5 = _construct_rank5_2x3x3()
        U6, V6, W6, ok6 = _construct_rank6_2x3x1()
        
        if ok5 and ok6:
            rank_composed = 5 + 5 + 6
            U_comp = np.zeros((nm, rank_composed))
            V_comp = np.zeros((mp, rank_composed))
            W_comp = np.zeros((pn, rank_composed))
            
            # Block 1: A * B1 (cols 0-2)
            U_comp[:, 0:5] = U5
            for j in range(3):
                for k in range(3):
                    V_comp[j*7+k, 0:5] = V5[j*3+k, :]
            for k in range(3):
                for i in range(2):
                    W_comp[k*2+i, 0:5] = W5[k*2+i, :]
            
            # Block 2: A * B2 (cols 3-5)
            U_comp[:, 5:10] = U5
            for j in range(3):
                for k in range(3, 6):
                    V_comp[j*7+k, 5:10] = V5[j*3+(k-3), :]
            for k in range(3):
                for i in range(2):
                    W_comp[k*2+i, 5:10] = W5[k*2+i, :]
            
            # Block 3: A * B3 (col 6)
            U_comp[:, 10:16] = U6
            for j in range(3):
                V_comp[j*7+6, 10:16] = V6[j, :]
            for i in range(2):
                W_comp[6*2+i, 10:16] = W6[i, :]
            
            if _verify_decomposition(U_comp, V_comp, W_comp, n, m, p, rank_composed, tol=1e-4):
                U_list = _snap_to_rational(U_comp)
                V_list = _snap_to_rational(V_comp)
                W_list = _snap_to_rational(W_comp)
                if _verify_decomposition(np.array(U_list, dtype=float), np.array(V_list, dtype=float), np.array(W_list, dtype=float), n, m, p, rank_composed, tol=1e-4):
                    best = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = rank_composed
    
    # Strategy 2: Try direct ALS for lower ranks
    if time.time() < deadline - 15:
        for rank in range(min(best_rank - 1, 18), 13, -1):
            if time.time() > deadline - 12:
                break
            
            found = False
            for attempt in range(5):
                if time.time() > deadline - 10:
                    break
                
                np.random.seed(rank * 1000 + attempt * 7)
                U = np.random.randn(nm, rank) * 0.3
                V = np.random.randn(mp, rank) * 0.3
                W = np.random.randn(pn, rank) * 0.3
                
                for it in range(400):
                    if time.time() > deadline - 8:
                        break
                    U3 = U.reshape(n, m, rank)
                    V3 = V.reshape(m, p, rank)
                    M = np.einsum('ijc,jkc->kic', U3, V3)
                    for k in range(p):
                        for i in range(n):
                            vec = M[k, i, :]
                            ns = np.dot(vec, vec)
                            if ns > 1e-14:
                                W[k*n+i, :] = vec / ns
                    W3 = W.reshape(p, n, rank)
                    N = np.einsum('jkc,kic->ijc', V3, W3)
                    for i in range(n):
                        for j in range(m):
                            vec = N[i, j, :]
                            ns = np.dot(vec, vec)
                            if ns > 1e-14:
                                U[i*m+j, :] = vec / ns
                    U3 = U.reshape(n, m, rank)
                    P = np.einsum('ijc,kic->jkc', U3, W3)
                    for j in range(m):
                        for k in range(p):
                            vec = P[j, k, :]
                            ns = np.dot(vec, vec)
                            if ns > 1e-14:
                                V[j*p+k, :] = vec / ns
                
                U3 = U.reshape(n, m, rank)
                V3 = V.reshape(m, p, rank)
                W3 = W.reshape(p, n, rank)
                recon = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
                err = np.max(np.abs(recon - 1.0))
                
                if err < 1e-6:
                    U_list = _snap_to_rational(U)
                    V_list = _snap_to_rational(V)
                    W_list = _snap_to_rational(W)
                    if _verify_decomposition(np.array(U_list, dtype=float), np.array(V_list, dtype=float), np.array(W_list, dtype=float), n, m, p, rank, tol=1e-4):
                        best = {"U": U_list, "V": V_list, "W": W_list}
                        best_rank = rank
                        found = True
                        break
            
            if found:
                continue
    
    return best
