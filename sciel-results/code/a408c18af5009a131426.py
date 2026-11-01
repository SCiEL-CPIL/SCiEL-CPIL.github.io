"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import least_squares

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                if abs(val - 1.0) > tol:
                    return False
    return True

def _build_from_terms(all_terms, n, m, p):
    """Build U, V, W matrices from a list of (u_vec, v_vec, w_vec) terms."""
    rank = len(all_terms)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    for r, (u_vec, v_vec, w_vec) in enumerate(all_terms):
        U[:, r] = u_vec
        V[:, r] = v_vec
        W[:, r] = w_vec
    return U, V, W

def _strassen_2x2_terms():
    """Return the 7 Strassen terms for 2x2x2 matrix multiplication."""
    terms = []
    # M1 = (a11+a22)(b11+b22)
    terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
    # M2 = (a21+a22)b11
    terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 1]))
    # M3 = a11(b12-b22)
    terms.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]))
    # M4 = a22(b21-b11)
    terms.append(([0, 0, 0, 1], [0, 0, 1, -1], [1, 1, 0, 0]))
    # M5 = (a11+a12)b22
    terms.append(([1, 1, 0, 0], [0, 0, 0, 1], [1, 1, 0, 0]))
    # M6 = (a21-a11)(b11+b12)
    terms.append(([1, 0, -1, 0], [1, 1, 0, 0], [0, 1, 0, 0]))
    # M7 = (a12-a22)(b21+b22)
    terms.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 1]))
    return terms

def _make_schoolbook_terms(n, m, p, k_start=0, k_end=None):
    """Make schoolbook terms for columns k_start to k_end-1."""
    if k_end is None:
        k_end = p
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(k_start, k_end):
                u_vec = [0.0] * (n * m)
                u_vec[i * m + j] = 1.0
                v_vec = [0.0] * (m * p)
                v_vec[j * p + k] = 1.0
                w_vec = [0.0] * (p * n)
                w_vec[k * n + i] = 1.0
                terms.append((u_vec, v_vec, w_vec))
    return terms

def _extend_2x2_term_to_2x5(u_vec, v_vec_2x2, w_vec_2x2, n=2, m=2, p=5):
    """Extend a 2x2x2 term to a 2x2x5 term (only uses k=0,1)."""
    u_full = list(u_vec)
    v_full = [0.0] * (m * p)
    for j in range(m):
        for k in range(2):
            v_full[j * p + k] = v_vec_2x2[j * 2 + k]
    w_full = [0.0] * (p * n)
    for k in range(2):
        for i in range(n):
            w_full[k * n + i] = w_vec_2x2[k * 2 + i]
    return u_full, v_full, w_full

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    n, m, p = DIMENSIONS  # 2, 2, 5
    nm, mp, pn = n*m, m*p, p*n
    
    # Build target tensor T
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    # Initial solution: Rank 19 (Strassen + Schoolbook)
    strassen_terms = _strassen_2x2_terms()
    all_terms = []
    for u_vec, v_vec, w_vec in strassen_terms:
        u_full, v_full, w_full = _extend_2x2_term_to_2x5(u_vec, v_vec, w_vec, n, m, p)
        all_terms.append((u_full, v_full, w_full))
    
    schoolbook_terms = _make_schoolbook_terms(n, m, p, k_start=2, k_end=5)
    all_terms.extend(schoolbook_terms)
    
    U, V, W = _build_from_terms(all_terms, n, m, p)
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback
        rank = n * m * p
        U_fb = np.zeros((nm, rank))
        V_fb = np.zeros((mp, rank))
        W_fb = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    # Try to reduce rank using optimization
    def make_residual(r):
        def residual(x):
            U = x[:nm*r].reshape(nm, r)
            V = x[nm*r:nm*r+mp*r].reshape(mp, r)
            W = x[nm*r+mp*r:].reshape(pn, r)
            R = np.einsum('ad,bd,cd->abc', U, V, W)
            return (R - T).flatten()
        return residual

    def schoolbook_init(r):
        # Initialize with schoolbook terms, zero-padding if r < nm*mp*pn
        U0 = np.zeros((nm, r))
        V0 = np.zeros((mp, r))
        W0 = np.zeros((pn, r))
        for c in range(r):
            if c < n*m*p:
                i = c // (m*p)
                rem = c % (m*p)
                j = rem // p
                k = rem % p
                U0[i*m+j, c] = 1.0
                V0[j*p+k, c] = 1.0
                W0[k*n+i, c] = 1.0
            else:
                # Random init for extra slots? Or just 0?
                # Better to use random small values to break symmetry
                U0[:, c] = np.random.randn(nm) * 0.1
                V0[:, c] = np.random.randn(mp) * 0.1
                W0[:, c] = np.random.randn(pn) * 0.1
        return np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])

    # Try ranks from best_rank - 1 down to 10
    for target_rank in range(best_rank - 1, 9, -1):
        if time.time() > deadline - 20:
            break
        
        residual_fn = make_residual(target_rank)
        
        # Try a few initializations
        found = False
        for attempt in range(10):
            if time.time() > deadline - 10:
                break
            
            if attempt == 0:
                # Use the current best solution truncated/padded?
                # Or schoolbook init
                x0 = schoolbook_init(target_rank)
            else:
                # Random init
                np.random.seed(attempt * 7919 + target_rank * 31)
                x0 = np.random.randn(nm*target_rank + mp*target_rank + pn*target_rank) * 0.1
            
            try:
                result = least_squares(residual_fn, x0, method='lm', max_nfev=10000)
                if result.cost < 1e-10:
                    # Found a solution
                    U_sol = result.x[:nm*target_rank].reshape(nm, target_rank)
                    V_sol = result.x[nm*target_rank:nm*target_rank+mp*target_rank].reshape(mp, target_rank)
                    W_sol = result.x[nm*target_rank+mp*target_rank:].reshape(pn, target_rank)
                    
                    # Verify
                    if _verify_decomposition(U_sol, V_sol, W_sol, n, m, p):
                        best_rank = target_rank
                        best_U, best_V, best_W = U_sol, V_sol, W_sol
                        found = True
                        break
            except Exception:
                pass
        
        if found:
            continue # Try lower rank
        else:
            # If not found, we can break early if we assume rank is monotonic?
            # Not necessarily, but likely.
            pass

    # Clean up: convert near-integers to exact integers
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    for mat in [U_list, V_list, W_list]:
        for row in mat:
            for idx, val in enumerate(row):
                if abs(val - round(val)) < 1e-10:
                    row[idx] = round(val)
                elif abs(val - (round(val*2)/2.0)) < 1e-10:
                    row[idx] = round(val*2)/2.0

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
