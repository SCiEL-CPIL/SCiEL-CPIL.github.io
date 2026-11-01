"""Search for a low-rank exact tensor decomposition of (2,2,8) matrix multiplication."""
import time
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [2, 2, 8]
N, M, P = DIMENSIONS
NM = N * M   # 4
MP = M * P   # 16
PN = P * N   # 16
SCHOOBOOK_RANK = N * M * P  # 32


def verify(U, V, W, n=N, m=M, p=P):
    """Verify that U, V, W form an exact decomposition. Returns True/False."""
    nm, mp, pn = n*m, m*p, p*n
    rank = len(U[0]) if U else 0
    # Check shapes
    if len(U) != nm or len(V) != mp or len(W) != pn:
        return False
    for row in U:
        if len(row) != rank:
            return False
    for row in V:
        if len(row) != rank:
            return False
    for row in W:
        if len(row) != rank:
            return False
    # Build a set of valid (a,b,c) indices
    valid = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid.add((i*m+j, j*p+k, k*n+i))
    # Check all entries
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                s = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                expected = 1 if (a, b, c) in valid else 0
                if s != expected:
                    return False
    return True


def verify_fast(U, V, W, n=N, m=M, p=P):
    """Faster verification using the structure. Only check necessary conditions."""
    nm, mp, pn = n*m, m*p, p*n
    rank = len(U[0]) if U else 0
    # Check the 32 required entries equal 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                if s != 1:
                    return False
    # For the zero entries, we can check a subset for speed, but for exactness we need all.
    # Given the small sizes (4*16*16=1024), full check is fine.
    valid = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid.add((i*m+j, j*p+k, k*n+i))
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                if (a, b, c) not in valid:
                    s = 0
                    for r in range(rank):
                        s += U[a][r] * V[b][r] * W[c][r]
                    if s != 0:
                        return False
    return True


def make_half_int_mat(rows, cols, rng):
    """Generate a random matrix with entries in {-1, -1/2, 0, 1/2, 1}."""
    vals = [0, 0, 0, 1, -1, Fraction(1,2), Fraction(-1,2)]
    return [[rng.choice(vals) for _ in range(cols)] for _ in range(rows)]


def schoolbook():
    """Return the schoolbook rank-32 decomposition."""
    n, m, p = N, M, P
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W


def _build_tensor():
    """Build the full 4x16x16 tensor T as a numpy array."""
    T = np.zeros((NM, MP, PN), dtype=np.int32)
    for i in range(N):
        for j in range(M):
            for k in range(P):
                a = i * M + j
                b = j * P + k
                c = k * N + i
                T[a, b, c] = 1
    return T


def _float_to_frac_list(mat_float):
    """Convert a float matrix to a list of lists of Fractions (rounding to nearest half-integer)."""
    rows, cols = mat_float.shape
    result = []
    for i in range(rows):
        row = []
        for j in range(cols):
            v = mat_float[i, j]
            fv = round(v * 2) / 2
            row.append(Fraction(fv).limit_denominator(2))
        result.append(row)
    return result


def _exact_residual_check(U, V, W):
    """Check if the decomposition is exact using Fraction arithmetic.
    Returns True if exact, False otherwise."""
    rank = len(U[0])
    # Check the 32 required entries
    for i in range(N):
        for j in range(M):
            for k in range(P):
                a = i * M + j
                b = j * P + k
                c = k * N + i
                s = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                if s != 1:
                    return False
    # Check a sample of zero entries
    valid = set()
    for i in range(N):
        for j in range(M):
            for k in range(P):
                valid.add((i*M+j, j*P+k, k*N+i))
    
    # Check all zero entries for correctness
    for a in range(NM):
        for b in range(MP):
            for c in range(PN):
                if (a, b, c) not in valid:
                    s = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                    if s != 0:
                        return False
    return True


def _local_search_float(Uf, Vf, Wf, T_float, deadline, max_iters=100):
    """Local search using float64 matrices with incremental residual updates."""
    rank = Uf.shape[1]
    Uf = Uf.copy()
    Vf = Vf.copy()
    Wf = Wf.copy()
    
    recon = np.einsum('ar,br,cr->abc', Uf, Vf, Wf)
    residual = T_float - recon
    best_norm = float(np.sum(residual**2))
    
    if best_norm < 1e-10:
        return Uf, Vf, Wf
    
    vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        improved = False
        
        for a in range(NM):
            for r in range(rank):
                old_val = Uf[a, r]
                for new_val in vals:
                    if abs(new_val - old_val) < 1e-12:
                        continue
                    delta = new_val - old_val
                    res_new = residual + delta * np.outer(Vf[:, r], Wf[:, r])
                    norm_new = float(np.sum(res_new**2))
                    if norm_new < best_norm - 1e-10:
                        Uf[a, r] = new_val
                        residual = res_new
                        best_norm = norm_new
                        improved = True
                        break
        
        for b in range(MP):
            for r in range(rank):
                old_val = Vf[b, r]
                for new_val in vals:
                    if abs(new_val - old_val) < 1e-12:
                        continue
                    delta = new_val - old_val
                    res_new = residual + delta * np.outer(Uf[:, r], Wf[:, r])
                    norm_new = float(np.sum(res_new**2))
                    if norm_new < best_norm - 1e-10:
                        Vf[b, r] = new_val
                        residual = res_new
                        best_norm = norm_new
                        improved = True
                        break
        
        for c in range(PN):
            for r in range(rank):
                old_val = Wf[c, r]
                for new_val in vals:
                    if abs(new_val - old_val) < 1e-12:
                        continue
                    delta = new_val - old_val
                    res_new = residual + delta * np.outer(Uf[:, r], Vf[:, r])
                    norm_new = float(np.sum(res_new**2))
                    if norm_new < best_norm - 1e-10:
                        Wf[c, r] = new_val
                        residual = res_new
                        best_norm = norm_new
                        improved = True
                        break
        
        if not improved:
            break
    
    return Uf, Vf, Wf


def search_deadline():
    """Main search with time limit."""
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s kill
    
    best = None
    best_rank = SCHOOBOOK_RANK
    
    U0, V0, W0 = schoolbook()
    if verify_fast(U0, V0, W0):
        best = (U0, V0, W0)
        best_rank = SCHOOBOOK_RANK
    
    rng = random.Random(42)
    T = _build_tensor()
    T_float = T.astype(np.float64)
    
    for target_rank in range(16, 32):
        if time.time() > deadline:
            break
        
        remaining = deadline - time.time()
        if remaining < 5:
            break
        
        if remaining > 60:
            n_trials = 400 if target_rank <= 18 else 250 if target_rank <= 22 else 120
        elif remaining > 30:
            n_trials = 150 if target_rank <= 18 else 80 if target_rank <= 22 else 40
        else:
            n_trials = 40 if target_rank <= 18 else 20 if target_rank <= 22 else 10
        
        for trial in range(n_trials):
            if time.time() > deadline:
                break
            
            seed = rng.randint(0, 2**31 - 1)
            np_rng = np.random.default_rng(seed)
            vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
            probs = np.array([0.3, 0.15, 0.2, 0.15, 0.2])
            
            Uf = np_rng.choice(vals, size=(NM, target_rank), p=probs)
            Vf = np_rng.choice(vals, size=(MP, target_rank), p=probs)
            Wf = np_rng.choice(vals, size=(PN, target_rank), p=probs)
            
            recon = np.einsum('ar,br,cr->abc', Uf, Vf, Wf)
            residual = T_float - recon
            norm = float(np.sum(residual**2))
            
            if norm < 1e-10:
                U_frac = _float_to_frac_list(Uf)
                V_frac = _float_to_frac_list(Vf)
                W_frac = _float_to_frac_list(Wf)
                if _exact_residual_check(U_frac, V_frac, W_frac):
                    best = (U_frac, V_frac, W_frac)
                    best_rank = target_rank
                    break
            
            if time.time() > deadline:
                break
            
            if norm < 100:
                Uf_new, Vf_new, Wf_new = _local_search_float(Uf, Vf, Wf, T_float, deadline, max_iters=60)
                
                recon_new = np.einsum('ar,br,cr->abc', Uf_new, Vf_new, Wf_new)
                residual_new = T_float - recon_new
                norm_new = float(np.sum(residual_new**2))
                
                if norm_new < 1e-10:
                    U_frac = _float_to_frac_list(Uf_new)
                    V_frac = _float_to_frac_list(Vf_new)
                    W_frac = _float_to_frac_list(Wf_new)
                    if _exact_residual_check(U_frac, V_frac, W_frac):
                        best = (U_frac, V_frac, W_frac)
                        best_rank = target_rank
                        break
        
        if best_rank <= target_rank:
            if best_rank < 16:
                break
    
    if best is not None:
        U, V, W = best
        return {"U": U, "V": V, "W": W}
    else:
        U, V, W = schoolbook()
        return {"U": U, "V": V, "W": W}


def construct():
    """Entry point: return the best decomposition found."""
    try:
        return search_deadline()
    except Exception:
        # Fallback to schoolbook
        U, V, W = schoolbook()
        return {"U": U, "V": V, "W": W}
