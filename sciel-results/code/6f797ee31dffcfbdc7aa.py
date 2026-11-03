"""
4x5x6 matrix multiplication tensor decomposition via Strassen block construction
with iterative refinement and column-removal.
"""
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0

def _time_left():
    return _DEADLINE - time.time()

def _build_strassen_2x2x3():
    """Strassen-like 2x2x3 decomposition (rank 8) using 3x3 output split."""
    # Use a known 2x2x3 rank-8 decomposition
    U = np.zeros((4, 8))
    V = np.zeros((6, 8))
    W = np.zeros((6, 8))
    
    # Standard 2x2x3 with 3 output columns per row pair
    # P1 = (a11+a22)*(b11+b22) -> contributes to C11,C12,C21,C22
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    # P2 = (a21+a22)*b11 -> C21,C22
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[4,1]=1; W[5,1]=-1
    # P3 = a11*(b12-b22) -> C12,C22  
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[1,2]=1; W[5,2]=1
    # P4 = a22*(b21-b11) -> C21,C11
    U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[4,3]=1
    # P5 = (a11+a12)*b22 -> C12,C22
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[1,4]=-1; W[5,4]=1
    # P6 = (a21-a11)*(b11+b12) -> C21,C22
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[4,5]=1
    # P7 = (a12-a22)*(b21+b22) -> C11,C12
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
    # P8 = a12*b21 -> C12
    U[1,7]=1; V[2,7]=1; W[1,7]=1
    
    return U, V, W

def _verify_decomp(U, V, W, n, m, p, tol=1e-9):
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = float(np.dot(U[i * m + j], V[j * p + k] * W[k * n + i]))
                if abs(total - 1.0) > tol:
                    return False
    return True

def _verify_np(U, V, W, n, m, p, tol=1e-4):
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return bool(np.all(np.abs(T - 1.0) < tol))

def _build_strassen_2x2():
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # P1 = (a11+a22)*(b11+b22)
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    # P2 = (a21+a22)*b11
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[1,1]=1; W[3,1]=-1
    # P3 = a11*(b12-b22)
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[2,2]=1; W[3,2]=1
    # P4 = a22*(b21-b11)
    U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[1,3]=1
    # P5 = (a11+a12)*b22
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[2,4]=1
    # P6 = (a21-a11)*(b11+b12)
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    # P7 = (a12-a22)*(b21+b22)
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
    
    return U, V, W

def _build_strassen_4x4x2():
    U2, V2, W2 = _build_strassen_2x2()
    rank = 28
    U = np.zeros((16, rank))
    V = np.zeros((8, rank))
    W = np.zeros((8, rank))
    
    def add_block(ai, aj, bj, ci, ck, base):
        for t in range(7):
            col = base + t
            for li in range(2):
                for lj in range(2):
                    la = li*2+lj
                    ca = U2[la, t]
                    if ca != 0:
                        U[(ai+li)*4+(aj+lj), col] += ca
            for lj in range(2):
                for lk in range(2):
                    lb = lj*2+lk
                    cb = V2[lb, t]
                    if cb != 0:
                        V[(bj+lj)*2+(ck+lk), col] += cb
            for lk in range(2):
                for li in range(2):
                    lc = lk*2+li
                    cc = W2[lc, t]
                    if cc != 0:
                        W[(ck+lk)*4+(ci+li), col] += cc
    
    add_block(0, 0, 0, 0, 0, 0)
    add_block(0, 2, 2, 0, 0, 7)
    add_block(2, 0, 0, 2, 0, 14)
    add_block(2, 2, 2, 2, 0, 21)
    return U, V, W

def _build_4x5x6():
    """
    Split m=5 into 4+1:
    - A_left (4x4) @ B_top (4x6): 3 Strassen 4x4x2 blocks, rank 84
    - A_right (4x1) @ B_bottom (1x6): schoolbook, rank 24
    Total rank: 108
    """
    n, m, p = 4, 5, 6
    U442, V442, W442 = _build_strassen_4x4x2()
    total_rank = 3*28 + 24
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    for blk in range(3):
        base = blk * 28
        ks = blk * 2
        for i in range(4):
            for j in range(4):
                U[i*5+j, base:base+28] = U442[i*4+j, :]
        for j in range(4):
            for k in range(6):
                if ks <= k < ks+2:
                    V[j*6+k, base:base+28] = V442[j*2+(k-ks), :]
        for k in range(6):
            if ks <= k < ks+2:
                for i in range(4):
                    W[k*4+i, base:base+28] = W442[(k-ks)*4+i, :]
    
    base = 84
    for i in range(4):
        for k in range(6):
            c = base + i*6+k
            U[i*5+4, c] = 1.0
            V[4*6+k, c] = 1.0
            W[k*4+i, c] = 1.0
    return U, V, W

def _build_4x5x6_v2():
    """
    Alternative split: m=5 into 3+2
    - A_left (4x3) @ B_top (3x6): 3 Strassen 2x2x2 blocks on 2x2 subblocks of 3x3, rank 3*7*2=42 for 2x2x2 on each 2x3 pair
    - Actually use 4x3x6 directly with Strassen on rows
    - A_right (4x2) @ B_bottom (2x6): schoolbook rank 4*2*6=48
    Try: 4x3x6 via two 2x3x6 Strassen blocks (rank 2*8*2=32) + 4x2x6 schoolbook (48) = 80? 
    Let's try a different approach: 4x5x6 = 4x4x6 + 4x1x6
    4x4x6 = 3 * 4x4x2 (Strassen rank 28 each) = 84
    4x1x6 = schoolbook 24
    Total 108. Same as before.
    
    Try: 4x5x6 = 4x3x6 + 4x2x6
    4x3x6: split 3 into 2+1. 4x2x6 via Strassen 2x2x2 on rows: 2 blocks of 2x2x6? 
    This is getting complex. Keep v1 and try refinement.
    """
    return _build_4x5x6()

def _build_schoolbook(n, m, p):
    rank = n*m*p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, c] = 1.0
                V[j*p+k, c] = 1.0
                W[k*n+i, c] = 1.0
                c += 1
    return U, V, W

def _iterative_refinement(n, m, p, initial_rank, max_iters=120, n_attempts=8):
    if _time_left() < 10.0:
        return None
    R = initial_rank
    for attempt in range(n_attempts):
        if _time_left() < 8.0:
            break
        np.random.seed(1000 + attempt * 13)
        U = np.random.randn(n*m, R) * 0.5
        V = np.random.randn(m*p, R) * 0.5
        W = np.random.randn(p*n, R) * 0.5
        for it in range(max_iters):
            if _time_left() < 5.0:
                break
            U3 = U.reshape(n, m, R)
            V3 = V.reshape(m, p, R)
            W3 = W.reshape(p, n, R)
            A = np.einsum('ijr,jkr->kir', U3, V3)
            norms = np.sum(A*A, axis=2, keepdims=True)
            if np.any(norms < 1e-12):
                break
            W = (A / norms).reshape(p*n, R)
            W3 = W.reshape(p, n, R)
            B = np.einsum('jkr,kir->ijr', V3, W3)
            norms = np.sum(B*B, axis=2, keepdims=True)
            if np.any(norms < 1e-12):
                break
            U = (B / norms).reshape(n*m, R)
            U3 = U.reshape(n, m, R)
            C = np.einsum('ijr,kir->jkr', U3, W3)
            norms = np.sum(C*C, axis=2, keepdims=True)
            if np.any(norms < 1e-12):
                break
            V = (C / norms).reshape(m*p, R)
            if it % 4 == 0:
                if _verify_np(U, V, W, n, m, p, 1e-6):
                    if _verify_decomp(U, V, W, n, m, p, 1e-8):
                        return U, V, W
        if _verify_np(U, V, W, n, m, p, 1e-6):
            if _verify_decomp(U, V, W, n, m, p, 1e-8):
                return U, V, W
    return None

def _greedy_column_merge(U, V, W, n, m, p, budget_s=20.0):
    """
    Try to express each term as a linear combination of others and eliminate it.
    For term r, solve for coefficients alpha_s such that:
    U[:,:,r] ⊗ V[:,:,r] ⊗ W[:,:,r] ≈ sum_s alpha_s * U[:,:,s] ⊗ V[:,:,s] ⊗ W[:,:,s]
    This is a hard problem, but we can try: fix U and V, solve for W contribution.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    R = U.shape[1]
    t0 = time.time()
    U3 = U.reshape(n, m, R)
    V3 = V.reshape(m, p, R)
    W3 = W.reshape(p, n, R)
    
    for _ in range(5):
        if _time_left() < 5.0 or (time.time() - t0) > budget_s or R < 3:
            break
        found = False
        for r in range(R):
            if _time_left() < 5.0:
                break
            # Try to find if term r is redundant: can we find a vector in the span
            # of other terms that matches its contribution?
            # Approximate: check if removing r and adding a small perturbation still works
            # Better: use least squares to see if term r's contribution is in the span of others
            target = np.einsum('ij,jk,ki->ijk', U3[:,:,r], V3[:,:,r], W3[:,:,r])
            # Build basis from other terms
            others = [s for s in range(R) if s != r]
            if len(others) < 1:
                continue
            # This is expensive; skip for now
            pass
        if not found:
            break
    return U, V, W

def _local_reduce(U, V, W, n, m, p, max_rounds=30, budget_s=35.0):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    R = U.shape[1]
    t0 = time.time()
    U3 = U.reshape(n, m, R)
    V3 = V.reshape(m, p, R)
    W3 = W.reshape(p, n, R)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    for _ in range(max_rounds):
        if _time_left() < 3.0 or (time.time() - t0) > budget_s or R < 2:
            break
        found = False
        # Vectorized: compute residual for all columns at once
        # For each column c, residual_c = T - U[:,:,c] * V[:,:,c] * W[:,:,c]
        # We want |residual_c - 1| < tol for all i,j,k
        for c in range(R):
            if _time_left() < 3.0:
                break
            res = T - np.einsum('ij,jk,ki->ijk', U3[:,:,c], V3[:,:,c], W3[:,:,c])
            if np.all(np.abs(res - 1.0) < 1e-6):
                mask = np.ones(R, dtype=bool); mask[c] = False
                U2, V2, W2 = U[:,mask], V[:,mask], W[:,mask]
                if _verify_decomp(U2, V2, W2, n, m, p, 1e-9):
                    U, V, W = U2, V2, W2
                    R -= 1
                    U3 = U.reshape(n, m, R)
                    V3 = V.reshape(m, p, R)
                    W3 = W.reshape(p, n, R)
                    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
                    found = True
                    break
        if not found:
            break
    return U, V, W

def _pairwise_reduce(U, V, W, n, m, p, max_rounds=10, budget_s=25.0):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    R = U.shape[1]
    t0 = time.time()
    U3 = U.reshape(n, m, R)
    V3 = V.reshape(m, p, R)
    W3 = W.reshape(p, n, R)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    for _ in range(max_rounds):
        if _time_left() < 3.0 or (time.time() - t0) > budget_s or R < 3:
            break
        found = False
        for c1 in range(R):
            if _time_left() < 3.0:
                break
            for c2 in range(c1+1, R):
                if _time_left() < 3.0:
                    break
                r1 = np.einsum('ij,jk,ki->ijk', U3[:,:,c1], V3[:,:,c1], W3[:,:,c1])
                r2 = np.einsum('ij,jk,ki->ijk', U3[:,:,c2], V3[:,:,c2], W3[:,:,c2])
                res = T - r1 - r2
                if np.all(np.abs(res - 1.0) < 1e-6):
                    mask = np.ones(R, dtype=bool); mask[c1]=False; mask[c2]=False
                    U2, V2, W2 = U[:,mask], V[:,mask], W[:,mask]
                    if _verify_decomp(U2, V2, W2, n, m, p, 1e-9):
                        U, V, W = U2, V2, W2
                        R -= 2
                        U3 = U.reshape(n, m, R)
                        V3 = V.reshape(m, p, R)
                        W3 = W.reshape(p, n, R)
                        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
                        found = True
                        break
            if found:
                break
        if not found:
            break
    return U, V, W

def _triple_reduce(U, V, W, n, m, p, max_rounds=5, budget_s=15.0):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    R = U.shape[1]
    t0 = time.time()
    U3 = U.reshape(n, m, R)
    V3 = V.reshape(m, p, R)
    W3 = W.reshape(p, n, R)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    for _ in range(max_rounds):
        if _time_left() < 3.0 or (time.time() - t0) > budget_s or R < 4:
            break
        found = False
        for c1 in range(R):
            if _time_left() < 3.0:
                break
            for c2 in range(c1+1, R):
                if _time_left() < 3.0:
                    break
                for c3 in range(c2+1, R):
                    if _time_left() < 3.0:
                        break
                    r1 = np.einsum('ij,jk,ki->ijk', U3[:,:,c1], V3[:,:,c1], W3[:,:,c1])
                    r2 = np.einsum('ij,jk,ki->ijk', U3[:,:,c2], V3[:,:,c2], W3[:,:,c2])
                    r3 = np.einsum('ij,jk,ki->ijk', U3[:,:,c3], V3[:,:,c3], W3[:,:,c3])
                    res = T - r1 - r2 - r3
                    if np.all(np.abs(res - 1.0) < 1e-6):
                        mask = np.ones(R, dtype=bool)
                        mask[c1]=False; mask[c2]=False; mask[c3]=False
                        U2, V2, W2 = U[:,mask], V[:,mask], W[:,mask]
                        if _verify_decomp(U2, V2, W2, n, m, p, 1e-9):
                            U, V, W = U2, V2, W2
                            R -= 3
                            U3 = U.reshape(n, m, R)
                            V3 = V.reshape(m, p, R)
                            W3 = W.reshape(p, n, R)
                            T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
                            found = True
                            break
                if found:
                    break
            if found:
                break
        if not found:
            break
    return U, V, W

def construct():
    n, m, p = 4, 5, 6
    best_rank = n * m * p
    best_U, best_V, best_W = _build_schoolbook(n, m, p)
    
    # Strassen-based construction (rank 108)
    try:
        if _time_left() > 40.0:
            U, V, W = _build_4x5x6()
            if _verify_decomp(U, V, W, n, m, p, 1e-9):
                if U.shape[1] < best_rank:
                    best_rank = U.shape[1]
                    best_U, best_V, best_W = U, V, W
    except Exception:
        pass
    
    # Column-removal refinement on Strassen decomposition
    try:
        if _time_left() > 25.0:
            cand = _local_reduce(best_U, best_V, best_W, n, m, p, max_rounds=40, budget_s=25.0)
            if _verify_decomp(cand[0], cand[1], cand[2], n, m, p, 1e-9):
                if cand[0].shape[1] < best_rank:
                    best_rank = cand[0].shape[1]
                    best_U, best_V, best_W = cand
    except Exception:
        pass
    
    # Pairwise removal
    try:
        if _time_left() > 15.0:
            cand = _pairwise_reduce(best_U, best_V, best_W, n, m, p, max_rounds=15, budget_s=15.0)
            if _verify_decomp(cand[0], cand[1], cand[2], n, m, p, 1e-9):
                if cand[0].shape[1] < best_rank:
                    best_rank = cand[0].shape[1]
                    best_U, best_V, best_W = cand
    except Exception:
        pass
    
    # Another local pass
    try:
        if _time_left() > 10.0:
            cand = _local_reduce(best_U, best_V, best_W, n, m, p, max_rounds=30, budget_s=10.0)
            if _verify_decomp(cand[0], cand[1], cand[2], n, m, p, 1e-9):
                if cand[0].shape[1] < best_rank:
                    best_rank = cand[0].shape[1]
                    best_U, best_V, best_W = cand
    except Exception:
        pass
    
    # Iterative refinement to find lower-rank decompositions
    try:
        if _time_left() > 20.0:
            for R in range(max(60, best_rank - 20), best_rank):
                if _time_left() < 12.0:
                    break
                cand = _iterative_refinement(n, m, p, initial_rank=R, max_iters=100, n_attempts=6)
                if cand is not None:
                    U, V, W = cand
                    if _verify_decomp(U, V, W, n, m, p, 1e-9):
                        if U.shape[1] < best_rank:
                            best_rank = U.shape[1]
                            best_U, best_V, best_W = U, V, W
                            break
    except Exception:
        pass
    
    # Final local pass after refinement
    try:
        if _time_left() > 8.0:
            cand = _local_reduce(best_U, best_V, best_W, n, m, p, max_rounds=25, budget_s=8.0)
            if _verify_decomp(cand[0], cand[1], cand[2], n, m, p, 1e-9):
                if cand[0].shape[1] < best_rank:
                    best_rank = cand[0].shape[1]
                    best_U, best_V, best_W = cand
    except Exception:
        pass
    
    # Triple removal
    try:
        if _time_left() > 5.0:
            cand = _triple_reduce(best_U, best_V, best_W, n, m, p, max_rounds=5, budget_s=5.0)
            if _verify_decomp(cand[0], cand[1], cand[2], n, m, p, 1e-9):
                if cand[0].shape[1] < best_rank:
                    best_rank = cand[0].shape[1]
                    best_U, best_V, best_W = cand
    except Exception:
        pass
    
    # Final local pass
    try:
        if _time_left() > 3.0:
            cand = _local_reduce(best_U, best_V, best_W, n, m, p, max_rounds=15, budget_s=3.0)
            if _verify_decomp(cand[0], cand[1], cand[2], n, m, p, 1e-9):
                if cand[0].shape[1] < best_rank:
                    best_rank = cand[0].shape[1]
                    best_U, best_V, best_W = cand
    except Exception:
        pass
    
    # Safety check
    if not _verify_decomp(best_U, best_V, best_W, n, m, p, 1e-9):
        U, V, W = _build_schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
