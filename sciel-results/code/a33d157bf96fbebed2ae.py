"""Search for a rank-minimal exact tensor decomposition of 2x2 * 2x3 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def _target_tensor(n, m, p):
    # T[a,b,d] = 1 iff a=i*m+j, b=j*p+k, d=k*n+i (matching i,j,k)
    T = np.zeros((n*m, m*p, p*n), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                d = k*n + i
                T[a, b, d] = 1
    return T

def _validate(U, V, W, T):
    # U: (n*m, r), V: (m*p, r), W: (p*n, r)
    r = U.shape[1]
    # Reconstruct: sum_c U[:,c] V[:,c] W[:,c]
    # Compute as: for each c, outer product
    rec = np.zeros(T.shape, dtype=object)
    for c in range(r):
        uc = U[:, c]
        vc = V[:, c]
        wc = W[:, c]
        # term = uc[:,None,None] * vc[None,:,None] * wc[None,None,:]
        term = uc[:, None, None] * vc[None, :, None] * wc[None, None, :]
        rec += term
    # Check exactness
    if not np.array_equal(rec, T):
        return False
    return True

def _validate_fast(U, V, W, T):
    # Faster: use einsum-like but exact with integers
    r = U.shape[1]
    # rec[a,b,d] = sum_c U[a,c] V[b,c] W[d,c]
    # We can compute per (a,b): rec[a,b,:] = (U[:,c]*V[b,c]) dot W... 
    # Actually rec[a,b,d] = sum_c U[a,c] V[b,c] W[d,c] = (U[a,:] * V[b,:]) dot W[d,:]
    # So for each (a,b), the vector over d is W.T @ (U[a,:]*V[b,:])
    Ua = U  # (nm, r)
    Vb = V  # (mp, r)
    Wd = W  # (pn, r)
    # rec[a,b,d] = sum_c Ua[a,c]*Vb[b,c]*Wd[d,c]
    # = (Ua[a,:]*Vb[b,:]) dot Wd[d,:]
    nm, mp, pn = T.shape
    ok = True
    for a in range(nm):
        for b in range(mp):
            prod = Ua[a, :] * Vb[b, :]  # (r,)
            # rec[a,b,d] = Wd[d,:] dot prod
            rec_d = Wd @ prod  # (pn,)
            if not np.array_equal(rec_d, T[a, b, :]):
                ok = False
                return False
    return ok

def _schoolbook(n, m, p):
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

def _try_random_search(n, m, p, T, deadline, max_rank_start=11, trials_per_rank=400, seed=0):
    rng = random.Random(seed)
    best = None
    for rank in range(max_rank_start, 0, -1):
        if time.time() > deadline - 5:
            break
        for trial in range(trials_per_rank):
            if time.time() > deadline - 5:
                break
            # Random initialization with small integers
            U = np.zeros((n*m, rank), dtype=int)
            V = np.zeros((m*p, rank), dtype=int)
            W = np.zeros((p*n, rank), dtype=int)
            for c in range(rank):
                # Each column: pick a pattern
                # W column: select which (i,k) pairs
                for d in range(p*n):
                    W[d, c] = rng.choice([-1, 0, 0, 0, 1])
                for a in range(n*m):
                    U[a, c] = rng.choice([-1, 0, 0, 0, 1])
                for b in range(m*p):
                    V[b, c] = rng.choice([-1, 0, 0, 0, 1])
            if _validate_fast(U, V, W, T):
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                return best
    return best

def _structured_search(n, m, p, T, deadline, max_rank=11, iterations=20000, seed=1):
    # Structured: parameterize by fixing W to sparse selectors and solving
    # For 2x2x3, try to find a rank-10 or rank-11 decomposition
    # Use a known ansatz: the multiplication tensor can be decomposed using
    # the fact that it's related to the "border rank" constructions.
    # We do a local search from a random valid-ish start.
    rng = random.Random(seed)
    nm, mp, pn = n*m, m*p, p*n
    best = None
    for rank in range(max_rank, 0, -1):
        if time.time() > deadline - 5:
            break
        # Start from a random sparse configuration
        U = np.zeros((nm, rank), dtype=int)
        V = np.zeros((mp, rank), dtype=int)
        W = np.zeros((pn, rank), dtype=int)
        for c in range(rank):
            # Assign each column a "role": pick a (i,k) for W
            i0 = rng.randrange(n)
            k0 = rng.randrange(p)
            d0 = k0*n + i0
            W[d0, c] = 1
            # U and V: pick (i,j) and (j,k)
            for j in range(m):
                if rng.random() < 0.5:
                    U[i0*m + j, c] = 1
                if rng.random() < 0.5:
                    V[j*p + k0, c] = 1
        # Local search: flip random entries to reduce error
        def error_count(U, V, W):
            # Count mismatches
            cnt = 0
            for a in range(nm):
                for b in range(mp):
                    prod = U[a, :] * V[b, :]
                    rec_d = W @ prod
                    for d in range(pn):
                        if rec_d[d] != T[a, b, d]:
                            cnt += 1
            return cnt
        cur_err = error_count(U, V, W)
        if cur_err == 0:
            best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            return best
        for it in range(iterations):
            if time.time() > deadline - 5:
                break
            # Pick a random matrix, row, col to flip
            mat = rng.choice([U, V, W])
            row = rng.randrange(mat.shape[0])
            col = rng.randrange(mat.shape[1])
            old = mat[row, col]
            mat[row, col] = old + 1
            new_err = error_count(U, V, W)
            if new_err < cur_err:
                cur_err = new_err
                if cur_err == 0:
                    best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    return best
            else:
                mat[row, col] = old
        # Try a few restarts at this rank
    return best

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin before 120s
    T = _target_tensor(n, m, p)
    
    # Fallback: schoolbook rank 12
    fallback = _schoolbook(n, m, p)
    
    best = None
    # Try structured search for rank < 12
    for seed in range(0, 8):
        if time.time() > deadline - 10:
            break
        res = _structured_search(n, m, p, T, deadline, max_rank=11, iterations=3000, seed=seed)
        if res is not None:
            best = res
            break
    if best is None:
        # Try random search
        for seed in range(0, 4):
            if time.time() > deadline - 10:
                break
            res = _try_random_search(n, m, p, T, deadline, max_rank_start=11, trials_per_rank=200, seed=seed+100)
            if res is not None:
                best = res
                break
    if best is None:
        best = fallback
    return best
# EVOLVE-BLOCK-END
