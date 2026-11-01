"""Matrix multiplication 2x3x7 rank-minimal search via systematic folding."""
import time
import itertools
import numpy as np

DIMENSIONS = [2, 3, 7]

def _build_target(n, m, p):
    """Build the target tensor as a 3D array."""
    nm, mp, pn = n*m, m*p, p*n
    target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0
    return target

def _verify_decomposition(U, V, W, target, tol=1e-9):
    """Verify that U,V,W exactly decompose the target tensor."""
    r = U.shape[1]
    R = np.zeros_like(target)
    for c in range(r):
        R += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
    return np.max(np.abs(R - target)) < tol

def _is_half_integer(mat, tol=1e-9):
    """Check if all entries are integer or half-integer."""
    return np.max(np.abs(mat * 2 - np.round(mat * 2))) < tol

def _schoolbook(n, m, p):
    """Return schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    nm, mp, pn = n*m, m*p, p*n
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    terms = []
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, idx] = 1
                V[j*p+k, idx] = 1
                W[k*n+i, idx] = 1
                terms.append((i, j, k))
                idx += 1
    return U, V, W, terms

def _effective_rank(U, V, W):
    """Count columns where at least one entry is non-zero."""
    r = U.shape[1]
    count = 0
    for c in range(r):
        if np.any(U[:, c] != 0) or np.any(V[:, c] != 0) or np.any(W[:, c] != 0):
            count += 1
    return count

def _compact(U, V, W):
    """Remove zero columns and return compacted matrices."""
    r = U.shape[1]
    keep = []
    for c in range(r):
        if np.any(U[:, c] != 0) or np.any(V[:, c] != 0) or np.any(W[:, c] != 0):
            keep.append(c)
    Uf = U[:, keep]
    Vf = V[:, keep]
    Wf = W[:, keep]
    return Uf, Vf, Wf, len(keep)

def _try_fold(U, V, W, terms, t1, t2, n, m, p):
    """Attempt to fold terms t1 and t2. Returns new (U,V,W,terms) or None."""
    i1, j1, k1 = terms[t1]
    i2, j2, k2 = terms[t2]
    
    # Folding requires i1 != i2 and j1 != j2
    if i1 == i2 or j1 == j2:
        return None
    
    # Make copies
    U2 = U.copy()
    V2 = V.copy()
    W2 = W.copy()
    terms2 = list(terms)
    
    # Zero out old contributions at t1 and t2
    U2[i1*m+j1, t1] = 0
    V2[j1*p+k1, t1] = 0
    W2[k1*n+i1, t1] = 0
    U2[i2*m+j2, t2] = 0
    V2[j2*p+k2, t2] = 0
    W2[k2*n+i2, t2] = 0
    
    # Add new folded terms
    U2[i1*m+j1, t1] = 1
    V2[j1*p+k2, t1] = 1
    W2[k2*n+i1, t1] = 1
    U2[i2*m+j2, t2] = 1
    V2[j2*p+k1, t2] = 1
    W2[k1*n+i2, t2] = 1
    
    terms2[t1] = (i1, j1, k2)
    terms2[t2] = (i2, j2, k1)
    
    return U2, V2, W2, terms2

def _greedy_fold_search(U, V, W, terms, target, n, m, p, deadline):
    """Greedily apply folds that reduce effective rank."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        cur_rank = _effective_rank(U, V, W)
        
        # Try all pairs of terms
        for t1 in range(len(terms)):
            if time.time() > deadline:
                break
            for t2 in range(t1 + 1, len(terms)):
                res = _try_fold(U, V, W, terms, t1, t2, n, m, p)
                if res is None:
                    continue
                U2, V2, W2, terms2 = res
                new_rank = _effective_rank(U2, V2, W2)
                if new_rank < cur_rank:
                    # Verify the folded decomposition is still valid
                    if _verify_decomposition(U2, V2, W2, target):
                        U, V, W, terms = U2, V2, W2, terms2
                        cur_rank = new_rank
                        improved = True
                        break
            if improved:
                break
    
    return U, V, W, terms

def _local_search(U, V, W, terms, target, n, m, p, deadline, max_iters=5000):
    """Local search: try single folds and accept if they maintain validity and reduce rank."""
    cur_rank = _effective_rank(U, V, W)
    best_U, best_V, best_W, best_terms = U.copy(), V.copy(), W.copy(), list(terms)
    best_rank = cur_rank
    
    iters = 0
    while time.time() < deadline and iters < max_iters:
        iters += 1
        # Pick a random pair
        t1 = np.random.randint(len(terms))
        t2 = np.random.randint(len(terms))
        if t1 == t2:
            continue
        
        res = _try_fold(U, V, W, terms, t1, t2, n, m, p)
        if res is None:
            continue
        U2, V2, W2, terms2 = res
        new_rank = _effective_rank(U2, V2, W2)
        
        if new_rank < best_rank and _verify_decomposition(U2, V2, W2, target):
            U, V, W, terms = U2, V2, W2, terms2
            best_U, best_V, best_W, best_terms = U.copy(), V.copy(), W.copy(), list(terms)
            best_rank = new_rank
    
    return best_U, best_V, best_W, best_terms, best_rank

def construct():
    start = time.time()
    deadline = start + 140  # safety margin under 150s
    
    n, m, p = DIMENSIONS
    target = _build_target(n, m, p)
    
    # Start with schoolbook
    U, V, W, terms = _schoolbook(n, m, p)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = n * m * p
    
    # Phase 1: Greedy folding
    U, V, W, terms = _greedy_fold_search(U, V, W, terms, target, n, m, p, deadline)
    U, V, W, rank1 = _compact(U, V, W)
    if rank1 < best_rank and _verify_decomposition(U, V, W, target):
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_rank = rank1
    
    # Phase 2: Local search from current state
    if time.time() < deadline - 10:
        U, V, W, terms, rank2 = _local_search(U, V, W, terms, target, n, m, p, deadline - 5, max_iters=3000)
        U, V, W, rank2 = _compact(U, V, W)
        if rank2 < best_rank and _verify_decomposition(U, V, W, target):
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            best_rank = rank2
    
    # Phase 3: Try multiple random restarts with local search
    if time.time() < deadline - 10:
        np.random.seed(42)
        for restart in range(5):
            if time.time() > deadline - 5:
                break
            # Start from schoolbook again
            U_r, V_r, W_r, terms_r = _schoolbook(n, m, p)
            # Apply a few random folds first to perturb
            for _ in range(20):
                t1 = np.random.randint(len(terms_r))
                t2 = np.random.randint(len(terms_r))
                if t1 == t2:
                    continue
                res = _try_fold(U_r, V_r, W_r, terms_r, t1, t2, n, m, p)
                if res is not None:
                    U_r, V_r, W_r, terms_r = res
                    break
            # Then local search
            U_r, V_r, W_r, terms_r, rank_r = _local_search(U_r, V_r, W_r, terms_r, target, n, m, p, deadline - 3, max_iters=2000)
            U_r, V_r, W_r, rank_r = _compact(U_r, V_r, W_r)
            if rank_r < best_rank and _verify_decomposition(U_r, V_r, W_r, target):
                best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                best_rank = rank_r
    
    # Final: ensure half-integer entries
    best_U = np.round(best_U * 2) / 2.0
    best_V = np.round(best_V * 2) / 2.0
    best_W = np.round(best_W * 2) / 2.0
    
    # Final verification
    if not _verify_decomposition(best_U, best_V, best_W, target):
        # Fallback to schoolbook
        U, V, W, _ = _schoolbook(n, m, p)
        best_U, best_V, best_W = U, V, W
        best_rank = n * m * p
    
    # Convert to list-of-lists format
    Uf = best_U.tolist()
    Vf = best_V.tolist()
    Wf = best_W.tolist()
    
    return {"U": Uf, "V": Vf, "W": Wf}
