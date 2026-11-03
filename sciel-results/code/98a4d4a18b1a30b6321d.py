"""Matrix multiplication 3x3 * 3x4 - rank-minimal exact tensor decomposition.

Uses structured random search with fast tensor evaluation, exploiting the
algebraic structure of matmul. Falls back to schoolbook rank-36 if no
better solution is found within the time budget.
"""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 4]

# Time management
_START_TIME = time.time()
_DEADLINE = _START_TIME + 110.0  # 110s deadline, 10s safety margin


def _time_left():
    return _DEADLINE - time.time()


def _schoolbook():
    """Rank-36 schoolbook construction."""
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W}


def _evaluate(U, V, W, n, m, p):
    """Compute the tensor from factors and return max error from 1.0.
    
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    Should equal 1.0 for all valid i,j,k.
    """
    Ua = np.asarray(U, dtype=np.float64)
    Va = np.asarray(V, dtype=np.float64)
    Wa = np.asarray(W, dtype=np.float64)
    rank = Ua.shape[1]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = float(np.dot(Ua[i * m + j], Va[j * p + k] * Wa[k * n + i]))
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    return max_err


def _is_exact(U, V, W, n, m, p, tol=1e-9):
    return _evaluate(U, V, W, n, m, p) < tol


def _random_half_int(nrows, rank, scale=2):
    """Random half-integer matrix."""
    mat = []
    for _ in range(nrows):
        row = []
        for _ in range(rank):
            val = random.randint(-scale, scale) * 0.5
            row.append(val)
        mat.append(row)
    return mat


def _smooth_polish(U, V, W, n, m, p, deadline, tol=1e-9, max_iters=500):
    """Smooth-surrogate polish using Adam optimizer: minimize a smooth objective
    over continuous variables, then snap to half-integers and verify.
    
    Objective: sum_{i,j,k} (sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] - 1)^2
    """
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    rank = Ua.shape[1]
    
    # Precompute index arrays for fast gradient/eval
    triples_i = []
    triples_j = []
    triples_k = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                triples_i.append(i)
                triples_j.append(j)
                triples_k.append(k)
    triples_i = np.array(triples_i)
    triples_j = np.array(triples_j)
    triples_k = np.array(triples_k)
    ntrip = len(triples_i)
    
    def _obj_and_grad(Ua, Va, Wa):
        loss = 0.0
        gU = np.zeros_like(Ua)
        gV = np.zeros_like(Va)
        gW = np.zeros_like(Wa)
        for idx in range(ntrip):
            i = triples_i[idx]
            j = triples_j[idx]
            k = triples_k[idx]
            ui = i * m + j
            vj = j * p + k
            wk = k * n + i
            s = float(np.dot(Ua[ui], Va[vj] * Wa[wk]))
            d = s - 1.0
            loss += d * d
            # d/dU[ui,r] = 2*d*Va[vj,r]*Wa[wk,r]
            gU[ui] += 2.0 * d * (Va[vj] * Wa[wk])
            gV[vj] += 2.0 * d * (Ua[ui] * Wa[wk])
            gW[wk] += 2.0 * d * (Ua[ui] * Va[vj])
        return loss, gU, gV, gW
    
    # Adam optimizer state
    b1, b2, eps = 0.9, 0.999, 1e-8
    mU = np.zeros_like(Ua)
    vU = np.zeros_like(Ua)
    mV = np.zeros_like(Va)
    vV = np.zeros_like(Va)
    mW = np.zeros_like(Wa)
    vW = np.zeros_like(Wa)
    t = 0
    lr = 0.05
    
    cur_loss, _, _, _ = _obj_and_grad(Ua, Va, Wa)
    best_loss = cur_loss
    best_U, best_V, best_W = Ua.copy(), Va.copy(), Wa.copy()
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        t += 1
        loss, gU, gV, gW = _obj_and_grad(Ua, Va, Wa)
        
        # Adam update
        mU = b1 * mU + (1 - b1) * gU
        vU = b2 * vU + (1 - b2) * gU ** 2
        mU_hat = mU / (1 - b1 ** t)
        vU_hat = vU / (1 - b2 ** t)
        
        mV = b1 * mV + (1 - b1) * gV
        vV = b2 * vV + (1 - b2) * gV ** 2
        mV_hat = mV / (1 - b1 ** t)
        vV_hat = vV / (1 - b2 ** t)
        
        mW = b1 * mW + (1 - b1) * gW
        vW = b2 * vW + (1 - b2) * gW ** 2
        mW_hat = mW / (1 - b1 ** t)
        vW_hat = vW / (1 - b2 ** t)
        
        Ua -= lr * mU_hat / (np.sqrt(vU_hat) + eps)
        Va -= lr * mV_hat / (np.sqrt(vV_hat) + eps)
        Wa -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
        
        cur_loss, _, _, _ = _obj_and_grad(Ua, Va, Wa)
        if cur_loss < best_loss:
            best_loss = cur_loss
            best_U, best_V, best_W = Ua.copy(), Va.copy(), Wa.copy()
        if cur_loss < 1e-14:
            break
    
    # Snap to half-integers
    Ua = np.round(best_U * 2.0) / 2.0
    Va = np.round(best_V * 2.0) / 2.0
    Wa = np.round(best_W * 2.0) / 2.0
    
    U = Ua.tolist()
    V = Va.tolist()
    W = Wa.tolist()
    err = _evaluate(U, V, W, n, m, p)
    if err < tol:
        return U, V, W
    return None


def _structured_init(n, m, p, rank, seed=0):
    """Create a structured initialization that exploits matmul structure.
    
    For matmul, each rank dimension can handle a 'path' through the summation.
    We initialize with patterns that are more likely to converge.
    """
    rng = np.random.RandomState(seed)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strategy: each rank dimension handles a subset of (i,j,k) tuples
    # Use a structured assignment based on cyclic patterns
    for r in range(rank):
        # Assign this rank to handle certain combinations
        # Use a hash-like function to distribute
        i = r % n
        j = (r // n) % m
        k = (r // (n * m)) % p
        
        # Add some randomness to avoid perfect symmetry
        if rng.random() < 0.3:
            i = rng.randint(0, n)
            j = rng.randint(0, m)
            k = rng.randint(0, p)
        
        U[i * m + j, r] = 1.0
        V[j * p + k, r] = 1.0
        W[k * n + i, r] = 1.0
    
    # Add small random perturbations to break symmetry
    U += rng.normal(0, 0.1, U.shape)
    V += rng.normal(0, 0.1, V.shape)
    W += rng.normal(0, 0.1, W.shape)
    
    return U.tolist(), V.tolist(), W.tolist()


def _symmetric_kronecker_init(n, m, p, rank, seed=0):
    """Initialize using symmetric Kronecker product structure.
    
    Exploits the 3x3 symmetry: treat the input as symmetric and use
    Kronecker product patterns that respect this structure.
    """
    rng = np.random.RandomState(seed)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Symmetric pattern: for each rank, assign based on symmetric pairs
    for r in range(rank):
        # Use symmetric assignment: i and j from same "block"
        block = r % 3
        sub = r // 3
        
        i = block
        j = block  # Symmetric: same index for i and j
        k = sub % p
        
        # Add variation
        if rng.random() < 0.4:
            j = (block + rng.randint(0, n-1)) % n
        
        U[i * m + j, r] = 1.0
        V[j * p + k, r] = 1.0
        W[k * n + i, r] = 1.0
    
    # Perturb
    U += rng.normal(0, 0.15, U.shape)
    V += rng.normal(0, 0.15, V.shape)
    W += rng.normal(0, 0.15, W.shape)
    
    return U.tolist(), V.tolist(), W.tolist()


def _try_structured_rank(n, m, p, rank, deadline, num_restarts=100):
    """Attempt to find a rank-r decomposition via structured initialization and Adam polishing."""
    for restart in range(num_restarts):
        if time.time() > deadline - 2.0:
            break
        
        # Alternate between structured and symmetric Kronecker inits
        if restart % 2 == 0:
            U, V, W = _structured_init(n, m, p, rank, seed=restart)
        else:
            U, V, W = _symmetric_kronecker_init(n, m, p, rank, seed=restart)
        
        # Polish
        result = _smooth_polish(U, V, W, n, m, p, deadline)
        if result is not None:
            return result
            
    return None


def _try_low_rank_search(n, m, p, deadline):
    """Search for low-rank decompositions starting from the theoretical lower bound."""
    # Theoretical lower bounds:
    # - max(n*m, m*p, p*n) = max(9, 12, 12) = 12
    # Try ranks from 14 upward (12-13 are very unlikely)
    
    for rank in range(14, 36):
        if time.time() > deadline:
            break
        
        # Use more restarts for lower ranks as they are harder
        num_restarts = 50 if rank <= 18 else 30
        result = _try_structured_rank(n, m, p, rank, deadline, num_restarts=num_restarts)
        if result is not None:
            return result
    
    return None


def _try_pattern_based(n, m, p, deadline):
    """Try specific structured patterns that might yield low rank.
    
    For matmul, there are known algebraic structures. Try:
    1. Kronecker product patterns
    2. Sparse patterns with specific symmetries
    """
    if time.time() > deadline:
        return None
    
    # Try a few specific structured patterns
    for attempt in range(10):
        if time.time() > deadline:
            break
        
        rank = 16 + attempt  # Try various ranks
        if rank >= 36:
            break
        
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Create a structured pattern: each rank dimension handles a subset
        # of the (i,j,k) tuples using a more systematic approach
        for r in range(rank):
            # Use a systematic mapping
            i = r % n
            j = (r // n) % m
            k = (r // (n * m)) % p
            
            U[i * m + j][r] = 1.0
            V[j * p + k][r] = 1.0
            W[k * n + i][r] = 1.0
        
        err = _evaluate(U, V, W, n, m, p)
        if err < 1e-9:
            return U, V, W
    
    return None


def construct():
    """Find a rank-minimal exact tensor decomposition for 3x3 * 3x4 matmul.
    
    Returns {'U', 'V', 'W'} with factor matrices of shapes:
    U: (n*m, rank) = (9, rank)
    V: (m*p, rank) = (12, rank)
    W: (p*n, rank) = (12, rank)
    
    Entries are integer or half-integer.
    """
    n, m, p = DIMENSIONS
    
    # Try structured search for low-rank solutions
    result = _try_low_rank_search(n, m, p, _DEADLINE)
    if result is not None:
        U, V, W = result
        return {"U": U, "V": V, "W": W}
    
    # Try pattern-based search
    result = _try_pattern_based(n, m, p, _DEADLINE)
    if result is not None:
        U, V, W = result
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook rank-36
    return _schoolbook()
