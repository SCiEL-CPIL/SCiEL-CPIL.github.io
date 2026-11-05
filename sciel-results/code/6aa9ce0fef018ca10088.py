"""Matrix multiplication 2x4x4: rank-minimal exact tensor decomposition.
Combines Strassen-28 baseline with finite-field exact rank search.
"""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]
TIME_LIMIT = 150.0
START_TIME = time.time()
SAFETY_MARGIN = 15.0


def _remaining_time():
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - START_TIME)


def _strassen_2x2():
    """Return the 7 Strassen multiplication patterns for 2x2 matrix multiplication."""
    patterns = [
        {'a': {0: 1, 3: 1}, 'b': {0: 1, 3: 1}, 'c': {0: 1, 3: 1}},
        {'a': {2: 1, 3: 1}, 'b': {0: 1}, 'c': {2: 1, 3: -1}},
        {'a': {0: 1}, 'b': {1: 1, 3: -1}, 'c': {1: 1, 3: 1}},
        {'a': {3: 1}, 'b': {2: 1, 0: -1}, 'c': {0: 1, 2: 1}},
        {'a': {0: 1, 1: 1}, 'b': {3: 1}, 'c': {0: -1, 1: 1}},
        {'a': {2: 1, 0: -1}, 'b': {0: 1, 1: 1}, 'c': {3: 1}},
        {'a': {1: 1, 3: -1}, 'b': {2: 1, 3: 1}, 'c': {0: 1}},
    ]
    return patterns


def _build_strassen_28():
    """Build the rank-28 decomposition via 4 independent Strassen 2x2 block products."""
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    strassen_patterns = _strassen_2x2()

    blocks = [
        (0, 0, 0, 0, 0, 0),
        (0, 0, 0, 2, 0, 2),
        (0, 2, 2, 0, 0, 0),
        (0, 2, 2, 2, 0, 2),
    ]

    col = 0
    for (a_r, a_c, b_r, b_c, c_r, c_c) in blocks:
        for pattern in strassen_patterns:
            for local_a_idx, coeff in pattern['a'].items():
                local_r = local_a_idx // 2
                local_c = local_a_idx % 2
                global_i = a_r + local_r
                global_j = a_c + local_c
                U[global_i * m + global_j][col] += coeff
            for local_b_idx, coeff in pattern['b'].items():
                local_r = local_b_idx // 2
                local_c = local_b_idx % 2
                global_j = b_r + local_r
                global_k = b_c + local_c
                V[global_j * p + global_k][col] += coeff
            for local_c_idx, coeff in pattern['c'].items():
                local_r = local_c_idx // 2
                local_c = local_c_idx % 2
                global_i = c_r + local_r
                global_k = c_c + local_c
                W[global_k * n + global_i][col] += coeff
            col += 1

    return U, V, W


def _build_tensor(n, m, p):
    """Build the multiplication tensor T[a,b,c]=1 with trace convention c=k*n+i."""
    nm, mp, pn = n * m, m * p, p * n
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for l in range(m):
            for k in range(p):
                a = i * m + l
                b = l * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T


def _verify(U, V, W, T, rank):
    """Max abs error between reconstructed tensor and T."""
    nm, mp, pn = U.shape[0], V.shape[0], W.shape[0]
    S = np.zeros((nm, mp, pn), dtype=float)
    for r in range(rank):
        S += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    return float(np.max(np.abs(S - T)))


def _verify_exact_mod(U, V, W, n, m, p, prime):
    """Verify exactness over a finite field using modular arithmetic."""
    nm, mp, pn = n * m, m * p, p * n
    rank = U.shape[1]
    
    # Build expected tensor mod prime
    T_mod = np.zeros((nm, mp, pn), dtype=np.int64)
    for i in range(n):
        for l in range(m):
            for k in range(p):
                a = i * m + l
                b = l * p + k
                c = k * n + i
                T_mod[a, b, c] = 1
    
    # Reconstruct
    S_mod = np.zeros((nm, mp, pn), dtype=np.int64)
    for r in range(rank):
        for a in range(nm):
            ua = U[a, r] % prime
            if ua == 0:
                continue
            for b in range(mp):
                vb = V[b, r] % prime
                if vb == 0:
                    continue
                uv = (ua * vb) % prime
                if uv == 0:
                    continue
                for c in range(pn):
                    wc = W[c, r] % prime
                    if wc != 0:
                        S_mod[a, b, c] = (S_mod[a, b, c] + uv * wc) % prime
    
    return np.all(S_mod == T_mod)


def _search_rank(prime, target_rank, n, m, p, max_iters=500, max_restarts=10):
    """Search for a rank-`target_rank` decomposition over finite field GF(prime).
    Uses alternating linear solves (ALS) with random restarts."""
    nm, mp, pn = n * m, m * p, p * n
    T_flat = np.zeros(nm * mp * pn, dtype=np.int64)
    for i in range(n):
        for l in range(m):
            for k in range(p):
                a = i * m + l
                b = l * p + k
                c = k * n + i
                T_flat[a * mp * pn + b * pn + c] = 1
    
    np.random.seed(42)
    
    for restart in range(max_restarts):
        if _remaining_time() < 5.0:
            break
        
        # Random initialization
        U = np.random.randint(1, prime, size=(nm, target_rank), dtype=np.int64) % prime
        V = np.random.randint(1, prime, size=(mp, target_rank), dtype=np.int64) % prime
        W = np.random.randint(1, prime, size=(pn, target_rank), dtype=np.int64) % prime
        
        for it in range(max_iters):
            if _remaining_time() < 5.0:
                break
            
            # Solve for U: for each a, solve linear system
            # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # For fixed a: T[a,b,c] over (b,c) = sum_r U[a,r] * (V[b,r]*W[c,r])
            # This is a linear system in U[a,:] with nm*mp*pn equations but only
            # mp*pn = 16 equations per a, and target_rank unknowns.
            
            # Build the system for U[a,:]
            # For each a: 16 equations (b in 0..15, c in 0..7), target_rank unknowns
            # A_U[a, (b,c), r] = V[b,r] * W[c,r]
            
            U_new = np.zeros_like(U)
            for a in range(nm):
                # Build 16 x target_rank matrix
                A_a = np.zeros((mp * pn, target_rank), dtype=np.int64)
                b_idx = 0
                for b in range(mp):
                    for c in range(pn):
                        for r in range(target_rank):
                            A_a[b_idx, r] = (V[b, r] * W[c, r]) % prime
                        b_idx += 1
                b_vec = T_flat[a * mp * pn : (a + 1) * mp * pn].astype(np.int64)
                
                # Solve A_a @ U[a,:] = b_vec mod prime
                # Use Gaussian elimination mod prime
                sol = _solve_mod(A_a, b_vec, prime)
                U_new[a, :] = sol
            
            U = U_new
            
            # Solve for V similarly
            V_new = np.zeros_like(V)
            for b in range(mp):
                A_b = np.zeros((nm * pn, target_rank), dtype=np.int64)
                eq_idx = 0
                for a in range(nm):
                    for c in range(pn):
                        for r in range(target_rank):
                            A_b[eq_idx, r] = (U[a, r] * W[c, r]) % prime
                        eq_idx += 1
                b_vec = np.zeros(nm * pn, dtype=np.int64)
                for a in range(nm):
                    for c in range(pn):
                        b_vec[a * pn + c] = T_flat[a * mp * pn + b * pn + c]
                
                sol = _solve_mod(A_b, b_vec, prime)
                V_new[b, :] = sol
            
            V = V_new
            
            # Solve for W
            W_new = np.zeros_like(W)
            for c in range(pn):
                A_c = np.zeros((nm * mp, target_rank), dtype=np.int64)
                eq_idx = 0
                for a in range(nm):
                    for b in range(mp):
                        for r in range(target_rank):
                            A_c[eq_idx, r] = (U[a, r] * V[b, r]) % prime
                        eq_idx += 1
                b_vec = np.zeros(nm * mp, dtype=np.int64)
                for a in range(nm):
                    for b in range(mp):
                        b_vec[a * mp + b] = T_flat[a * mp * pn + b * pn + c]
                
                sol = _solve_mod(A_c, b_vec, prime)
                W_new[c, :] = sol
            
            W = W_new
            
            # Check convergence
            if _verify_exact_mod(U, V, W, n, m, p, prime):
                return U, V, W
    
    return None


def _solve_mod(A, b, prime):
    """Solve A @ x = b mod prime using Gaussian elimination.
    A is (m, n), b is (m,). Returns x of shape (n,)."""
    m, n = A.shape
    # Augmented matrix
    M = np.column_stack([A, b]).astype(np.int64)
    
    # Gaussian elimination
    for col in range(min(m, n)):
        # Find pivot
        pivot = -1
        for row in range(col, m):
            if M[row, col] % prime != 0:
                pivot = row
                break
        if pivot == -1:
            # No pivot, column is free; set to 0
            continue
        # Swap
        M[[col, pivot]] = M[[pivot, col]]
        # Normalize
        inv_pivot = _modinv(M[col, col] % prime, prime)
        M[col, :] = (M[col, :] * inv_pivot) % prime
        # Eliminate
        for row in range(m):
            if row != col and M[row, col] % prime != 0:
                factor = M[row, col] % prime
                M[row, :] = (M[row, :] - factor * M[col, :]) % prime
    
    # Extract solution
    x = np.zeros(n, dtype=np.int64)
    for i in range(min(m, n)):
        x[i] = M[i, n] % prime
    return x


def _modinv(a, p):
    """Modular inverse of a mod p (p prime) using Fermat's little theorem."""
    return pow(a, p - 2, p)


def _build_schoolbook_32():
    """Build the schoolbook rank-32 decomposition."""
    n, m, p = 2, 4, 4
    rank = 32
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
    return np.array(U, dtype=float), np.array(V, dtype=float), np.array(W, dtype=float)


def construct():
    t0 = time.time()
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    T = _build_tensor(n, m, p)

    # Start with Strassen-28 as baseline
    U, V, W = _build_strassen_28()
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    best_rank = 28
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    err = _verify(U, V, W, T, best_rank)
    if err > 1e-9:
        U, V, W = _build_schoolbook_32()
        best_rank = 32
        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to find lower rank via finite field search
    # Try rank 24, 25, 26, 27 in order
    prime = 1000003  # A prime
    
    for target_rank in [24, 25, 26, 27]:
        if _remaining_time() < 10.0:
            break
        
        result = _search_rank(prime, target_rank, n, m, p, max_iters=200, max_restarts=5)
        if result is not None:
            U_sol, V_sol, W_sol = result
            # Convert to float for output
            U_f = U_sol.astype(float)
            V_f = V_sol.astype(float)
            W_f = W_sol.astype(float)
            
            # Verify over reals
            err = _verify(U_f, V_f, W_f, T, target_rank)
            if err < 1e-6:
                best_rank = target_rank
                best_result = {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}
                break
    
    return best_result

