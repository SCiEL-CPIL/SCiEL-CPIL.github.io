"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 7]
DEADLINE_SECONDS = 135.0  # Safety margin before 150s kill

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    for _ in range(5):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        C_computed = np.zeros((n, p))
        for c in range(rank):
            uc = U[:, c]
            vc = V[:, c]
            wc = W[:, c]
            A_c = uc.reshape(n, m)
            B_c = vc.reshape(m, p)
            # W index: W[k*n+i, c] corresponds to C[i,k]
            C_c = np.zeros((n, p))
            for idx in range(p * n):
                k, i = idx // n, idx % n
                C_c[i, k] = wc[idx]
            C_computed += C_c
        if not np.allclose(C_computed, C_expected, atol=tol, rtol=tol):
            return False
    return True

def _build_schoolbook(n, m, p):
    """Build the standard schoolbook decomposition."""
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

def _is_nice_half_integer(x, tol=1e-9):
    """Check if x is close to an integer or half-integer."""
    doubled = 2.0 * x
    rounded = np.round(doubled)
    return abs(doubled - rounded) < tol

def _generate_structured_patterns(n, m, p, seed=42):
    """Generate diverse structured patterns for U and V."""
    np.random.seed(seed)
    U_patterns = []
    V_patterns = []
    
    # Basic unit patterns
    for i in range(n):
        for j in range(m):
            u_pat = np.zeros(n*m)
            u_pat[i*m+j] = 1
            U_patterns.append(u_pat)
    for j in range(m):
        for k in range(p):
            v_pat = np.zeros(m*p)
            v_pat[j*p+k] = 1
            V_patterns.append(v_pat)
    
    # Row-wise patterns
    for i in range(n):
        u_pat = np.zeros(n*m)
        for j in range(m):
            u_pat[i*m+j] = 1
        U_patterns.append(u_pat)
    for j in range(m):
        v_pat = np.zeros(m*p)
        for k in range(p):
            v_pat[j*p+k] = 1
        V_patterns.append(v_pat)
    
    # Random sparse patterns with {-1, 0, 1}
    for _ in range(200):
        u_pat = np.random.choice([-1, 0, 1], size=n*m)
        v_pat = np.random.choice([-1, 0, 1], size=m*p)
        U_patterns.append(u_pat)
        V_patterns.append(v_pat)
    
    # Random dense patterns with small integers
    for _ in range(100):
        u_pat = np.random.randint(-2, 3, size=n*m).astype(float)
        v_pat = np.random.randint(-2, 3, size=m*p).astype(float)
        U_patterns.append(u_pat)
        V_patterns.append(v_pat)
    
    # Winograd-style: use DFT-like structures
    # For 4x4 blocks, use 2x2 Strassen-like patterns extended
    for _ in range(50):
        u_pat = np.zeros(n*m)
        v_pat = np.zeros(m*p)
        # Random sign patterns on blocks
        for i in range(n):
            for j in range(m):
                u_pat[i*m+j] = np.random.choice([-1, 1])
        for j in range(m):
            for k in range(p):
                v_pat[j*p+k] = np.random.choice([-1, 1])
        U_patterns.append(u_pat)
        V_patterns.append(v_pat)
    
    return U_patterns, V_patterns

def _compute_W_from_UV(U_mat, V_mat, n, m, p):
    """Compute W from given U and V matrices. Returns W and validity flag."""
    rank = U_mat.shape[1]
    W_mat = np.zeros((p*n, rank))
    valid = True
    for c in range(rank):
        uc = U_mat[:, c]
        vc = V_mat[:, c]
        for k in range(p):
            for i in range(n):
                denom = 0.0
                for j in range(m):
                    denom += uc[i*m+j] * vc[j*p+k]
                if abs(denom) < 1e-12:
                    valid = False
                    break
                W_mat[k*n+i, c] = 1.0 / denom
            if not valid:
                break
        if not valid:
            break
    return W_mat, valid

def _check_nice_W(W_mat, tol=1e-9):
    """Check if all entries in W are close to integers or half-integers."""
    W_doubled = W_mat * 2.0
    W_rounded = np.round(W_doubled)
    return np.allclose(W_doubled, W_rounded, atol=tol)

def _try_structured_search(n, m, p, start_time, deadline):
    """Search for low-rank decompositions using structured patterns."""
    np.random.seed(42)
    best_rank = n * m * p
    best_result = _build_schoolbook(n, m, p)
    
    U_patterns, V_patterns = _generate_structured_patterns(n, m, p)
    
    # Try different target ranks from lower to higher
    for target_rank in range(50, 113):
        if time.time() > deadline:
            break
        
        max_attempts = 150 if target_rank < 80 else 80
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Randomly select target_rank patterns
            indices = np.random.choice(len(U_patterns), size=target_rank, replace=False)
            U_mat = np.array([U_patterns[idx] for idx in indices], dtype=float).T
            V_mat = np.array([V_patterns[idx] for idx in indices], dtype=float).T
            
            W_mat, valid = _compute_W_from_UV(U_mat, V_mat, n, m, p)
            if not valid:
                continue
            
            if not _check_nice_W(W_mat):
                continue
            
            # Convert to exact values
            U_list = [[int(round(U_mat[i, c])) for c in range(target_rank)] for i in range(n*m)]
            V_list = [[int(round(V_mat[i, c])) for c in range(target_rank)] for i in range(m*p)]
            W_list = [[float(int(round(W_mat[i, c] * 2))) / 2.0 for c in range(target_rank)] for i in range(p*n)]
            
            if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    return best_result

def _try_direct_random_search(n, m, p, start_time, deadline):
    """Direct random search with small integer entries."""
    np.random.seed(123)
    best_rank = n * m * p
    best_result = _build_schoolbook(n, m, p)
    
    for target_rank in range(50, 113):
        if time.time() > deadline:
            break
        
        max_attempts = 100
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            U_rand = np.random.randint(-2, 3, size=(n*m, target_rank)).astype(float)
            V_rand = np.random.randint(-2, 3, size=(m*p, target_rank)).astype(float)
            
            W_mat, valid = _compute_W_from_UV(U_rand, V_rand, n, m, p)
            if not valid:
                continue
            
            if not _check_nice_W(W_mat):
                continue
            
            U_list = [[int(round(U_rand[i, c])) for c in range(target_rank)] for i in range(n*m)]
            V_list = [[int(round(V_rand[i, c])) for c in range(target_rank)] for i in range(m*p)]
            W_list = [[float(int(round(W_mat[i, c] * 2))) / 2.0 for c in range(target_rank)] for i in range(p*n)]
            
            if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                return {"U": list(map(list, U_list)), "V": list(map(list, V_list)), "W": list(map(list, W_list))}
    
    return best_result

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    best = _build_schoolbook(n, m, p)
    
    # Try structured search first
    try:
        result = _try_structured_search(n, m, p, start_time, deadline)
        if result is not None:
            if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
                best = result
    except Exception:
        pass
    
    # Try direct random search if time remains
    if time.time() < deadline - 5:
        try:
            result = _try_direct_random_search(n, m, p, start_time, deadline)
            if result is not None:
                if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
                    # Compare ranks
                    if len(result["U"][0]) < len(best["U"][0]):
                        best = result
        except Exception:
            pass
    
    return best
# EVOLVE-BLOCK-END
