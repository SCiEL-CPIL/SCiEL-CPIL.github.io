"""Search for a low-rank exact tensor decomposition of (2,2,8) matrix multiplication."""
import time
from fractions import Fraction
import numpy as np

DIMENSIONS = [2, 2, 8]
N, M, P = DIMENSIONS
NM = N * M   # 4
MP = M * P   # 16
PN = P * N   # 16

def _build_target_tensor(n=N, m=M, p=P):
    """Build the target multiplication tensor T[a,b,c] as a numpy array."""
    nm, mp, pn = n * m, m * p, p * n
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def verify(U, V, W, n=N, m=M, p=P):
    """Verify that U, V, W form an exact decomposition. Returns True/False."""
    nm, mp, pn = n * m, m * p, p * n
    if not U or not V or not W:
        return False
    rank = len(U[0]) if U else 0
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
    # Fast verification via tensor contraction
    Ua = np.array(U, dtype=np.float64)   # (nm, rank)
    Va = np.array(V, dtype=np.float64)   # (mp, rank)
    Wa = np.array(W, dtype=np.float64)   # (pn, rank)
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # Reshape and contract
    Ua_r = Ua[:, :, None]  # (nm, rank, 1)
    Va_r = Va[None, :, :]  # (1, mp, rank)
    prod = Ua_r * Va_r      # (nm, mp, rank)
    T_computed = np.einsum('abr,cr->abc', prod, Wa)
    T_target = _build_target_tensor(n, m, p)
    return np.allclose(T_computed, T_target, atol=1e-9)

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

def strassen_decomposition_228():
    """
    Construct a rank-28 decomposition for M_{2,2,8} using 4 blocks of Strassen M_{2,2,2}.
    """
    n, m, p = N, M, P
    # Correct Strassen terms for M_{2,2,2}
    terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]
    
    rank = 28
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    
    for block_idx in range(4):
        for term_idx in range(7):
            col = block_idx * 7 + term_idx
            u, v, w = terms[term_idx]
            
            for i in range(4):
                U[i][col] = u[i]
                
            for j in range(2):
                for k_local in range(2):
                    k_global = 2 * block_idx + k_local
                    b_idx = j * P + k_global
                    v_idx = j * 2 + k_local
                    V[b_idx][col] = v[v_idx]
                    
            for i in range(2):
                for k_local in range(2):
                    k_global = 2 * block_idx + k_local
                    c_idx = k_global * N + i
                    w_idx = i * 2 + k_local
                    W[c_idx][col] = w[w_idx]
                    
    return U, V, W

def _try_rank24_search(deadline):
    """
    Search for a rank-24 decomposition by perturbing the rank-28 Strassen
    decomposition and attempting to eliminate 4 redundant terms.
    
    Strategy: The 2x2x8 tensor is a Kronecker product of the 2x2x2 tensor
    (rank 7) with a rank-4 selection tensor. We try to find a rank-24
    decomposition by allowing small perturbations and checking if the
    resulting tensor can be expressed with fewer terms.
    """
    import random
    
    # Start from the rank-28 decomposition
    U, V, W = strassen_decomposition_228()
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    
    T_target = _build_target_tensor()
    
    # Try to find a rank-24 sub-decomposition by zeroing out 4 columns
    # and checking if we can adjust the remaining 24 to still match
    best_rank = 28
    best_decomp = (U, V, W)
    
    # Try various subsets of 4 columns to remove
    cols = list(range(28))
    random.seed(42)
    
    # First, try simple: just zero out 4 columns and see if it still works
    # (unlikely, but quick check)
    for attempt in range(200):
        if time.time() > deadline:
            break
        remove = random.sample(cols, 4)
        keep = [c for c in cols if c not in remove]
        
        U24 = Ua[:, keep]
        V24 = Va[:, keep]
        W24 = Wa[:, keep]
        
        # Check if this 24-term sub-decomposition still works
        T_test = np.einsum('ar,br,cr->abc', U24, V24, W24)
        if np.allclose(T_test, T_target, atol=1e-9):
            # Found a valid rank-24 decomposition!
            best_rank = 24
            best_decomp = (U24.tolist(), V24.tolist(), W24.tolist())
            break
    
    # If not found by simple removal, try a more sophisticated approach:
    # Allow small perturbations of all 28 terms, then try to project to rank 24
    if best_rank == 28:
        for attempt in range(100):
            if time.time() > deadline:
                break
            # Create perturbed version
            U_pert = Ua.copy()
            V_pert = Va.copy()
            W_pert = Wa.copy()
            
            # Small random perturbations
            eps = 0.01
            U_pert += np.random.normal(0, eps, U_pert.shape)
            V_pert += np.random.normal(0, eps, V_pert.shape)
            W_pert += np.random.normal(0, eps, W_pert.shape)
            
            # Try to find a rank-24 approximation by SVD-like approach
            # This is a heuristic - not guaranteed to find exact solution
            # For now, just check if the perturbed version is still valid
            T_pert = np.einsum('ar,br,cr->abc', U_pert, V_pert, W_pert)
            if np.allclose(T_pert, T_target, atol=1e-9):
                # Still valid with 28 terms, but we want 24
                # Try to identify 4 nearly-redundant terms
                pass
    
    if best_rank < 28:
        return best_decomp
    return None

def construct():
    """Entry point: return the best decomposition found."""
    deadline = time.time() + 130  # safety margin before 150s kill
    
    # First try the known rank-28 Strassen decomposition
    best_result = None
    try:
        U, V, W = strassen_decomposition_228()
        if verify(U, V, W):
            best_result = {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Try to find a lower-rank decomposition
    if time.time() < deadline - 10:
        try:
            result = _try_rank24_search(deadline)
            if result is not None:
                U24, V24, W24 = result
                if verify(U24, V24, W24):
                    return {"U": U24, "V": V24, "W": W24}
        except Exception:
            pass
    
    # Return best found or fallback to schoolbook
    if best_result is not None:
        return best_result
    
    U, V, W = schoolbook()
    return {"U": U, "V": V, "W": W}
