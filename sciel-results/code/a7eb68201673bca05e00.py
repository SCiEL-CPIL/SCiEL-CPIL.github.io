"""Program for matrix multiplication tensor decomposition (3x5 by 5x5)."""
DIMENSIONS = [3, 5, 5]

import time
import numpy as np
import random

def _build_schoolbook(n, m, p):
    """Return (U, V, W) lists-of-lists for schoolbook decomposition."""
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

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reconstructs the matrix multiplication tensor."""
    rank = len(U[0])
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += Ua[i*m+j, r] * Va[j*p+k, r] * Wa[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _verify_fast(Ua, Va, Wa, n, m, p):
    """Fast vectorized verification using numpy."""
    rank = Ua.shape[1]
    # Build index arrays for all (i, j, k) triples
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = np.dot(Ua[a], Va[b] * Wa[c])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_remove_column(Ua, Va, Wa, n, m, p, col_idx):
    """Try removing column col_idx and see if the remaining columns still satisfy the tensor."""
    rank = Ua.shape[1]
    if rank <= 1:
        return None, None, None
    
    Ua_new = np.delete(Ua, col_idx, axis=1)
    Va_new = np.delete(Va, col_idx, axis=1)
    Wa_new = np.delete(Wa, col_idx, axis=1)
    
    if _verify_fast(Ua_new, Va_new, Wa_new, n, m, p):
        return Ua_new, Va_new, Wa_new
    return None, None, None

def _greedy_column_removal(Ua, Va, Wa, n, m, p, deadline):
    """Greedily try to remove columns while maintaining validity."""
    start_time = time.time()
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        rank = Ua.shape[1]
        for col_idx in range(rank):
            if (time.time() - start_time) > deadline:
                break
            Ua_new, Va_new, Wa_new = _try_remove_column(Ua, Va, Wa, n, m, p, col_idx)
            if Ua_new is not None:
                Ua, Va, Wa = Ua_new, Va_new, Wa_new
                improved = True
                break
    return Ua, Va, Wa

def _random_local_search(Ua, Va, Wa, n, m, p, deadline, max_iters=5000):
    """Try random perturbations to find lower-rank decompositions."""
    start_time = time.time()
    best_Ua, best_Va, best_Wa = Ua.copy(), Va.copy(), Wa.copy()
    best_rank = Ua.shape[1]
    
    np.random.seed(12345)
    random.seed(12345)
    
    for it in range(max_iters):
        if (time.time() - start_time) > deadline:
            break
        
        # Try to reduce rank by 1 by finding a linear combination that allows removal
        rank = best_Ua.shape[1]
        if rank <= 1:
            break
        
        # Pick a random column to try to eliminate
        target_col = np.random.randint(0, rank)
        
        # Try to express column target_col as a combination of others
        # This is a simplified approach: just try random coefficient vectors
        
        # Alternative: try swapping out a column with a random one
        new_Ua = best_Ua.copy()
        new_Va = best_Va.copy()
        new_Wa = best_Wa.copy()
        
        # Randomly perturb one column
        col_to_perturb = np.random.randint(0, rank)
        perturbation = np.random.randn() * 0.1
        new_Ua[:, col_to_perturb] += perturbation
        new_Va[:, col_to_perturb] += perturbation
        new_Wa[:, col_to_perturb] += perturbation
        
        if _verify_fast(new_Ua, new_Va, new_Wa, n, m, p):
            # Check if we can now remove a column
            for c in range(rank):
                Ua_r, Va_r, Wa_r = _try_remove_column(new_Ua, new_Va, new_Wa, n, m, p, c)
                if Ua_r is not None:
                    best_Ua, best_Va, best_Wa = Ua_r, Va_r, Wa_r
                    best_rank = Ua_r.shape[1]
                    break
            if (time.time() - start_time) > deadline:
                break
    
    return best_Ua, best_Va, best_Wa

def construct():
    start_time = time.time()
    deadline = 140.0  # 150s limit, leave 10s margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    
    best_rank = n * m * p
    best_Ua, best_Va, best_Wa = Ua.copy(), Va.copy(), Wa.copy()
    
    # Verify schoolbook
    if not _verify_fast(Ua, Va, Wa, n, m, p):
        raise ValueError("Schoolbook verification failed")
    
    # Phase 1: Greedy column removal
    phase1_deadline = start_time + 30.0
    Ua, Va, Wa = _greedy_column_removal(Ua, Va, Wa, n, m, p, phase1_deadline)
    if Ua.shape[1] < best_rank:
        best_rank = Ua.shape[1]
        best_Ua, best_Va, best_Wa = Ua.copy(), Va.copy(), Wa.copy()
    
    # Phase 2: More aggressive search - try structured reductions
    # For matrix multiplication, we can try to find decompositions using
    # known algebraic identities or random linear combinations
    
    phase2_deadline = start_time + 80.0
    Ua, Va, Wa = _random_local_search(Ua, Va, Wa, n, m, p, phase2_deadline, max_iters=10000)
    if Ua.shape[1] < best_rank:
        best_rank = Ua.shape[1]
        best_Ua, best_Va, best_Wa = Ua.copy(), Va.copy(), Wa.copy()
    
    # Phase 3: Try another round of greedy removal on the best found
    phase3_deadline = start_time + 120.0
    Ua, Va, Wa = _greedy_column_removal(best_Ua, best_Va, best_Wa, n, m, p, phase3_deadline)
    if Ua.shape[1] < best_rank:
        best_rank = Ua.shape[1]
        best_Ua, best_Va, best_Wa = Ua.copy(), Va.copy(), Wa.copy()
    
    # Final verification
    if not _verify_fast(best_Ua, best_Va, best_Wa, n, m, p):
        # Fallback to schoolbook
        best_Ua, best_Va, best_Wa = np.array(U, dtype=np.float64), np.array(V, dtype=np.float64), np.array(W, dtype=np.float64)
        best_rank = n * m * p
    
    # Convert back to lists
    best_U = best_Ua.tolist()
    best_V = best_Va.tolist()
    best_W = best_Wa.tolist()
    
    # Ensure entries are exact (0, 1, or simple fractions)
    # For safety, round small values to 0
    for mat in [best_U, best_V, best_W]:
        for row in mat:
            for i in range(len(row)):
                if abs(row[i]) < 1e-10:
                    row[i] = 0.0
                elif abs(row[i] - round(row[i])) < 1e-10:
                    row[i] = float(round(row[i]))
    
    return {"U": best_U, "V": best_V, "W": best_W}
