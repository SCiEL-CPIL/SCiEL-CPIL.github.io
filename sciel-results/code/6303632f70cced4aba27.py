"""Program for matrix multiplication tensor decomposition (3x5 by 5x5)."""
DIMENSIONS = [3, 5, 5]

import time
import numpy as np
import random
from itertools import combinations

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

def _try_remove_columns(Ua, Va, Wa, n, m, p, col_indices):
    """Try removing multiple columns at once."""
    rank = Ua.shape[1]
    if len(col_indices) >= rank:
        return None, None, None
    
    # Sort indices for consistent removal
    col_indices = sorted(col_indices, reverse=True)
    Ua_new = Ua.copy()
    Va_new = Va.copy()
    Wa_new = Wa.copy()
    
    for idx in col_indices:
        Ua_new = np.delete(Ua_new, idx, axis=1)
        Va_new = np.delete(Va_new, idx, axis=1)
        Wa_new = np.delete(Wa_new, idx, axis=1)
    
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

def _structured_local_search(Ua, Va, Wa, n, m, p, deadline, max_iters=5000):
    """Structured search: try to find linear dependencies among columns."""
    start_time = time.time()
    best_Ua, best_Va, best_Wa = Ua.copy(), Va.copy(), Wa.copy()
    best_rank = Ua.shape[1]
    
    np.random.seed(42)
    random.seed(42)
    
    for it in range(max_iters):
        if (time.time() - start_time) > deadline:
            break
        
        rank = best_Ua.shape[1]
        if rank <= 1:
            break
        
        # Strategy 1: Try to express one column as a linear combination of others
        # Pick a random target column
        target_col = np.random.randint(0, rank)
        
        # Try to find coefficients such that replacing target_col with 
        # a combination of others still works
        # This is equivalent to checking if the tensor can be satisfied
        # without the target column
        
        # First, try simple removal
        Ua_r, Va_r, Wa_r = _try_remove_column(best_Ua, best_Va, best_Wa, n, m, p, target_col)
        if Ua_r is not None:
            best_Ua, best_Va, best_Wa = Ua_r, Va_r, Wa_r
            best_rank = Ua_r.shape[1]
            continue
        
        # Strategy 2: Try to merge two columns by finding a linear relation
        # Pick another column
        other_col = np.random.randint(0, rank)
        if other_col == target_col:
            other_col = (other_col + 1) % rank
        
        # Try to see if we can replace both columns with a single combined column
        # This requires finding scalars a, b such that the new column works
        # Simplified: try a=1, b=1 (sum) or a=1, b=-1 (difference)
        for a, b in [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 0.5), (0.5, 1.0)]:
            new_Ua = best_Ua.copy()
            new_Va = best_Va.copy()
            new_Wa = best_Wa.copy()
            
            # Replace target_col with a*target + b*other
            new_Ua[:, target_col] = a * best_Ua[:, target_col] + b * best_Ua[:, other_col]
            new_Va[:, target_col] = a * best_Va[:, target_col] + b * best_Va[:, other_col]
            new_Wa[:, target_col] = a * best_Wa[:, target_col] + b * best_Wa[:, other_col]
            
            # Now try to remove other_col
            Ua_r, Va_r, Wa_r = _try_remove_column(new_Ua, new_Va, new_Wa, n, m, p, other_col)
            if Ua_r is not None:
                best_Ua, best_Va, best_Wa = Ua_r, Va_r, Wa_r
                best_rank = Ua_r.shape[1]
                break
        
        # Strategy 3: Try random linear combinations of 2-3 columns
        if (time.time() - start_time) > deadline:
            break
        
        num_cols = min(3, rank - 1)  # Keep at least 1 column
        if num_cols >= 2:
            cols_to_combine = np.random.choice(rank, size=num_cols, replace=False)
            coeffs = np.random.randn(num_cols)
            # Normalize to avoid numerical issues
            coeffs = coeffs / np.linalg.norm(coeffs)
            
            new_Ua = best_Ua.copy()
            new_Va = best_Va.copy()
            new_Wa = best_Wa.copy()
            
            # Replace first column in cols_to_combine with combination
            target = cols_to_combine[0]
            new_Ua[:, target] = sum(c * best_Ua[:, col] for c, col in zip(coeffs, cols_to_combine))
            new_Va[:, target] = sum(c * best_Va[:, col] for c, col in zip(coeffs, cols_to_combine))
            new_Wa[:, target] = sum(c * best_Wa[:, col] for c, col in zip(coeffs, cols_to_combine))
            
            # Try to remove the other columns
            for col in cols_to_combine[1:]:
                Ua_r, Va_r, Wa_r = _try_remove_column(new_Ua, new_Va, new_Wa, n, m, p, col)
                if Ua_r is not None:
                    new_Ua, new_Va, new_Wa = Ua_r, Va_r, Wa_r
            
            if new_Ua.shape[1] < best_rank and _verify_fast(new_Ua, new_Va, new_Wa, n, m, p):
                best_Ua, best_Va, best_Wa = new_Ua, new_Va, new_Wa
                best_rank = new_Ua.shape[1]
    
    return best_Ua, best_Va, best_Wa

def _multi_column_removal_search(Ua, Va, Wa, n, m, p, deadline, max_pairs=1000):
    """Try removing pairs of columns that might be redundant together."""
    start_time = time.time()
    best_Ua, best_Va, best_Wa = Ua.copy(), Va.copy(), Wa.copy()
    best_rank = Ua.shape[1]
    
    rank = best_Ua.shape[1]
    attempts = 0
    
    while attempts < max_pairs and (time.time() - start_time) < deadline:
        # Pick two random columns to try removing
        if best_rank <= 2:
            break
        cols = np.random.choice(best_rank, size=2, replace=False)
        
        Ua_r, Va_r, Wa_r = _try_remove_columns(best_Ua, best_Va, best_Wa, n, m, p, cols.tolist())
        if Ua_r is not None:
            best_Ua, best_Va, best_Wa = Ua_r, Va_r, Wa_r
            best_rank = Ua_r.shape[1]
        
        attempts += 1
    
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
    
    # Phase 1: Greedy column removal (single columns)
    phase1_deadline = start_time + 20.0
    Ua, Va, Wa = _greedy_column_removal(Ua, Va, Wa, n, m, p, phase1_deadline)
    if Ua.shape[1] < best_rank:
        best_rank = Ua.shape[1]
        best_Ua, best_Va, best_Wa = Ua.copy(), Va.copy(), Wa.copy()
    
    # Phase 2: Multi-column removal search
    phase2_deadline = start_time + 40.0
    Ua, Va, Wa = _multi_column_removal_search(best_Ua, best_Va, best_Wa, n, m, p, phase2_deadline)
    if Ua.shape[1] < best_rank:
        best_rank = Ua.shape[1]
        best_Ua, best_Va, best_Wa = Ua.copy(), Va.copy(), Wa.copy()
    
    # Phase 3: Structured local search with linear combinations
    phase3_deadline = start_time + 100.0
    Ua, Va, Wa = _structured_local_search(best_Ua, best_Va, best_Wa, n, m, p, phase3_deadline, max_iters=20000)
    if Ua.shape[1] < best_rank:
        best_rank = Ua.shape[1]
        best_Ua, best_Va, best_Wa = Ua.copy(), Va.copy(), Wa.copy()
    
    # Phase 4: Final greedy pass on best found
    phase4_deadline = start_time + 130.0
    Ua, Va, Wa = _greedy_column_removal(best_Ua, best_Va, best_Wa, n, m, p, phase4_deadline)
    if Ua.shape[1] < best_rank:
        best_rank = Ua.shape[1]
        best_Ua, best_Va, best_Wa = Ua.copy(), Va.copy(), Wa.copy()
    
    # Phase 5: Final multi-column pass
    phase5_deadline = start_time + 140.0
    Ua, Va, Wa = _multi_column_removal_search(best_Ua, best_Va, best_Wa, n, m, p, phase5_deadline, max_pairs=500)
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
