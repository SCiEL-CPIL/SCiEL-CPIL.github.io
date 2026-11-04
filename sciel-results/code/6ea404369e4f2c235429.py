# EVOLVE-BLOCK-START
import numpy as np
import time


def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))


def _compute_delta_matrix(a, b, p):
    """
    Compute the full n x n matrix of swap deltas.
    D[i, j] is the cost change if we swap positions i and j.
    Uses O(n^3) matrix multiplications, which is fast for n=100.
    """
    n = len(p)
    Bp = b[np.ix_(p, p)]
    
    # M1[i, j] = sum_k a[k, i] b[p[k], p[j]]
    M1 = a.T @ Bp
    
    # M2[i, j] = sum_k a[i, k] b[p[k], p[j]]
    M2 = a @ Bp
    
    # Term 1 matrix: T1[i, j] = M1[i, j] - M1[i, i] - M1[j, j] + M1[j, i]
    # But we need to exclude k=i and k=j from the sum.
    # Let's compute the raw sum first, then subtract corrections.
    
    # Raw T1[i, j] = sum_k a[k,i] b[p[k],p[j]] - sum_k a[k,i] b[p[k],p[i]]
    #               - sum_k a[k,j] b[p[k],p[j]] + sum_k a[k,j] b[p[k],p[i]]
    # = M1[i, j] - M1[i, i] - M1[j, j] + M1[j, i]
    
    T1_raw = M1 - np.diag(M1)[:, None] - np.diag(M1)[None, :] + M1.T
    
    # Corrections for T1:
    # k=i: (a[i,i] - a[i,j]) * (b[p[i],p[j]] - b[p[i],p[i]])
    #      = (a[i,i] - a[i,j]) * (Bp[i,j] - Bp[i,i])
    # k=j: (a[j,i] - a[j,j]) * (b[p[j],p[j]] - b[p[j],p[i]])
    #      = (a[j,i] - a[j,j]) * (Bp[j,j] - Bp[j,i])
    
    diag_a = np.diag(a)
    diag_Bp = np.diag(Bp)
    
    # corr_i[i, j] = (diag_a[i] - a[i,j]) * (Bp[i,j] - diag_Bp[i])
    C1 = (diag_a[:, None] - a) * (Bp - diag_Bp[:, None])
    
    # C2[i, j] = (a[j,i] - a[j,j]) * (Bp[j,j] - Bp[j,i])
    # a[j,i] is a.T[i, j]
    # a[j,j] is diag_a[j]
    # Bp[j,j] is diag_Bp[j]
    # Bp[j,i] is Bp.T[i, j] (Bp is symmetric? No, b is symmetric in QAP usually, but let's assume general)
    # Actually b is symmetric in standard QAP instances.
    # Let's stick to the derivation from the inspiration which assumed symmetry or handled it generally.
    # The inspiration code: C2 = (a.T - diag_a[None, :]) * (diag_Bp[None, :] - Bp.T)
    # Let's verify indices.
    # We want term for k=j in T1 sum: (a[j,i] - a[j,j]) * (b[p[j],p[j]] - b[p[j],p[i]])
    # In matrix D[i,j], this correction is added to T1[i,j].
    # a[j,i] is element (j,i) of a. In matrix a.T, this is (i,j).
    # a[j,j] is diag_a[j].
    # b[p[j],p[j]] is Bp[j,j] = diag_Bp[j].
    # b[p[j],p[i]] is Bp[j,i] = Bp.T[i,j] (if b symmetric, Bp symmetric).
    # So C2[i,j] should be (a.T[i,j] - diag_a[j]) * (diag_Bp[j] - Bp.T[i,j])
    C2 = (a.T - diag_a[None, :]) * (diag_Bp[None, :] - Bp.T)
    
    T1 = T1_raw - C1 - C2
    
    # Term 2 matrix: T2[i, j] = M2[i, j] - M2[i, i] - M2[j, j] + M2[j, i]
    # T2[i, j] = sum_k (a[i,k] - a[j,k]) * (b[p[j],p[k]] - b[p[i],p[k]])
    # = M2[i, j] - M2[i, i] - M2[j, j] + M2[j, i]
    
    T2_raw = M2 - np.diag(M2)[:, None] - np.diag(M2)[None, :] + M2.T
    
    # Corrections for T2:
    # k=i: (a[i,i] - a[j,i]) * (b[p[j],p[i]] - b[p[i],p[i]])
    #      = (a[i,i] - a[j,i]) * (Bp[j,i] - Bp[i,i])
    # k=j: (a[i,j] - a[j,j]) * (b[p[j],p[j]] - b[p[i],p[j]])
    #      = (a[i,j] - a[j,j]) * (Bp[j,j] - Bp[i,j])
    
    # C3[i, j] = (a[i,i] - a[j,i]) * (Bp[j,i] - Bp[i,i])
    # a[i,i] is diag_a[i]
    # a[j,i] is a.T[i, j]
    # Bp[j,i] is Bp.T[i, j]
    # Bp[i,i] is diag_Bp[i]
    C3 = (diag_a[:, None] - a.T) * (Bp.T - diag_Bp[:, None])
    
    # C4[i, j] = (a[i,j] - a[j,j]) * (Bp[j,j] - Bp[i,j])
    # a[i,j] is a[i, j]
    # a[j,j] is diag_a[j]
    # Bp[j,j] is diag_Bp[j]
    # Bp[i,j] is Bp[i, j]
    C4 = (a - diag_a[None, :]) * (diag_Bp[None, :] - Bp)
    
    T2 = T2_raw - C3 - C4
    
    # Term 3 matrix: T3[i, j] = (a[i,j] - a[j,i]) * (b[p[j],p[i]] - b[p[i],p[j]])
    # = (a[i,j] - a[j,i]) * (Bp[j,i] - Bp[i,j])
    T3 = (a - a.T) * (Bp.T - Bp)
    
    D = T1 + T2 + T3
    
    # The diagonal should be 0 (swapping with self).
    np.fill_diagonal(D, 0)
    
    return D


def _local_search_steepest(a, b, p, deadline, best_cost, best_p):
    """
    Perform steepest descent local search using the full delta matrix.
    """
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    
    if current_cost < best_cost:
        best_cost = current_cost
        best_p = p.copy()
        
    improved = True
    # Limit iterations to prevent infinite loops in case of bugs or precision issues, though unlikely with ints
    max_iter = 1000
    
    while improved:
        if time.time() >= deadline:
            break
            
        D = _compute_delta_matrix(a, b, p)
        
        # Find the best improving move
        min_val = np.min(D)
        
        if min_val < 0:
            min_idx = np.argmin(D)
            i, j = np.unravel_index(min_idx, D.shape)
            
            # Apply swap
            p[i], p[j] = p[j], p[i]
            current_cost += min_val
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
                
            improved = True
        else:
            improved = False
            
    return best_cost, best_p, p, current_cost


def _simulated_annealing(a, b, p, deadline, best_cost, best_p, rng, t_start=1000.0, t_end=1.0, alpha=0.999):
    """
    Simulated Annealing using fast delta matrix lookups.
    """
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    
    if current_cost < best_cost:
        best_cost = current_cost
        best_p = p.copy()
        
    t = t_start
    step = 0
    
    while time.time() < deadline and t > t_end:
        # Compute delta matrix once per step (expensive but allows O(1) delta lookup)
        # For n=100, this is ~1-2ms. 115s / 2ms = ~50k steps.
        # This is a reasonable number for SA.
        D = _compute_delta_matrix(a, b, p)
        
        i = rng.integers(n)
        j = rng.integers(n)
        if i == j:
            continue
            
        delta = D[i, j]
        
        # Accept if improving or with probability exp(-delta/t)
        if delta < 0 or rng.random() < np.exp(-delta / max(t, 1e-10)):
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
                
        t *= alpha
        step += 1
        
        # Occasionally re-center or just let it cool
        if step % 1000 == 0 and time.time() > deadline - 5:
             break

    return best_cost, best_p, p, current_cost


def solve(a, b):
    a = np.ascontiguousarray(a, dtype=np.int64)
    b = np.ascontiguousarray(b, dtype=np.int64)
    n = len(a)
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin before 120s kill

    rng = np.random.default_rng(42)
    
    best_cost = np.iinfo(np.int64).max
    best_p = None
    
    # --- Constructive starts ---
    
    # Strategy 1: Rank matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p1 = np.empty(n, dtype=np.int64)
    p1[heavy] = central
    c1 = _qap_cost(a, b, p1)
    if c1 < best_cost:
        best_cost = c1
        best_p = p1.copy()
        
    # Strategy 2: Random
    p2 = rng.permutation(n).astype(np.int64)
    c2 = _qap_cost(a, b, p2)
    if c2 < best_cost:
        best_cost = c2
        best_p = p2.copy()
        
    # Strategy 3: Greedy Nearest Neighbor
    p3 = np.zeros(n, dtype=np.int64)
    used = np.zeros(n, dtype=bool)
    a_row_sums = a.sum(axis=1)
    b_col_sums = b.sum(axis=1)
    order = np.argsort(-a_row_sums, kind="stable")
    for idx in order:
        if not np.any(used):
            p3[idx] = int(np.argmin(b_col_sums))
        else:
            placed = np.where(~used)[0]
            scores = np.zeros(n, dtype=np.float64)
            for j in placed:
                scores += a[idx, j] * b[:, p3[j]] + a[j, idx] * b[p3[j], :]
            candidates = np.where(~used)[0]
            p3[idx] = int(candidates[np.argmin(scores[candidates])])
        used[p3[idx]] = True
    c3 = _qap_cost(a, b, p3)
    if c3 < best_cost:
        best_cost = c3
        best_p = p3.copy()
        
    # --- Local Search from best start ---
    p = best_p.copy()
    best_cost, best_p, p, cur_cost = _local_search_steepest(a, b, p, deadline, best_cost, best_p)
    
    # --- Simulated Annealing from current best ---
    if time.time() < deadline - 10.0:
        p = best_p.copy()
        # Use a higher initial temperature for exploration
        best_cost, best_p, p, cur_cost = _simulated_annealing(a, b, p, deadline, best_cost, best_p, rng, t_start=5000.0, t_end=10.0, alpha=0.999)
        
    # --- Final Local Search to polish the SA result ---
    if time.time() < deadline - 2.0:
        p = best_p.copy()
        best_cost, best_p, p, cur_cost = _local_search_steepest(a, b, p, deadline, best_cost, best_p)
        
    # Final verification
    final_cost = _qap_cost(a, b, best_p)
    if final_cost < best_cost:
        best_cost = final_cost
        
    return best_p.tolist()
# EVOLVE-BLOCK-END
