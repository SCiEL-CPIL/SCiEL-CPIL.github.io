"""Search-based solver for QAPLIB wil100: Hungarian Init + GNN-inspired surrogate + Vectorized SA."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _swap_delta(a, b, p, i, j):
    """
    Compute the change in cost when swapping positions i and j in permutation p.
    O(n) vectorized computation.
    """
    pi = p[i]
    pj = p[j]
    
    # Direct terms (i,j pair)
    # Original: a[i,j]*b[pi,pj] + a[j,i]*b[pj,pi]
    # New: a[i,j]*b[pj,pi] + a[j,i]*b[pi,pj]
    # Delta = a[i,j]*(b[pj,pi]-b[pi,pj]) + a[j,i]*(b[pi,pj]-b[pj,pi])
    #        = (a[i,j]-a[j,i])*(b[pj,pi]-b[pi,pj])
    delta = (a[i, j] - a[j, i]) * (b[pj, pi] - b[pi, pj])
    
    # Diagonal terms (usually 0 for QAP but handle generally)
    # Original: a[i,i]*b[pi,pi] + a[j,j]*b[pj,pj]
    # New: a[i,i]*b[pj,pj] + a[j,j]*b[pi,pi]
    # Delta = a[i,i]*(b[pj,pj]-b[pi,pi]) + a[j,j]*(b[pi,pi]-b[pj,pj])
    delta += a[i, i] * (b[pj, pj] - b[pi, pi])
    delta += a[j, j] * (b[pi, pi] - b[pj, pj])
    
    # Cross terms with all other k
    # Term 1: sum_{k != i,j} (a[i,k] - a[j,k]) * (b[pj, p[k]] - b[pi, p[k]])
    # Term 2: sum_{k != i,j} (a[k,i] - a[k,j]) * (b[p[k], pj] - b[p[k], pi])
    
    # Compute over all k, then subtract k=i and k=j contributions
    
    # Term 1 over all k:
    diff_a_row = a[i, :] - a[j, :]
    diff_b_col = b[pj, p] - b[pi, p]
    sum_T1 = np.dot(diff_a_row, diff_b_col)
    
    # Subtract k=i: (a[i,i]-a[j,i]) * (b[pj,pi]-b[pi,pi])
    sum_T1 -= (a[i, i] - a[j, i]) * (b[pj, pi] - b[pi, pi])
    # Subtract k=j: (a[i,j]-a[j,j]) * (b[pj,pj]-b[pi,pj])
    sum_T1 -= (a[i, j] - a[j, j]) * (b[pj, pj] - b[pi, pj])
    
    # Term 2 over all k:
    diff_a_col = a[:, i] - a[:, j]
    diff_b_row = b[p, pj] - b[p, pi]
    sum_T2 = np.dot(diff_a_col, diff_b_row)
    
    # Subtract k=i: (a[i,i]-a[i,j]) * (b[pi,pj]-b[pi,pi])
    sum_T2 -= (a[i, i] - a[i, j]) * (b[pi, pj] - b[pi, pi])
    # Subtract k=j: (a[j,i]-a[j,j]) * (b[pj,pj]-b[pj,pi])
    sum_T2 -= (a[j, i] - a[j, j]) * (b[pj, pj] - b[pj, pi])
    
    delta += sum_T1 + sum_T2
    return int(delta)

def _gnn_embed(a, b, steps=3):
    """
    Compute node embeddings for facilities (from A) and locations (from B)
    using a simple message-passing mechanism (GNN-inspired).
    Returns embeddings of shape (n, d) where d = n for simplicity, or reduced.
    Here we use the matrix powers or repeated multiplication to capture higher-order structure.
    """
    n = len(a)
    
    # Normalize matrices to prevent explosion
    a_max = np.max(np.abs(a))
    b_max = np.max(np.abs(b))
    if a_max > 0:
        a_norm = a / a_max
    else:
        a_norm = a.copy()
    if b_max > 0:
        b_norm = b / b_max
    else:
        b_norm = b.copy()
    
    # Initialize embeddings as row sums (degree-like)
    e_f = a_norm.sum(axis=1)
    e_l = b_norm.sum(axis=1)
    
    # Message passing steps
    for _ in range(steps):
        # Update facility embeddings based on neighbor flows
        # e_f_new[i] = sum_j a_norm[i,j] * e_f[j]  (self-loop included)
        e_f = a_norm @ e_f
        # Update location embeddings based on neighbor distances
        e_l = b_norm @ e_l
        
    # Normalize embeddings
    e_f_norm = np.linalg.norm(e_f)
    e_l_norm = np.linalg.norm(e_l)
    if e_f_norm > 0:
        e_f = e_f / e_f_norm
    if e_l_norm > 0:
        e_l = e_l / e_l_norm
        
    return e_f, e_l

def _hungarian_initial(a, b):
    """
    Create an initial permutation using Hungarian algorithm on a linear surrogate.
    Combines row-sum centrality with GNN-inspired embeddings.
    Surrogate cost: C[i, j] = -w1 * a_row_sums[i] * b_row_sums[j] - w2 * dot(e_f[i], e_l[j])
    """
    n = len(a)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    # GNN embeddings
    e_f, e_l = _gnn_embed(a, b, steps=2)
    
    # Construct surrogate cost matrix
    # We want to assign high a_row_sums to high b_row_sums
    # And high similarity between embeddings
    # Cost[i, j] = - (w1 * a_row_sums[i] * b_row_sums[j] + w2 * e_f[i] * e_l[j])
    
    # Normalize row sums for combination
    ars_max = np.max(np.abs(a_row_sums))
    brs_max = np.max(np.abs(b_row_sums))
    if ars_max > 0:
        ars_norm = a_row_sums / ars_max
    else:
        ars_norm = a_row_sums
    if brs_max > 0:
        brs_norm = b_row_sums / brs_max
    else:
        brs_norm = b_row_sums
        
    # Weighting
    w1 = 1.0
    w2 = 0.5
    
    # Outer product for row sums
    term1 = w1 * np.outer(ars_norm, brs_norm)
    
    # Dot product for embeddings (element-wise since embeddings are scalars here)
    term2 = w2 * np.outer(e_f, e_l)
    
    C = -(term1 + term2)
    
    # Solve assignment problem
    row_ind, col_ind = linear_sum_assignment(C)
    
    # row_ind is facilities, col_ind is locations
    # p[facility] = location
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    
    return p

def _sa_polish(a, b, p, time_limit):
    """Simulated annealing polish phase with vectorized delta."""
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    start_time = time.time()
    
    # Cooling schedule
    t0 = 5000.0
    t_final = 0.1
    
    while time.time() - start_time < time_limit:
        elapsed = time.time() - start_time
        frac = elapsed / time_limit
        temp = t0 * (t_final / t0) ** frac
        
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
        
        delta = _swap_delta(a, b, p, i, j)
        
        # Acceptance criterion
        if delta < 0 or np.random.random() < np.exp(-delta / max(temp, 1e-10)):
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
    
    return best_p, best_cost

def _local_search(a, b, p, time_limit):
    """First-improvement local search (faster than best-improvement for n=100)."""
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Find first improving swap
        for i in range(n):
            for j in range(i + 1, n):
                delta = _swap_delta(a, b, p, i, j)
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
            
    return best_p, best_cost

def solve(a, b):
    n = len(a)
    deadline = 115.0  # Safety margin before 120s limit
    start = time.time()
    
    np.random.seed(42)
    
    best_p = None
    best_cost = np.inf
    
    # Strategy 1: Hungarian initial solution with GNN surrogate
    p_hung = _hungarian_initial(a, b)
    cost_hung = _qap_cost(a, b, p_hung)
    best_p = p_hung.copy()
    best_cost = cost_hung
    
    # Strategy 2: Local search polish on Hungarian solution
    time_remaining = deadline - (time.time() - start)
    if time_remaining > 5:
        p_ls, cost_ls = _local_search(a, b, p_hung.copy(), min(5.0, time_remaining * 0.2))
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
    
    # Strategy 3: SA polish from best solution
    time_remaining = deadline - (time.time() - start)
    if time_remaining > 10:
        p_sa, cost_sa = _sa_polish(a, b, best_p.copy(), time_remaining - 5)
        if cost_sa < best_cost:
            best_cost = cost_sa
            best_p = p_sa.copy()
    
    return best_p.tolist()

# EVOLVE-BLOCK-END
