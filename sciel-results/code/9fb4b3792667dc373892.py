"""
Hungarian Initialization + Vectorized Simulated Annealing solver for QAPLIB wil100.
Strategy:
1. Compute facility flow centrality (row sums of A) and location distance centrality (row sums of B).
2. Construct a linear surrogate cost matrix C[i, j] = -a_row_sums[i] * b_row_sums[j] 
   (minimizing C maximizes the product of centralities, aligning high-flow facilities 
   with high-distance locations in a globally optimal linear assignment).
3. Solve the assignment problem using the Hungarian algorithm (scipy.optimize.linear_sum_assignment) 
   to get a structurally coherent initial permutation p.
4. Perform Simulated Annealing with vectorized delta evaluation for 2-opt swaps.
   - Precompute row/col differences for fast delta calculation.
   - Use a cooling schedule that allows escape from local minima.
5. Return the best permutation found.
"""
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
    delta = (a[i, j] - a[j, i]) * (b[pj, pi] - b[pi, pj])
    
    # Diagonal terms (usually 0 for QAP but handle generally)
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

def _hungarian_initial(a, b):
    """
    Create an initial permutation using Hungarian algorithm on a linear surrogate.
    Surrogate cost: C[i, j] = -a_row_sums[i] * b_row_sums[j]
    Minimizing C maximizes the product of flow centrality and distance centrality.
    """
    n = len(a)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    # Construct surrogate cost matrix
    # We want to assign high a_row_sums to high b_row_sums
    # So cost[i, j] = -a_row_sums[i] * b_row_sums[j]
    # Minimizing this cost maximizes the product sum
    C = -np.outer(a_row_sums, b_row_sums)
    
    # Solve assignment problem
    row_ind, col_ind = linear_sum_assignment(C)
    
    # row_ind is facilities, col_ind is locations
    # p[facility] = location
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    
    return p

def solve(a, b):
    n = len(a)
    total_deadline = 115.0  # Safety margin before 120s limit
    start_time = time.time()
    
    np.random.seed(42)  # Fixed seed for reproducibility within a run
    
    # Phase 1: Hungarian initialization
    p = _hungarian_initial(a, b)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    # Phase 2: Simulated Annealing
    T_start = 1000.0
    T_end = 1.0
    cooling_rate = 0.999  # Geometric cooling
    
    elapsed = time.time() - start_time
    while time.time() - start_time < total_deadline:
        # Update temperature based on elapsed time fraction
        frac = min(1.0, elapsed / total_deadline)
        T = T_start * (T_end / T_start) ** frac
        
        # Generate random swap
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
        
        delta = _swap_delta(a, b, p, i, j)
        
        # Acceptance criterion
        if delta < 0 or np.random.rand() < np.exp(-delta / max(T, 1e-10)):
            p[i], p[j] = p[j], p[i]
            current_cost += int(delta)
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        
        elapsed = time.time() - start_time
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
