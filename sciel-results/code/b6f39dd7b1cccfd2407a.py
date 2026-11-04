"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    n = len(a)
    
    # Compute row and column sums for both matrices
    a_row_sums = a.sum(axis=1)
    a_col_sums = a.sum(axis=0)
    b_row_sums = b.sum(axis=1)
    b_col_sums = b.sum(axis=0)
    
    # Linear relaxation: minimize sum_{i,j} a[i,j] * b_row_sums[j] + a_col_sums[i] * b[j,?]
    # Actually, a common linearization is to use the fact that for a permutation P,
    # cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # A lower bound / linear approximation: sum_i (a_row_sums[i] * b_row_sums[p[i]])
    # or more precisely, we can use the fact that if we fix p, the cost is quadratic.
    
    # Better approach: use a greedy + local search hybrid.
    # Start with a rank-matching heuristic, then improve with 2-opt style swaps.
    
    # Initial assignment: match high-flow facilities to low-distance locations
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # Compute initial cost
    def compute_cost(p):
        # sum_{i,j} a[i,j] * b[p[i], p[j]]
        # For efficiency, we can compute this as:
        # Let P be the permutation matrix. Cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
        # = sum_i sum_j a[i,j] * b[p[i], p[j]]
        # We can vectorize: for each i, sum_j a[i,j] * b[p[i], p[j]] = a[i,:] @ b[p[i], p]
        # So cost = sum_i (a[i,:] @ b[p[i], p])
        # = sum_i np.dot(a[i], b[p[i], p])
        total = 0
        for i in range(n):
            total += np.dot(a[i], b[p[i], p])
        return total
    
    current_cost = compute_cost(p)
    
    # Local search: try swaps and 3-cycles
    # For 256 facilities, O(n^2) swaps per iteration is ~65k evaluations.
    # Each evaluation is O(n) if we do it naively, but we can do incremental updates.
    
    # For a swap of p[i] and p[j], the change in cost is:
    # Old contribution involving i and j:
    #   sum_k a[i,k] * b[p[i], p[k]] + sum_k a[k,i] * b[p[k], p[i]] +
    #   sum_k a[j,k] * b[p[j], p[k]] + sum_k a[k,j] * b[p[k], p[j]]
    # New contribution:
    #   sum_k a[i,k] * b[p[j], p[k]] + sum_k a[k,i] * b[p[k], p[j]] +
    #   sum_k a[j,k] * b[p[i], p[k]] + sum_k a[k,j] * b[p[k], p[i]]
    # Plus the i-j and j-i terms change.
    
    # This is complex. Let's use a simpler approach: try all swaps but only evaluate
    # the delta efficiently.
    
    # Actually, for n=256, let's try a simpler local search with limited iterations.
    # Each full pass over all pairs is O(n^2 * n) = O(n^3) = ~16M operations, which is fine.
    
    # But we need to be careful about time. Let's do a few passes.
    
    improved = True
    max_passes = 10
    pass_count = 0
    
    while improved and pass_count < max_passes:
        improved = False
        pass_count += 1
        for i in range(n):
            for j in range(i + 1, n):
                # Try swapping p[i] and p[j]
                old_pi, old_pj = p[i], p[j]
                p[i], p[j] = p[j], p[i]
                new_cost = compute_cost(p)
                if new_cost < current_cost:
                    current_cost = new_cost
                    improved = True
                else:
                    p[i], p[j] = old_pi, old_pj
    
    return p.tolist()
# EVOLVE-BLOCK-END
