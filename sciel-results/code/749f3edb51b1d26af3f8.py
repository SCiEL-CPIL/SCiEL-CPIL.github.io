import numpy as np
import time
import random

def solve(a, b):
    """
    Solve the QAP instance using Tabu Search.
    a: Flow matrix (n, n)
    b: Distance matrix (n, n)
    Returns: Permutation p of 0..n-1
    """
    n = len(a)
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit

    # 1. Initial Solution: Greedy Construction
    # We start with a random permutation and improve it with a simple local search
    # or use a greedy approach. For robustness, we'll use a random start and 
    # immediately enter the tabu search loop, which will quickly improve it.
    # A better start: sort by row sums of A and B inversely.
    a_sums = a.sum(axis=1)
    b_sums = b.sum(axis=1)
    
    # Rank matching: High flow -> Low distance
    order_a = np.argsort(-a_sums)
    order_b = np.argsort(b_sums)
    
    p = np.empty(n, dtype=np.int64)
    p[order_a] = order_b
    
    # Convert to list for easier manipulation if needed, but we'll keep as numpy array
    # and manage indices carefully.
    
    # Precompute some helper data if possible, but dynamic updates are better.
    
    # Calculate initial cost
    def calc_cost(p):
        # p is a numpy array of locations for each facility
        # Cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
        # We can compute this as:
        # C = sum_i sum_j a[i,j] b[p_i, p_j]
        # This is equivalent to trace(A @ B_p) where B_p is permuted B?
        # Let P be permutation matrix. B_p = P B P^T.
        # Cost = trace(A P B P^T) = trace(P^T A P B)
        # Simpler: just do the double sum or use einsum.
        # np.einsum('ij,ik,jk->', a, b[p][:, p]) is O(n^3) if not careful, 
        # but b[p][:, p] is O(n^2) to construct.
        # Let's use the efficient calculation:
        # C = sum_i (a[i, :] * b[p[i], p]).sum()
        # This is O(n^2).
        perm_b = b[np.ix_(p, p)]
        return np.sum(a * perm_b)

    current_cost = calc_cost(p)
    best_cost = current_cost
    best_p = p.copy()
    
    # Tabu Search Parameters
    tabu_tenure = 15
    tabu_list = {}  # (i, j) -> iteration when it becomes non-tabu
    
    iteration = 0
    
    while time.time() < deadline:
        # Find the best non-tabu move
        best_delta = -np.inf
        best_i, best_j = -1, -1
        
        # We can sample a subset of moves if n is large, but n=100 is small.
        # Iterate over all pairs i < j
        for i in range(n):
            for j in range(i + 1, n):
                # Check if move (i, j) is tabu
                # We store (i, j) with i < j
                if (i, j) in tabu_list and tabu_list[(i, j)] > iteration:
                    continue
                
                # Calculate delta for swapping i and j
                u = p[i]
                v = p[j]
                
                # Delta formula derived earlier:
                # Delta = sum_{k != i, j} (a[i, k] - a[j, k]) * (b[v, p[k]] - b[u, p[k]]) 
                #         + (a[j, i] - a[i, j]) * (b[u, v] - b[v, u])
                
                # Vectorized calculation for the sum part
                # Mask to exclude i and j
                mask = np.ones(n, dtype=bool)
                mask[i] = False
                mask[j] = False
                
                # a_diff = a[i, mask] - a[j, mask]
                # b_diff = b[v, p[mask]] - b[u, p[mask]]
                # sum_part = np.dot(a_diff, b_diff)
                
                # To speed up, we can precompute row differences?
                # a[i, :] - a[j, :] is O(n).
                # b[v, p] - b[u, p] is O(n).
                
                a_diff = a[i, mask] - a[j, mask]
                b_diff = b[v, p[mask]] - b[u, p[mask]]
                sum_part = np.dot(a_diff, b_diff)
                
                interaction_part = (a[j, i] - a[i, j]) * (b[u, v] - b[v, u])
                
                delta = sum_part + interaction_part
                
                if delta < best_delta:
                    best_delta = delta
                    best_i, best_j = i, j
        
        # If no valid move found (should not happen unless all tabu), break or clear tabu
        if best_i == -1:
            # Clear tabu list or reduce tenure? For now, break.
            break
        
        # Apply the move
        p[best_i], p[best_j] = p[best_j], p[best_i]
        current_cost += best_delta
        
        # Update best
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            
        # Update Tabu List
        tabu_list[(best_i, best_j)] = iteration + tabu_tenure
        
        # Optional: Aspiration criterion - if move improves best, allow it even if tabu
        # We already skip tabu moves in the selection loop, so this is handled by
        # potentially relaxing the tabu check if we implement aspiration.
        # For simplicity, we stick to strict tabu.
        
        iteration += 1
        
        # Periodically clean up tabu list to avoid memory bloat
        if iteration % 100 == 0:
            # Remove expired tabu entries
            keys_to_remove = [k for k, v in tabu_list.items() if v <= iteration]
            for k in keys_to_remove:
                del tabu_list[k]

    return best_p.tolist()
