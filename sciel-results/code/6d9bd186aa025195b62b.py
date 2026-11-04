"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linear_sum_assignment
import time

def _qap_cost(p, a, b):
    """Calculate the exact cost of permutation p."""
    # p is a list or array of locations for each facility
    # Cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # Vectorized:
    # b_perm = b[np.ix_(p, p)]
    # return np.sum(a * b_perm)
    b_perm = b[np.ix_(p, p)]
    return np.sum(a * b_perm)

def _swap_delta(p, a, b, u, v):
    """
    Calculate the change in cost if we swap locations u and v in permutation p.
    p[i] is the location of facility i.
    So p[x] = u and p[y] = v.
    We need to find x and y.
    """
    # Find facilities x and y at locations u and v
    # p is an array where p[i] = location.
    # We need inverse mapping: inv_p[loc] = fac
    # This is O(n) to compute, but we can pass it or compute it.
    # For efficiency in the loop, we should maintain inv_p.
    pass

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # 1. Initialization using Hungarian Algorithm
    # Surrogate cost matrix C where C[i,j] = sum_k A[i,k] * B[j,k]
    # This approximates the interaction of facility i at location j with all other facilities.
    # C = A @ B.T
    C = a @ b.T
    # Solve Linear Assignment Problem
    # We want to minimize sum C[i, p[i]]
    row_ind, col_ind = linear_sum_assignment(C)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    
    # Calculate initial cost
    current_cost = _qap_cost(p, a, b)
    best_p = p.copy()
    best_cost = current_cost
    
    # Maintain inverse permutation for fast lookup
    # inv_p[loc] = fac
    inv_p = np.empty(n, dtype=np.int64)
    inv_p[p] = np.arange(n)
    
    def get_swap_delta(p, inv_p, a, b, u, v):
        """
        Calculate delta cost for swapping locations u and v.
        x = inv_p[u], y = inv_p[v]
        """
        x = inv_p[u]
        y = inv_p[v]
        
        # Terms involving only x and y
        # (x,x): A[xx] * (B[vv] - B[uu])
        # (y,y): A[yy] * (B[uu] - B[vv])
        # (x,y): A[xy] * (B[vu] - B[uv])
        # (y,x): A[yx] * (B[uv] - B[vu])
        
        A_xx = a[x, x]
        A_yy = a[y, y]
        A_xy = a[x, y]
        A_yx = a[y, x]
        
        B_uu = b[u, u]
        B_vv = b[v, v]
        B_uv = b[u, v]
        B_vu = b[v, u]
        
        delta_diag = A_xx * (B_vv - B_uu) + A_yy * (B_uu - B_vv)
        delta_off_diag = A_xy * (B_vu - B_uv) + A_yx * (B_uv - B_vu)
        
        # Terms involving k != x, y
        # Sum_{k != x,y} (A[x,k] - A[y,k]) * (B[v, p[k]] - B[u, p[k]])
        # + Sum_{k != x,y} (A[k,x] - A[k,y]) * (B[p[k], v] - B[p[k], u])
        
        # Vectorized calculation
        # We need p[k] for all k != x, y
        # Let's just compute for all k and subtract x,y terms later, or mask.
        # Masking is easier.
        
        # Create a mask for k != x, y
        # But we can just compute the full sum and subtract the x,y parts if they are included.
        # The formula derived earlier assumed k != x,y.
        # If we sum over all k, we include k=x and k=y.
        
        # Let's compute:
        # S1 = sum_k (A[x,k] - A[y,k]) * (B[v, p[k]] - B[u, p[k]])
        # S2 = sum_k (A[k,x] - A[k,y]) * (B[p[k], v] - B[p[k], u])
        
        # We can compute these using dot products.
        # Let diff_A_row = a[x, :] - a[y, :]
        # Let diff_A_col = a[:, x] - a[:, y]
        
        # For S1:
        # We need B[v, p[k]] - B[u, p[k]].
        # This is (b[v, :] - b[u, :])[p[k]]
        # Let diff_B_row = b[v, :] - b[u, :]
        # Then S1 = sum_k diff_A_row[k] * diff_B_row[p[k]]
        
        # For S2:
        # We need B[p[k], v] - B[p[k], u].
        # This is (b[:, v] - b[:, u])[p[k]]
        # Let diff_B_col = b[:, v] - b[:, u]
        # Then S2 = sum_k diff_A_col[k] * diff_B_col[p[k]]
        
        diff_A_row = a[x, :] - a[y, :]
        diff_B_row = b[v, :] - b[u, :]
        S1 = np.dot(diff_A_row, diff_B_row[p])
        
        diff_A_col = a[:, x] - a[:, y]
        diff_B_col = b[:, v] - b[:, u]
        S2 = np.dot(diff_A_col, diff_B_col[p])
        
        # The sums S1 and S2 include k=x and k=y.
        # We must subtract the terms for k=x and k=y because they were handled in delta_diag/off_diag?
        # Let's check.
        # In the full cost change, the terms for k=x and k=y in the "interaction with others" part
        # correspond to interactions between x/y and x/y themselves.
        # Specifically:
        # For k=x in S1: (A[xx]-A[yx]) * (B[v, p[x]] - B[u, p[x]]) = (A[xx]-A[yx]) * (B[v, u] - B[u, u])
        # For k=y in S1: (A[xy]-A[yy]) * (B[v, p[y]] - B[u, p[y]]) = (A[xy]-A[yy]) * (B[v, v] - B[u, v])
        
        # Similarly for S2.
        # It is safer to just compute the full delta using the vectorized sums S1+S2 and the diagonal terms,
        # BUT we must ensure we don't double count or miss the self-terms.
        
        # Actually, the formula:
        # Delta = Delta_self + S1 + S2
        # Where Delta_self handles (x,x), (y,y), (x,y), (y,x).
        # S1 and S2 handle interactions with k != x,y.
        # If S1 and S2 are computed over ALL k, they include k=x and k=y.
        # So we must subtract the k=x and k=y contributions from S1 and S2.
        
        # k=x contribution to S1:
        # term_kx_S1 = (a[x,x] - a[y,x]) * (b[v, u] - b[u, u])
        # k=y contribution to S1:
        # term_ky_S1 = (a[x,y] - a[y,y]) * (b[v, v] - b[u, v])
        
        # k=x contribution to S2:
        # term_kx_S2 = (a[x,x] - a[x,y]) * (b[u, v] - b[u, u])  # Note: diff_A_col[x] = a[x,x]-a[x,y]
        # Wait, diff_A_col[k] = a[k,x] - a[k,y].
        # For k=x: a[x,x] - a[x,y].
        # diff_B_col[p[x]] = diff_B_col[u] = b[u,v] - b[u,u].
        # So term_kx_S2 = (a[x,x] - a[x,y]) * (b[u,v] - b[u,u])
        
        # k=y contribution to S2:
        # term_ky_S2 = (a[y,x] - a[y,y]) * (b[v,v] - b[v,u])
        
        # Let's subtract these.
        
        term_kx_S1 = (a[x, x] - a[y, x]) * (b[v, u] - b[u, u])
        term_ky_S1 = (a[x, y] - a[y, y]) * (b[v, v] - b[u, v])
        
        term_kx_S2 = (a[x, x] - a[x, y]) * (b[u, v] - b[u, u])
        term_ky_S2 = (a[y, x] - a[y, y]) * (b[v, v] - b[v, u])
        
        S1_adj = S1 - term_kx_S1 - term_ky_S1
        S2_adj = S2 - term_kx_S2 - term_ky_S2
        
        total_delta = delta_diag + delta_off_diag + S1_adj + S2_adj
        return total_delta

    # ILS Loop
    while time.time() < deadline:
        # Local Search: Best Improvement
        improved = True
        while improved:
            improved = False
            best_delta = 0
            best_u = -1
            best_v = -1
            
            # Check all pairs u, v
            # Optimization: Only check u < v
            for u in range(n):
                for v in range(u + 1, n):
                    delta = get_swap_delta(p, inv_p, a, b, u, v)
                    if delta < best_delta:
                        best_delta = delta
                        best_u = u
                        best_v = v
            
            if best_u != -1:
                # Apply swap
                x = inv_p[best_u]
                y = inv_p[best_v]
                
                # Update p
                p[x] = best_v
                p[y] = best_u
                
                # Update inv_p
                inv_p[best_u] = y
                inv_p[best_v] = x
                
                # Update cost
                current_cost += best_delta
                improved = True
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
                    
        # Perturbation
        # Randomly swap 5-10 pairs
        num_swaps = np.random.randint(5, 11)
        for _ in range(num_swaps):
            u, v = np.random.choice(n, 2, replace=False)
            x = inv_p[u]
            y = inv_p[v]
            
            p[x] = v
            p[y] = u
            inv_p[u] = y
            inv_p[v] = x
            
            # Update cost (could use delta, but for perturbation we just recalculate or use delta)
            # Using delta is faster
            delta = get_swap_delta(p, inv_p, a, b, u, v)
            # Note: get_swap_delta assumes current state.
            # We just swapped, so the delta calculated is for the state BEFORE swap?
            # No, get_swap_delta calculates delta for swapping u,v in the CURRENT p.
            # So if we call it BEFORE swapping, we get the delta.
            # I applied the swap above. I should calculate delta BEFORE swapping.
            
            # Let's refactor: Calculate delta, then swap.
            # But I already swapped in the loop above.
            # Let's just recalculate cost for perturbation to be safe, or fix the order.
            # For 5-10 swaps, recalculation is O(n^2) which is fine.
            pass
        
        # Recalculate cost after perturbation to be accurate
        current_cost = _qap_cost(p, a, b)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            
    return best_p.tolist()

# EVOLVE-BLOCK-END
