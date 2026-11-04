import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    def compute_cost(p):
        return int(np.sum(a * b[np.ix_(p, p)]))

    def fast_swap_delta(p, u, v):
        pu = p[u]
        pv = p[v]
        
        # d_uv[k] = b[pv, k] - b[pu, k]
        d_uv = b[pv, :] - b[pu, :]
        # d_vu[k] = b[k, pv] - b[k, pu]
        d_vu = b[:, pv] - b[:, pu]
        
        # s1 = sum_k (a[u,k] - a[v,k]) * d_uv[p[k]]
        # s2 = sum_k (a[k,u] - a[k,v]) * d_vu[p[k]]
        
        s1 = np.dot(a[u, :] - a[v, :], d_uv[p])
        s2 = np.dot(a[:, u] - a[:, v], d_vu[p])
        
        # Corrections for k=u and k=v
        s1 -= (a[u, u] - a[v, u]) * d_uv[pu]
        s1 -= (a[u, v] - a[v, v]) * d_uv[pv]
        
        s2 -= (a[u, u] - a[u, v]) * d_vu[pu]
        s2 -= (a[v, u] - a[v, v]) * d_vu[pv]
        
        # Cross and diagonal terms
        # a[u,u] term: a[u,u] * (b[pv,pv] - b[pu,pu])
        # a[v,v] term: a[v,v] * (b[pu,pu] - b[pv,pv])
        # Sum: (a[u,u] - a[v,v]) * (b[pv,pv] - b[pu,pu])
        diag_term = (a[u, u] - a[v, v]) * (b[pv, pv] - b[pu, pu])
        
        # a[u,v] term: a[u,v] * (b[pv,pu] - b[pu,pv])
        # a[v,u] term: a[v,u] * (b[pu,pv] - b[pv,pu])
        # Sum: (a[u,v] - a[v,u]) * (b[pv,pu] - b[pu,pv])
        cross_term = (a[u, v] - a[v, u]) * (b[pv, pu] - b[pu, pv])
        
        return int(s1 + s2 + diag_term + cross_term)

    def local_search(p, cost, max_passes=50):
        """
        Vectorized Best-Improvement 2-opt Local Search.
        For each i, finds the best j to swap with.
        """
        p = p.copy()
        improved = True
        passes = 0
        
        while improved and passes < max_passes:
            improved = False
            passes += 1
            
            for i in range(n - 1):
                if time.time() > deadline:
                    return p, cost
                
                # Find best j > i to swap with i
                best_delta = 0
                best_j = -1
                
                # We can vectorize the delta calculation for all j > i
                # But the delta formula is complex. 
                # Let's stick to a loop over j for now, but optimize the inner loop?
                # Or use the vectorized approach from inspiration?
                # The inspiration's vectorized approach was:
                # diff_A = A_sym[i] - A_sym[js]
                # diff_D = D[js] - D[i]
                # S = np.sum(diff_A * diff_D, axis=1)
                # This calculates sum_k (a[i,k]+a[k,i] - a[j,k]-a[k,j]) * (b[p[i],p[k]] - b[p[j],p[k]])?
                # No, D is b[np.ix_(p,p)]. D[i, k] = b[p[i], p[k]].
                # So diff_D[k] = b[p[i], p[k]] - b[p[j], p[k]].
                # The formula for delta is roughly 2 * sum (a_sym[i,k] - a_sym[j,k]) * (b[p[i],p[k]] - b[p[j],p[k]])?
                # Let's verify.
                # Delta = sum_{k != i,j} (a[i,k]-a[j,k]) (b[pj,pk] - b[pi,pk]) + ...
                # Note b[pj,pk] - b[pi,pk] = -(b[pi,pk] - b[pj,pk]).
                # So sum part is - sum (a[i,k]-a[j,k]) (b[pi,pk] - b[pj,pk]).
                # If A is symmetric, a[i,k]-a[j,k] = A[i,k]-A[j,k].
                # So sum part is - sum (A[i,k]-A[j,k]) (D[i,k] - D[j,k]).
                # = - (sum A[i,k]D[i,k] - sum A[i,k]D[j,k] - sum A[j,k]D[i,k] + sum A[j,k]D[j,k])
                # This is getting messy.
                
                # Let's use the explicit O(n) delta in a loop, but try to make it fast.
                # For n=256, inner loop is 256. Outer is 256. Total 65k deltas per pass.
                # Each delta is ~10 numpy ops. 650k ops. Very fast.
                
                for j in range(i + 1, n):
                    delta = fast_swap_delta(p, i, j)
                    if delta < best_delta:
                        best_delta = delta
                        best_j = j
                
                if best_j != -1 and best_delta < 0:
                    p[i], p[best_j] = p[best_j], p[i]
                    cost += best_delta
                    improved = True
                    # Restart from i=0 to ensure best improvement global
                    break
            
        return p, cost

    def hungarian_init():
        """
        Initialize using Hungarian algorithm on surrogate cost.
        Surrogate: C[i,j] = sum_k a[i,k]*b[j,k] + sum_k a[k,i]*b[k,j]
        This approximates the cost contribution of facility i at location j
        assuming other facilities are optimally placed relative to this term.
        """
        # C1[i,j] = sum_k a[i,k] b[j,k] = (a @ b)[i,j]
        # C2[i,j] = sum_k a[k,i] b[k,j] = (a.T @ b)[i,j] ? No.
        # a[k,i] is column i of a. b[k,j] is column j of b.
        # sum_k a[k,i] b[k,j] = (a.T @ b)[i,j] is wrong.
        # (a.T @ b)[i,j] = sum_k a.T[i,k] b[k,j] = sum_k a[k,i] b[k,j]. Correct.
        
        C = (a @ b) + (a.T @ b)
        row_ind, col_ind = linear_sum_assignment(C)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        return p

    def hungarian_reopt(p):
        """
        Re-optimize using Hungarian algorithm on the current assignment's linearized cost.
        C[i,j] = cost of putting i at j, given others are fixed at p.
        """
        # Cost of i at j:
        # sum_{k != i} a[i,k] b[j, p[k]] + sum_{k != i} a[k,i] b[p[k], j]
        
        # Let's compute M1[i,j] = sum_k a[i,k] b[j, p[k]]
        # B_p = b[np.ix_(np.arange(n), p)]  # B_p[j,k] = b[j, p[k]]
        B_p = b[:, p] # shape (n, n), B_p[j,k] = b[j, p[k]]
        # M1 = a @ B_p.T ?
        # (a @ B_p.T)[i,j] = sum_k a[i,k] B_p.T[k,j] = sum_k a[i,k] B_p[j,k] = sum_k a[i,k] b[j, p[k]]. Correct.
        M1 = a @ B_p.T
        
        # Let's compute M2[i,j] = sum_k a[k,i] b[p[k], j]
        # Bp = b[np.ix_(p, np.arange(n))] # Bp[k,j] = b[p[k], j]
        Bp = b[p, :] # shape (n, n), Bp[k,j] = b[p[k], j]
        # M2 = a.T @ Bp ?
        # (a.T @ Bp)[i,j] = sum_k a.T[i,k] Bp[k,j] = sum_k a[k,i] b[p[k], j]. Correct.
        M2 = a.T @ Bp
        
        C = M1 + M2
        
        # Exclude diagonal? No, LAP handles assignment.
        # The term where k=i in the sums above is included in M1 and M2.
        # M1[i,i] includes a[i,i] b[i, p[i]].
        # M2[i,i] includes a[i,i] b[p[i], i].
        # The actual cost term for (i,i) is a[i,i] b[p[i], p[i]].
        # The linearization approximates the interaction.
        
        row_ind, col_ind = linear_sum_assignment(C)
        new_p = np.empty(n, dtype=np.int64)
        new_p[row_ind] = col_ind
        return new_p

    # 1. Initial Solutions
    p1 = hungarian_init()
    cost1 = compute_cost(p1)
    p1, cost1 = local_search(p1, cost1)
    
    # Rank based init
    a_row_sums = a.sum(axis=1)
    b_col_sums = b.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_col_sums, kind="stable")
    p2 = np.empty(n, dtype=np.int64)
    p2[heavy] = central
    cost2 = compute_cost(p2)
    p2, cost2 = local_search(p2, cost2)
    
    best_p = p1.copy()
    best_cost = cost1
    if cost2 < best_cost:
        best_p = p2.copy()
        best_cost = cost2
        
    # 2. Iterated Local Search with Hungarian Re-optimization
    rng = np.random.default_rng(42)
    no_improve_count = 0
    max_no_improve = 10
    
    while time.time() < deadline - 5:
        # Perturb
        p = best_p.copy()
        n_swaps = rng.integers(2, 6)
        if no_improve_count > 5:
            n_swaps += rng.integers(2, 5)
            
        for _ in range(n_swaps):
            i, j = rng.choice(n, 2, replace=False)
            p[i], p[j] = p[j], p[i]
            
        p, cost = local_search(p, compute_cost(p))
        
        if cost < best_cost:
            best_p = p.copy()
            best_cost = cost
            no_improve_count = 0
        else:
            no_improve_count += 1
            
            # If stuck, try Hungarian Re-optimization
            if no_improve_count % 3 == 0:
                if time.time() > deadline - 10:
                    break
                h_p = hungarian_reopt(best_p)
                h_cost = compute_cost(h_p)
                h_p, h_cost = local_search(h_p, h_cost)
                
                if h_cost < best_cost:
                    best_p = h_p.copy()
                    best_cost = h_cost
                    no_improve_count = 0
                else:
                    # If Hungarian also fails, increase perturbation
                    no_improve_count += 2

    return best_p.tolist()
