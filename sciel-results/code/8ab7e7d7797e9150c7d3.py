"""
QAP Solver: Eigenvector Ordering + Greedy Refinement + Local Search
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve(a, b):
    n = len(a)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    
    # 1. Compute "spine" coordinates using dominant eigenvectors
    # This projects the high-dimensional flow/distance matrices onto a 1D axis
    # that preserves the most significant structure (variance).
    
    # For symmetric matrices, the dominant eigenvector (largest eigenvalue)
    # captures the main "direction" of the matrix.
    
    def get_dominant_eigenvector(M):
        # M is assumed symmetric
        # Use power iteration or numpy's eigh for speed
        # For n=100, eigh is fast enough
        vals, vecs = np.linalg.eigh(M)
        # The eigenvalues are sorted in ascending order.
        # The dominant one is the last one.
        # However, for QAP, we often care about the structure that minimizes cost.
        # Let's check if we should use the largest or smallest magnitude?
        # Usually, the "layout" problem relates to the largest eigenvalue for positive matrices.
        # If the matrix has negative entries, it might be the smallest.
        # Let's take the one with the largest absolute value?
        # Or just the largest?
        # Let's try the largest eigenvalue first.
        idx = np.argmax(np.abs(vals))
        v = vecs[:, idx]
        # Ensure consistent sign
        if v[np.argmax(np.abs(v))] < 0:
            v = -v
        return v
    
    try:
        v_a = get_dominant_eigenvector(a)
        v_b = get_dominant_eigenvector(b)
    except Exception:
        # Fallback to row sums if eig fails
        v_a = a.sum(axis=1)
        v_b = b.sum(axis=1)

    # Project facilities and locations onto these 1D axes
    # Facility i has coordinate v_a[i]
    # Location j has coordinate v_b[j]
    
    # The "Slate DP" or Ordering Strategy:
    # We assume that the optimal assignment preserves the relative order of coordinates.
    # i.e., if facility i has a smaller v_a coordinate than facility k,
    # it should be assigned to a location with a smaller v_b coordinate.
    
    # This reduces the search space to checking if the sorted orders match.
    # Specifically, we sort facilities by v_a and locations by v_b.
    # The "canonical" assignment is mapping sorted_facilities[k] to sorted_locations[k].
    
    order_a = np.argsort(v_a, kind='stable')
    order_b = np.argsort(v_b, kind='stable')
    
    # Candidate 1: The "Perfect Order" Assignment
    # Facility at rank k in A gets Location at rank k in B.
    p_order = np.empty(n, dtype=np.int64)
    p_order[order_a] = order_b
    # p_order[i] is the location assigned to facility i.
    # If i = order_a[k], then p_order[i] = order_b[k].
    
    # Candidate 2: The "Reversed Order" Assignment
    # Sometimes the best layout is anti-correlated.
    p_rev = np.empty(n, dtype=np.int64)
    p_rev[order_a] = order_b[::-1]
    
    # Candidate 3: Hungarian on projected costs (refinement of the spine)
    # We can create a cost matrix based on the 1D distances.
    # Cost[i, j] = (v_a[i] - v_b[j])^2
    # This is a linear assignment problem that respects the 1D structure but allows "crossings" if beneficial in the 1D projection.
    cost_proj = (v_a[:, None] - v_b[None, :])**2
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(cost_proj)
    p_proj = np.empty(n, dtype=np.int64)
    p_proj[row_ind] = col_ind
    
    # Candidate 4: Standard Greedy (Row Sums)
    # High flow to low distance sum (central)
    a_sums = a.sum(axis=1)
    b_sums = b.sum(axis=1)
    order_a_sum = np.argsort(-a_sums, kind='stable') # Descending flow
    order_b_sum = np.argsort(b_sums, kind='stable')   # Ascending distance (central)
    p_greedy = np.empty(n, dtype=np.int64)
    p_greedy[order_a_sum] = order_b_sum
    
    # Evaluate all candidates
    candidates = [p_order, p_rev, p_proj, p_greedy]
    
    def calc_cost(p):
        return int(np.sum(a * b[np.ix_(p, p)]))
    
    best_p = None
    best_cost = float('inf')
    for cand in candidates:
        c = calc_cost(cand)
        if c < best_cost:
            best_cost = c
            best_p = cand.copy()
    
    p = best_p.copy()
    current_cost = best_cost
    
    # 2. Local Search Refinement
    # Even with the "spine" constraint, the optimal solution might deviate slightly.
    # We use efficient 2-swap local search to polish the result.
    
    def get_delta(i, j, p):
        u = p[i]
        v = p[j]
        
        # Delta for swapping i and j
        # Terms involving i or j change.
        # Old: a[i,j]b[u,v] + a[j,i]b[v,u]
        #      + sum_{k!=i,j} a[i,k]b[u,p[k]] + a[k,i]b[p[k],u]
        #      + sum_{k!=i,j} a[j,k]b[v,p[k]] + a[k,j]b[p[k],v]
        # New: a[i,j]b[v,u] + a[j,i]b[u,v]
        #      + sum_{k!=i,j} a[i,k]b[v,p[k]] + a[k,i]b[p[k],v]
        #      + sum_{k!=i,j} a[j,k]b[u,p[k]] + a[k,j]b[p[k],u]
        
        # Delta = (a[i,j]b[v,u] + a[j,i]b[u,v]) - (a[i,j]b[u,v] + a[j,i]b[v,u])
        #       + sum_{k!=i,j} [ a[i,k](b[v,p[k]]-b[u,p[k]]) + a[k,i](b[p[k],v]-b[p[k],u])
        #                       + a[j,k](b[u,p[k]]-b[v,p[k]]) + a[k,j](b[p[k],u]-b[p[k],v]) ]
        
        # Let D_b[k] = b[v, p[k]] - b[u, p[k]]
        # Note: b[p[k], v] - b[p[k], u] is not the same as D_b[k] unless b is symmetric.
        # QAP matrices are usually symmetric, but let's be safe.
        # Actually, for wil100, b is symmetric. a is symmetric.
        # If a and b are symmetric:
        # a[i,k] = a[k,i] and b[v,p[k]] = b[p[k],v].
        # So the term for k is:
        # a[i,k] (b[v,p[k]] - b[u,p[k]]) + a[i,k] (b[p[k],v] - b[p[k],u])
        # = 2 * a[i,k] * (b[v,p[k]] - b[u,p[k]])
        
        # Let's assume symmetry for speed, as is standard for QAPLIB.
        # If not symmetric, this delta is incorrect.
        # Given the task description "wil100", it is symmetric.
        
        diff_b = b[v, p] - b[u, p]
        
        # Sum over all k
        sum_all = np.dot(a[i] - a[j], diff_b)
        
        # Corrections for k=i and k=j
        # k=i term in sum_all: (a[i,i]-a[j,i]) * (b[v,p[i]] - b[u,p[i]]) = (a[i,i]-a[j,i]) * (b[v,u] - b[u,u])
        term_i = (a[i,i] - a[j,i]) * (b[v,u] - b[u,u])
        
        # k=j term in sum_all: (a[i,j]-a[j,j]) * (b[v,p[j]] - b[u,p[j]]) = (a[i,j]-a[j,j]) * (b[v,v] - b[u,v])
        term_j = (a[i,j] - a[j,j]) * (b[v,v] - b[u,v])
        
        # The diagonal swap term:
        # Old: a[i,j]b[u,v] + a[j,i]b[v,u]
        # New: a[i,j]b[v,u] + a[j,i]b[u,v]
        # Since a[i,j]=a[j,i] and b[u,v]=b[v,u], these are identical!
        # So the diagonal swap term contributes 0 to the delta?
        # Wait.
        # Old cost part: a[i,j]*b[p[i],p[j]] + a[j,i]*b[p[j],p[i]]
        # New cost part: a[i,j]*b[p[j],p[i]] + a[j,i]*b[p[i],p[j]]
        # If a and b are symmetric, a[i,j]=a[j,i] and b[u,v]=b[v,u].
        # Old: a[i,j]*b[u,v] + a[i,j]*b[u,v] = 2*a[i,j]*b[u,v]
        # New: a[i,j]*b[v,u] + a[i,j]*b[v,u] = 2*a[i,j]*b[v,u]
        # Since b[u,v] == b[v,u], the delta is 0.
        
        # So delta = 2 * (sum_all - term_i - term_j)
        # Let's verify.
        # sum_all = sum_k (a[i,k]-a[j,k]) * (b[v,p[k]] - b[u,p[k]])
        # = sum_{k!=i,j} (a[i,k]-a[j,k]) * (b[v,p[k]] - b[u,p[k]]) + term_i_val + term_j_val
        # where term_i_val is the value at k=i and term_j_val at k=j.
        
        # The actual change for k!=i,j is 2 * (a[i,k]-a[j,k]) * (b[v,p[k]] - b[u,p[k]])?
        # Let's re-derive.
        # Change for pair (i,k):
        # Old: a[i,k]b[u,p[k]] + a[k,i]b[p[k],u] = 2*a[i,k]*b[u,p[k]]
        # New: a[i,k]b[v,p[k]] + a[k,i]b[p[k],v] = 2*a[i,k]*b[v,p[k]]
        # Delta_ik = 2*a[i,k]*(b[v,p[k]] - b[u,p[k]])
        
        # Change for pair (j,k):
        # Old: a[j,k]b[v,p[k]] + a[k,j]b[p[k],v] = 2*a[j,k]*b[v,p[k]]
        # New: a[j,k]b[u,p[k]] + a[k,j]b[p[k],u] = 2*a[j,k]*b[u,p[k]]
        # Delta_jk = 2*a[j,k]*(b[u,p[k]] - b[v,p[k]]) = -2*a[j,k]*(b[v,p[k]] - b[u,p[k]])
        
        # Total Delta for k!=i,j:
        # 2 * (a[i,k] - a[j,k]) * (b[v,p[k]] - b[u,p[k]])
        
        # So yes, Delta = 2 * sum_{k!=i,j} (a[i,k]-a[j,k]) * (b[v,p[k]] - b[u,p[k]])
        
        # sum_all includes k=i and k=j.
        # So sum_{k!=i,j} ... = sum_all - term_i - term_j
        
        delta = 2 * (sum_all - term_i - term_j)
        return int(delta)

    def local_search(p, current_cost, max_iters=50):
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            for i in range(n):
                for j in range(i + 1, n):
                    delta = get_delta(i, j, p)
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        current_cost += delta
                        improved = True
        return p, current_cost

    start_time = time.time()
    timeout = 115.0
    
    # Perform local search
    p, current_cost = local_search(p, current_cost, max_iters=100)
    
    if current_cost < best_cost:
        best_cost = current_cost
        best_p = p.copy()
        
    # If time remains, try ILS (Iterated Local Search)
    if time.time() - start_time < timeout - 10:
        rng = np.random.RandomState(42)
        
        # Define perturbation strength
        # We want to stay close to the "spine" but allow deviations.
        # A simple random swap perturbation is fine.
        
        while time.time() - start_time < timeout - 5:
            # Perturb
            num_swaps = 5
            for _ in range(num_swaps):
                i = rng.randint(n)
                j = rng.randint(n)
                if i == j:
                    continue
                delta = get_delta(i, j, p)
                p[i], p[j] = p[j], p[i]
                current_cost += delta
            
            # Local Search
            p, current_cost = local_search(p, current_cost, max_iters=20)
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
            else:
                # Revert to best
                p = best_p.copy()
                current_cost = best_cost
                
    return best_p.tolist()
# EVOLVE-BLOCK-END
