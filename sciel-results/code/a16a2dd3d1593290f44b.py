"""
Strategy: Frank-Wolfe Relaxation + Hungarian Discretization + Vectorized Local Search
This approach treats the QAP as a continuous optimization problem over the Birkhoff polytope
(space of doubly stochastic matrices) using the Frank-Wolfe (Conditional Gradient) method.
It then discretizes the solution using the Hungarian algorithm to get a permutation,
and refines it with a fast vectorized local search.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    def calc_cost(p):
        """Calculate exact cost for permutation p."""
        # p is a list/array of length n. Facility i is at location p[i].
        # Cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
        # This is trace(A * B_P) where B_P is permuted B.
        # B_P[i,j] = B[p[i], p[j]]
        # Using numpy indexing: b[np.ix_(p, p)] gives matrix M where M[i,j] = b[p[i], p[j]]
        b_perm = b[np.ix_(p, p)]
        return int(np.sum(a * b_perm))

    def get_delta(p, i, j):
        """
        Calculate delta cost for swapping facilities i and j in permutation p.
        p[i] -> u, p[j] -> v.
        Returns the change in cost if we swap p[i] and p[j].
        """
        u = p[i]
        v = p[j]
        
        # delta = 2 * sum_{k != i, j} (a[i,k] - a[j,k]) * (b[v, p[k]] - b[u, p[k]])
        # + terms for k=i and k=j handled separately or included in sum with care.
        
        # Vectorized calculation
        # diff_a = a[i] - a[j] (vector of length n)
        # For each k, term is (a[i,k] - a[j,k]) * (b[v, p[k]] - b[u, p[k]])
        
        # b_v_p = b[v, p] -> array where index k is b[v, p[k]]
        # b_u_p = b[u, p] -> array where index k is b[u, p[k]]
        b_v_p = b[v, p]
        b_u_p = b[u, p]
        diff_b = b_v_p - b_u_p
        
        diff_a = a[i] - a[j]
        
        # Sum over all k
        sum_all = np.sum(diff_a * diff_b)
        
        # We need to subtract the contributions from k=i and k=j because the formula 
        # 2*sum_{k!=i,j} assumes those are excluded, but our sum_all includes them.
        # Actually, the standard swap delta formula is:
        # Delta = 2 * sum_{k != i, j} (a[i,k] - a[j,k]) * (b[p[j], p[k]] - b[p[i], p[k]])
        #        + (a[i,i] - a[j,i]) * (b[p[j], p[i]] - b[p[i], p[i]])  ... wait, let's derive.
        
        # Let C(p) = sum_{i,j} a[i,j] b[p[i], p[j]]
        # Swap i, j. Let p' be new perm. p'[i]=v, p'[j]=u, p'[k]=p[k] for k!=i,j.
        # C(p') - C(p) = sum_{x,y} a[x,y] (b[p'[x], p'[y]] - b[p[x], p[y]])
        # Non-zero terms are only where x or y is i or j.
        # Terms:
        # 1. x=i, y=i: a[i,i] (b[v,v] - b[u,u])
        # 2. x=i, y=j: a[i,j] (b[v,u] - b[u,v])
        # 3. x=j, y=i: a[j,i] (b[u,v] - b[v,u])
        # 4. x=j, y=j: a[j,j] (b[u,u] - b[v,v])
        # 5. x=i, y=k (k!=i,j): a[i,k] (b[v, p[k]] - b[u, p[k]])
        # 6. x=j, y=k (k!=i,j): a[j,k] (b[u, p[k]] - b[v, p[k]])
        # 7. x=k, y=i (k!=i,j): a[k,i] (b[p[k], v] - b[p[k], u])
        # 8. x=k, y=j (k!=i,j): a[k,j] (b[p[k], u] - b[p[k], v])
        
        # Since A and B are typically symmetric in QAP instances (wil100 is symmetric),
        # a[i,j]=a[j,i] and b[u,v]=b[v,u].
        # Let's assume symmetry for speed, but the code should be robust.
        # If symmetric:
        # Term 2 + Term 3 = (a[i,j] + a[j,i]) (b[v,u] - b[u,v]) = 0 if symmetric.
        # Term 1 + Term 4 = (a[i,i] - a[j,j]) (b[v,v] - b[u,u])
        # Terms 5+8: a[i,k](b[v,p[k]] - b[u,p[k]]) + a[k,j](b[p[k],u] - b[p[k],v])
        #   = (a[i,k] - a[k,j]) (b[v,p[k]] - b[u,p[k]])  (if b symmetric)
        # Terms 6+7: a[j,k](b[u,p[k]] - b[v,p[k]]) + a[k,i](b[p[k],v] - b[p[k],u])
        #   = (a[j,k] - a[k,i]) (b[u,p[k]] - b[v,p[k]])
        #   = (a[i,k] - a[j,k]) (b[v,p[k]] - b[u,p[k]]) (using symmetry a[k,i]=a[i,k], a[k,j]=a[j,k])
        
        # So if symmetric:
        # Delta = (a[i,i] - a[j,j]) (b[v,v] - b[u,u]) 
        #         + 2 * sum_{k != i, j} (a[i,k] - a[j,k]) (b[v, p[k]] - b[u, p[k]])
        
        # Let's compute the sum over all k first:
        # S = sum_{k} (a[i,k] - a[j,k]) (b[v, p[k]] - b[u, p[k]])
        # S includes k=i and k=j.
        # Term k=i: (a[i,i] - a[j,i]) (b[v, u] - b[u, u])
        # Term k=j: (a[i,j] - a[j,j]) (b[v, v] - b[u, v])
        
        # If symmetric, a[j,i]=a[i,j], b[v,u]=b[u,v].
        # Term k=i: (a[i,i] - a[i,j]) (b[v, u] - b[u, u])
        # Term k=j: (a[i,j] - a[j,j]) (b[v, v] - b[u, v])
        
        # The formula derived above for symmetric case:
        # Delta = (a[i,i] - a[j,j]) (b[v,v] - b[u,u]) + 2 * (S - Term_k_i - Term_k_j)
        
        # Let's verify with non-symmetric case just to be safe? 
        # The task says "a and b are this task's QAPLIB matrices". QAPLIB instances are symmetric.
        # wil100 is symmetric.
        
        term_k_i = (a[i, i] - a[j, i]) * (b[v, u] - b[u, u])
        term_k_j = (a[i, j] - a[j, j]) * (b[v, v] - b[u, v])
        
        sum_all = np.dot(a[i] - a[j], b[v, p] - b[u, p])
        
        delta = (a[i, i] - a[j, j]) * (b[v, v] - b[u, u]) + 2 * (sum_all - term_k_i - term_k_j)
        
        return delta

    def local_search_vectorized(p, current_cost):
        """
        Best Improvement Local Search.
        For each i, find best j to swap with. Apply best swap. Repeat until no improvement.
        """
        p = p.copy()
        improved = True
        while improved:
            if time.time() > deadline:
                break
            best_delta = 0
            best_i = -1
            best_j = -1
            
            for i in range(n):
                if time.time() > deadline:
                    break
                u = p[i]
                a_i = a[i]
                b_u_p = b[u, p]
                
                # We want to find j > i that minimizes delta (most negative)
                # To avoid checking j < i, we can just check all j != i, 
                # but checking all pairs is O(n^3) per pass. 
                # n=100, n^3 = 10^6. Vectorized dot products are fast.
                # Let's try to vectorize the inner loop for fixed i.
                
                # For fixed i, we want min over j of delta(i, j).
                # delta(i, j) = 2 * dot(a[i] - a[j], b[p[j], p] - b[u, p]) - corrections.
                # This is hard to fully vectorize over j because b[p[j], p] changes with j.
                # However, we can precompute b[:, p] ? No, p is a permutation.
                # b[p[j], p] is the row of b indexed by p[j], evaluated at indices p.
                
                # Let's stick to a loop over j for now, but optimize the inner calculation.
                # n=100, loop is small.
                
                for j in range(i + 1, n):
                    v = p[j]
                    # delta calculation
                    # We can reuse some calculations?
                    # b_v_p = b[v, p]
                    # diff_b = b_v_p - b_u_p
                    # diff_a = a_i - a[j]
                    # sum_all = np.dot(diff_a, diff_b)
                    # ...
                    
                    # Optimization: Precompute b_v_p? No, v changes.
                    # But we can precompute b_u_p outside j loop (done).
                    
                    diff_b = b[v, p] - b_u_p
                    diff_a = a_i - a[j]
                    sum_all = np.dot(diff_a, diff_b)
                    
                    term_k_i = (a[i, i] - a[j, i]) * (b[v, u] - b[u, u])
                    term_k_j = (a[i, j] - a[j, j]) * (b[v, v] - b[u, v])
                    
                    delta = (a[i, i] - a[j, j]) * (b[v, v] - b[u, u]) + 2 * (sum_all - term_k_i - term_k_j)
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_i = i
                        best_j = j
            
            if best_i != -1:
                i, j = best_i, best_j
                p[i], p[j] = p[j], p[i]
                current_cost += best_delta
                improved = True
            else:
                improved = False
                
        return p, current_cost

    def frank_wolfe(a, b, max_iter=500, tol=1e-6):
        """
        Frank-Wolfe solver for QAP relaxation.
        Minimize F(X) = <A, X^T B X> over doubly stochastic X.
        Gradient G = 2 * B X A.
        Linearization: min <G, Y> over Y in Birkhoff.
        Solution Y* is given by Hungarian algorithm on cost matrix G.
        Step size: gamma_t = 2/(t+2) or line search.
        """
        n = len(a)
        # Initialize X as identity or random doubly stochastic
        # Let's start with Identity (permutation)
        X = np.eye(n)
        
        for t in range(max_iter):
            if time.time() > deadline:
                break
            
            # Gradient
            # G = 2 * B X A
            G = 2.0 * (b @ X) @ a
            
            # Solve linear subproblem: min <G, Y> s.t. Y in Birkhoff
            # This is equivalent to min sum G[i,j] Y[i,j]
            # Hungarian algorithm solves min sum C[i,j] P[i,j]
            # Here we want min sum G[i,j] Y[i,j].
            # So cost matrix for Hungarian is G.
            row_ind, col_ind = linear_sum_assignment(G)
            Y = np.zeros((n, n))
            Y[row_ind, col_ind] = 1.0
            
            # Step size
            # gamma_t = 2 / (t + 2)
            gamma = 2.0 / (t + 2)
            
            X_new = (1 - gamma) * X + gamma * Y
            
            # Check convergence
            if np.linalg.norm(X_new - X, ord='fro') < tol:
                X = X_new
                break
            X = X_new
            
        return X

    # 1. Frank-Wolfe Relaxation
    X = frank_wolfe(a, b, max_iter=200)
    
    # 2. Discretize using Hungarian
    # We want to find permutation P minimizing <A, P^T B P>.
    # This is not directly solvable by Hungarian on A or B.
    # However, we can use the continuous solution X to guide a greedy or Hungarian-based discretization.
    # A common heuristic: Solve Hungarian on the "effective cost" matrix derived from X.
    # Or simply, take the permutation closest to X?
    # Better: Use the gradient at X to form a cost matrix for Hungarian?
    # Actually, a standard approach is to just run Hungarian on A? No, that ignores B.
    
    # Let's use a "rank-based" initialization from X.
    # Or, better: Use the Frank-Wolfe result to initialize a local search.
    # But we need a permutation.
    # Let's solve Hungarian on a matrix that approximates the QAP objective.
    # One approximation: C[i,j] = sum_k A[i,k] B[j,k] ? No.
    
    # Alternative Discretization:
    # The Frank-Wolfe iterates are convex combinations of vertices (permutations).
    # We can keep track of the vertices visited and pick the best one.
    # Let's modify frank_wolfe to return the best permutation seen.
    
    # Let's re-implement FW to track best permutation.
    def frank_wolfe_best_perm(a, b, max_iter=300):
        n = len(a)
        X = np.eye(n)
        best_perm = np.arange(n)
        best_cost = calc_cost(best_perm)
        
        for t in range(max_iter):
            if time.time() > deadline:
                break
            
            G = 2.0 * (b @ X) @ a
            row_ind, col_ind = linear_sum_assignment(G)
            Y = np.zeros((n, n))
            Y[row_ind, col_ind] = 1.0
            
            # Y is a permutation matrix. The permutation is col_ind (row i goes to col col_ind[i])
            # Wait, linear_sum_assignment returns row_ind, col_ind such that cost is min sum C[row_ind[k], col_ind[k]].
            # So Y[row_ind[k], col_ind[k]] = 1.
            # This means facility row_ind[k] is assigned to location col_ind[k].
            # So perm[facility] = location.
            # perm[row_ind[k]] = col_ind[k].
            perm = np.empty(n, dtype=np.int64)
            perm[row_ind] = col_ind
            
            cost = calc_cost(perm)
            if cost < best_cost:
                best_cost = cost
                best_perm = perm.copy()
            
            gamma = 2.0 / (t + 2)
            X_new = (1 - gamma) * X + gamma * Y
            X = X_new
            
        return best_perm, best_cost

    if time.time() < deadline - 20:
        p, current_cost = frank_wolfe_best_perm(a, b, max_iter=200)
    else:
        # Fallback to rank matching
        flow_sums = a.sum(axis=1)
        dist_sums = b.sum(axis=1)
        heavy_facilities = np.argsort(-flow_sums, kind="stable")
        central_locations = np.argsort(dist_sums, kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy_facilities] = central_locations
        current_cost = calc_cost(p)

    # 3. Local Search Polish
    p, current_cost = local_search_vectorized(p, current_cost)
    
    # 4. ILS with Random Restarts if time permits
    best_p = p.copy()
    best_cost = current_cost
    
    while time.time() < deadline - 5:
        # Perturb
        p = best_p.copy()
        current_cost = best_cost
        num_swaps = 5
        for _ in range(num_swaps):
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j: continue
            delta = get_delta(p, i, j)
            p[i], p[j] = p[j], p[i]
            current_cost += delta
        
        p, current_cost = local_search_vectorized(p, current_cost)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
        else:
            # Restart from best
            p = best_p.copy()
            current_cost = best_cost

    return best_p.tolist()
# EVOLVE-BLOCK-END
