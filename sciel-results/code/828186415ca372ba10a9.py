"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq

def solve(a, b):
    n = len(a)
    t_start = time.time()
    deadline = t_start + 115.0  # safety margin before 120s kill
    
    # Precompute row sums of a for delta computation
    a_row_sums = a.sum(axis=1)  # a_row_sums[i] = sum_j a[i,j]
    
    # Spectral embedding initialization
    # Compute Laplacians
    # For QAP, we want to align the "structure" of a and b
    # Use the smallest non-trivial eigenvectors of the Laplacian
    # L = D - W
    # For a: weights are a[i,j], for b: weights are -b[i,j] (since we want to minimize distance, 
    #        but spectral embedding usually works with similarity. 
    #        Actually, let's stick to the standard approach: 
    #        Embed facility i based on its connections in graph A.
    #        Embed location j based on its connections in graph B.
        #        Then match closest points.
    
    # A simpler, robust spectral init:
    # 1. Compute eigenvectors of a (or a - mean) and b (or b - mean)
    # 2. Project onto principal components
    # 3. Match
    
    # Let's use the "rank matching" as a fallback if spectral fails or is too slow,
    # but try spectral first.
    
    # Compute row sums for rank matching fallback
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    # Spectral Embedding
    try:
        # Center the matrices to remove bias from overall magnitude
        # Use only the off-diagonal part? No, keep full.
        # We want to find a low-dim representation.
        # Use symmetric part if not symmetric? QAP matrices are symmetric.
        
        # Eigendecomposition of a and b
        # For n=100, this is fast.
        eigvals_a, eigvecs_a = np.linalg.eigh(a)
        eigvals_b, eigvecs_b = np.linalg.eigh(b)
        
        # We want to match the "shape" of the graphs.
        # Typically, we take the first k eigenvectors corresponding to the smallest eigenvalues 
        # (Fiedler vectors) for clustering, or largest for principal components.
        # For QAP, the cost is sum a_ij b_p(i)p(j).
        # If a has high flow between i,j, we want b_p(i)p(j) to be small.
        # So high a_ij should correspond to low b_ij.
        
        # Let's project facilities onto the subspace spanned by the top eigenvectors of A.
        # And locations onto the top eigenvectors of B.
        # But we need to align the coordinate systems.
        
        # Alternative: Use the "Monge" property or rank correlation.
        # The rank matching seed is decent. Let's try to improve it with a spectral projection 
        # that captures 2D structure.
        
        # Take top 10 eigenvectors of A and B (largest eigenvalues).
        k = min(10, n)
        # eigh returns ascending order. So take the last k.
        U_a = eigvecs_a[:, -k:]
        U_b = eigvecs_b[:, -k:]
        
        # Project
        P_a = a @ U_a  # n x k
        P_b = b @ U_b  # n x k
        
        # We need to find a rotation/alignment between P_a and P_b?
        # Or just match points in these spaces?
        # Since the eigenvectors are not aligned, direct matching might be poor.
        
        # A better spectral approach for QAP:
        # Solve the relaxation: minimize trace(X^T A X B) where X is a permutation matrix.
        # Relax X to orthogonal matrix U.
        # min trace(U^T A U B)
        # Let V = U B^{1/2}. Then min trace(U^T A V) s.t. U^T U = I, V^T V = B.
        # This is the Procrustes problem.
        
        # Steps:
        # 1. Eigen-decompose B: B = Q_B Lambda_B Q_B^T
        # 2. B^{1/2} = Q_B Lambda_B^{1/2} Q_B^T
        # 3. We want to maximize trace(U^T A U B) for maximization, but we minimize cost.
        #    Cost = sum a_ij b_p(i)p(j) = trace(P^T A P B) where P is perm matrix.
        #    Minimize trace(P^T A P B).
        #    Let U be orthogonal. Minimize trace(U^T A U B).
        #    This is equivalent to maximizing trace(U^T (-A) U B).
        #    Let C = -A.
        #    Maximize trace(U^T C U B).
        #    Let V = U B^{1/2}. Then V^T V = B^{1/2} U^T U B^{1/2} = B.
        #    Maximize trace(U^T C V).
        #    This is the orthogonal Procrustes problem: max trace(U^T C V) s.t. U orthogonal, V^T V = B.
        #    Actually, the standard Procrustes is max trace(U^T M) s.t. U orthogonal.
        #    Here we have two variables.
        
        # Simpler: 
        # 1. Compute eigendecomposition of B: B = Q_b diag(lambda_b) Q_b^T
        # 2. Let B^{1/2} = Q_b diag(sqrt(lambda_b)) Q_b^T (assuming B is PSD? Distances are not PSD).
        #    Distances are not necessarily PSD. So this relaxation might not work directly.
        
        # Let's stick to a simpler spectral init:
        # 1. Compute the first few Fiedler vectors of A and B.
        # 2. Use them to define a 2D embedding.
        # 3. Match points by nearest neighbor in this embedding.
        
        # Fiedler vectors are the eigenvectors corresponding to the 2nd smallest eigenvalue.
        # For a distance matrix, the smallest eigenvalue is 0 (for constant vector).
        # So we take eigenvectors for indices 1 and 2 (0-indexed) from eigh.
        
        # Let's try k=4 dimensions.
        k = 4
        # eigh returns ascending. Indices 1, 2, 3, 4 are the smallest non-trivial.
        # But for distances, large distances are "far". 
        # The Laplacian of a graph with weights w_ij = -dist_ij?
        
        # Let's just use the top eigenvectors of A and B and match by correlation.
        # Or, simpler: use the rank matching but with a local search.
        # The rank matching is O(n log n). The local search is the main cost.
        
        # Let's use a hybrid init:
        # 1. Rank matching (seed).
        # 2. Spectral matching (top eigenvectors).
        # 3. Pick the better one.
        
    except:
        pass
    
    # Rank Matching Init
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p_rank = np.empty(n, dtype=np.int64)
    p_rank[heavy] = central
    
    # Spectral Init
    p_spec = p_rank.copy() # fallback
    try:
        eigvals_a, eigvecs_a = np.linalg.eigh(a)
        eigvals_b, eigvecs_b = np.linalg.eigh(b)
        
        # Use the eigenvectors corresponding to the largest eigenvalues (principal components)
        # But for QAP, we want to align the "structure".
        # Let's take the first 5 eigenvectors (smallest eigenvalues) which capture the smooth structure.
        # Actually, for a distance matrix, the smallest eigenvalues correspond to the most global structure.
        
        k = 5
        U_a = eigvecs_a[:, :k]
        U_b = eigvecs_b[:, :k]
        
        # Project
        P_a = a @ U_a
        P_b = b @ U_b
        
        # Now we have two sets of points in k-dim space.
        # We want to find a permutation p that minimizes sum ||P_a[i] - P_b[p[i]]||^2 ?
        # This is the Assignment Problem.
        # For n=100, Hungarian algorithm is O(n^3) = 1e6, which is fast.
        
        # Cost matrix for assignment: C[i, j] = ||P_a[i] - P_b[j]||^2
        # We want to minimize sum C[i, p[i]].
        
        # Compute cost matrix
        # P_a shape (n, k), P_b shape (n, k)
        # C[i, j] = sum_d (P_a[i, d] - P_b[j, d])^2
        #          = sum_d P_a[i,d]^2 + sum_d P_b[j,d]^2 - 2 sum_d P_a[i,d] P_b[j,d]
        
        A2 = np.sum(P_a**2, axis=1)[:, np.newaxis]
        B2 = np.sum(P_b**2, axis=1)[np.newaxis, :]
        AB = P_a @ P_b.T
        
        C_assign = A2 + B2 - 2 * AB
        
        # Solve Assignment Problem
        # scipy.optimize.linear_sum_assignment
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(C_assign)
        
        p_spec = np.empty(n, dtype=np.int64)
        p_spec[row_ind] = col_ind
        
        # Evaluate cost of p_rank and p_spec
        cost_rank = _qap_cost(a, b, p_rank)
        cost_spec = _qap_cost(a, b, p_spec)
        
        if cost_spec < cost_rank:
            p = p_spec
        else:
            p = p_rank
            
    except:
        p = p_rank
        
    # Local Search: Tabu Search with First Improvement
    p = _tabu_search(a, b, p, deadline)
    
    return p.tolist()


def _qap_cost(a, b, p):
    # Cost = sum_{i,j} a[i,j] b[p[i], p[j]]
    # Can be computed as trace(P^T A P B)
    # Or simply:
    # sum_i sum_j a[i,j] b[p[i], p[j]]
    # This is O(n^2).
    # For n=100, 10000 ops. Very fast.
    
    # Vectorized:
    # b_p = b[np.ix_(p, p)]
    # cost = np.sum(a * b_p)
    
    b_p = b[np.ix_(p, p)]
    return np.sum(a * b_p)


def _swap_delta(a, b, p, i, j):
    # Compute the change in cost if we swap p[i] and p[j].
    # Let x = p[i], y = p[j].
    # New p: p'[i] = y, p'[j] = x.
    # Old cost contribution involving i or j:
    # C_old = sum_k a[i,k] b[x, p[k]] + sum_k a[k,i] b[p[k], x] 
    #       + sum_k a[j,k] b[y, p[k]] + sum_k a[k,j] b[p[k], y]
    #       - a[i,j] b[x, y] - a[j,i] b[y, x]  (to avoid double counting if i!=j, but sum covers all)
    # Actually, the total cost is sum_{u,v} a[u,v] b[p[u], p[v]].
    # Terms involving i or j:
    # 1. u=i, v=j: a[i,j] b[x, y]
    # 2. u=j, v=i: a[j,i] b[y, x]
    # 3. u=i, v=k (k!=i,j): a[i,k] b[x, p[k]]
    # 4. u=k, v=i (k!=i,j): a[k,i] b[p[k], x]
    # 5. u=j, v=k (k!=i,j): a[j,k] b[y, p[k]]
    # 6. u=k, v=j (k!=i,j): a[k,j] b[p[k], y]
    
    # New cost contribution:
    # 1. u=i, v=j: a[i,j] b[y, x]
    # 2. u=j, v=i: a[j,i] b[x, y]
    # 3. u=i, v=k: a[i,k] b[y, p[k]]
    # 4. u=k, v=i: a[k,i] b[p[k], y]
    # 5. u=j, v=k: a[j,k] b[x, p[k]]
    # 6. u=k, v=j: a[k,j] b[p[k], x]
    
    # Delta = C_new - C_old
    
    x = p[i]
    y = p[j]
    
    # Since a and b are symmetric, a[i,j]=a[j,i], b[x,y]=b[y,x].
    # Term 1+2 old: 2 * a[i,j] * b[x,y]
    # Term 1+2 new: 2 * a[i,j] * b[y,x] = 2 * a[i,j] * b[x,y] (same!)
    # So the cross term between i and j does NOT change if a and b are symmetric.
    # Wait, b[x,y] is the distance between location x and y. It is symmetric.
    # So swapping i and j does not change the cost between i and j.
    
    # The change comes from the interactions with other nodes k.
    # For a fixed k (k!=i,j):
    # Old: a[i,k] b[x, p[k]] + a[k,i] b[p[k], x] + a[j,k] b[y, p[k]] + a[k,j] b[p[k], y]
    #     = 2 * a[i,k] b[x, p[k]] + 2 * a[j,k] b[y, p[k]]  (using symmetry of a and b)
    # New: a[i,k] b[y, p[k]] + a[k,i] b[p[k], y] + a[j,k] b[x, p[k]] + a[k,j] b[p[k], x]
    #     = 2 * a[i,k] b[y, p[k]] + 2 * a[j,k] b[x, p[k]]
    
    # Delta_k = 2 * (a[i,k] (b[y, p[k]] - b[x, p[k]]) + a[j,k] (b[x, p[k]] - b[y, p[k]]))
    #         = 2 * (a[i,k] - a[j,k]) * (b[y, p[k]] - b[x, p[k]])
    
    # Sum over all k != i, j.
    
    # Vectorized:
    # diff_a = a[i, :] - a[j, :]
    # diff_b = b[y, :] - b[x, :]
    # delta = 2 * np.sum(diff_a * diff_b)
    # But we must exclude k=i and k=j?
    # If k=i: diff_a[i] = a[i,i] - a[j,i]. diff_b[i] = b[y, p[i]] - b[x, p[i]] = b[y, x] - b[x, x].
    # This term is included in the sum. Does it correspond to the actual change?
    # The formula derived assumed k != i, j.
    # Let's check the k=i term in the sum:
    # 2 * (a[i,i] - a[j,i]) * (b[y, x] - b[x, x])
    # In the actual cost change, the term involving (i,i) is a[i,i] b[x,x] -> a[i,i] b[y,y].
    # The term involving (j,i) is a[j,i] b[y,x] -> a[j,i] b[x,x].
    # This is getting complicated. It is safer to compute the delta by summing over all k and subtracting the i,j terms if they are incorrectly included, or just trust the vectorized sum if the derivation holds for all k.
    
    # Let's re-verify for k=i.
    # Old terms with k=i:
    # (i,i): a[i,i] b[x,x]
    # (j,i): a[j,i] b[y,x]
    # New terms with k=i:
    # (i,i): a[i,i] b[y,y]
    # (j,i): a[j,i] b[x,x]
    # Change: a[i,i](b[y,y]-b[x,x]) + a[j,i](b[x,x]-b[y,x])
    
    # Formula term for k=i:
    # 2 * (a[i,i] - a[j,i]) * (b[y,x] - b[x,x])
    # = 2 a[i,i] b[y,x] - 2 a[i,i] b[x,x] - 2 a[j,i] b[y,x] + 2 a[j,i] b[x,x]
    # Since b[y,x]=b[x,y], and b[x,x]=b[y,y]=0 (distances to self are 0), this becomes:
    # 2 a[i,i] b[x,y] - 0 - 2 a[j,i] b[x,y] + 0 = 2 (a[i,i] - a[j,i]) b[x,y]
    
    # Actual change:
    # a[i,i](0-0) + a[j,i](b[x,x]-b[y,x]) = 0 + a[j,i](0 - b[x,y]) = -a[j,i] b[x,y]
    
    # They don't match. So the vectorized sum over ALL k is incorrect.
    # We must exclude k=i and k=j.
    
    # delta = 2 * sum_{k!=i,j} (a[i,k] - a[j,k]) * (b[y, p[k]] - b[x, p[k]])
    
    # Implementation:
    # Create a mask or just sum and subtract the i,j terms.
    
    diff_a = a[i, :] - a[j, :]
    # p is the current permutation.
    # b[y, p[k]] - b[x, p[k]]
    # Let v_y = b[y, p] (vector of distances from y to all locations p[k])
    # Let v_x = b[x, p]
    # diff_b[k] = v_y[k] - v_x[k]
    
    v_y = b[y, p]
    v_x = b[x, p]
    diff_b = v_y - v_x
    
    total_delta = 2.0 * np.dot(diff_a, diff_b)
    
    # Subtract contributions for k=i and k=j
    # k=i term: 2 * (a[i,i] - a[j,i]) * (b[y, p[i]] - b[x, p[i]])
    # p[i] = x
    # term_i = 2 * (a[i,i] - a[j,i]) * (b[y, x] - b[x, x])
    # k=j term: 2 * (a[i,j] - a[j,j]) * (b[y, p[j]] - b[x, p[j]])
    # p[j] = y
    # term_j = 2 * (a[i,j] - a[j,j]) * (b[y, y] - b[x, y])
    
    # Since b[x,x]=b[y,y]=0:
    # term_i = 2 * (a[i,i] - a[j,i]) * b[y, x]
    # term_j = 2 * (a[i,j] - a[j,j]) * (-b[x, y]) = -2 * (a[i,j] - a[j,j]) * b[x, y]
    
    # b[y,x] = b[x,y]
    
    term_i = 2.0 * (a[i, i] - a[j, i]) * b[y, x]
    term_j = -2.0 * (a[i, j] - a[j, j]) * b[x, y]
    
    total_delta -= (term_i + term_j)
    
    return total_delta


def _tabu_search(a, b, p, deadline):
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    best_cost = current_cost
    best_p = p.copy()
    
    tabu_tenure = 10
    tabu_list = {} # (i, j) -> iteration when it becomes non-tabu
    iteration = 0
    
    # Precompute neighbors? No, just iterate.
    # First Improvement: scan i=0..n-1, j=i+1..n-1, take first improving non-tabu move.
    
    while time.time() < deadline:
        iteration += 1
        found = False
        
        # First improvement
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i+1, n):
                key = (i, j)
                if key in tabu_list and tabu_list[key] > iteration:
                    continue
                
                delta = _swap_delta(a, b, p, i, j)
                new_cost = current_cost + delta
                
                # Aspiration: if new_cost < best_cost, allow even if tabu
                if new_cost < best_cost or (new_cost < current_cost and (key not in tabu_list or tabu_list[key] <= iteration)):
                    # Perform swap
                    p[i], p[j] = p[j], p[i]
                    current_cost = new_cost
                    tabu_list[key] = iteration + tabu_tenure
                    found = True
                    break
            
            if found:
                break
        
        if not found:
            # No improving move found. Perturbation or restart?
            # Simple perturbation: swap 3 random pairs
            for _ in range(3):
                i, j = np.random.choice(n, 2, replace=False)
                p[i], p[j] = p[j], p[i]
            current_cost = _qap_cost(a, b, p)
            # Clear tabu list? Or keep it?
            # Let's keep it but maybe increase tenure?
            continue
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
    
    return best_p
# EVOLVE-BLOCK-END
