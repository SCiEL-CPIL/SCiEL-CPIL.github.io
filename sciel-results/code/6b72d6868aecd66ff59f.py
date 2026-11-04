"""QAP tai256c solver: iterated local search with Hungarian re-optimization."""
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    n = len(a)
    deadline = time.time() + 115.0  # safety margin under 120s

    def compute_cost(p):
        return np.sum(a * b[np.ix_(p, p)])

    def fast_swap_delta(p, u, v):
        pu = p[u]
        pv = p[v]

        x, y = pu, pv
        # Compute efficiently using precomputed differences
        # sum_{k!=u,v} (a[u,k]-a[v,k])*(b[y,p[k]]-b[x,p[k]])
        diff_a_row = a[u, :] - a[v, :]  # shape (n,)
        diff_b_col = b[y, p] - b[x, p]  # shape (n,)
        s1 = np.dot(diff_a_row, diff_b_col)
        # Subtract k=u and k=v terms
        s1 -= diff_a_row[u] * diff_b_col[u]
        s1 -= diff_a_row[v] * diff_b_col[v]

        # sum_{k!=u,v} (a[k,u]-a[k,v])*(b[p[k],y]-b[p[k],x])
        diff_a_col = a[:, u] - a[:, v]  # shape (n,)
        diff_b_row = b[p, y] - b[p, x]  # shape (n,)
        s2 = np.dot(diff_a_col, diff_b_row)
        # Subtract k=u and k=v terms
        s2 -= diff_a_col[u] * diff_b_row[u]
        s2 -= diff_a_col[v] * diff_b_row[v]

        # Cross term: 2*(a[u,v]-a[v,u])*(b[y,x]-b[x,y])
        cross = 2.0 * (a[u, v] - a[v, u]) * (b[y, x] - b[x, y])

        # Self term: (a[u,u]-a[v,v])*(b[y,y]-b[x,x])
        self_term = (a[u, u] - a[v, v]) * (b[y, y] - b[x, x])

        return s1 + s2 + cross + self_term

    def local_search(p, cost, max_passes=100):
        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            if passes % 3 == 0 and time.time() > deadline:
                return p, cost
            for i in range(n):
                if passes % 10 == 0 and time.time() > deadline:
                    return p, cost
                for j in range(i + 1, n):
                    delta = fast_swap_delta(p, i, j)
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        cost += delta
                        improved = True
                        break
                if improved:
                    break
        if time.time() > deadline:
            return p, cost
        return p, cost

    def hungarian_reopt(p):
        """Use Hungarian algorithm on a linearized cost matrix to escape local optima."""
        B_p = b[np.ix_(np.arange(n), p)]
        M1 = a @ B_p.T  # M1[i,j] = sum_k a[i,k]*b[j,p[k]]
        
        Bp2 = b[np.ix_(p, np.arange(n))]
        M2 = a.T @ Bp2  # M2[i,j] = sum_k a[k,i]*b[p[k],j]
        
        C = M1 + M2
        
        # Solve assignment
        row_ind, col_ind = linear_sum_assignment(C)
        new_p = np.empty(n, dtype=np.int64)
        new_p[row_ind] = col_ind
        return new_p

    def spectral_init():
        """Generate initial solution using spectral clustering / graph partitioning ideas."""
        # Use the flow matrix to create a similarity graph
        # Facilities with high mutual flow should be close in the assignment
        # Use eigenvectors of the flow matrix for ordering
        
        # Symmetrize flow matrix for spectral methods
        a_sym = (a + a.T) / 2.0
        
        # Get the second eigenvector (Fiedler vector) for 1D ordering
        # Use scipy's sparse eigenvalue solver or numpy's full eigendecomposition
        try:
            from scipy.sparse.linalg import eigsh
            from scipy.sparse import csc_matrix
            A_sp = csc_matrix(a_sym)
            # Get smallest magnitude eigenvalues (excluding the trivial one)
            vals, vecs = eigsh(A_sp, k=3, which='SM')
            # Use the eigenvector corresponding to the second smallest eigenvalue
            # Sort facilities by their coordinate in this eigenvector
            idx = np.argsort(vecs[:, 1])
            return idx.astype(np.int64)
        except:
            # Fallback: use rank matching
            a_row_sums = a.sum(axis=1)
            b_col_sums = b.sum(axis=1)
            heavy = np.argsort(-a_row_sums, kind="stable")
            central = np.argsort(b_col_sums, kind="stable")
            p = np.empty(n, dtype=np.int64)
            p[heavy] = central
            return p

    # Initial solution: try multiple strategies
    candidates = []
    
    # 1. Spectral initialization
    p_spectral = spectral_init()
    candidates.append(p_spectral)
    
    # 2. Rank matching
    a_row_sums = a.sum(axis=1)
    b_col_sums = b.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_col_sums, kind="stable")
    p_rank = np.empty(n, dtype=np.int64)
    p_rank[heavy] = central
    candidates.append(p_rank)
    
    # 3. Random permutations
    for _ in range(5):
        candidates.append(np.random.permutation(n).astype(np.int64))
    
    # Evaluate all candidates and pick the best
    best_p = None
    best_cost = float('inf')
    for cand in candidates:
        c = compute_cost(cand)
        if c < best_cost:
            best_cost = c
            best_p = cand.copy()
    
    current_p = best_p.copy()
    current_cost = best_cost
    
    # First local search
    current_p, current_cost = local_search(current_p, current_cost)
    if current_cost < best_cost:
        best_p = current_p.copy()
        best_cost = current_cost

    # Iterated Local Search with periodic Hungarian re-optimization
    iteration = 0
    while time.time() < deadline:
        iteration += 1

        # Perturb: random swaps with adaptive intensity
        p = best_p.copy()
        # Increase perturbation strength over time to explore more
        base_swaps = 3 + int(iteration * 0.1)
        n_swaps = np.random.randint(base_swaps, base_swaps + 8)
        for _ in range(n_swaps):
            i, j = np.random.choice(n, 2, replace=False)
            p[i], p[j] = p[j], p[i]

        # Local search from perturbed
        p, cost = local_search(p, compute_cost(p))

        # Every few iterations, try Hungarian re-optimization
        if iteration % 3 == 0 and time.time() < deadline - 40:
            h_p = hungarian_reopt(p)
            h_cost = compute_cost(h_p)
            h_p, h_cost = local_search(h_p, h_cost)
            if h_cost < cost:
                p, cost = h_p, h_cost
        
        # Additional: try a 2-opt style move (swap two facilities)
        if time.time() < deadline - 20 and iteration % 5 == 0:
            # Try random pair swaps and keep if better
            for _ in range(20):
                if time.time() > deadline - 20:
                    break
                i, j = np.random.choice(n, 2, replace=False)
                p[i], p[j] = p[j], p[i]
                new_cost = compute_cost(p)
                if new_cost < cost:
                    cost = new_cost
                else:
                    p[i], p[j] = p[j], p[i]  # revert
                    # Local search from here
                    p, cost = local_search(p, cost)
                    break

        if cost < best_cost:
            best_p = p.copy()
            best_cost = cost

    return best_p.tolist()
