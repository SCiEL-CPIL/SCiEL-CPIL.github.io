"""QAP tai256c solver: continuous embedding + fast 2-opt with O(n) delta."""
import numpy as np
import time
import scipy.linalg
import scipy.optimize


def solve(a, b):
    n = len(a)
    deadline = time.time() + 115.0  # safety margin under 120s

    def compute_cost(p):
        """Compute exact QAP cost for permutation p."""
        return np.sum(a * b[np.ix_(p, p)])

    # --- Fast 2-opt with O(n) delta computation ---
    # cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # When swapping p[u] and p[v], only terms involving u or v change.
    # delta = sum_j a[u,j]*(b[pv,p[j]]-b[pu,p[j]]) + sum_j a[v,j]*(b[pu,p[j]]-b[pv,p[j]])
    #       + sum_i a[i,u]*(b[p[i],pv]-b[p[i],pu]) + sum_i a[i,v]*(b[p[i],pu]-b[p[i],pv])
    #       - (terms counted twice for i=u or j=u etc.)
    # More carefully:
    # Let pu = p[u], pv = p[v]
    # Terms affected: all (i,j) where i in {u,v} or j in {u,v}
    #
    # Before:
    #   sum_j a[u,j]*b[pu,p[j]] + sum_j a[v,j]*b[pv,p[j]]
    #   + sum_i a[i,u]*b[p[i],pu] + sum_i a[i,v]*b[p[i],pv]
    #   - a[u,u]*b[pu,pu] - a[u,v]*b[pu,pv]  (counted twice in first two sums)
    #   - a[v,u]*b[pv,pu] - a[v,v]*b[pv,pv]  (counted twice in last two sums)
    #   Wait, this is getting confusing. Let me be very precise.
    #
    # Total cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # After swapping p[u]<->p[v]:
    #   p'[u] = pv, p'[v] = pu, p'[k] = p[k] for k != u,v
    #
    # Delta = sum_{i,j} a[i,j] * (b[p'[i], p'[j]] - b[p[i], p[j]])
    #
    # The difference is nonzero only when i in {u,v} or j in {u,v}.
    #
    # Case 1: i=u, j not in {u,v}:
    #   a[u,j] * (b[pv, p[j]] - b[pu, p[j]])
    # Case 2: i=v, j not in {u,v}:
    #   a[v,j] * (b[pu, p[j]] - b[pv, p[j]])
    # Case 3: i not in {u,v}, j=u:
    #   a[i,u] * (b[p[i], pv] - b[p[i], pu])
    # Case 4: i not in {u,v}, j=v:
    #   a[i,v] * (b[p[i], pu] - b[p[i], pv])
    # Case 5: i=u, j=u:
    #   a[u,u] * (b[pv, pv] - b[pu, pu])
    # Case 6: i=u, j=v:
    #   a[u,v] * (b[pv, pu] - b[pu, pv])
    # Case 7: i=v, j=u:
    #   a[v,u] * (b[pu, pv] - b[pv, pu])
    # Case 8: i=v, j=v:
    #   a[v,v] * (b[pu, pu] - b[pv, pv])
    #
    # Combine cases 1&2: sum_{j not in {u,v}} (a[u,j]-a[v,j]) * (b[pv,p[j]] - b[pu,p[j]])
    # Combine cases 3&4: sum_{i not in {u,v}} (a[i,u]-a[i,v]) * (b[p[i],pv] - b[p[i],pu])
    #
    # So:
    # delta = sum_{j not in {u,v}} (a[u,j]-a[v,j]) * (b[pv,p[j]] - b[pu,p[j]])
    #       + sum_{i not in {u,v}} (a[i,u]-a[i,v]) * (b[p[i],pv] - b[p[i],pu])
    #       + a[u,u]*(b[pv,pv]-b[pu,pu]) + a[v,v]*(b[pu,pu]-b[pv,pv])
    #       + a[u,v]*(b[pv,pu]-b[pu,pv]) + a[v,u]*(b[pu,pv]-b[pv,pu])
    #
    # Note: (b[pv,pu]-b[pu,pv]) = -(b[pu,pv]-b[pv,pu]) if b is symmetric, but not in general.
    #
    # Let d = b[pv,:] - b[pu,:]  (row vector, length n)
    # Then case 1&2: sum_{j not in {u,v}} (a[u,j]-a[v,j]) * d[p[j]]
    #   = sum_j (a[u,j]-a[v,j]) * d[p[j]] - (a[u,u]-a[v,u])*d[pv] - (a[u,v]-a[v,v])*d[pu]
    #
    # Similarly for the other sum with column differences.
    #
    # Let me precompute:
    #   For each location l, define row_b[l] = b[l, :]
    #   d_row = b[pv, :] - b[pu, :]
    #   d_col = b[:, pv] - b[:, pu]
    #
    # sum_{j not in {u,v}} (a[u,j]-a[v,j]) * d_row[p[j]]
    #   = sum_j (a[u,j]-a[v,j]) * d_row[p[j]] - (a[u,u]-a[v,u])*d_row[pv] - (a[u,v]-a[v,v])*d_row[pu]
    #
    # sum_{i not in {u,v}} (a[i,u]-a[i,v]) * d_col[p[i]]
    #   = sum_i (a[i,u]-a[i,v]) * d_col[p[i]] - (a[u,u]-a[u,v])*d_col[pv] - (a[v,u]-a[v,v])*d_col[pu]
    #
    # Plus the diagonal and cross terms.

    def swap_delta(p, u, v):
        """Compute delta cost when swapping p[u] and p[v]. O(n)."""
        pu = p[u]
        pv = p[v]
        if pu == pv:
            return 0

        d_row = b[pv, :] - b[pu, :]  # shape (n,)
        d_col = b[:, pv] - b[:, pu]  # shape (n,)

        au = a[u, :]  # shape (n,)
        av = a[v, :]  # shape (n,)
        a_u = a[:, u]  # shape (n,)
        a_v = a[:, v]  # shape (n,)

        # Sum over all j of (a[u,j]-a[v,j]) * d_row[p[j]]
        s1 = np.dot(au - av, d_row[p])
        # Subtract j=u and j=v terms
        s1 -= (au[u] - av[u]) * d_row[pv]
        s1 -= (au[v] - av[v]) * d_row[pu]

        # Sum over all i of (a[i,u]-a[i,v]) * d_col[p[i]]
        s2 = np.dot(a_u - a_v, d_col[p])
        # Subtract i=u and i=v terms
        s2 -= (a_u[u] - a_v[u]) * d_col[pv]
        s2 -= (a_u[v] - a_v[v]) * d_col[pu]

        # Diagonal and cross terms
        diag_cross = (a[u,u] * (b[pv,pv] - b[pu,pu])
                     + a[v,v] * (b[pu,pu] - b[pv,pv])
                     + a[u,v] * (b[pv,pu] - b[pu,pv])
                     + a[v,u] * (b[pu,pv] - b[pv,pu]))

        return s1 + s2 + diag_cross

    def local_search(p, cost, max_passes=50):
        """2-opt local search with O(n) delta. Returns (p, cost)."""
        improved = True
        passes = 0
        while improved and passes < max_passes and time.time() < deadline:
            improved = False
            passes += 1
            for i in range(n):
                if time.time() > deadline:
                    return p, cost
                for j in range(i + 1, n):
                    d = swap_delta(p, i, j)
                    if d < 0:
                        p[i], p[j] = p[j], p[i]
                        cost += d
                        improved = True
        return p, cost

    # --- Continuous embedding initialization ---
    # Model: embed facilities as points X (n x d) in R^d.
    # Energy: E = sum_{i,j} a[i,j] * ||x_i - x_j||^2  (this is a relaxation)
    # But we also want to match the distance matrix b.
    # Better: use the QAP objective directly in continuous space.
    #
    # Actually, a better approach: use spectral methods.
    # The QAP can be relaxed by replacing the permutation matrix P with a doubly stochastic matrix.
    # cost = sum_{i,j} a[i,j] * b[p[i], p[j]] = trace(A P^T B P) (in matrix form)
    #
    # Let's use a simpler continuous relaxation:
    # Minimize F(X) = sum_{i,j} a[i,j] * (||x_i - x_j||^2 - d_ij)^2
    # where d_ij are target distances from b. But b is not necessarily a distance matrix.
    #
    # Alternative: use the fact that for the QAP, we can use the following approach:
    # 1. Compute the "flow graph" Laplacian L_a = D_a - A where A = a (symmetrized)
    # 2. Compute the "distance graph" similarly
    # 3. Use spectral embedding to find a good initial permutation
    #
    # Let's try: symmetrize a and b, compute eigenvectors, use them for initialization.

    # Symmetrize
    a_sym = (a + a.T) / 2.0
    b_sym = (b + b.T) / 2.0

    # Compute Laplacian of flow graph
    d_a = a_sym.sum(axis=1)
    L_a = np.diag(d_a) - a_sym

    # Compute Laplacian of distance graph
    d_b = b_sym.sum(axis=1)
    L_b = np.diag(d_b) - b_sym

    # Get eigenvectors
    try:
        eigvals_a, eigvecs_a = scipy.linalg.eigh(L_a)
        eigvals_b, eigvecs_b = scipy.linalg.eigh(L_b)
    except Exception:
        eigvecs_a = np.eye(n)
        eigvecs_b = np.eye(n)

    # Use the first few non-trivial eigenvectors for embedding
    # Skip the first eigenvector (constant)
    k = min(8, n - 1)
    X_a = eigvecs_a[:, 1:k+1]  # n x k
    X_b = eigvecs_b[:, 1:k+1]  # n x k

    # Project to get a good initial permutation
    # Use the embedding to match facilities to locations
    # Sort facilities by their embedding coordinates and match to locations

    # Simple approach: use the embedding to define a "similarity" and match
    # For now, use rank matching on the embeddings
    # Actually, let's just use the eigenvector-based initialization:
    # Sort facilities by their first eigenvector coordinate, do same for locations

    # Better: use the continuous embedding to define a cost and optimize
    # Let's try a different approach: use the embedding to get an initial permutation
    # via nearest-neighbor matching in the embedding space

    # For each facility i, compute its "position" in the embedding
    # For each location j, compute its "position"
    # Match them greedily or via Hungarian

    # Actually, let's just use a simple spectral initialization:
    # Sort by the second eigenvector (Fiedler vector) of each graph
    if n > 1:
        # Fiedler vector of flow graph
        fiedler_a = eigvecs_a[:, 1]
        # Fiedler vector of distance graph
        fiedler_b = eigvecs_b[:, 1]

        # Sort facilities by Fiedler vector of a
        order_a = np.argsort(fiedler_a)
        # Sort locations by Fiedler vector of b
        order_b = np.argsort(fiedler_b)

        # Initial permutation: facility at rank i in a goes to location at rank i in b
        p_init = np.empty(n, dtype=np.int64)
        p_init[order_a] = order_b
    else:
        p_init = np.array([0], dtype=np.int64)

    # Also try rank-matching initialization
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p_rank = np.empty(n, dtype=np.int64)
    p_rank[heavy] = central

    # Also try random initialization
    p_random = np.random.permutation(n)

    # Evaluate all three initializations and pick the best
    candidates = [p_init.copy(), p_rank.copy(), p_random.copy()]
    best_p = None
    best_cost = np.inf

    for cand in candidates:
        c = compute_cost(cand)
        if c < best_cost:
            best_cost = c
            best_p = cand.copy()

    # Local search from the best initialization
    best_p, best_cost = local_search(best_p, best_cost)

    # Iterated local search with continuous embedding perturbation
    rng = np.random.RandomState(42)

    while time.time() < deadline - 5.0:  # leave 5s margin
        # Perturb: random 2-5 swaps
        p = best_p.copy()
        n_swaps = rng.randint(2, 6)
        for _ in range(n_swaps):
            i, j = rng.choice(n, 2, replace=False)
            p[i], p[j] = p[j], p[i]

        # Local search
        p, cost = local_search(p, compute_cost(p), max_passes=30)

        if cost < best_cost:
            best_p = p.copy()
            best_cost = cost

    return best_p.tolist()
