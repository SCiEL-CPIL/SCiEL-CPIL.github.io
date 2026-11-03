"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def solve(a, b):
    n = len(a)
    deadline = time.time() + 145.0  # safety margin before 150s kill

    # Precompute for fast cost updates
    # cost(p) = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # We can compute this as: for each i, sum_j a[i,j] * b[p[i], p[j]]
    # Let's precompute row sums of a: a_row_sum[i] = sum_j a[i,j]
    # But the structure is: cost = sum_i sum_j a[i,j] * b[p[i], p[j]]
    # = sum_i (a[i,:] @ b[p[i], :])  ... no wait, b[p[i], p[j]] for all j
    # Actually: cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # For a fixed permutation p, this is: sum_i sum_j a[i,j] * b[p[i], p[j]]
    # = sum_i (a[i,:] * something) ... let me think differently.
    # cost = sum_i sum_j a[i,j] * b[p[i], p[j]]
    # Let c_i = sum_j a[i,j] * b[p[i], p[j]]
    # Then cost = sum_i c_i
    # c_i depends on p[i] and all p[j] for j != i.
    #
    # For swap-based local search:
    # If we swap p[u] and p[v], the change in cost can be computed incrementally.
    #
    # Let's define: for facility i at location l = p[i], its contribution involves:
    # sum_j a[i,j] * b[l, p[j]]  (as row i)
    # sum_j a[j,i] * b[p[j], l]  (as column i, since a[j,i]*b[p[j],l])
    #
    # Actually the full cost is:
    # C = sum_{i,j} a[i,j] * b[p[i], p[j]]
    #
    # Let's precompute for the current permutation p:
    # For each location l, and for each facility i, what is the "interaction"?
    #
    # A more efficient approach for swap delta:
    # When we swap p[u] <-> p[v], let x = p[u], y = p[v].
    # The only terms that change are those involving i=u or j=u or i=v or j=v.
    #
    # Let's define:
    #   contrib(i, j) = a[i,j] * b[p[i], p[j]]
    #
    # After swapping u and v (so p'[u]=y, p'[v]=x, others same):
    # Changed terms: all (i,j) where i in {u,v} or j in {u,v}.
    #
    # Delta = sum_{i,j: i in {u,v} or j in {u,v}} [contrib'(i,j) - contrib(i,j)]
    #
    # Let's expand. The set of changed (i,j) pairs:
    # (u,u), (u,v), (v,u), (v,v), and for all k not in {u,v}: (u,k), (k,u), (v,k), (k,v)
    #
    # Before swap:
    #   (u,u): a[u,u]*b[x,x]
    #   (u,v): a[u,v]*b[x,y]
    #   (v,u): a[v,u]*b[y,x]
    #   (v,v): a[v,v]*b[y,y]
    #   (u,k): a[u,k]*b[x, p[k]]
    #   (k,u): a[k,u]*b[p[k], x]
    #   (v,k): a[v,k]*b[y, p[k]]
    #   (k,v): a[k,v]*b[p[k], y]
    #
    # After swap:
    #   (u,u): a[u,u]*b[y,y]
    #   (u,v): a[u,v]*b[y,x]
    #   (v,u): a[v,u]*b[x,y]
    #   (v,v): a[v,v]*b[x,x]
    #   (u,k): a[u,k]*b[y, p[k]]
    #   (k,u): a[k,u]*b[p[k], y]
    #   (v,k): a[v,k]*b[x, p[k]]
    #   (k,v): a[k,v]*b[p[k], x]
    #
    # Delta for diagonal terms:
    #   a[u,u]*(b[y,y]-b[x,x]) + a[v,v]*(b[x,x]-b[y,y])
    #   = (a[u,u]-a[v,v])*(b[y,y]-b[x,x])
    #
    # Delta for (u,v) and (v,u):
    #   a[u,v]*(b[y,x]-b[x,y]) + a[v,u]*(b[x,y]-b[y,x])
    #   = (a[u,v]-a[v,u])*(b[y,x]-b[x,y])
    #
    # Delta for cross terms with k not in {u,v}:
    #   sum_k [a[u,k]*(b[y,p[k]]-b[x,p[k]]) + a[k,u]*(b[p[k],y]-b[p[k],x])
    #          + a[v,k]*(b[x,p[k]]-b[y,p[k]]) + a[k,v]*(b[p[k],x]-b[p[k],y])]
    #   = sum_k [(a[u,k]-a[v,k])*(b[y,p[k]]-b[x,p[k]])
    #           + (a[k,u]-a[k,v])*(b[p[k],y]-b[p[k],x])]
    #
    # We can precompute for each location l:
    #   For a given pair of locations (x, y), define:
    #   diff_row[k] = b[y, p[k]] - b[x, p[k]]  (for each k)
    #   diff_col[k] = b[p[k], y] - b[p[k], x]  (for each k)
    #
    # But this is O(n) per swap, which is good.
    #
    # Let's precompute:
    #   a_row_diff[u,v] = a[u,:] - a[v,:]  (vector of length n)
    #   a_col_diff[u,v] = a[:,u] - a[:,v]  (vector of length n)
    #   b_row_diff[x,y] = b[y,:] - b[x,:]  (vector of length n)
    #   b_col_diff[x,y] = b[:,y] - b[:,x]  (vector of length n)
    #
    # Then for swap of u,v with x=p[u], y=p[v]:
    #   cross_delta = sum_k (a_row_diff[u,v][k] * b_row_diff[x,y][p[k]]
    #                        + a_col_diff[u,v][k] * b_col_diff[x,y][p[k]])
    #
    # This is still O(n) but with large constant. Let's just do it directly.

    def compute_cost(p):
        """Compute the full QAP cost."""
        total = 0
        for i in range(n):
            li = p[i]
            for j in range(n):
                total += a[i, j] * b[li, p[j]]
        return total

    def swap_delta(p, u, v):
        """Compute the cost delta when swapping p[u] and p[v]."""
        x = p[u]
        y = p[v]
        if x == y:
            return 0

        # Diagonal terms
        delta = (a[u, u] - a[v, v]) * (b[y, y] - b[x, x])
        # Off-diagonal (u,v) and (v,u)
        delta += (a[u, v] - a[v, u]) * (b[y, x] - b[x, y])

        # Cross terms with k not in {u, v}
        # We need: sum_k [(a[u,k]-a[v,k])*(b[y,p[k]]-b[x,p[k]]) + (a[k,u]-a[k,v])*(b[p[k],y]-b[p[k],x])]
        # for k not in {u,v}
        # Note: when k=u or k=v, p[k] is x or y, so we need to exclude them.
        # Let's just iterate over all k and handle it.
        for k in range(n):
            if k == u or k == v:
                continue
            pk = p[k]
            delta += (a[u, k] - a[v, k]) * (b[y, pk] - b[x, pk])
            delta += (a[k, u] - a[k, v]) * (b[pk, y] - b[pk, x])
        return delta

    # Initial solution: rank-matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p_list = p.tolist()

    best_cost = compute_cost(p)
    best_p = p.copy()

    # Local search: 2-opt style swaps
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() >= deadline:
                break
            for j in range(i + 1, n):
                d = swap_delta(p, i, j)
                if d < 0:
                    # Perform swap
                    p[i], p[j] = p[j], p[i]
                    best_cost += d
                    best_p = p.copy()
                    improved = True
                    if time.time() >= deadline:
                        break
        if time.time() >= deadline:
            break

    # Iterated Local Search: perturb and re-optimize
    while time.time() < deadline:
        # Perturbation: random double swap
        num_swaps = 2 + np.random.randint(0, 3)
        perturbed = best_p.copy()
        for _ in range(num_swaps):
            i, j = np.random.choice(n, 2, replace=False)
            perturbed[i], perturbed[j] = perturbed[j], perturbed[i]

        # Local search from perturbed
        curr_cost = compute_cost(perturbed)
        curr_p = perturbed

        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(n):
                if time.time() >= deadline:
                    break
                for j in range(i + 1, n):
                    d = swap_delta(curr_p, i, j)
                    if d < 0:
                        curr_p[i], curr_p[j] = curr_p[j], curr_p[i]
                        curr_cost += d
                        improved = True
                        if time.time() >= deadline:
                            break
                if time.time() >= deadline:
                    break
            if time.time() >= deadline:
                break

        if curr_cost < best_cost:
            best_cost = curr_cost
            best_p = curr_p.copy()

        if time.time() >= deadline:
            break

    return best_p.tolist()
# EVOLVE-BLOCK-END
