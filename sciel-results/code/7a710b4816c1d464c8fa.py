"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def solve(a, b):
    n = len(a)
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)

    deadline = time.time() + 115.0  # safety margin before 120s kill

    # ---------- cost helpers ----------
    def compute_cost(p):
        # sum_{i,j} a[i,j] * b[p[i], p[j]]
        return float(np.einsum('ij,kl,il,jl->', a, b, p, p, optimize=True))

    def compute_cost_int(p):
        # exact integer cost using int64
        return int(np.einsum('ij,kl,il,jl->', a.astype(np.int64), b.astype(np.int64), p, p, optimize=True))

    # ---------- vectorized 2-swap delta ----------
    def swap_delta(p, i, j):
        """Return delta in cost if we swap positions i and j (i.e., p[i] and p[j] exchange)."""
        u = p[i]
        v = p[j]
        # Direct terms
        d = a[i, j] * (b[v, u] - b[u, v]) + a[j, i] * (b[u, v] - b[v, u])
        # Cross terms with all other k
        # For each k != i, j: a[i,k]*(b[v, p[k]] - b[u, p[k]]) + a[j,k]*(b[u, p[k]] - b[v, p[k]])
        # = (a[i,k] - a[j,k]) * (b[v, p[k]] - b[u, p[k]])
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        mask[j] = False
        ks = np.where(mask)[0]
        diff_a = a[i, ks] - a[j, ks]
        diff_b = b[v, p[ks]] - b[u, p[ks]]
        d += float(np.dot(diff_a, diff_b))
        return d

    # ---------- local search: random 2-swap hill climbing ----------
    def local_search(p, max_iters=200000):
        p = p.copy()
        best_cost = compute_cost(p)
        no_improve = 0
        rng = np.random.RandomState(42)
        for it in range(max_iters):
            if time.time() > deadline:
                break
            if no_improve > 5000:
                break
            i, j = rng.randint(0, n, size=2)
            if i == j:
                continue
            d = swap_delta(p, i, j)
            if d < -1e-10:
                p[i], p[j] = p[j], p[i]
                best_cost += d
                no_improve = 0
            else:
                no_improve += 1
        return p, best_cost

    # ---------- GRASP construction ----------
    def grasp_construct(rng, rcl_size=10):
        """Build a permutation via greedy randomized construction."""
        p = np.full(n, -1, dtype=np.int64)
        available = np.arange(n)
        # place facilities in random order
        order = rng.permutation(n)
        placed = []
        for fac in order:
            # compute cost of placing fac at each available location
            costs = np.zeros(len(available))
            for idx, loc in enumerate(available):
                c = 0.0
                for j in placed:
                    c += a[fac, j] * b[loc, p[j]] + a[j, fac] * b[p[j], loc]
                costs[idx] = c
            # RCL: top rcl_size by cost
            k = min(rcl_size, len(available))
            if k == len(available):
                choice_idx = rng.randint(len(available))
            else:
                top_k = np.argpartition(costs, k)[:k]
                choice_idx = top_k[rng.randint(k)]
            p[fac] = available[choice_idx]
            available = np.delete(available, choice_idx)
            placed.append(fac)
        return p

    # ---------- initial solutions ----------
    # Rank-matching start
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p0 = np.empty(n, dtype=np.int64)
    p0[heavy] = central

    best_p = p0.copy()
    best_cost = compute_cost(best_p)

    # Local search from rank-matching
    best_p, best_cost = local_search(best_p)

    # GRASP iterations
    rng = np.random.RandomState(123)
    grasp_iter = 0
    while time.time() < deadline - 5.0:
        grasp_iter += 1
        p_g = grasp_construct(rng, rcl_size=8 + (grasp_iter % 4) * 2)
        p_ls, cost_ls = local_search(p_g, max_iters=100000)
        if cost_ls < best_cost - 1e-10:
            best_cost = cost_ls
            best_p = p_ls

    # Final polish: more local search iterations
    best_p, best_cost = local_search(best_p, max_iters=500000)

    # Verify exact integer cost
    final_p = best_p.astype(np.int64)
    # Make sure it's a valid permutation
    assert len(final_p) == n
    assert set(final_p.tolist()) == set(range(n)), "Invalid permutation"

    return final_p.tolist()
# EVOLVE-BLOCK-END
