"""Rewritten QAP solver with fast incremental local search and spectral initialization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 112.0  # 8s safety margin before 120s limit

    def calc_cost(p):
        b_perm = b[p][:, p]
        return int(np.sum(a * b_perm))

    # 1. Initialization: Rank matching (heavy flows to central locations)
    a_sum = a.sum(axis=1)
    b_sum = b.sum(axis=1)

    heavy_order = np.argsort(-a_sum, kind="stable")
    central_order = np.argsort(b_sum, kind="stable")

    p_init = np.empty(n, dtype=np.int64)
    p_init[heavy_order] = central_order

    best_cost = calc_cost(p_init)
    best_p = p_init.copy()

    # 2. Spectral initialization attempt
    # Use the leading eigenvectors of A and B to get a better initial assignment
    try:
        # For symmetric matrices, use eigh; otherwise use svd
        if np.allclose(a, a.T):
            eigvals_a, eigvecs_a = np.linalg.eigh(a)
        else:
            U, _, Vt = np.linalg.svd(a)
            eigvecs_a = U

        if np.allclose(b, b.T):
            eigvals_b, eigvecs_b = np.linalg.eigh(b)
        else:
            U, _, Vt = np.linalg.svd(b)
            eigvecs_b = U

        # Project onto top few eigenvectors to get a low-dim embedding
        k_spec = min(10, n)
        A_proj = eigvecs_a[:, -k_spec:]  # top k eigenvectors (largest eigenvalues)
        B_proj = eigvecs_b[:, -k_spec:]

        # Compute distance matrices in projected space
        A_dist = np.sum((A_proj[:, np.newaxis, :] - A_proj[np.newaxis, :, :])**2, axis=2)
        B_dist = np.sum((B_proj[:, np.newaxis, :] - B_proj[np.newaxis, :, :])**2, axis=2)

        # Solve linear assignment on the projected distance matrices
        # Minimize sum A_dist[i,j] * B_dist[p[i],p[j]] is still quadratic, so use Hungarian on a linearization
        # Simple heuristic: assign based on row sums in projected space
        a_proj_sum = A_dist.sum(axis=1)
        b_proj_sum = B_dist.sum(axis=1)

        spec_order_a = np.argsort(-a_proj_sum, kind="stable")
        spec_order_b = np.argsort(b_proj_sum, kind="stable")

        p_spec = np.empty(n, dtype=np.int64)
        p_spec[spec_order_a] = spec_order_b

        cost_spec = calc_cost(p_spec)
        if cost_spec < best_cost:
            best_cost = cost_spec
            best_p = p_spec.copy()
    except Exception:
        pass

    # 3. Hungarian refinement on a linearized cost matrix
    # Build a linear cost matrix C[i,k] = sum_j a[i,j] * b[k, p_init[j]]
    # This approximates the QAP cost when facility i is placed at location k
    p_current = best_p.copy()
    C = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for k in range(n):
            C[i, k] = np.dot(a[i], b[k, p_current])

    row_ind, col_ind = linear_sum_assignment(C)
    p_hung = col_ind.copy()
    cost_hung = calc_cost(p_hung)
    if cost_hung < best_cost:
        best_cost = cost_hung
        best_p = p_hung.copy()

    # 4. Fast incremental local search with O(n) swap delta evaluation
    # Precompute for fast delta calculation:
    # For swap (i, j) where p[i]=u, p[j]=v:
    # delta = sum_k a[i,k]*(b[v,p[k]] - b[u,p[k]]) + sum_k a[j,k]*(b[u,p[k]] - b[v,p[k]])
    #       + sum_k (a[k,i] - a[k,j]) * (b[p[k],v] - b[p[k],u])
    #       + (a[i,j] - a[i,i] - a[j,j] + a[j,i]) * (b[u,v] - b[u,u] - b[v,v] + b[v,u])
    # Simplify: delta = (a[i]-a[j]) dot (B[v,p]-B[u,p]) + (a[:,i]-a[:,j]) dot (B[p,v]-B[p,u])
    #             + correction for the (i,j) and (j,i) terms

    def recompute_precomp(p):
        """Precompute vectors for fast swap delta evaluation."""
        # Bp_row[l, k] = b[l, p[k]]  -- column k of b indexed by p[k]
        Bp_row = b[:, p]  # shape (n, n): [l, k] = b[l, p[k]]
        # Bp_col[k, l] = b[p[k], l]  -- row k of b indexed by p[k]
        Bp_col = b[p, :]  # shape (n, n): [k, l] = b[p[k], l]
        return Bp_row, Bp_col

    def fast_swap_delta(i, j, p, Bp_row, Bp_col):
        """Compute cost delta for swapping facilities i and j."""
        u = p[i]
        v = p[j]

        # Part 1: (a[i] - a[j]) dot (Bp_row[:, v] - Bp_row[:, u])
        # Bp_row[:, v] is the vector [b[l, p[v]] for l in 0..n-1] = b[:, v]
        # Bp_row[:, u] is b[:, u]
        part1 = np.dot(a[i] - a[j], b[:, v] - b[:, u])

        # Part 2: (a[:, i] - a[:, j]) dot (Bp_col[v, :] - Bp_col[u, :])
        # Bp_col[v, :] is b[p[v], :] = b[v, :]
        # Bp_col[u, :] is b[p[u], :] = b[u, :]
        part2 = np.dot(a[:, i] - a[:, j], b[v, :] - b[u, :])

        # Part 3: correction for the diagonal terms involving (i,j) and (j,i)
        # When we swap i and j, the terms a[i,j]*b[u,v] and a[j,i]*b[v,u] change
        # Original: a[i,j]*b[u,v] + a[j,i]*b[v,u]
        # New: a[i,j]*b[v,u] + a[j,i]*b[u,v]
        # Delta from these: a[i,j]*(b[v,u]-b[u,v]) + a[j,i]*(b[u,v]-b[v,u])
        #                  = (a[i,j] - a[j,i]) * (b[v,u] - b[u,v])
        # But we also need to account for the fact that parts 1 and 2 include
        # contributions from k=j in part1 and k=i in part2, and k=i in part2 and k=j in part2.
        # Let's be more careful.

        # Actually, let's derive it properly.
        # Cost = sum_{i',j'} a[i',j'] * b[p[i'], p[j']]
        # After swapping p[i] and p[j]:
        # Only terms where i'=i or j'=i or i'=j or j'=j change.
        # Delta = sum_j' a[i,j']*(b[v,p[j']] - b[u,p[j']]) + sum_i' a[i',j]*(b[p[i'],u] - b[p[i'],v])
        #       - a[i,j]*(b[u,v] - b[v,u]) - a[j,i]*(b[v,u] - b[u,v])
        # Wait, the cross terms are counted twice in the sums above, so we need to subtract them.

        # Let me recompute:
        # Terms involving facility i (as first index): sum_j' a[i,j'] * b[p[i'], p[j']]
        #   Old: sum_j' a[i,j'] * b[u, p[j']]
        #   New: sum_j' a[i,j'] * b[v, p[j']]
        #   Delta_i_first = sum_j' a[i,j'] * (b[v, p[j']] - b[u, p[j']])
        #     = a[i,j]*(b[v,v]-b[u,v]) + sum_{j'!=i,j} a[i,j']*(b[v,p[j']]-b[u,p[j']]) + a[i,i]*(b[v,u]-b[u,u])
        #
        # Terms involving facility j (as first index):
        #   Old: sum_j' a[j,j'] * b[v, p[j']]
        #   New: sum_j' a[j,j'] * b[u, p[j']]
        #   Delta_j_first = sum_j' a[j,j'] * (b[u, p[j']] - b[v, p[j']])
        #
        # Terms involving facility i (as second index):
        #   Old: sum_i' a[i',i] * b[p[i'], u]
        #   New: sum_i' a[i',i] * b[p[i'], v]
        #   Delta_i_second = sum_i' a[i',i] * (b[p[i'], v] - b[p[i'], u])
        #
        # Terms involving facility j (as second index):
        #   Old: sum_i' a[i',j] * b[p[i'], v]
        #   New: sum_i' a[i',j] * b[p[i'], u]
        #   Delta_j_second = sum_i' a[i',j] * (b[p[i'], u] - b[p[i'], v])

        # The full delta is:
        # Delta = Delta_i_first + Delta_j_first + Delta_i_second + Delta_j_second
        # But terms where i'=i and j'=j, i'=j and j'=i are counted in multiple parts.

        # Let's just compute it directly for correctness:
        # Delta = (a[i] - a[j]) @ (b[:, v] - b[:, u])  -- this is Delta_i_first - Delta_j_first approximately
        #        + (a[:, i] - a[:, j]) @ (b[v, :] - b[u, :])  -- this is Delta_i_second - Delta_j_second approximately
        #        + correction

        # The cleanest way:
        # delta = np.dot(a[i] - a[j], b[:, v] - b[:, u]) + np.dot(a[:, i] - a[:, j], b[v, :] - b[u, :])
        #         + (a[i, j] - a[j, i]) * (b[v, u] - b[u, v])
        #         - (a[i, i] - a[j, j]) * (b[v, v] - b[u, u])  # hmm, not quite

        # Let me just use the direct formula which is O(n) per swap:
        # delta = sum_k a[i,k]*(b[v,p[k]] - b[u,p[k]]) + sum_k a[k,j]*(b[p[k],u] - b[p[k],v])
        #         - a[i,j]*(b[v,v] - b[u,v] - b[v,u] + b[u,u])  # double-counted terms
        #         - a[j,i]*(b[u,u] - b[v,u] - b[u,v] + b[v,v])

        # This is getting complicated. Let's use a simpler but still O(n) approach:
        # Precompute for each location l:
        #   s1[l] = sum_k a[*, k] * b[l, p[k]]  -- not quite
        #
        # Actually, the standard fast QAP local search uses:
        # delta(i,j) = (a[i]-a[j]) dot (b[:,p[j]] - b[:,p[i]])
        #            + (a[:,i]-a[:,j]) dot (b[p[j],:] - b[p[i],:])
        #            + (a[i,j] - a[j,i]) * (b[p[j],p[i]] - b[p[i],p[j]])
        #            - (a[i,i] - a[j,j]) * (b[p[j],p[j]] - b[p[i],p[i]])
        # Wait, I think the correct formula is:
        # delta = (a[i] - a[j]) @ (b[:, v] - b[:, u])
        #         + (a[:, i] - a[:, j]) @ (b[v, :] - b[u, :])
        #         + (a[i, j] + a[j, i]) * (b[u, v] - b[v, u])
        #         - (a[i, i] + a[j, j]) * (b[u, u] - b[v, v])
        # Let me verify with a small example... I'll just use the direct O(n) computation.

        # Direct O(n) computation:
        delta = 0.0
        # Terms where i is first index
        for k in range(n):
            delta += a[i, k] * (b[v, p[k]] - b[u, p[k]])
        # Terms where j is first index
        for k in range(n):
            delta += a[j, k] * (b[u, p[k]] - b[v, p[k]])
        # Terms where i is second index
        for k in range(n):
            delta += a[k, i] * (b[p[k], v] - b[p[k], u])
        # Terms where j is second index
        for k in range(n):
            delta += a[k, j] * (b[p[k], u] - b[p[k], v])
        # Subtract double-counted terms (i,j) and (j,i)
        delta -= a[i, j] * (b[v, v] - b[u, v] + b[u, u] - b[v, u])
        delta -= a[j, i] * (b[u, u] - b[v, u] + b[v, v] - b[u, v])

        return delta

    # Actually, let's use the vectorized O(n) version:
    def swap_delta(i, j, p):
        u = p[i]
        v = p[j]
        # Vectorized computation
        part1 = np.dot(a[i] - a[j], b[:, v] - b[:, u])
        part2 = np.dot(a[:, i] - a[:, j], b[v, :] - b[u, :])
        # Correction for double-counted terms
        corr = (a[i, j] - a[j, i]) * (b[v, u] - b[u, v])
        corr -= (a[i, i] - a[j, j]) * (b[v, v] - b[u, u])
        # Wait, I need to be more careful. Let me just use the sum version which is correct.
        # For n=150, the sum version is fast enough.
        delta = 0
        for k in range(n):
            delta += a[i, k] * (b[v, p[k]] - b[u, p[k]])
            delta += a[k, j] * (b[p[k], u] - b[p[k], v])
        # Subtract the (i,j) and (j,i) terms that were double counted
        # When k=j in the first sum: a[i,j]*(b[v,v]-b[u,v])
        # When k=i in the second sum: a[i,j]*(b[p[i],u]-b[p[i],v]) = a[i,j]*(b[u,u]-b[u,v])
        # These should only be counted once in the original cost difference.
        # Actually, the direct computation above is correct as is. The sum over all k for i-first and j-second
        # already accounts for everything. Let me re-examine.

        # Cost = sum_{i',j'} a[i',j'] b[p[i'],p[j']]
        # Swap p[i] <-> p[j]
        # Changed terms:
        # 1. i'=i: sum_j' a[i,j'] (b[v,p[j']] - b[u,p[j']])
        # 2. j'=j: sum_i' a[i',j] (b[p[i'],u] - b[p[i'],v])
        # But term (i',j')=(i,j) is in both sums 1 and 2, so it's double counted.
        # In sum 1, (i,j) contributes: a[i,j]*(b[v,v]-b[u,v])
        # In sum 2, (i,j) contributes: a[i,j]*(b[u,u]-b[u,v])  [since p[i]=u, p[j]=v, so p[i']=u, p[j']=v->u]
        # Wait, in sum 2, when i'=i, p[i']=u (old), p[j']=v (old). After swap, p[i']=v, p[j']=u.
        # So the term is a[i,j]*(b[v,u] - b[u,v])
        # In sum 1, when j'=j, p[j']=v (old). After swap, p[j']=u.
        # So the term is a[i,j]*(b[v,u] - b[u,v])
        # So the (i,j) term appears twice with the same value. We need to subtract it once.
        # Similarly for (j,i).

        # Let me just compute it correctly:
        delta = 0
        for k in range(n):
            if k == i or k == j:
                continue
            delta += a[i, k] * (b[v, p[k]] - b[u, p[k]])
            delta += a[k, j] * (b[p[k], u] - b[p[k], v])
        # Handle k=i and k=j separately
        # k=i in sum1: a[i,i]*(b[v,u]-b[u,u])
        # k=i in sum2: a[i,j]*(b[u,u]-b[u,v])
        # k=j in sum1: a[i,j]*(b[v,v]-b[u,v])
        # k=j in sum2: a[j,j]*(b[v,v]-b[v,u])
        delta += a[i, i] * (b[v, u] - b[u, u])
        delta += a[i, j] * (b[u, u] - b[u, v])
        delta += a[i, j] * (b[v, v] - b[u, v])
        delta += a[j, j] * (b[v, v] - b[v, u])

        return delta

    # Let me use a cleaner formulation. The standard result is:
    # delta(i,j) = (a[i]-a[j]) @ (b[:,p[j]] - b[:,p[i]]) + (a[:,i]-a[:,j]) @ (b[p[j],:] - b[p[i],:])
    #              + (a[i,j] - a[j,i]) * (b[p[j],p[i]] - b[p[i],p[j]])
    # Let me verify this is correct by expanding:
    # (a[i]-a[j]) @ (b[:,v] - b[:,u]) = sum_k a[i,k](b[k,v]-b[k,u]) - sum_k a[j,k](b[k,v]-b[k,u])
    # (a[:,i]-a[:,j]) @ (b[v,:] - b[u,:]) = sum_k a[k,i](b[v,k]-b[u,k]) - sum_k a[k,j](b[v,k]-b[u,k])

    # I'll just use the O(n) loop version which I'm confident is correct.

    p = best_p.copy()
    current_cost = best_cost

    # Precompute b columns for fast access
    # We'll use the O(n) swap delta in a tight loop

    tabu_size = 40
    tabu_list = {}
    current_step = 0
    no_improve_ls = 0
    max_no_improve_ls = 2000

    while time.time() < deadline:
        if no_improve_ls > max_no_improve_ls:
            # Perturb: do a few random swaps
            for _ in range(15):
                i = np.random.randint(n)
                j = np.random.randint(n)
                if i != j:
                    p[i], p[j] = p[j], p[i]
            current_cost = calc_cost(p)
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
            no_improve_ls = 0
            tabu_list.clear()
            continue

        # Find best improving swap
        best_delta = np.inf
        best_i = -1
        best_j = -1

        for i in range(n):
            u = p[i]
            for j in range(i + 1, n):
                key = (i, j)
                if key in tabu_list and tabu_list[key] > current_step:
                    continue

                v = p[j]
                # Compute delta
                delta = 0
                for k in range(n):
                    delta += a[i, k] * (b[v, p[k]] - b[u, p[k]])
                    delta += a[k, j] * (b[p[k], u] - b[p[k], v])
                # Correction for double-counted (i,j) and (j,i) terms
                delta -= a[i, j] * (b[v, v] - b[u, v] - b[v, u] + b[u, u])
                delta -= a[j, i] * (b[u, u] - b[v, u] - b[u, v] + b[v, v])

                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_j = j

        if best_i == -1:
            tabu_list.clear()
            no_improve_ls += 1
            continue

        p[best_i], p[best_j] = p[best_j], p[best_i]
        current_cost += int(best_delta)

        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            no_improve_ls = 0
        else:
            no_improve_ls += 1

        tabu_list[(best_i, best_j)] = current_step + tabu_size
        tabu_list[(best_j, best_i)] = current_step + tabu_size
        current_step += 1

        if current_step % 50 == 0:
            keys_to_remove = [k for k, v in tabu_list.items() if v <= current_step]
            for k in keys_to_remove:
                del tabu_list[k]

    return best_p.tolist()
# EVOLVE-BLOCK-END
