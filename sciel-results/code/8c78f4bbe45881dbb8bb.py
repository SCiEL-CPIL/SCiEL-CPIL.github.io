# EVOLVE-BLOCK-START
"""Residual-connectivity TSP constructive heuristic (1-tree estimator).

Structural idea (from the MST-based program): score each candidate by the
*residual* cost of the remaining subgraph, not just the incremental step.

Refinement (from the fast/softmin program): fully-vectorized O(n^2) scoring
with a richer parameter space and a tight time budget, so we can afford to
actually search for good parameters and then build many tours.

The residual-connectivity term uses a deterministic minimum-1-tree estimator:
for candidate c, the cheapest way to keep the remaining cities connected is a
1-tree = (two cheapest edges incident to c) + (cheapest edge of every other
remaining city).  A 1-tree over the remaining set is always >= its MST, so this
is a principled lower-bound proxy for the MST residual, computed in O(n^2)
instead of O(n^2) per candidate with a heavy constant.
"""
import numpy as np
import time

_start_time = time.time()
_LIMIT = 145.0  # safety margin before the 150s kill


def _is_deadline_passed():
    return (time.time() - _start_time) > _LIMIT


def _build_tour(distance_matrix, start_node, alpha=0.0, beta=0.0, gamma=0.0, delta=0.0):
    """Build a full TSP tour.  Returns (tour_length, list_of_nodes_in_order).

    score(c) = d(cur,c)
               + alpha * 1tree_residual(unvisited \ {c})
               + beta  * d(c, dest)
               + gamma * min_{u in unvisited \ {c}} d(c, u)
               + delta * (d(c,dest) - min_u d(u,dest))   [isolation penalty]
    """
    n = distance_matrix.shape[0]
    current = start_node
    total_dist = 0.0
    remaining = list(range(1, n))
    tour = [start_node]

    while remaining:
        rem = np.asarray(remaining, dtype=np.int64)
        k = len(rem)
        d_cur = distance_matrix[current, rem]
        scores = d_cur.astype(float).copy()

        if k > 1:
            sub = distance_matrix[np.ix_(rem, rem)]
            np.fill_diagonal(sub, np.inf)

            if alpha > 0 and k > 2:
                # 1-tree residual for candidate i: two cheapest edges of rem[i]
                # (to other remaining) + cheapest edge of every other remaining node.
                min1 = np.min(sub, axis=1)          # cheapest edge of each remaining node
                min2 = np.partition(sub, 1, axis=1)[:, 1]  # 2nd-cheapest edge of each
                two_cheapest = min1 + min2          # two cheapest edges of each node
                # cost to connect rem[i] into the 1-tree:
                #   two cheapest of rem[i]  +  cheapest of every other node
                # = two_cheapest[i] + (sum(min1) - min1[i])
                scores += alpha * (two_cheapest + (min1.sum() - min1))

            if gamma > 0:
                scores += gamma * min1

            if delta > 0:
                d_dest = distance_matrix[rem, start_node]
                scores += delta * (d_dest - d_dest.min())

        if beta > 0:
            scores += beta * distance_matrix[rem, start_node]

        best_idx = int(np.argmin(scores))
        nxt = int(rem[best_idx])
        total_dist += distance_matrix[current, nxt]
        remaining.pop(best_idx)
        current = nxt
        tour.append(nxt)

    total_dist += distance_matrix[current, start_node]
    return total_dist, tour


def _fit_params(distance_matrix, start_node, time_budget=5.0):
    """Search for good (alpha, beta, gamma, delta) by building full tours."""
    if _is_deadline_passed():
        return (0.0, 0.0, 0.0, 0.0)

    alpha_values = [0.0, 0.05, 0.10, 0.15, 0.25, 0.40, 0.60]
    beta_values = [0.0, 0.05, 0.10]
    gamma_values = [0.0, 0.05, 0.10]
    delta_values = [0.0, 0.05]

    best = (0.0, 0.0, 0.0, 0.0)
    best_len = float("inf")

    t0 = time.time()
    deadline = t0 + min(time_budget, max(1.0, _LIMIT - (time.time() - _start_time) - 8.0))

    for a in alpha_values:
        if time.time() > deadline:
            break
        for b in beta_values:
            if time.time() > deadline:
                break
            for g in gamma_values:
                if time.time() > deadline:
                    break
                for d in delta_values:
                    if time.time() > deadline:
                        break
                    L, _ = _build_tour(distance_matrix, 0, a, b, g, d)
                    if L < best_len:
                        best_len = L
                        best = (a, b, g, d)

    # Local refinement around the best found (cheaper: only 3x3x2x2).
    if time.time() < deadline - 1.5:
        ba, bb, bg, bd = best
        for sa in (max(0.0, ba - 0.05), ba, min(0.8, ba + 0.05)):
            for sb in (max(0.0, bb - 0.025), bb, min(0.3, bb + 0.025)):
                for sg in (max(0.0, bg - 0.025), bg, min(0.3, bg + 0.025)):
                    for sd in (max(0.0, bd - 0.025), bd, min(0.2, bd + 0.025)):
                        if time.time() > deadline:
                            break
                        L, _ = _build_tour(distance_matrix, 0, sa, sb, sg, sd)
                        if L < best_len:
                            best_len = L
                            best = (sa, sb, sg, sd)

    return best


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """Pick the next city while a TSP tour is built step by step.

    Model-and-optimize: on a new instance, search for good parameters and keep
    the best full tour found; then replay that tour step by step.  If the
    evaluator's start node is not the one we optimised from, fall back to
    greedy scoring with the fitted parameters.
    """
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    n_total = distance_matrix.shape[0]
    is_new_instance = (n_unvisited >= n_total - 1)

    # --- new instance: search for the best tour and parameters -------------
    if is_new_instance:
        if _is_deadline_passed():
            # Not enough time to search; greedy nearest-neighbour from here.
            d = distance_matrix[current_node, unvisited_nodes]
            return int(unvisited_nodes[int(np.argmin(d))])

        tb = 8.0 if n_total <= 50 else 4.0
        params = _fit_params(distance_matrix, 0, time_budget=tb)
        alpha, beta, gamma, delta = params

        # Build the best tour we can afford, starting from node 0.
        tour, _ = _build_tour(distance_matrix, 0, alpha, beta, gamma, delta)

        # Try a couple of alternative starts if we still have budget.
        best_len, best_tour = float("inf"), tour
        L0, _ = _build_tour(distance_matrix, 0, alpha, beta, gamma, delta)
        best_len, best_tour = L0, tour
        for s in (1, 2):
            if _is_deadline_passed() or (time.time() - _start_time) > _LIMIT - 12.0:
                break
            Ls, ts = _build_tour(distance_matrix, s, alpha, beta, gamma, delta)
            if Ls < best_len:
                best_len, best_tour = Ls, ts

        # Cache the plan for this instance.
        global _PLAN, _PLAN_PARAMS, _PLAN_START
        _PLAN = best_tour
        _PLAN_PARAMS = (alpha, beta, gamma, delta)
        _PLAN_START = 0
        return _next_from_plan(current_node, destination_node, unvisited_nodes,
                               distance_matrix)

    # --- mid-tour: replay the cached plan if possible ----------------------
    return _next_from_plan(current_node, destination_node, unvisited_nodes,
                           distance_matrix)


# per-instance cached plan
_PLAN = None
_PLAN_PARAMS = None
_PLAN_START = None


def _next_from_plan(current_node, destination_node, unvisited_nodes, distance_matrix):
    """Replay the cached tour: return the successor of current_node in the plan.

    If the plan does not contain current_node (evaluator used a different start
    or a different tour so far), fall back to greedy scoring with fitted params.
    """
    if _PLAN is not None:
        try:
            idx = _PLAN.index(int(current_node))
            # successor in the cycle
            succ = _PLAN[(idx + 1) % len(_PLAN)]
            if succ in set(unvisited_nodes.tolist()):
                return int(succ)
        except (ValueError, AttributeError):
            pass

    # Fallback: greedy scoring with fitted parameters.
    if _is_deadline_passed():
        d = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[int(np.argmin(d))])

    alpha, beta, gamma, delta = _PLAN_PARAMS if _PLAN_PARAMS else (0.1, 0.05, 0.05, 0.0)
    rem = np.asarray(unvisited_nodes, dtype=np.int64)
    k = len(rem)
    d_cur = distance_matrix[current_node, rem]
    scores = d_cur.astype(float).copy()

    if k > 1:
        sub = distance_matrix[np.ix_(rem, rem)]
        np.fill_diagonal(sub, np.inf)
        min1 = np.min(sub, axis=1)
        if alpha > 0 and k > 2:
            min2 = np.partition(sub, 1, axis=1)[:, 1]
            two_cheapest = min1 + min2
            scores += alpha * (two_cheapest + (min1.sum() - min1))
        if gamma > 0:
            scores += gamma * min1
        if delta > 0:
            d_dest = distance_matrix[rem, destination_node]
            scores += delta * (d_dest - d_dest.min())
    if beta > 0:
        scores += beta * distance_matrix[rem, destination_node]

    return int(rem[int(np.argmin(scores))])

# EVOLVE-BLOCK-END
