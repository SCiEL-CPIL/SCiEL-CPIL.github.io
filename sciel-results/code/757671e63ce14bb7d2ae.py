"""Rewritten program: MILP-based exact optimizer for reciprocal-sum independent set."""
# EVOLVE-BLOCK-START
import time
from math import lcm, gcd

def construct():
    t0 = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    # Build conflict graph: edge (i,j) if lcm(i,j) <= 256
    # We want an independent set maximizing sum of 1/i
    n = 256
    nodes = list(range(2, 257))  # 2..256 inclusive
    N = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}

    # Precompute conflicts
    conflict = [[False] * N for _ in range(N)]
    for i in range(N):
        a = nodes[i]
        for j in range(i + 1, N):
            b = nodes[j]
            if lcm(a, b) <= 256:
                conflict[i][j] = True
                conflict[j][i] = True

    # Try to use a MILP solver if available
    solved = False
    best_set = set()
    best_val = 0.0

    # First, get a good initial solution via greedy
    greedy_set = _greedy(nodes, conflict, idx)
    best_set = set(greedy_set)
    best_val = sum(1.0 / v for v in greedy_set)

    # Try scipy.optimize.milp if available (SciPy >= 1.7)
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import numpy as np

        # Variables: x_i in {0,1} for each node i
        # Maximize sum (1/nodes[i]) * x_i
        # Subject to: x_i + x_j <= 1 for all conflicting pairs (i,j)
        #             x_i in {0,1}

        c = np.array([-1.0 / nodes[i] for i in range(N)])  # minimize -sum

        # Constraints: for each edge, x_i + x_j <= 1
        edges = []
        for i in range(N):
            for j in range(i + 1, N):
                if conflict[i][j]:
                    edges.append((i, j))

        M = len(edges)
        if M > 0:
            A = np.zeros((M, N))
            for k, (i, j) in enumerate(edges):
                A[k, i] = 1.0
                A[k, j] = 1.0
            cons = LinearConstraint(A, -np.inf, np.ones(M))
        else:
            cons = LinearConstraint(np.zeros((1, N)), 0, 0)

        integrality = np.ones(N)
        bounds = Bounds(0, 1)

        res = milp(c, constraints=cons, integrality=integrality, bounds=bounds)

        if res.status == 0 and res.x is not None:
            sol = [i for i in range(N) if res.x[i] > 0.5]
            sol_set = {nodes[i] for i in sol}
            sol_val = sum(1.0 / v for v in sol_set)
            if sol_val > best_val:
                best_val = sol_val
                best_set = sol_set
            solved = True
        elif res.status == 0:
            solved = True
    except Exception:
        pass

    # If MILP didn't solve or wasn't available, do local search
    if time.time() < DEADLINE:
        best_set, best_val = _local_search(nodes, conflict, idx, best_set, best_val, DEADLINE)

    # Final polish: try adding any node that fits, then removing low-value nodes
    if time.time() < DEADLINE:
        best_set, best_val = _polish(nodes, conflict, idx, best_set, best_val, DEADLINE)

    return sorted(best_set)


def _greedy(nodes, conflict, idx):
    """Greedy: pick nodes in order of decreasing 1/value (i.e., increasing value)."""
    order = sorted(range(len(nodes)), key=lambda i: nodes[i])  # smallest first = largest 1/x
    selected = set()
    for i in order:
        if all(i not in _conflict_set(nodes, conflict, j) for j in selected):
            # Check against all currently selected
            ok = True
            for j in selected:
                if conflict[i][j]:
                    ok = False
                    break
            if ok:
                selected.add(i)
    return [nodes[i] for i in selected]


def _conflict_set(nodes, conflict, i):
    """Return set of indices conflicting with i."""
    return set(j for j in range(len(nodes)) if conflict[i][j])


def _local_search(nodes, conflict, idx, best_set, best_val, deadline):
    """Local search: try single swaps and additions."""
    N = len(nodes)
    best_set = set(best_set)
    best_val = best_val
    t0 = time.time()

    while time.time() < deadline:
        improved = False

        # Try adding a node
        for i in range(N):
            if i in best_set:
                continue
            if all(not conflict[i][j] for j in best_set):
                new_val = best_val + 1.0 / nodes[i]
                if new_val > best_val:
                    best_set.add(i)
                    best_val = new_val
                    improved = True
                    break

        if improved:
            continue

        # Try single swap: remove one, add one
        for rem in list(best_set):
            for add in range(N):
                if add in best_set:
                    continue
                # Check if add conflicts with any node in best_set \ {rem}
                ok = True
                for j in best_set:
                    if j == rem:
                        continue
                    if conflict[add][j]:
                        ok = False
                        break
                if ok:
                    new_val = best_val - 1.0 / nodes[rem] + 1.0 / nodes[add]
                    if new_val > best_val + 1e-12:
                        best_set.discard(rem)
                        best_set.add(add)
                        best_val = new_val
                        improved = True
                        break
            if improved:
                break

        if not improved:
            break

    return best_set, best_val


def _polish(nodes, conflict, idx, best_set, best_val, deadline):
    """Polish: try 2-for-1 and 1-for-2 swaps."""
    N = len(nodes)
    best_set = set(best_set)
    best_val = best_val
    t0 = time.time()

    # Try 2-for-1: remove one node, add two
    for rem in list(best_set):
        if time.time() > deadline:
            break
        # Find pairs (a,b) not in best_set, not conflicting with each other or with best_set\{rem}
        candidates = [i for i in range(N) if i not in best_set and not conflict[i][rem]]
        # Filter: must not conflict with any node in best_set \ {rem}
        filtered = []
        for i in candidates:
            ok = True
            for j in best_set:
                if j == rem:
                    continue
                if conflict[i][j]:
                    ok = False
                    break
            if ok:
                filtered.append(i)

        for a_idx in range(len(filtered)):
            if time.time() > deadline:
                break
            a = filtered[a_idx]
            for b_idx in range(a_idx + 1, len(filtered)):
                b = filtered[b_idx]
                if conflict[a][b]:
                    continue
                new_val = best_val - 1.0 / nodes[rem] + 1.0 / nodes[a] + 1.0 / nodes[b]
                if new_val > best_val + 1e-12:
                    best_set.discard(rem)
                    best_set.add(a)
                    best_set.add(b)
                    best_val = new_val
                    break
            if new_val > best_val + 1e-12:
                break

    # Try 1-for-2: remove two nodes, add one
    for rem1 in list(best_set):
        if time.time() > deadline:
            break
        for rem2 in list(best_set):
            if rem2 <= rem1:
                continue
            if time.time() > deadline:
                break
            # Find a node that doesn't conflict with best_set \ {rem1, rem2}
            for add in range(N):
                if add in best_set:
                    continue
                ok = True
                for j in best_set:
                    if j == rem1 or j == rem2:
                        continue
                    if conflict[add][j]:
                        ok = False
                        break
                if ok:
                    new_val = best_val - 1.0 / nodes[rem1] - 1.0 / nodes[rem2] + 1.0 / nodes[add]
                    if new_val > best_val + 1e-12:
                        best_set.discard(rem1)
                        best_set.discard(rem2)
                        best_set.add(add)
                        best_val = new_val
                        break

    return best_set, best_val
# EVOLVE-BLOCK-END
