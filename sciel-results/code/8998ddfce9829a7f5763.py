"""Long simple path in pinned G(100,1.8/100) graph.
Combines: diverse greedy (incl. forced-move) + two-end extension + insert/extend
local improvement + iterative-deepening exact DFS (search for a specific target
length with pruning) to push the path as long as possible within the time budget.
"""
import time
import random

N = 100
EDGES = [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]]

ADJ = [set() for _ in range(N)]
DEG = [0] * N
for u, v in EDGES:
    ADJ[u].add(v)
    ADJ[v].add(u)
    DEG[u] += 1
    DEG[v] += 1


def find_components():
    visited = set()
    comps = []
    for s in range(N):
        if s in visited:
            continue
        comp = set()
        stack = [s]
        while stack:
            u = stack.pop()
            if u in visited:
                continue
            visited.add(u)
            comp.add(u)
            for v in ADJ[u]:
                if v not in visited:
                    stack.append(v)
        comps.append(comp)
    return comps


def greedy_warnsdorff(start, rng):
    seen = {start}
    path = [start]
    while True:
        cand = ADJ[path[-1]] - seen
        if not cand:
            break
        v = min(cand, key=lambda w: (len(ADJ[w] - seen), rng.random()))
        path.append(v)
        seen.add(v)
    return path


def greedy_max_degree(start, rng):
    seen = {start}
    path = [start]
    while True:
        cand = ADJ[path[-1]] - seen
        if not cand:
            break
        v = max(cand, key=lambda w: (DEG[w], rng.random()))
        path.append(v)
        seen.add(v)
    return path


def greedy_random(start, rng):
    seen = {start}
    path = [start]
    while True:
        cand = list(ADJ[path[-1]] - seen)
        if not cand:
            break
        v = rng.choice(cand)
        path.append(v)
        seen.add(v)
    return path


def greedy_mixed(start, rng):
    seen = {start}
    path = [start]
    while True:
        cand = ADJ[path[-1]] - seen
        if not cand:
            break
        w = rng.random() * 2
        v = max(cand, key=lambda x: (-len(ADJ[x] - seen), DEG[x] * w, rng.random()))
        path.append(v)
        seen.add(v)
    return path


def greedy_forced(start, rng):
    """Forced-move: always take the unique unvisited neighbor if one exists;
    otherwise random tie-break. Escapes single-Warnsdorff local minima."""
    seen = {start}
    path = [start]
    while True:
        cand = ADJ[path[-1]] - seen
        if not cand:
            break
        forced = [w for w in cand if len(ADJ[w] - seen) == 1]
        if forced:
            v = forced[0]
        else:
            v = rng.choice(list(cand))
        path.append(v)
        seen.add(v)
    return path


def extend_both_ends(path, seen):
    path = list(path)
    seen = set(seen)
    improved = True
    while improved:
        improved = False
        while True:
            cand = ADJ[path[0]] - seen
            if not cand:
                break
            v = min(cand, key=lambda w: (len(ADJ[w] - seen), w))
            path.insert(0, v)
            seen.add(v)
            improved = True
        while True:
            cand = ADJ[path[-1]] - seen
            if not cand:
                break
            v = min(cand, key=lambda w: (len(ADJ[w] - seen), w))
            path.append(v)
            seen.add(v)
            improved = True
    return path


def local_improve(path, seen, rng, deadline):
    """Insert unvisited vertices into the path / extend both ends."""
    path = list(path)
    seen = set(seen)
    unvisited = set(range(N)) - seen
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for u in list(unvisited):
            found = False
            for i in range(len(path) - 1):
                if path[i] in ADJ[u] and path[i + 1] in ADJ[u]:
                    path.insert(i + 1, u)
                    seen.add(u)
                    unvisited.remove(u)
                    improved = True
                    found = True
                    break
            if not found:
                if path[0] in ADJ[u]:
                    path.insert(0, u)
                    seen.add(u)
                    unvisited.remove(u)
                    improved = True
                elif path[-1] in ADJ[u]:
                    path.append(u)
                    seen.add(u)
                    unvisited.remove(u)
                    improved = True
        while True:
            cand = ADJ[path[0]] - seen
            if not cand:
                break
            v = min(cand, key=lambda w: (len(ADJ[w] - seen), w))
            path.insert(0, v)
            seen.add(v)
            improved = True
        while True:
            cand = ADJ[path[-1]] - seen
            if not cand:
                break
            v = min(cand, key=lambda w: (len(ADJ[w] - seen), w))
            path.append(v)
            seen.add(v)
            improved = True
    return path


def _dfs_target(start, target_len, deadline, node_budget, best_holder):
    """Iterative-deepening DFS: find a simple path of exactly target_len vertices
    starting at `start`. Returns the path list or None. Uses Warnsdorff ordering,
    dead-end pruning and a 'can we still reach the target?' connectivity check."""
    seen = {start}
    path = [start]
    nodes = [0]

    def rec():
        if nodes[0] > node_budget or time.time() > deadline:
            return False
        if len(path) == target_len:
            best_holder.clear()
            best_holder.extend(path)
            return True
        # Not enough vertices left to reach target? (only if we restrict to comp; here global)
        if N - len(path) < target_len - len(path):
            return False
        cand = ADJ[path[-1]] - seen
        if not cand:
            return False
        # Dead-end pruning: avoid moving to a vertex that has no further unvisited
        # neighbor unless it is the final vertex we need.
        remaining = target_len - len(path)
        ordered = sorted(cand, key=lambda w: (len(ADJ[w] - seen), w))
        for v in ordered:
            nodes[0] += 1
            if nodes[0] > node_budget or time.time() > deadline:
                return False
            # If this is not the last step, v must have a continuation.
            if remaining > 1:
                if len(ADJ[v] - seen - {path[-1]}) == 0:
                    continue  # v would be a dead end before target
            path.append(v)
            seen.add(v)
            if rec():
                return True
            path.pop()
            seen.remove(v)
        return False

    return rec()


def construct():
    t0 = time.time()
    deadline = t0 + 85.0  # 90s kill; keep 5s safety margin

    comps = find_components()
    comps.sort(key=len, reverse=True)

    best_path = []
    rng = random.Random(42)

    greedy_fns = [greedy_warnsdorff, greedy_max_degree, greedy_random,
                  greedy_mixed, greedy_forced]

    for comp in comps:
        if time.time() > deadline:
            break
        comp_size = len(comp)
        comp_list = sorted(comp)
        target = comp_size

        # Phase 1: diverse greedy starts from every vertex in the component
        for start in comp_list:
            if time.time() > deadline:
                break
            for func in greedy_fns:
                if time.time() > deadline:
                    break
                p = func(start, rng)
                p = extend_both_ends(p, set(p))
                if len(p) > len(best_path):
                    best_path = p[:]

        # Phase 2: local improvement (insert/extend) on the best
        if best_path and len(best_path) < target and time.time() < deadline:
            p = local_improve(best_path, set(best_path), rng, deadline)
            if len(p) > len(best_path):
                best_path = p

        # Phase 3: iterative-deepening exact DFS to push past current best.
        # Search for target_len = best_len+1, +2, ... up to comp_size.
        if len(best_path) < target:
            for target_len in range(len(best_path) + 1, target + 1):
                if time.time() > deadline:
                    break
                found = False
                # Try a handful of promising starts (endpoints + a few vertices)
                starts = [best_path[0], best_path[-1]] + comp_list[:12]
                starts = list(dict.fromkeys(starts))
                for start in starts:
                    if time.time() > deadline:
                        break
                    remaining = deadline - time.time()
                    if remaining < 0.2:
                        break
                    # Node budget scales with remaining time (cap to avoid blowup)
                    node_budget = int(min(200000, remaining * 6000))
                    holder = []
                    if _dfs_target(start, target_len, deadline, node_budget, holder):
                        best_path = holder[:]
                        found = True
                        # Polish the new best
                        if len(best_path) < target and time.time() < deadline:
                            p2 = local_improve(best_path, set(best_path), rng, deadline)
                            if len(p2) > len(best_path):
                                best_path = p2
                        break
                if not found:
                    # If we cannot find a path of this length, longer ones are
                    # at least as hard; stop deepening.
                    break
                if len(best_path) >= target:
                    break

    if not best_path:
        best_path = [0]

    # Verify path validity
    for i in range(len(best_path) - 1):
        if best_path[i + 1] not in ADJ[best_path[i]]:
            return [0]
    if len(set(best_path)) != len(best_path):
        seen = set()
        fixed = []
        for v in best_path:
            if v not in seen:
                fixed.append(v)
                seen.add(v)
        return fixed

    return best_path
