"""
Longest simple path in a pinned G(100, 0.018) graph.
Uses a multi-phase approach:
  1. Greedy + Warnsdorff from all starts
  2. Iterative-deepening DFS with strong pruning (connectivity-based)
  3. 2-opt / segment-reversal local search
  4. Path stitching
  5. Final extension passes

The key improvement over the seed: use a proper connectivity check (BFS/DFS on
remaining graph) to prune branches where the unvisited subgraph becomes
disconnected in a way that prevents a longer path.  Also add an iterative-
deepening DFS that targets a specific length, which is much more effective
than a single unbounded DFS on sparse graphs.
"""

import time
import sys
from collections import deque

# ---- Pinned graph (same as seed) ----
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

N = P['n']
EDGES = P['edges']

# Build adjacency (use list-of-sets for fast membership, and list-of-lists for iteration)
adj_set = [set() for _ in range(N)]
for u, v in EDGES:
    adj_set[u].add(v)
    adj_set[v].add(u)

adj = [sorted(adj_set[i]) for i in range(N)]
DEG = [len(adj[i]) for i in range(N)]

# Pre-sort neighbours by degree (ascending) for Warnsdorff heuristic
adj_sorted = [sorted(adj[i], key=lambda w: (DEG[w], w)) for i in range(N)]

# Bitmask helpers: N=100, use Python integers as bitsets
def bit(i):
    return 1 << i

def is_set(mask, i):
    return (mask >> i) & 1

def set_bit(mask, i):
    return mask | (1 << i)

def clear_bit(mask, i):
    return mask & ~(1 << i)

def popcount(mask):
    return bin(mask).count('1')

# Precompute adjacency bitmask
adj_mask = [0] * N
for i in range(N):
    m = 0
    for j in adj[i]:
        m |= bit(j)
    adj_mask[i] = m

def path_is_valid(path):
    if not path:
        return False
    if len(path) != len(set(path)):
        return False
    for i in range(len(path) - 1):
        if path[i + 1] not in adj_set[path[i]]:
            return False
    return True

def greedy_grow(start):
    """Grow a path greedily using Warnsdorff-like heuristic."""
    path = [start]
    seen = {start}
    while True:
        cur = path[-1]
        candidates = adj_set[cur] - seen
        if not candidates:
            break
        v = min(candidates, key=lambda w: (len(adj_set[w] - seen), DEG[w], w))
        path.append(v)
        seen.add(v)
    return path

def try_extend_both_ends(path):
    """Extend a path from both ends repeatedly."""
    path = list(path)
    seen = set(path)
    changed = True
    while changed:
        changed = False
        cur_l = path[0]
        cands_l = adj_set[cur_l] - seen
        cur_r = path[-1]
        cands_r = adj_set[cur_r] - seen
        if cands_l and cands_r:
            if len(cands_l) <= len(cands_r):
                v = min(cands_l, key=lambda w: (len(adj_set[w] - seen), DEG[w], w))
                path.insert(0, v)
            else:
                v = min(cands_r, key=lambda w: (len(adj_set[w] - seen), DEG[w], w))
                path.append(v)
            seen.add(v)
            changed = True
        elif cands_l:
            v = min(cands_l, key=lambda w: (len(adj_set[w] - seen), DEG[w], w))
            path.insert(0, v)
            seen.add(v)
            changed = True
        elif cands_r:
            v = min(cands_r, key=lambda w: (len(adj_set[w] - seen), DEG[w], w))
            path.append(v)
            seen.add(v)
            changed = True
    return path

def connectivity_prune(path, seen_mask, deadline):
    """
    Check whether the unvisited subgraph (induced by vertices not in seen_mask)
    is connected, or at least whether the endpoints of the current path can
    reach all unvisited vertices.  If the unvisited graph is split into
    components and one of them is "trapped" (not reachable from either end),
    we prune.
    Returns True if pruning is safe to continue (i.e., no forced trap).
    """
    if time.time() > deadline:
        return True  # don't waste time on pruning

    unvisited = [v for v in range(N) if not is_set(seen_mask, v)]
    if len(unvisited) <= 1:
        return True

    # BFS from the two ends of the path through unvisited vertices
    # Actually, we need to check: can we still visit all unvisited vertices
    # in a single path extension?  A sufficient (but not necessary) condition:
    # the unvisited subgraph must be connected, OR each component must be
    # adjacent to at least one endpoint of the current path.
    # For speed, just check if unvisited subgraph is connected.
    
    # Build unvisited set as a set for fast lookup
    uv_set = set(unvisited)
    if not uv_set:
        return True
    
    # BFS from first unvisited vertex
    start_v = unvisited[0]
    visited = {start_v}
    queue = deque([start_v])
    while queue:
        u = queue.popleft()
        for w in adj[u]:
            if w in uv_set and w not in visited:
                visited.add(w)
                queue.append(w)
    
    if len(visited) < len(unvisited):
        # Disconnected: check if each component touches an endpoint
        # Find all components
        remaining = uv_set - visited
        while remaining:
            comp_start = next(iter(remaining))
            comp = {comp_start}
            q2 = deque([comp_start])
            while q2:
                u = q2.popleft()
                for w in adj[u]:
                    if w in remaining and w not in comp:
                        comp.add(w)
                        q2.append(w)
            remaining -= comp
            # Check if this component is adjacent to path[0] or path[-1]
            touches_end = False
            for v in comp:
                if path[0] in adj_set[v] or path[-1] in adj_set[v]:
                    touches_end = True
                    break
            if not touches_end:
                return False  # trapped component -> prune
    return True

def dfs_targeted(start, target_len, deadline, best_path, best_len):
    """
    DFS trying to find a path of exactly target_len starting at `start`.
    Uses Warnsdorff ordering and connectivity pruning.
    Returns (found, best_path, best_len).
    """
    found = False
    path = [start]
    seen_mask = bit(start)

    def _dfs(cur, depth):
        nonlocal found, best_path, best_len
        if time.time() > deadline:
            return
        if depth > best_len:
            best_len = depth
            best_path = list(path)
        if depth == target_len:
            found = True
            return
        # Prune: can't reach target
        remaining = N - depth
        if depth + remaining < target_len:
            return
        # Connectivity prune (only do it when remaining is small enough to matter)
        if remaining > 1 and remaining <= 30:
            if not connectivity_prune(path, seen_mask, deadline):
                return
        # Try neighbours in Warnsdorff order
        for nxt in adj_sorted[cur]:
            if time.time() > deadline:
                return
            if is_set(seen_mask, nxt):
                continue
            # Additional prune: if nxt is a leaf in the remaining graph and
            # we still need more vertices, skip it (it would be the last vertex)
            unvisited_after = 0
            for nb in adj[nxt]:
                if not is_set(seen_mask, nb) and nb != cur:
                    unvisited_after += 1
            if unvisited_after == 0 and depth + 1 < target_len:
                continue  # would be a dead end before target

            seen_mask = set_bit(seen_mask, nxt)
            path.append(nxt)
            _dfs(nxt, depth + 1)
            path.pop()
            seen_mask = clear_bit(seen_mask, nxt)
            if found:
                return

    _dfs(start, 1)
    return found, best_path, best_len

def iterative_deepening_dfs(start, min_len, max_len, deadline, best_path, best_len):
    """
    Iterative deepening: try to find a path of length L for L = max_len, max_len-1, ...
    Stop early if we find one.  Returns (best_path, best_len).
    """
    for L in range(max_len, min_len - 1, -1):
        if time.time() > deadline:
            break
        found, best_path, best_len = dfs_targeted(start, L, deadline, best_path, best_len)
        if found:
            break
    return best_path, best_len

def local_search_improve(path, deadline):
    """
    2-opt: reverse segments to expose new extension opportunities.
    Also try "1-opt" (remove internal vertex if its neighbours are connected).
    """
    path = list(path)
    best = list(path)
    best_len = len(path)
    improved = True
    max_rounds = 15

    while improved and time.time() < deadline:
        improved = False
        n = len(best)

        # 2-opt: reverse segment [i..j]
        for i in range(n - 1):
            if time.time() > deadline:
                return best
            for j in range(i + 1, n):
                if time.time() > deadline:
                    return best
                left_ok = (i == 0) or (best[j] in adj_set[best[i - 1]])
                right_ok = (j == n - 1) or (best[i] in adj_set[best[j + 1]])
                if left_ok and right_ok:
                    new_path = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                    # Quick validity
                    valid = True
                    for k in range(len(new_path) - 1):
                        if new_path[k + 1] not in adj_set[new_path[k]]:
                            valid = False
                            break
                    if valid:
                        ext = try_extend_both_ends(new_path)
                        if len(ext) > best_len:
                            best_len = len(ext)
                            best = ext
                            improved = True

        # 1-opt: remove internal vertex
        n = len(best)
        for i in range(1, n - 1):
            if time.time() > deadline:
                return best
            if best[i + 1] in adj_set[best[i - 1]]:
                new_path = best[:i] + best[i + 1:]
                valid = True
                for k in range(len(new_path) - 1):
                    if new_path[k + 1] not in adj_set[new_path[k]]:
                        valid = False
                        break
                if valid:
                    ext = try_extend_both_ends(new_path)
                    if len(ext) > best_len:
                        best_len = len(ext)
                        best = ext
                        improved = True

    return best

def stitch_paths(path1, path2, deadline):
    """Try to stitch two paths into one longer path."""
    best = max(path1, path2, key=len)
    best_len = len(best)

    # Direct end-to-end
    if path1[-1] in adj_set[path2[0]]:
        c = path1 + path2[1:]
        if len(set(c)) == len(c) and len(c) > best_len:
            best, best_len = c, len(c)
    if path2[-1] in adj_set[path1[0]]:
        c = path2 + path1[1:]
        if len(set(c)) == len(c) and len(c) > best_len:
            best, best_len = c, len(c)

    # Insert path1 into a gap in path2
    for i in range(len(path2) - 1):
        if time.time() > deadline:
            break
        if path1[0] in adj_set[path2[i]] and path1[-1] in adj_set[path2[i + 1]]:
            c = path2[:i + 1] + path1 + path2[i + 1:]
            if len(set(c)) == len(c) and len(c) > best_len:
                best, best_len = c, len(c)
        if path1[-1] in adj_set[path2[i]] and path1[0] in adj_set[path2[i + 1]]:
            c = path2[:i + 1] + path1[::-1] + path2[i + 1:]
            if len(set(c)) == len(c) and len(c) > best_len:
                best, best_len = c, len(c)

    # Insert path2 into a gap in path1
    for i in range(len(path1) - 1):
        if time.time() > deadline:
            break
        if path2[0] in adj_set[path1[i]] and path2[-1] in adj_set[path1[i + 1]]:
            c = path1[:i + 1] + path2 + path1[i + 1:]
            if len(set(c)) == len(c) and len(c) > best_len:
                best, best_len = c, len(c)
        if path2[-1] in adj_set[path1[i]] and path2[0] in adj_set[path1[i + 1]]:
            c = path1[:i + 1] + path2[::-1] + path1[i + 1:]
            if len(set(c)) == len(c) and len(c) > best_len:
                best, best_len = c, len(c)

    return best

def beam_search_path(start, beam_width, max_depth, deadline):
    """Beam search for longest path."""
    beam = [([start], {start})]
    best_path = [start]

    for depth in range(max_depth):
        if time.time() > deadline:
            break
        if not beam:
            break
        new_beam = []
        for path, seen in beam:
            cur = path[-1]
            candidates = sorted(adj_set[cur] - seen, key=lambda w: (len(adj_set[w] - seen), w))
            for v in candidates:
                new_path = path + [v]
                new_seen = seen | {v}
                new_beam.append((new_path, new_seen))
                if len(new_path) > len(best_path):
                    best_path = list(new_path)
        if not new_beam:
            break
        new_beam.sort(key=lambda x: len(adj_set[x[0][-1]] - x[1]))
        beam = new_beam[:beam_width]

    return best_path

def construct():
    start_time = time.time()
    # Safety margin: evaluator kills at 90s; use 80s as our deadline
    deadline = start_time + 80

    best_path = []
    best_len = 0
    all_paths = []

    # ---- Phase 1: Greedy from all starts ----
    for s in range(N):
        if time.time() > deadline - 30:
            break
        p = greedy_grow(s)
        p = try_extend_both_ends(p)
        if len(p) > best_len:
            best_len = len(p)
            best_path = list(p)
        if len(p) >= max(1, best_len - 5):
            all_paths.append(list(p))

    # ---- Phase 2: Iterative-deepening DFS from promising starts ----
    # Target: find a path longer than current best
    if best_len > 0 and time.time() < deadline - 15:
        # Pick diverse starting points: endpoints of best path, high-degree nodes
        dfs_starts = list(set([best_path[0], best_path[-1]]))
        high_deg = sorted(range(N), key=lambda x: -DEG[x])[:8]
        dfs_starts.extend(high_deg)
        dfs_starts = list(set(dfs_starts))

        # Also add some low-degree nodes (leaves in the graph) as they might
        # be endpoints of a long path
        low_deg = sorted(range(N), key=lambda x: DEG[x])[:5]
        dfs_starts.extend(low_deg)
        dfs_starts = list(set(dfs_starts))

        remaining_time = deadline - 15 - time.time()
        per_start = max(1.5, remaining_time / max(1, len(dfs_starts)))

        for s in dfs_starts:
            if time.time() > deadline - 15:
                break
            # Try to find a path of length best_len + 1, then best_len + 2, etc.
            # But iterative deepening from high target down is more efficient
            target_max = min(N, best_len + 10)
            target_min = max(best_len + 1, 5)
            if target_min <= target_max:
                p, bl = iterative_deepening_dfs(s, target_min, target_max,
                                                 min(time.time() + per_start, deadline - 15),
                                                 best_path, best_len)
                if bl > best_len:
                    best_len = bl
                    best_path = list(p)
                    all_paths.append(list(p))

    # ---- Phase 3: Beam search from promising starts ----
    if time.time() < deadline - 10:
        promising = list(set([best_path[0], best_path[-1]]))
        promising.extend(sorted(range(N), key=lambda x: -DEG[x])[:6])
        promising.extend(sorted(range(N), key=lambda x: DEG[x])[:4])
        promising = list(set(promising))
        for s in promising:
            if time.time() > deadline - 10:
                break
            p = beam_search_path(s, beam_width=100, max_depth=N, deadline=deadline - 10)
            p = try_extend_both_ends(p)
            if len(p) > best_len:
                best_len = len(p)
                best_path = list(p)
            if len(p) >= max(1, best_len - 5):
                all_paths.append(list(p))

    # ---- Phase 4: 2-opt local search ----
    if best_len > 0 and time.time() < deadline - 5:
        improved = local_search_improve(best_path, deadline - 5)
        if len(improved) > best_len:
            best_len = len(improved)
            best_path = improved
        best_path = try_extend_both_ends(best_path)
        if len(best_path) > best_len:
            best_len = len(best_path)

    # ---- Phase 5: Stitch paths ----
    if len(all_paths) > 1 and time.time() < deadline - 5:
        # Deduplicate
        seen_set = set()
        unique_paths = []
        for p in all_paths:
            key = tuple(p)
            if key not in seen_set:
                seen_set.add(key)
                unique_paths.append(p)
        # Also add reversed versions
        for p in list(unique_paths):
            rev = p[::-1]
            key = tuple(rev)
            if key not in seen_set:
                seen_set.add(key)
                unique_paths.append(rev)

        for i in range(len(unique_paths)):
            if time.time() > deadline - 5:
                break
            for j in range(i + 1, len(unique_paths)):
                if time.time() > deadline - 5:
                    break
                c = stitch_paths(unique_paths[i], unique_paths[j], deadline - 5)
                if len(c) > best_len:
                    best_len = len(c)
                    best_path = c

    # ---- Phase 6: Another round of local search + extension ----
    if best_len > 0 and time.time() < deadline - 2:
        improved = local_search_improve(best_path, deadline - 2)
        if len(improved) > best_len:
            best_len = len(improved)
            best_path = improved
        best_path = try_extend_both_ends(best_path)
        if len(best_path) > best_len:
            best_len = len(best_path)

    # ---- Phase 7: Final targeted DFS to squeeze out one more vertex ----
    if best_len < N and time.time() < deadline:
        # Try to extend by 1 from the current best path
        # Remove the last vertex and try a different choice
        if len(best_path) >= 2:
            # Try removing last vertex and exploring alternatives
            truncated = best_path[:-1]
            seen_t = set(truncated)
            cur = truncated[-1]
            alts = sorted(adj_set[cur] - seen_t, key=lambda w: (len(adj_set[w] - seen_t), w))
            for a in alts:
                if time.time() > deadline:
                    break
                cand = truncated + [a]
                cand = try_extend_both_ends(cand)
                if len(cand) > best_len:
                    best_len = len(cand)
                    best_path = cand

    # ---- Final validation ----
    if not best_path:
        best_path = [0]
    if not path_is_valid(best_path):
        for u, v in EDGES:
            best_path = [u, v]
            break
        else:
            best_path = [0]

    return best_path
