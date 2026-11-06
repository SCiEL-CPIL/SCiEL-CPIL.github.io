"""Multi-phase longest-path search for the pinned G(100, 1.8/100) graph."""
import time
import random

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

N = P['n']
# Build adjacency
_adj = [set() for _ in range(N)]
for u, v in P['edges']:
    _adj[u].add(v)
    _adj[v].add(u)
# Precompute degree
_deg = [len(_adj[i]) for i in range(N)]

def _greedy_extend(path, seen, heuristic='min_deg'):
    """Extend path greedily from the end. Returns new path (list) and seen (set)."""
    while True:
        last = path[-1]
        cands = _adj[last] - seen
        if not cands:
            break
        if heuristic == 'min_deg':
            v = min(cands, key=lambda w: (_deg[w], w))
        elif heuristic == 'min_unseen':
            v = min(cands, key=lambda w: (len(_adj[w] - seen), w))
        elif heuristic == 'max_deg':
            v = max(cands, key=lambda w: (_deg[w], w))
        else:  # random
            v = random.Random(hash((last, tuple(sorted(seen))[:5]))).choice(sorted(cands))
        path.append(v)
        seen.add(v)
    return path, seen

def _extend_both_ends(path):
    """Try to extend a path from both ends repeatedly."""
    seen = set(path)
    changed = True
    while changed:
        changed = False
        # Extend from start
        while True:
            first = path[0]
            cands = _adj[first] - seen
            if not cands:
                break
            v = min(cands, key=lambda w: (_deg[w], w))
            path.insert(0, v)
            seen.add(v)
            changed = True
        # Extend from end
        while True:
            last = path[-1]
            cands = _adj[last] - seen
            if not cands:
                break
            v = min(cands, key=lambda w: (_deg[w], w))
            path.append(v)
            seen.add(v)
            changed = True
    return path

def _two_opt(path):
    """Try 2-opt reversals to find a longer valid path (rearrange internal segment)."""
    n = len(path)
    if n < 4:
        return path
    best = path[:]
    best_len = n
    # Try reversing a subpath [i..j] and check if it's still a valid path
    for i in range(1, n - 2):
        for j in range(i + 1, n - 1):
            # Reverse path[i..j]
            new_path = path[:i] + path[i:j+1][::-1] + path[j+1:]
            # Check validity: all consecutive pairs must be edges
            valid = True
            for k in range(len(new_path) - 1):
                if new_path[k+1] not in _adj[new_path[k]]:
                    valid = False
                    break
            if valid and len(new_path) > best_len:
                best = new_path
                best_len = len(new_path)
    return best

def _local_search(path):
    """Try to improve path by swapping/replacing vertices."""
    seen = set(path)
    n = len(path)
    improved = True
    while improved:
        improved = False
        # Try inserting a new vertex between consecutive pairs
        for i in range(n - 1):
            a, b = path[i], path[i+1]
            # Find a vertex w not in seen that connects to both a and b
            for w in _adj[a]:
                if w not in seen and w in _adj[b]:
                    new_path = path[:i+1] + [w] + path[i+1:]
                    # Validate
                    valid = True
                    for k in range(len(new_path) - 1):
                        if new_path[k+1] not in _adj[new_path[k]]:
                            valid = False
                            break
                    if valid:
                        path = new_path
                        seen.add(w)
                        n = len(path)
                        improved = True
                        break
            if improved:
                break
        if not improved:
            # Try removing a vertex and reinserting elsewhere
            for i in range(1, n - 1):
                w = path[i]
                a, b = path[i-1], path[i+1]
                # Remove w
                trial = path[:i] + path[i+1:]
                # Check if a-b is an edge (if not, we need w to stay)
                if b not in _adj[a]:
                    continue
                # Try to reinsert w elsewhere
                for j in range(len(trial) - 1):
                    x, y = trial[j], trial[j+1]
                    if w in _adj[x] and w in _adj[y] and w not in trial:
                        new_path = trial[:j+1] + [w] + trial[j+1:]
                        valid = True
                        for k in range(len(new_path) - 1):
                            if new_path[k+1] not in _adj[new_path[k]]:
                                valid = False
                                break
                        if valid and len(new_path) > len(path):
                            path = new_path
                            seen = set(path)
                            n = len(path)
                            improved = True
                            break
                if improved:
                    break
    return path

def _combine_paths(path1, path2):
    """Try to combine two paths into a longer path."""
    seen1 = set(path1)
    seen2 = set(path2)
    if seen1 & seen2:
        return None
    # Try to connect end of path1 to start of path2
    # or end of path1 to end of path2, etc.
    for end1 in [path1[0], path1[-1]]:
        for end2 in [path2[0], path2[-1]]:
            if end2 in _adj[end1]:
                # Merge
                p1 = path1 if path1[-1] == end1 else path1[::-1]
                p2 = path2 if path2[0] == end2 else path2[::-1]
                return p1 + p2[1:]
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    rng = random.Random(748)

    best_path = []

    # Phase 1: Greedy from all starts with multiple heuristics
    for heuristic in ['min_deg', 'min_unseen', 'max_deg']:
        if time.time() > deadline:
            break
        for s in range(N):
            if time.time() > deadline:
                break
            path = [s]
            seen = {s}
            path, seen = _greedy_extend(path, seen, heuristic)
            if len(path) > len(best_path):
                best_path = path[:]

    # Phase 2: Extend both ends of best path
    if best_path and time.time() < deadline:
        best_path = _extend_both_ends(best_path[:])

    # Phase 3: Local search (insert new vertices)
    if best_path and time.time() < deadline:
        best_path = _local_search(best_path[:])

    # Phase 4: 2-opt rearrangement
    if best_path and time.time() < deadline:
        best_path = _two_opt(best_path[:])

    # Phase 5: Multiple greedy passes with different random seeds, keep best
    if time.time() < deadline:
        for trial in range(200):
            if time.time() > deadline:
                break
            for s in range(N):
                if time.time() > deadline:
                    break
                path = [s]
                seen = {s}
                # Randomized greedy
                while True:
                    last = path[-1]
                    cands = _adj[last] - seen
                    if not cands:
                        break
                    v = rng.choice(sorted(cands))
                    path.append(v)
                    seen.add(v)
                if len(path) > len(best_path):
                    best_path = path[:]
                    # Extend both ends
                    best_path = _extend_both_ends(best_path[:])

    # Phase 6: Try to combine two best sub-paths
    if time.time() < deadline:
        # Generate several candidate paths
        candidates = [best_path[:]]
        for s in range(0, N, 5):
            if time.time() > deadline:
                break
            path = [s]
            seen = {s}
            path, seen = _greedy_extend(path, seen, 'min_deg')
            path = _extend_both_ends(path[:])
            if path not in candidates:
                candidates.append(path[:])
        # Try combining
        for i in range(len(candidates)):
            if time.time() > deadline:
                break
            for j in range(i + 1, len(candidates)):
                if time.time() > deadline:
                    break
                merged = _combine_paths(candidates[i], candidates[j])
                if merged and len(merged) > len(best_path):
                    best_path = merged[:]
                    best_path = _local_search(best_path[:])

    # Final: ensure best_path is valid
    if not best_path:
        # Fallback: single edge
        best_path = [P['edges'][0][0], P['edges'][0][1]]

    # Validate
    for i in range(len(best_path) - 1):
        assert best_path[i+1] in _adj[best_path[i]], f"Invalid edge at {i}"
    assert len(set(best_path)) == len(best_path), "Duplicate vertices"

    return best_path
