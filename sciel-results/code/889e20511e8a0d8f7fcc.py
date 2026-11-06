"""Direct approach: find longest simple path in the pinned graph using
depth-first search with strong pruning, plus multi-start greedy and local search."""

import time
import sys

# Pinned graph data from seed
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

N = P['n']
EDGES = P['edges']

# Build adjacency
adj = [set() for _ in range(N)]
for u, v in EDGES:
    adj[u].add(v)
    adj[v].add(u)

DEG = [len(adj[i]) for i in range(N)]

def path_is_valid(path):
    if not path:
        return False
    if len(path) != len(set(path)):
        return False
    for i in range(len(path) - 1):
        if path[i+1] not in adj[path[i]]:
            return False
    return True

def greedy_grow(start):
    """Grow a path greedily using Warnsdorff-like heuristic (least remaining degree)."""
    path = [start]
    seen = {start}
    while True:
        cur = path[-1]
        candidates = adj[cur] - seen
        if not candidates:
            break
        v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
        path.append(v)
        seen.add(v)
    return path

def try_extend_both_ends(path):
    """Extend a path from both ends repeatedly, always choosing the end with the
    most constrained (fewest) candidate to avoid dead-ends."""
    path = list(path)
    seen = set(path)
    changed = True
    while changed:
        changed = False
        cur_l = path[0]
        cands_l = adj[cur_l] - seen
        cur_r = path[-1]
        cands_r = adj[cur_r] - seen
        if cands_l and cands_r:
            # Pick the end with fewer candidates (more constrained)
            if len(cands_l) <= len(cands_r):
                v = min(cands_l, key=lambda w: (len(adj[w] - seen), w))
                path.insert(0, v)
            else:
                v = min(cands_r, key=lambda w: (len(adj[w] - seen), w))
                path.append(v)
            seen.add(v)
            changed = True
        elif cands_l:
            v = min(cands_l, key=lambda w: (len(adj[w] - seen), w))
            path.insert(0, v)
            seen.add(v)
            changed = True
        elif cands_r:
            v = min(cands_r, key=lambda w: (len(adj[w] - seen), w))
            path.append(v)
            seen.add(v)
            changed = True
    return path

def local_search_improve(path, deadline):
    """Improve a path via 2-opt style moves: try reversing segments, swapping
    internal edges, and re-extending. Returns the best path found."""
    path = list(path)
    seen = set(path)
    best = list(path)
    best_len = len(path)
    
    # Try 2-opt: reverse a segment [i..j] and check if the new connections are valid
    n = len(path)
    for i in range(n - 1):
        if time.time() > deadline:
            break
        for j in range(i + 1, n):
            if time.time() > deadline:
                break
            # Check if reversing path[i..j] creates valid edges
            # Before: path[i-1]--path[i] and path[j]--path[j+1]
            # After:  path[i-1]--path[j] and path[i]--path[j+1]
            left_ok = (i == 0) or (path[j] in adj[path[i-1]])
            right_ok = (j == n - 1) or (path[i] in adj[path[j+1]])
            if left_ok and right_ok:
                new_path = path[:i] + path[i:j+1][::-1] + path[j+1:]
                # Check all edges in new_path
                valid = True
                for k in range(len(new_path) - 1):
                    if new_path[k+1] not in adj[new_path[k]]:
                        valid = False
                        break
                if valid:
                    # Try extending
                    ext = try_extend_both_ends(new_path)
                    if len(ext) > best_len:
                        best_len = len(ext)
                        best = ext
    return best

def dfs_longest_path(start, deadline, best_len):
    """
    Depth-first search for longest simple path starting at `start`.
    Uses time-limited search with pruning.
    Returns the best path found.
    """
    best_path = [start]
    best_len_local = 1
    
    path = [start]
    seen = [False] * N
    seen[start] = True
    
    # Sort neighbors for each node to try low-degree first (Warnsdorff)
    neighbors_sorted = [sorted(adj[i], key=lambda w: (len(adj[w]), w)) for i in range(N)]
    
    # Use iterative deepening or just DFS with time limit
    # We'll do a DFS that tries to extend, with backtracking
    
    # To make DFS manageable, we use a recursion limit and time check
    # But Python recursion depth for 100 nodes is fine
    
    def _dfs(cur, depth):
        nonlocal best_path, best_len_local
        if time.time() > deadline:
            return
        
        if depth > best_len_local:
            best_len_local = depth
            best_path = list(path)
        
        # Try extending from current end
        cur = path[-1]
        for nxt in neighbors_sorted[cur]:
            if time.time() > deadline:
                return
            if not seen[nxt]:
                seen[nxt] = True
                path.append(nxt)
                _dfs(nxt, depth + 1)
                path.pop()
                seen[nxt] = False
    
    _dfs(start, 1)
    return best_path

def dfs_from_both_ends(start, deadline):
    """
    Try to find longest path by growing from both ends simultaneously.
    Start with a single vertex, extend both ends greedily, then use DFS on the shorter end.
    """
    # First grow greedily
    path = greedy_grow(start)
    path = try_extend_both_ends(path)
    best_path = list(path)
    best_len = len(path)
    
    # Now try DFS extension from both ends
    # We'll try to extend the left end or right end via DFS
    seen = set(path)
    
    # Try extending left end
    def try_extend_end(end_idx, deadline):
        nonlocal best_path, best_len
        if time.time() > deadline:
            return
        cur = path[end_idx]
        candidates = sorted(adj[cur] - seen, key=lambda w: (len(adj[w] - seen), w))
        for nxt in candidates:
            if time.time() > deadline:
                return
            if end_idx == 0:
                path.insert(0, nxt)
            else:
                path.append(nxt)
            seen.add(nxt)
            
            if len(path) > best_len:
                best_len = len(path)
                best_path = list(path)
            
            # Try to extend further from this new end
            try_extend_end(end_idx, deadline)
            
            if end_idx == 0:
                path.pop(0)
            else:
                path.pop()
            seen.discard(nxt)
    
    # This can be expensive, so limit it
    # Just do one level of extension
    for end_idx in [0, -1]:
        if time.time() > deadline:
            break
        cur = path[end_idx]
        candidates = sorted(adj[cur] - seen, key=lambda w: (len(adj[w] - seen), w))
        for nxt in candidates:
            if time.time() > deadline:
                break
            if end_idx == 0:
                path.insert(0, nxt)
            else:
                path.append(nxt)
            seen.add(nxt)
            
            if len(path) > best_len:
                best_len = len(path)
                best_path = list(path)
            
            # Try one more level
            cur2 = path[end_idx]
            cands2 = sorted(adj[cur2] - seen, key=lambda w: (len(adj[w] - seen), w))
            for nxt2 in cands2:
                if time.time() > deadline:
                    break
                if end_idx == 0:
                    path.insert(0, nxt2)
                else:
                    path.append(nxt2)
                seen.add(nxt2)
                
                if len(path) > best_len:
                    best_len = len(path)
                    best_path = list(path)
                
                # Extend both ends greedily
                extended = try_extend_both_ends(path)
                if len(extended) > best_len:
                    best_len = len(extended)
                    best_path = list(extended)
                
                if end_idx == 0:
                    path.pop(0)
                else:
                    path.pop()
                seen.discard(nxt2)
            
            if end_idx == 0:
                path.pop(0)
            else:
                path.pop()
            seen.discard(nxt)
    
    return best_path

def beam_search_path(start, beam_width, max_depth, deadline):
    """Beam search for longest path starting from start."""
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
            candidates = sorted(adj[cur] - seen, key=lambda w: (len(adj[w] - seen), w))
            for v in candidates:
                new_path = path + [v]
                new_seen = seen | {v}
                new_beam.append((new_path, new_seen))
                if len(new_path) > len(best_path):
                    best_path = list(new_path)
        if not new_beam:
            break
        # Sort by potential: prefer paths where the end has fewer unvisited neighbors
        new_beam.sort(key=lambda x: len(adj[x[0][-1]] - x[1]))
        beam = new_beam[:beam_width]
    
    return best_path

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    best_path = []
    best_len = 0
    
    # Phase 1: Greedy from all starts
    for start in range(N):
        if time.time() > deadline:
            break
        path = greedy_grow(start)
        path = try_extend_both_ends(path)
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
    
    # Phase 2: DFS from the best starting point (with time limit)
    if best_len > 0 and time.time() < deadline - 15:
        # Use the endpoints of the best path as starting points for DFS
        dfs_start = best_path[0]
        dfs_path = dfs_longest_path(dfs_start, min(time.time() + 10, deadline - 15), best_len)
        if len(dfs_path) > best_len:
            best_len = len(dfs_path)
            best_path = list(dfs_path)
        # Also try from the other end
        if time.time() < deadline - 10:
            dfs_start2 = best_path[-1]
            dfs_path2 = dfs_longest_path(dfs_start2, min(time.time() + 8, deadline - 10), best_len)
            if len(dfs_path2) > best_len:
                best_len = len(dfs_path2)
                best_path = list(dfs_path2)
    
    # Phase 3: Beam search from promising starts
    if time.time() < deadline - 10:
        promising_starts = list(set([best_path[0], best_path[-1], 0, 3, 8, 9, 19, 24]))
        for start in promising_starts:
            if time.time() > deadline - 10:
                break
            path = beam_search_path(start, beam_width=50, max_depth=N, deadline=deadline - 10)
            path = try_extend_both_ends(path)
            if len(path) > best_len:
                best_len = len(path)
                best_path = list(path)
    
    # Phase 4: Local search (2-opt) on the best path, then final extension
    if best_len > 0 and time.time() < deadline - 2:
        improved = local_search_improve(best_path, deadline - 2)
        if len(improved) > best_len:
            best_len = len(improved)
            best_path = improved
    
    # Phase 5: Final end-extension
    if best_len > 0 and time.time() < deadline:
        improved = try_extend_both_ends(best_path)
        if len(improved) > best_len:
            best_len = len(improved)
            best_path = improved
    
    # Phase 6: One more round of local search if we have time
    if best_len > 0 and time.time() < deadline - 1:
        improved = local_search_improve(best_path, deadline - 1)
        if len(improved) > best_len:
            best_len = len(improved)
            best_path = improved
        improved = try_extend_both_ends(best_path)
        if len(improved) > best_len:
            best_len = len(improved)
            best_path = improved
    
    # Ensure valid nonempty path
    if not best_path:
        best_path = [0]
    
    if not path_is_valid(best_path):
        for u, v in EDGES:
            best_path = [u, v]
            break
        else:
            best_path = [0]
    
    return best_path
