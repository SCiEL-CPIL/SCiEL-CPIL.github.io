"""Focused Branch-and-Bound DFS with Connectivity Pruning for Longest Path."""

import time
import sys
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.sparse.csgraph import connected_components

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

# Precompute sorted neighbors by degree (Warnsdorff ordering)
NEIGHBORS_SORTED = [sorted(adj[i], key=lambda w: (DEG[w], w)) for i in range(N)]


def path_is_valid(path):
    if not path:
        return False
    if len(path) != len(set(path)):
        return False
    for i in range(len(path) - 1):
        if path[i + 1] not in adj[path[i]]:
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
        v = min(candidates, key=lambda w: (len(adj[w] - seen), DEG[w], w))
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
            if len(cands_l) <= len(cands_r):
                v = min(cands_l, key=lambda w: (len(adj[w] - seen), DEG[w], w))
                path.insert(0, v)
            else:
                v = min(cands_r, key=lambda w: (len(adj[w] - seen), DEG[w], w))
                path.append(v)
            seen.add(v)
            changed = True
        elif cands_l:
            v = min(cands_l, key=lambda w: (len(adj[w] - seen), DEG[w], w))
            path.insert(0, v)
            seen.add(v)
            changed = True
        elif cands_r:
            v = min(cands_r, key=lambda w: (len(adj[w] - seen), DEG[w], w))
            path.append(v)
            seen.add(v)
            changed = True
    return path


def check_connectivity(unvisited_mask, current_node):
    """
    Check if all unvisited nodes are in the same connected component as current_node
    in the graph induced by unvisited nodes + current_node.
    If not, moving to a neighbor that separates them is bad.
    """
    # Build a small adjacency for unvisited + current
    # We can use BFS from current_node on the unvisited subgraph
    # Note: current_node is NOT in unvisited_mask, but it is the entry point.
    
    if not any(unvisited_mask):
        return True

    # BFS from current_node
    visited = set()
    stack = [current_node]
    visited.add(current_node)
    
    while stack:
        node = stack.pop()
        for nb in adj[node]:
            if nb == current_node:
                continue
            if nb in visited:
                continue
            if unvisited_mask[nb]:
                visited.add(nb)
                stack.append(nb)
    
    # Check if all unvisited nodes are in visited
    for i in range(N):
        if unvisited_mask[i] and i not in visited:
            return False
    return True


def dfs_longest_path(start, deadline, best_len):
    """
    Depth-first search for longest simple path starting at `start`.
    Uses time-limited search with strong connectivity pruning.
    Returns the best path found.
    """
    best_path = [start]
    best_len_local = max(1, best_len)

    path = [start]
    seen = [False] * N
    seen[start] = True
    unvisited_mask = [True] * N
    unvisited_mask[start] = False
    unvisited_count = N - 1

    def _dfs(cur, depth):
        nonlocal best_path, best_len_local, unvisited_count

        if time.time() > deadline:
            return

        if depth > best_len_local:
            best_len_local = depth
            best_path = list(path)

        # Pruning: if we can't beat best even visiting all remaining
        if depth + unvisited_count <= best_len_local:
            return

        # Get candidates
        candidates = []
        for nxt in NEIGHBORS_SORTED[cur]:
            if not seen[nxt]:
                candidates.append(nxt)
        
        if not candidates:
            return

        # Sort candidates by Warnsdorff (fewest unvisited neighbors)
        # We need to count unvisited neighbors for each candidate
        def count_unvisited_neighbors(node):
            count = 0
            for nb in adj[node]:
                if not seen[nb] and nb != cur: # cur is seen, but we are looking at unvisited
                    if unvisited_mask[nb]:
                        count += 1
            return count

        candidates.sort(key=lambda x: count_unvisited_neighbors(x))

        for nxt in candidates:
            if time.time() > deadline:
                return
            
            # Connectivity Pruning:
            # If we move to nxt, will the remaining unvisited nodes still be connected to nxt?
            # We simulate: nxt becomes part of path (seen).
            # The "entry" to the unvisited component is now nxt.
            # We need to check if all *other* unvisited nodes are reachable from nxt via unvisited nodes.
            
            # Optimization: Only perform expensive check if unvisited_count > 0
            if unvisited_count > 0:
                # Temporarily mark nxt as seen to check connectivity of the rest
                # Actually, we just need to check if the unvisited nodes (excluding nxt) are connected to nxt.
                # Let's do a quick check.
                
                # If nxt has no unvisited neighbors (other than cur which is seen), it's a leaf.
                # If it's a leaf, we can only go there if it's the last step or we don't care about the rest (but we do).
                # If unvisited_count > 1, and nxt is a leaf in the unvisited graph, we cannot visit it and then visit others.
                # So we should prune if count_unvisited_neighbors(nxt) == 0 and unvisited_count > 1.
                
                unv_nb_count = 0
                for nb in adj[nxt]:
                    if unvisited_mask[nb] and nb != cur:
                        unv_nb_count += 1
                
                if unv_nb_count == 0 and unvisited_count > 1:
                    # Dead end, but we have more nodes to visit. Prune.
                    continue

                # Full connectivity check (expensive, so maybe do it less often or optimize)
                # For N=100, BFS is fast.
                # Let's try the BFS check.
                
                # Save state? No, we can just run BFS.
                # BFS from nxt on unvisited_mask (excluding nxt itself, but including its unvisited neighbors)
                # Actually, we want to know if all unvisited nodes (except nxt) are reachable from nxt.
                
                # Create a temporary mask? Or just use a set.
                # Let's use a set for speed in Python for small graphs.
                reachable = set()
                stack = [nxt]
                reachable.add(nxt)
                
                while stack:
                    node = stack.pop()
                    for nb in adj[node]:
                        if nb in reachable:
                            continue
                        if unvisited_mask[nb]:
                            reachable.add(nb)
                            stack.append(nb)
                
                # Check if all unvisited nodes (except nxt) are in reachable
                # Note: nxt is in reachable.
                # We need to ensure no unvisited node is isolated from nxt.
                
                # Instead of iterating all N, we can check the size.
                # The number of nodes in `reachable` that are unvisited (including nxt) should be unvisited_count.
                # Wait, `reachable` includes `nxt` and any unvisited neighbors.
                # `unvisited_count` is the count of nodes with `unvisited_mask[i] == True`.
                # `nxt` is currently `unvisited_mask[nxt] == True`.
                # So the size of `reachable` (which only contains unvisited nodes and nxt) should be `unvisited_count`.
                
                if len(reachable) < unvisited_count:
                    # Some unvisited nodes are not reachable from nxt.
                    # This means moving to nxt would disconnect the graph.
                    # We cannot complete a path visiting all nodes.
                    # Since we want the *longest* path, and we are pruning based on "can we beat best",
                    # if this move disconnects, we can't visit the disconnected part.
                    # So the max possible length from this branch is depth + 1 + (size of connected component).
                    # If that is <= best_len, prune.
                    # But even simpler: if we disconnect, we can't visit ALL unvisited.
                    # If we just want to find *a* long path, we might still want to go to nxt if the connected component is large enough.
                    # However, standard "longest path" algorithms often prune if connectivity is lost, assuming we want to visit as many as possible.
                    # Let's refine: If len(reachable) + depth <= best_len, prune.
                    if depth + len(reachable) <= best_len_local:
                        continue
                    # Otherwise, we might still find a good path in the reachable component.
                    # But wait, if we disconnect, we lose the other part.
                    # So the max we can get is depth + len(reachable).
                    # If that doesn't beat best, prune.
                    # We already checked `depth + unvisited_count <= best_len` at the start.
                    # Here `len(reachable) < unvisited_count`.
                    # So `depth + len(reachable) < depth + unvisited_count`.
                    # It might still be > best_len.
                    # So we allow it, but we know we can't visit the rest.
                    # This effectively reduces the search space.
                    pass # Don't prune if it might still beat best, but it's a "dead end" for full coverage.

            seen[nxt] = True
            unvisited_mask[nxt] = False
            path.append(nxt)
            unvisited_count -= 1
            _dfs(nxt, depth + 1)
            path.pop()
            unvisited_mask[nxt] = True
            seen[nxt] = False
            unvisited_count += 1

    _dfs(start, 1)
    return best_path


def construct():
    start_time = time.time()
    deadline = start_time + 85

    best_path = []
    best_len = 0

    # Phase 1: Get a strong lower bound
    # Try greedy from all nodes
    for start in range(N):
        path = greedy_grow(start)
        path = try_extend_both_ends(path)
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
            
    # Also try a few random starts or high degree starts to diversify
    high_deg = sorted(range(N), key=lambda x: -DEG[x])[:10]
    for start in high_deg:
        path = greedy_grow(start)
        path = try_extend_both_ends(path)
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)

    # Phase 2: DFS with Pruning
    # Start DFS from the endpoints of the best greedy path, and a few other promising nodes
    dfs_starts = [best_path[0], best_path[-1]]
    # Add high degree nodes
    dfs_starts.extend(high_deg[:5])
    # Add degree 1 nodes (must be endpoints)
    for v in range(N):
        if DEG[v] == 1:
            dfs_starts.append(v)
            
    dfs_starts = list(set(dfs_starts))
    
    # Allocate time for DFS
    time_elapsed = time.time() - start_time
    remaining_time = 85 - time_elapsed
    
    # Use 80% of remaining time for DFS, leaving 20% for final polish
    dfs_deadline = start_time + 85 * 0.8
    
    # Try DFS from each start
    for start in dfs_starts:
        if time.time() > dfs_deadline:
            break
        # Give each start a fair share, but allow more if it's promising?
        # Just use the global deadline.
        dfs_path = dfs_longest_path(start, dfs_deadline, best_len)
        if len(dfs_path) > best_len:
            best_len = len(dfs_path)
            best_path = list(dfs_path)
            
    # Phase 3: Final Polish
    if best_len > 0 and time.time() < deadline:
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
