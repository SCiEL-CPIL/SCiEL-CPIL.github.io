"""
Combined strategy: Skeleton-based DFS (Chain Compression) with Backtracking + Refinement.
This structural change compresses degree-2 chains into weighted edges, reducing the search space.
"""
import time
import random
import sys

# Increase recursion limit for DFS
sys.setrecursionlimit(10000)

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

N = P['n']
_adj = [set() for _ in range(N)]
for u, v in P['edges']:
    _adj[u].add(v)
    _adj[v].add(u)

# Precompute degrees and neighbor lists
_deg = [len(_adj[i]) for i in range(N)]
_nbrs = [list(_adj[i]) for i in range(N)]

# Find connected components
def find_components():
    visited = [False] * N
    comps = []
    for i in range(N):
        if not visited[i]:
            comp = []
            stack = [i]
            visited[i] = True
            while stack:
                node = stack.pop()
                comp.append(node)
                for nb in _adj[node]:
                    if not visited[nb]:
                        visited[nb] = True
                        stack.append(nb)
            comps.append(comp)
    return comps

_components = find_components()
# Sort by size, largest first
_components.sort(key=len, reverse=True)
_largest_comp = _components[0] if _components else []

# Global best path
_best_path = []
_deadline = 0

# Skeleton Construction
# 1. Identify branching nodes (degree != 2). Degree 0 is isolated, Degree 1 is leaf, Degree >= 3 is branch.
# 2. Compress paths of degree-2 nodes between branching/leaf nodes into weighted edges.

# Map original node to skeleton node ID if it's a branching/leaf node
# We only care about the largest component for the main search, but let's build skeleton for all for simplicity?
# No, just largest component to save time.

_skel_node_map = {} # original_id -> skel_id
_skel_id_to_orig = [] # skel_id -> original_id
_skel_edges = [] # list of (skel_id_u, skel_id_v, weight, path_vertices)
_skel_adj = {} # skel_id -> list of (neighbor_skel_id, edge_index)

def build_skeleton():
    global _skel_node_map, _skel_id_to_orig, _skel_edges, _skel_adj
    
    _skel_node_map = {}
    _skel_id_to_orig = []
    _skel_edges = []
    _skel_adj = {}
    
    # Identify skeleton nodes in the largest component
    for node in _largest_comp:
        if _deg[node] != 2:
            skel_id = len(_skel_id_to_orig)
            _skel_node_map[node] = skel_id
            _skel_id_to_orig.append(node)
            _skel_adj[skel_id] = []

    if not _skel_id_to_orig:
        return

    # Find chains
    visited_chain_edges = set()
    
    for start_node in _largest_comp:
        if _deg[start_node] == 2:
            continue # We start chains from non-degree-2 nodes or we need to handle cycles of degree-2 nodes?
            # In G(n,p), cycles of degree-2 nodes are possible but rare. 
            # If a component is entirely degree-2 nodes (a cycle), it won't be captured by starting from non-2 nodes.
            # Let's check if there are any degree-2 nodes not visited.
            
    # Better approach: Iterate over all edges. If an edge connects two degree-2 nodes, it's part of a chain.
    # We need to find maximal chains.
    
    # Let's identify "chain edges" (both endpoints degree 2)
    # And "skeleton edges" (at least one endpoint degree != 2)
    
    # Actually, simpler:
    # For each node with degree != 2, explore its neighbors.
    # If neighbor is degree 2, walk the chain until we hit a node with degree != 2.
    
    for start_node in _largest_comp:
        if _deg[start_node] == 2:
            continue
            
        start_skel_id = _skel_node_map[start_node]
        
        for nb in _nbrs[start_node]:
            # Only process each edge once from the "start" side?
            # To avoid duplicates, we can enforce start_node < nb or just check if we've seen this chain.
            # Since we iterate all start_nodes, we might find the same chain from both ends.
            # Let's use a set of visited chain start pairs or just check if the edge (start_node, nb) is already in skel_edges.
            
            # Check if we already added this chain
            # We can store the first node of the chain in the edge tuple to check uniqueness
            # Or just build a set of frozensets of nodes in the chain? Too expensive.
            # Let's just check if (start_node, nb) is already processed.
            # We can maintain a set of processed edges (u, v) where u < v.
            
            pass # Placeholder, logic below

    # Reset and do it properly
    _skel_node_map = {}
    _skel_id_to_orig = []
    _skel_edges = []
    _skel_adj = {}
    
    for node in _largest_comp:
        if _deg[node] != 2:
            skel_id = len(_skel_id_to_orig)
            _skel_node_map[node] = skel_id
            _skel_id_to_orig.append(node)
            _skel_adj[skel_id] = []

    processed_edges = set()
    
    for start_node in _largest_comp:
        if _deg[start_node] == 2:
            continue
            
        start_skel_id = _skel_node_map[start_node]
        
        for nb in _nbrs[start_node]:
            if nb not in _skel_node_map and _deg[nb] == 2:
                # Start of a chain
                # Check if we already processed this chain from the other side?
                # The chain will end at some other non-degree-2 node.
                # We can check if the edge (start_node, nb) is in processed_edges.
                if (start_node, nb) in processed_edges or (nb, start_node) in processed_edges:
                    continue
                processed_edges.add((start_node, nb))
                
                # Walk the chain
                path = [start_node, nb]
                prev = start_node
                curr = nb
                while _deg[curr] == 2:
                    nexts = [x for x in _nbrs[curr] if x != prev]
                    if not nexts:
                        break # Should not happen in a valid chain unless isolated loop?
                    next_node = nexts[0]
                    path.append(next_node)
                    prev, curr = curr, next_node
                    if curr in _skel_node_map:
                        break # Reached a branching node
                    # If curr is degree 2, continue
                    # If curr is degree 1 (leaf), it should be in _skel_node_map
                
                # If we didn't reach a skel node, it means we hit a leaf?
                # Leaves have degree 1, so _deg[curr] == 1 != 2, so loop stops.
                # And leaves are added to _skel_node_map.
                
                if curr not in _skel_node_map:
                    # This shouldn't happen if _skel_node_map contains all non-degree-2 nodes in the component.
                    # But if the chain is a cycle of degree-2 nodes, we would loop forever or fail to terminate.
                    # Let's add a safety check: if len(path) > N, break.
                    # Also, if we don't terminate at a skel node, we might be in a cycle.
                    # In G(100, 0.018), cycles of degree-2 nodes are very unlikely to be the whole component or large parts.
                    # If we are in a cycle, we can't easily compress it without more logic.
                    # For now, if we don't hit a skel node, we treat the last node as a "virtual" skel node?
                    # Or just skip this chain?
                    # Let's assume we hit a skel node.
                    pass
                
                end_skel_id = _skel_node_map.get(curr)
                
                if end_skel_id is not None:
                    # Add edge to skeleton
                    weight = len(path) - 1 # Number of edges
                    # Store the path for reconstruction
                    # Note: The path includes start_node and curr.
                    # The "internal" nodes are path[1:-1]
                    edge_idx = len(_skel_edges)
                    _skel_edges.append((start_skel_id, end_skel_id, weight, path))
                    _skel_adj[start_skel_id].append((end_skel_id, edge_idx))
                    _skel_adj[end_skel_id].append((start_skel_id, edge_idx))
                else:
                    # This case implies a cycle of degree-2 nodes or an error.
                    # We can create a self-loop or a virtual node.
                    # For simplicity, let's create a virtual node for the end of the chain if it's not a skel node.
                    # This handles cycles of degree-2 nodes.
                    virtual_id = len(_skel_id_to_orig)
                    _skel_id_to_orig.append(curr) # Map to the last node
                    _skel_node_map[curr] = virtual_id
                    _skel_adj[virtual_id] = []
                    
                    weight = len(path) - 1
                    edge_idx = len(_skel_edges)
                    _skel_edges.append((start_skel_id, virtual_id, weight, path))
                    _skel_adj[start_skel_id].append((virtual_id, edge_idx))
                    _skel_adj[virtual_id].append((start_skel_id, edge_idx))

    # Handle isolated degree-2 cycles (if any) that were not connected to any non-degree-2 node.
    # These would be components consisting entirely of degree-2 nodes.
    # Since we only built skeleton for _largest_comp, and if _largest_comp has non-degree-2 nodes,
    # any degree-2 cycle would be a separate component or attached to a non-degree-2 node.
    # If _largest_comp is entirely degree-2 nodes (a cycle), _skel_id_to_orig would be empty.
    # In that case, we fall back to original DFS.

build_skeleton()

def _dfs_skeleton(start_skel, seen_skel, path_skel, current_weight):
    """
    DFS on the skeleton graph.
    path_skel: list of (skel_id, edge_idx_used_to_enter)
    """
    global _best_path, _deadline, _skel_edges, _skel_adj
    
    # Update best path if current weight is larger
    # We need to reconstruct the path to update _best_path
    # Reconstruction is expensive, so only do it if current_weight > len(_best_path)
    if current_weight > len(_best_path):
        # Reconstruct
        orig_path = []
        for i, (sid, eidx) in enumerate(path_skel):
            if i == 0:
                # Start node
                orig_path.append(_skel_id_to_orig[sid])
            else:
                # Add the path from the edge
                # The edge connects path_skel[i-1][0] to sid
                # We need to traverse the edge in the correct direction
                e = _skel_edges[eidx]
                u, v, w, p = e
                if u == path_skel[i-1][0] and v == sid:
                    # Traverse u -> v
                    # p is [u, ..., v]
                    # We already added u in previous step?
                    # No, we added the skel node u in the previous iteration.
                    # The edge path p contains u and v and intermediates.
                    # We should add p[1:] (excluding u which is already there)
                    orig_path.extend(p[1:])
                elif v == path_skel[i-1][0] and u == sid:
                    # Traverse v -> u
                    # p is [u, ..., v]
                    # We need to reverse it: [v, ..., u]
                    # We already added v in previous step.
                    # So we add reversed(p)[1:]
                    orig_path.extend(list(reversed(p))[1:])
                else:
                    # Error in path tracking
                    pass
        
        if len(orig_path) > len(_best_path):
            _best_path = orig_path
            # Verify validity?
            # for i in range(len(_best_path)-1):
            #    if _best_path[i+1] not in _adj[_best_path[i]]:
            #       raise ValueError("Invalid reconstructed path")

    if time.time() > _deadline:
        return

    cur_skel = path_skel[-1][0]
    
    # Get neighbors
    cands = []
    for nb_skel, eidx in _skel_adj[cur_skel]:
        if nb_skel not in seen_skel:
            cands.append((nb_skel, eidx))
            
    if not cands:
        return

    # Heuristic: Prefer edges with larger weights? Or fewer options?
    # In a tree, order doesn't matter for completeness.
    # In a graph with cycles, Warnsdorff-like heuristic might help.
    # Let's sort by weight descending to find long paths quickly.
    cands.sort(key=lambda x: _skel_edges[x[1]][2], reverse=True)
    
    for nb_skel, eidx in cands:
        seen_skel.add(nb_skel)
        path_skel.append((nb_skel, eidx))
        _dfs_skeleton(nb_skel, seen_skel, path_skel, current_weight + _skel_edges[eidx][2])
        path_skel.pop()
        seen_skel.remove(nb_skel)
        
        if time.time() > _deadline:
            return

def _refine_path(path):
    """
    Refine the path by trying to extend ends and insert unvisited vertices.
    """
    if not path:
        return path
        
    seen = set(path)
    improved = True
    while improved:
        improved = False
        
        # 1. Try to extend from both ends using Warnsdorff
        # Extend tail
        while True:
            cur = path[-1]
            cands = [v for v in _nbrs[cur] if v not in seen]
            if not cands:
                break
            best = min(cands, key=lambda v: (sum(1 for nb in _nbrs[v] if nb not in seen), v))
            path.append(best)
            seen.add(best)
            improved = True
            
        # Extend head
        while True:
            cur = path[0]
            cands = [v for v in _nbrs[cur] if v not in seen]
            if not cands:
                break
            best = min(cands, key=lambda v: (sum(1 for nb in _nbrs[v] if nb not in seen), v))
            path.insert(0, best)
            seen.add(best)
            improved = True

        # 2. Try to insert an unvisited vertex into the path
        unvisited = [v for v in range(N) if v not in seen]
        inserted = False
        for v in unvisited:
            for i in range(len(path) - 1):
                a, b = path[i], path[i+1]
                if v in _adj[a] and v in _adj[b]:
                    path.insert(i+1, v)
                    seen.add(v)
                    inserted = True
                    break
            if inserted:
                break
        
        if inserted:
            improved = True
            
    return path

def construct():
    global _best_path, _deadline
    
    t0 = time.time()
    # Safety margin: 90s limit, stop at 85s to ensure return
    _deadline = t0 + 85.0
    
    _best_path = []
    
    if not _largest_comp:
        return [0] if N > 0 else []
        
    # Check if skeleton is built and non-empty
    if _skel_id_to_orig:
        # Run DFS on skeleton from all possible start nodes
        # To avoid redundant work, we can just run from all skel nodes.
        # The number of skel nodes is small.
        for start_skel in range(len(_skel_id_to_orig)):
            if time.time() > _deadline - 2.0:
                break
            # Initialize seen and path
            seen_skel = {start_skel}
            path_skel = [(start_skel, -1)] # -1 indicates start, no edge
            # Initial weight is 0? No, the start node itself is 1 vertex.
            # The weight of edges is the number of edges.
            # Total vertices = 1 + sum(weights)
            # So we track sum(weights) and add 1 at the end?
            # Or just track vertices directly.
            # Let's track vertices.
            # Start vertex count = 1.
            # When we take an edge of weight w, we add w vertices?
            # Edge weight w means there are w edges, so w+1 vertices in the path segment.
            # But the start vertex of the segment is already counted.
            # So we add w vertices.
            # Example: A-B (weight 1). Start A (1). Add B (1). Total 2.
            # Example: A-C-B (weight 2). Start A (1). Add C, B (2). Total 3.
            # So yes, add weight.
            
            _dfs_skeleton(start_skel, seen_skel, path_skel, 1)
            
    # If skeleton search didn't find anything or we didn't have a skeleton, fall back to original DFS
    if not _best_path or len(_best_path) < 2:
        # Fallback to original DFS logic if skeleton failed or is empty
        # This handles cases where _largest_comp is a cycle of degree-2 nodes
        # or if skeleton building failed.
        
        def _dfs_search(start, seen, path, depth=0):
            global _best_path, _deadline
            if len(path) > len(_best_path):
                _best_path = list(path)
            if time.time() > _deadline:
                return
            cur = path[-1]
            cands = [v for v in _nbrs[cur] if v not in seen]
            if not cands:
                return
            cands.sort(key=lambda v: (sum(1 for nb in _nbrs[v] if nb not in seen), v))
            for v in cands:
                seen.add(v)
                path.append(v)
                _dfs_search(v, seen, path, depth + 1)
                path.pop()
                seen.remove(v)
                if time.time() > _deadline:
                    return

        starts = list(_largest_comp)
        starts.sort(key=lambda x: -_deg[x])
        for start in starts:
            if time.time() > _deadline - 5.0:
                break
            _dfs_search(start, {start}, [start])

    # Refine the best path found
    if _best_path:
        refined = _refine_path(_best_path)
        if len(refined) > len(_best_path):
            _best_path = refined

    if not _best_path:
        if _largest_comp:
            _best_path = [_largest_comp[0]]
        else:
            _best_path = [0]
            
    return _best_path

# For local testing/debugging, one might call construct()
if __name__ == "__main__":
    path = construct()
    print(f"Path length: {len(path)}")
    print(f"Path: {path}")
    # Verify validity
    for i in range(len(path) - 1):
        if path[i+1] not in _adj[path[i]]:
            raise ValueError(f"Invalid edge: {path[i]} - {path[i+1]}")
    if len(set(path)) != len(path):
        raise ValueError("Path is not simple")
