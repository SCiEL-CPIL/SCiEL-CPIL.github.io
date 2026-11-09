import time
import random
from random import Random
import sys

def instance_trees(n=20, seed=611):
    rng = Random(seed)
    trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)])
            continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code:
            degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x))
            degree[leaf] -= 1
            degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w))
        trees.append(edges)
    return trees

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # Leave 5s for cleanup/return
    
    trees = instance_trees(20, 611)
    n = 20
    
    # Sort indices by tree size descending
    # trees[i] has i+1 vertices
    indices = list(range(20))
    indices.sort(key=lambda i: len(trees[i]), reverse=True)
    
    # Reorder trees for processing, but keep original index for result
    sorted_trees = [trees[i] for i in indices]
    
    # Precompute adjacency and degrees for each tree
    tree_adj = []
    tree_deg = []
    for t in sorted_trees:
        adj = {v: [] for v in range(len(t) + 1) if t or True} # Vertices 0..k-1
        # Actually, vertices are 0..k-1
        k = len(t) + 1
        adj = {v: [] for v in range(k)}
        deg = [0] * k
        for u, v in t:
            adj[u].append(v)
            adj[v].append(u)
            deg[u] += 1
            deg[v] += 1
        tree_adj.append(adj)
        tree_deg.append(deg)
        
    # Backtracking state
    used_edges = set()
    result = [None] * 20
    best_result = None
    best_score = 0
    
    # We will try to find a full packing
    # If we fail, we fall back to greedy
    
    def backtrack(idx):
        nonlocal best_result, best_score, used_edges, result
        
        if time.time() > deadline:
            return False
            
        if idx == 0:
            # All trees embedded
            # Construct result
            # Note: result is filled in the caller or here?
            # It's easier to maintain result in the recursion
            # But we need to map back to original indices
            # Let's store the mapping for each sorted tree index
            # Actually, let's just return True and let the caller construct result
            # Or better, update best_result here
            # Since we are modifying result list in place, we need to be careful with backtracking
            # Let's use a separate list for current solution
            return True
            
        tree_edges = sorted_trees[idx]
        k = len(tree_edges) + 1
        adj = tree_adj[idx]
        deg = tree_deg[idx]
        
        # Determine order of vertices to place in this tree
        # Sort by degree descending
        verts = list(range(k))
        verts.sort(key=lambda v: deg[v], reverse=True)
        
        # Map tree vertex -> host vertex
        mapping = {}
        used_hosts = set()
        
        def place_tree_vertex(pos):
            nonlocal mapping, used_hosts
            
            if time.time() > deadline:
                return False
                
            if pos == len(verts):
                # All vertices of this tree placed
                # Check if all edges are valid (they should be, due to pruning)
                # Add edges to used_edges
                new_edges = []
                for u, v in tree_edges:
                    hu = mapping[u]
                    hv = mapping[v]
                    e = (min(hu, hv), max(hu, hv))
                    if e in used_edges:
                        # Should not happen if pruning is correct
                        return False
                    new_edges.append(e)
                
                # Recurse to next tree
                # Save state
                for e in new_edges:
                    used_edges.add(e)
                    
                # Update result for this tree
                # Original index for sorted_trees[idx] is indices[idx]
                orig_idx = indices[idx]
                # mapping has tree vertex -> host vertex
                # We need a list of length k where position i is image of vertex i
                res_list = [mapping[v] for v in range(k)]
                result[orig_idx] = res_list
                
                if backtrack(idx - 1):
                    return True
                else:
                    # Backtrack
                    for e in new_edges:
                        used_edges.remove(e)
                    result[orig_idx] = None
                    return False
            
            v = verts[pos]
            
            # Find candidate host vertices
            # If pos == 0, any host is valid (symmetry breaking: fix to 0? No, K20 is symmetric, but we can fix one to reduce search)
            # Let's fix the first vertex of the largest tree to 0 to break symmetry
            if idx == len(sorted_trees) - 1 and pos == 0:
                candidates = [0]
            else:
                candidates = [h for h in range(n) if h not in used_hosts]
                
            # Heuristic: sort candidates?
            # Maybe try hosts that have more available edges to already placed neighbors?
            # Or just randomize to avoid bias?
            # Let's try deterministic first, then random if needed?
            # For now, iterate in order.
            
            for h in candidates:
                # Check constraints with already placed neighbors
                valid = True
                for u in adj[v]:
                    if u in mapping:
                        hu = mapping[u]
                        e = (min(h, hu), max(h, hu))
                        if e in used_edges:
                            valid = False
                            break
                if not valid:
                    continue
                    
                # Place v at h
                mapping[v] = h
                used_hosts.add(h)
                
                if place_tree_vertex(pos + 1):
                    return True
                else:
                    # Backtrack
                    del mapping[v]
                    used_hosts.remove(h)
                    
            return False
            
        return place_tree_vertex(0)
        
    # Run backtracking
    # We need to handle the case where backtracking fails or times out
    # If it succeeds, result is filled.
    # If it fails, we need a fallback.
    
    # To get a fallback, we can run a greedy algorithm
    # Let's try backtracking first
    success = backtrack(len(sorted_trees) - 1)
    
    if success:
        # Verify and return
        for i in range(20):
            if result[i] is not None:
                result[i] = [int(x) for x in result[i]]
        return result
    else:
        # Fallback: Greedy
        # Reset
        used_edges = set()
        result = [None] * 20
        rng = Random(42)
        
        # Try greedy in order of decreasing size
        for idx in indices:
            if time.time() > deadline - 2.0:
                break
            tree_edges = trees[idx]
            k = len(tree_edges) + 1
            
            if not tree_edges:
                result[idx] = [0]
                continue
                
            # Try to embed greedily with random restarts
            # Simple greedy: assign vertices in BFS order
            # Build adj
            adj = {v: [] for v in range(k)}
            for u, v in tree_edges:
                adj[u].append(v)
                adj[v].append(u)
            
            root = 0
            order = [root]
            parent = {root: -1}
            queue = [root]
            idx_q = 0
            while idx_q < len(queue):
                u = queue[idx_q]
                idx_q += 1
                for w in adj[u]:
                    if w not in parent:
                        parent[w] = u
                        queue.append(w)
                        order.append(w)
                        
            found = False
            for attempt in range(100):
                mapping = {}
                host_used = set()
                success_g = True
                host_root = rng.choice(range(n))
                mapping[root] = host_root
                host_used.add(host_root)
                
                for v in order[1:]:
                    p = parent[v]
                    host_p = mapping[p]
                    candidates = [h for h in range(n) if h not in host_used]
                    valid = []
                    for h in candidates:
                        e = (min(host_p, h), max(host_p, h))
                        if e not in used_edges:
                            valid.append(h)
                    if not valid:
                        success_g = False
                        break
                    host_v = rng.choice(valid)
                    mapping[v] = host_v
                    host_used.add(host_v)
                    
                if success_g:
                    found = True
                    # Add edges
                    for u, v in tree_edges:
                        hu, hv = mapping[u], mapping[v]
                        e = (min(hu, hv), max(hu, hv))
                        used_edges.add(e)
                    result[idx] = [mapping[v] for v in range(k)]
                    break
                    
            if not found:
                # Skip this tree? Or stop?
                # If we skip, we might miss edges.
                # But if we can't embed it, we can't use its edges.
                # We continue to try smaller trees.
                pass
                
        for i in range(20):
            if result[i] is not None:
                result[i] = [int(x) for x in result[i]]
        return result
