"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from random import Random

def _instance_trees(n=20, seed=611):
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

def _embed_tree_greedy(tree_edges, k, used_edges, rng):
    """Try to embed tree with k vertices into K20 using unused edges."""
    if k == 1:
        return [0]  # single vertex, no edges
    if k == 2:
        # Find any unused edge
        for u in range(20):
            for v in range(u + 1, 20):
                if (u, v) not in used_edges and (v, u) not in used_edges:
                    return [u, v]
        return None
    
    # Build adjacency of tree
    adj = [[] for _ in range(k)]
    for u, v in tree_edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Greedy: pick a root, do DFS assigning host vertices
    # Try multiple random orderings
    for attempt in range(500):
        # Random permutation of tree vertices for ordering
        order = list(range(k))
        rng.shuffle(order)
        # But we need to respect tree structure - do a BFS/DFS from a random root
        root = rng.randint(0, k - 1)
        # Get BFS order from root
        visited = {root}
        queue = [root]
        bfs_order = []
        while queue:
            cur = queue.pop(0)
            bfs_order.append(cur)
            for nb in adj[cur]:
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        
        # Assign host vertices greedily
        assignment = {}  # tree vertex -> host vertex
        available = set(range(20))
        
        # Start with root
        root_host = rng.choice(list(available))
        assignment[root] = root_host
        available.discard(root_host)
        
        success = True
        for tv in bfs_order:
            if tv == root:
                continue
            # Find parent of tv in BFS
            parent = None
            for nb in adj[tv]:
                if nb in assignment:
                    parent = nb
                    break
            if parent is None:
                success = False
                break
            phost = assignment[parent]
            # Try to assign tv to a host adjacent to phost with unused edge
            candidates = [h for h in range(20) if h in available and h != phost and 
                         (phost, h) not in used_edges and (h, phost) not in used_edges]
            if not candidates:
                success = False
                break
            # Choose randomly or by some heuristic
            # Prefer vertices with more available neighbors for future
            tv_host = rng.choice(candidates)
            assignment[tv] = tv_host
            available.discard(tv_host)
        
        if not success:
            continue
        
        # Verify all edges are unused
        valid = True
        for u, v in tree_edges:
            hu, hv = assignment[u], assignment[v]
            if (hu, hv) in used_edges or (hv, hu) in used_edges:
                valid = False
                break
        if valid:
            # Return in order: position i is image of vertex i
            result = [assignment[i] for i in range(k)]
            return result
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    trees = _instance_trees(20, 611)
    
    best_result = [None] * 19 + [list(range(20))]
    best_value = 19  # T20 embedded
    
    for trial in range(200):
        if time.time() > deadline:
            break
        
        rng = Random(trial)
        used_edges = set()
        result = [None] * 20
        total_edges = 0
        success = True
        
        # Embed trees in order T1 to T20
        for k in range(1, 21):
            if time.time() > deadline:
                success = False
                break
            tree_edges = trees[k - 1]
            if not tree_edges:
                # T1: single vertex
                result[k - 1] = [0]
                continue
            
            embedding = _embed_tree_greedy(tree_edges, k, used_edges, rng)
            if embedding is None:
                success = False
                break
            
            result[k - 1] = embedding
            # Add edges to used set
            for u, v in tree_edges:
                hu, hv = embedding[u], embedding[v]
                used_edges.add((hu, hv))
                used_edges.add((hv, hu))
            total_edges += len(tree_edges)
        
        if success and total_edges > best_value:
            best_value = total_edges
            best_result = result[:]
            if best_value == 190:
                break
    
    # Ensure types are correct
    final = []
    for entry in best_result:
        if entry is None:
            final.append(None)
        else:
            final.append([int(x) for x in entry])
    
    return final
# EVOLVE-BLOCK-END
