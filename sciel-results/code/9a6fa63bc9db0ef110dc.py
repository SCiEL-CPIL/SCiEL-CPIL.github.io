"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)]); continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code: degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
    return trees

def construct():
    deadline = time.time() + 110  # safety margin before 120s limit
    trees = instance_trees(20, 611)
    # trees[k-1] is the edge list for T_k
    
    # Strategy: try to pack trees from largest to smallest using randomized backtracking
    # Each tree T_k has k vertices (0..k-1) and k-1 edges
    # We need an injective mapping of tree vertices to host vertices 0..19
    # such that no two embedded trees share an edge in K20
    
    # Build adjacency for each tree
    tree_adj = []
    for k in range(1, 21):
        adj = [[] for _ in range(k)]
        for (u, v) in trees[k-1]:
            adj[u].append(v)
            adj[v].append(u)
        tree_adj.append(adj)
    
    # Order: try larger trees first (more constrained)
    # But T1 is trivial, T2 is a single edge
    # Let's try from k=20 down to k=1
    order = list(range(1, 21))  # k values, we'll process in reverse
    order.reverse()  # 20, 19, ..., 1
    
    # Multiple restarts with different random seeds
    best_result = [None] * 20
    best_edges = 0
    
    for attempt in range(500):
        if time.time() > deadline:
            break
        
        rng = Random(attempt)
        # Track used edges in K20
        used_edges = set()
        # result[k-1] = list of host vertices for T_k, or None
        result = [None] * 20
        success = True
        
        for k in order:
            if time.time() > deadline:
                success = False
                break
            
            if k == 1:
                # T1 is a single vertex, any unused vertex works
                # Actually T1 has 0 edges, so it doesn't conflict with anything
                # But it still needs a vertex assignment
                # Find any vertex not used by other trees? No, vertices can be shared.
                # Only edges must be disjoint.
                # So T1 just needs any vertex.
                result[0] = [rng.randint(0, 19)]
                continue
            
            if k == 2:
                # T2 is a single edge (0,1)
                # Find any unused edge
                found = False
                for _ in range(200):
                    a = rng.randint(0, 19)
                    b = rng.randint(0, 19)
                    if a == b:
                        continue
                    edge = (min(a,b), max(a,b))
                    if edge not in used_edges:
                        used_edges.add(edge)
                        result[1] = [a, b]
                        found = True
                        break
                if not found:
                    # Try systematically
                    for a in range(20):
                        for b in range(a+1, 20):
                            edge = (a, b)
                            if edge not in used_edges:
                                used_edges.add(edge)
                                result[1] = [a, b]
                                found = True
                                break
                        if found:
                            break
                if not found:
                    success = False
                    break
                continue
            
            # For k >= 3: use backtracking to embed the tree
            # The tree has k vertices and k-1 edges
            # We need to assign each tree vertex to a distinct host vertex
            # such that for each tree edge (u,v), the host edge (map[u], map[v]) is not in used_edges
            
            adj = tree_adj[k-1]
            k_val = k
            mapping = [-1] * k_val  # tree vertex -> host vertex
            host_used = [False] * 20  # which host vertices are assigned
            
            # Use DFS/backtracking
            # Order tree vertices by degree (highest first) for better pruning
            deg = [len(adj[i]) for i in range(k_val)]
            vertex_order = sorted(range(k_val), key=lambda x: -deg[x])
            
            # We'll assign in vertex_order
            # For each tree vertex, try host vertices that are:
            #   - not already used
            #   - for all already-assigned neighbors, the edge to this host is not in used_edges
            
            def backtrack(idx):
                if time.time() > deadline:
                    return False
                if idx == len(vertex_order):
                    return True
                
                tv = vertex_order[idx]
                # Get already-assigned neighbors of tv
                assigned_neighbors = []
                for nb in adj[tv]:
                    if mapping[nb] != -1:
                        assigned_neighbors.append(nb)
                
                # Try host vertices
                candidates = list(range(20))
                rng.shuffle(candidates)
                
                for hv in candidates:
                    if host_used[hv]:
                        continue
                    # Check edges to assigned neighbors
                    ok = True
                    for nb in assigned_neighbors:
                        hnb = mapping[nb]
                        edge = (min(hv, hnb), max(hv, hnb))
                        if edge in used_edges:
                            ok = False
                            break
                    if not ok:
                        continue
                    
                    # Tentatively assign
                    mapping[tv] = hv
                    host_used[hv] = True
                    
                    if backtrack(idx + 1):
                        return True
                    
                    # Undo
                    mapping[tv] = -1
                    host_used[hv] = False
                
                return False
            
            # Try with a node limit to avoid infinite loops
            # Simple approach: just call backtrack
            if backtrack(0):
                # Record the mapping
                result[k-1] = [mapping[i] for i in range(k_val)]
                # Add all tree edges to used_edges
                for (u, v) in trees[k-1]:
                    hu, hv = mapping[u], mapping[v]
                    used_edges.add((min(hu, hv), max(hu, hv)))
            else:
                success = False
                break
        
        if success:
            # Count total edges
            total_edges = sum(len(trees[k-1]) for k in range(1, 21) if result[k-1] is not None)
            if total_edges > best_edges:
                best_edges = total_edges
                best_result = result[:]
                # If we got all trees, we're done
                if best_edges == 190:
                    break
    
    return best_result
# EVOLVE-BLOCK-END
