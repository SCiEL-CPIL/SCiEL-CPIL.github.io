"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

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
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    trees = instance_trees(20, 611)
    num_trees = 20
    
    # Pre-compute tree adjacency and edge lists
    tree_data = []
    for i in range(num_trees):
        k = i + 1
        if k == 1:
            tree_data.append({'edges': [], 'adj': [[]], 'size': 1})
        elif k == 2:
            tree_data.append({'edges': [(0, 1)], 'adj': [[1], [0]], 'size': 2})
        else:
            edges = trees[i]
            adj = [[] for _ in range(k)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            tree_data.append({'edges': edges, 'adj': adj, 'size': k})
    
    # Order: largest trees first (most constrained)
    order = list(range(19, -1, -1))
    
    best_score = 0
    best_mappings = [None] * 20
    
    def try_full_packing(rng, max_nodes_explored=500000):
        """
        Backtracking search to embed all trees.
        Returns (score, mappings) or (0, None) if no full packing found.
        """
        used_edges = set()
        mappings = [None] * num_trees
        nodes_explored = [0]
        
        def embed_tree(tree_idx, assignment, depth):
            """
            Recursively embed tree_idx using backtracking.
            assignment: list of length k, where assignment[v] = host vertex for tree vertex v
            depth: current depth in DFS (number of vertices assigned)
            Returns True if successful, False otherwise.
            """
            if nodes_explored[0] > max_nodes_explored:
                return False
            
            k = tree_data[tree_idx]['size']
            if k == 1:
                # T1: map vertex 0 to any available vertex (doesn't use edges)
                for h in range(20):
                    if h not in set(assignment):
                        assignment[0] = h
                        return True
                return False
            
            if k == 2:
                # T2: need one available edge
                for u in range(20):
                    if u in set(assignment):
                        continue
                    for v in range(u + 1, 20):
                        if v in set(assignment):
                            continue
                        e_key = (u, v)
                        if e_key not in used_edges:
                            assignment[0] = u
                            assignment[1] = v
                            used_edges.add(e_key)
                            return True
                return False
            
            # For k > 2: use BFS/DFS-based embedding
            adj = tree_data[tree_idx]['adj']
            edges = tree_data[tree_idx]['edges']
            
            # Order tree vertices: start from highest degree vertex
            degrees = [len(adj[v]) for v in range(k)]
            # Create ordering: root first, then children
            root = max(range(k), key=lambda v: degrees[v])
            
            # Build a spanning tree order (BFS from root)
            visited = [False] * k
            order_list = [root]
            visited[root] = True
            queue = [root]
            while queue:
                u = queue.pop(0)
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        order_list.append(v)
                        queue.append(v)
            
            # Now assign in order_list
            def dfs_assign(idx):
                if nodes_explored[0] > max_nodes_explored:
                    return False
                if idx == len(order_list):
                    # All vertices assigned, verify edges
                    for u, v in edges:
                        hu = assignment[u]
                        hv = assignment[v]
                        e_key = (min(hu, hv), max(hu, hv))
                        if e_key in used_edges:
                            return False
                    # All good
                    return True
                
                tv = order_list[idx]
                # Get already-assigned neighbors
                assigned_neighbors = []
                for nb in adj[tv]:
                    if nb in assignment:
                        assigned_neighbors.append(nb)
                
                # Determine candidate host vertices
                if assigned_neighbors:
                    # Must be adjacent to all assigned neighbors' hosts
                    # In K20, this means the host must be different from all assigned neighbors' hosts
                    # AND the edges to those hosts must not be used
                    candidates = []
                    for h in range(20):
                        if h in set(assignment):
                            continue
                        # Check edges to assigned neighbors
                        valid = True
                        for nb in assigned_neighbors:
                            hn = assignment[nb]
                            e_key = (min(h, hn), max(h, hn))
                            if e_key in used_edges:
                                valid = False
                                break
                        if valid:
                            candidates.append(h)
                else:
                    # No assigned neighbors yet (shouldn't happen for idx > 0 in BFS order)
                    candidates = [h for h in range(20) if h not in set(assignment)]
                
                # Shuffle candidates for randomization
                rng.shuffle(candidates)
                
                for h in candidates:
                    nodes_explored[0] += 1
                    assignment[tv] = h
                    if dfs_assign(idx + 1):
                        return True
                    del assignment[tv]
                
                return False
            
            # Initialize assignment
            assignment = [None] * k
            success = dfs_assign(0)
            if success:
                # Record the mapping
                mappings[tree_idx] = list(assignment)
                # Add all edges to used_edges
                for u, v in edges:
                    hu = assignment[u]
                    hv = assignment[v]
                    e_key = (min(hu, hv), max(hu, hv))
                    used_edges.add(e_key)
            return success
        
        # Embed trees in order
        for tree_idx in order:
            k = tree_data[tree_idx]['size']
            if k == 1:
                # T1: assign any vertex
                for h in range(20):
                    if h not in set(m[0] for m in mappings if m is not None and len(m) > 0):
                        mappings[tree_idx] = [h]
                        break
                continue
            
            # Use backtracking for this tree
            local_assignment = [None] * k
            success = embed_tree(tree_idx, local_assignment, 0)
            
            if not success:
                # Try a simpler approach for small trees
                if k <= 5:
                    # Random sampling
                    for _ in range(1000):
                        hosts = rng.sample(range(20), k)
                        perm = hosts[:]
                        rng.shuffle(perm)
                        valid = True
                        new_edges = []
                        for u, v in tree_data[tree_idx]['edges']:
                            hu = perm[u]
                            hv = perm[v]
                            e_key = (min(hu, hv), max(hu, hv))
                            if e_key in used_edges:
                                valid = False
                                break
                            new_edges.append(e_key)
                        if valid:
                            mappings[tree_idx] = perm
                            used_edges.update(new_edges)
                            break
                    if mappings[tree_idx] is None:
                        return 0, None
                else:
                    return 0, None
        
        # Calculate score
        score = sum(len(tree_data[i]['edges']) for i in range(num_trees) if mappings[i] is not None)
        return score, mappings
    
    # Run multiple restarts
    for restart in range(500):
        if time.time() > deadline:
            break
        
        rng = Random(restart * 7919 + 104729)
        score, mappings = try_full_packing(rng, max_nodes_explored=200000)
        
        if score > best_score:
            best_score = score
            best_mappings = [m[:] if m else None for m in mappings]
            
        if best_score == 190:
            break
    
    # If we haven't found a full packing, try a greedy approach for remaining trees
    if best_score < 190:
        # Try to fill in missing trees with greedy random sampling
        used_edges = set()
        for i in range(num_trees):
            if best_mappings[i] is not None:
                for u, v in tree_data[i]['edges']:
                    hu = best_mappings[i][u]
                    hv = best_mappings[i][v]
                    used_edges.add((min(hu, hv), max(hu, hv)))
        
        for i in range(num_trees):
            if best_mappings[i] is not None:
                continue
            k = tree_data[i]['size']
            if k == 1:
                for h in range(20):
                    if h not in set(m[v] for m in best_mappings if m is not None for v in range(len(m))):
                        best_mappings[i] = [h]
                        break
                continue
            if k == 2:
                for u in range(20):
                    for v in range(u + 1, 20):
                        if (u, v) not in used_edges:
                            best_mappings[i] = [u, v]
                            used_edges.add((u, v))
                            break
                    if best_mappings[i] is not None:
                        break
                continue
            
            # Random sampling for larger trees
            found = False
            for _ in range(5000):
                hosts = list(range(20))
                rng = Random(hash((i, _)) % (2**32))
                rng.shuffle(hosts)
                selected = hosts[:k]
                perm = selected[:]
                rng.shuffle(perm)
                valid = True
                new_edges = []
                for u, v in tree_data[i]['edges']:
                    hu = perm[u]
                    hv = perm[v]
                    e_key = (min(hu, hv), max(hu, hv))
                    if e_key in used_edges:
                        valid = False
                        break
                    new_edges.append(e_key)
                if valid:
                    best_mappings[i] = perm
                    used_edges.update(new_edges)
                    found = True
                    break
        
        # Recalculate score
        best_score = sum(len(tree_data[i]['edges']) for i in range(num_trees) if best_mappings[i] is not None)
    
    # Final conversion
    final_result = []
    for i in range(20):
        if best_mappings[i] is None:
            final_result.append(None)
        else:
            final_result.append([int(x) for x in best_mappings[i]])
    
    return final_result

# EVOLVE-BLOCK-END
