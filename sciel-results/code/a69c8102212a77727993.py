"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random, sample, shuffle

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
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
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
    return trees

def construct():
    deadline = time.time() + 145
    trees = instance_trees(20, 611)
    
    # Order trees by size descending (T20 first, T19, ..., T2)
    # T1 is trivial (no edges, single vertex)
    order = list(range(19, -1, -1))  # indices 19 down to 0 (T20 to T1)
    
    best_edges = 0
    best_result = [None] * 20
    
    # All edges in K20
    all_edges = set()
    for i in range(20):
        for j in range(i+1, 20):
            all_edges.add((i, j))
    
    used_edges = set()
    
    for trial in range(5000):
        if time.time() > deadline:
            break
            
        rng = Random(trial)
        used_edges = set()
        result = [None] * 20
        success = True
        total_edges = 0
        
        for idx in order:
            k = idx + 1  # tree T_k
            if k == 1:
                # T1 has no edges, just needs a vertex
                # Pick any unused vertex (all vertices are available for T1)
                result[0] = [0]
                continue
            if k == 2:
                # T2 has one edge (0,1)
                # Find an unused edge
                candidates = [e for e in all_edges if e not in used_edges]
                if not candidates:
                    success = False
                    break
                e = rng.choice(candidates)
                used_edges.add(e)
                # Map vertex 0 -> e[0], vertex 1 -> e[1]
                result[1] = [e[0], e[1]]
                total_edges += 1
                continue
            
            # For T_k (k >= 3), we need to embed a tree with k-1 edges
            # The tree has k vertices labeled 0..k-1
            # We need to assign each tree vertex to a host vertex in 0..19
            # such that all tree edges map to unused K20 edges
            # and no two tree vertices map to the same host vertex
            
            # Strategy: try random permutations of 20 host vertices for the k tree vertices
            # Pick k distinct host vertices, then try all k! assignments? No, too many.
            # Instead: pick a random subset of k host vertices, then try random permutations
            
            tree_edges = trees[idx]
            
            # Try to embed by random assignment
            found = False
            for attempt in range(200):
                # Pick k random distinct host vertices
                host_verts = sample(range(20), k)
                # Random permutation
                perm = host_verts[:]
                rng.shuffle(perm)
                
                # Check if this assignment works
                valid = True
                edge_map = {}
                for (u, v) in tree_edges:
                    hu, hv = perm[u], perm[v]
                    edge = (min(hu, hv), max(hu, hv))
                    if edge in used_edges:
                        valid = False
                        break
                    edge_map[(u, v)] = edge
                if valid:
                    # Also check that all edges are distinct (they should be since tree is a tree)
                    found = True
                    break
            
            if not found:
                # Try a different approach: greedy vertex placement
                # Build the tree adjacency
                adj = [[] for _ in range(k)]
                for (u, v) in tree_edges:
                    adj[u].append(v)
                    adj[v].append(u)
                
                # Find a leaf to start
                leaves = [v for v in range(k) if len(adj[v]) == 1]
                if not leaves:
                    success = False
                    break
                start = rng.choice(leaves)
                
                # BFS/DFS from start, assigning host vertices
                # Use backtracking with limited depth
                assignment = [-1] * k
                assignment[start] = rng.choice(range(20))
                
                # Build edge constraints as we go
                def backtrack(node, parent, depth):
                    if time.time() > deadline:
                        return False
                    if depth > 300:
                        return False
                    if node is not None:
                        # Assign host vertices to children
                        children = [c for c in adj[node] if c != parent]
                        # Shuffle children for randomization
                        rng.shuffle(children)
                        for child in children:
                            # Try each unused host vertex
                            used_hosts = set(assignment[i] for i in range(k) if assignment[i] >= 0)
                            candidates = [h for h in range(20) if h not in used_hosts]
                            rng.shuffle(candidates)
                            found_child = False
                            for h in candidates:
                                # Check edge (node, child) maps to unused edge
                                hu = assignment[node]
                                edge = (min(hu, h), max(hu, h))
                                if edge in used_edges:
                                    continue
                                assignment[child] = h
                                if backtrack(child, node, depth + 1):
                                    found_child = True
                                    break
                                assignment[child] = -1
                            if not found_child:
                                return False
                    return True
                
                if backtrack(start, -1, 0):
                    # Verify all assignments are valid
                    valid = True
                    for i in range(k):
                        if assignment[i] < 0:
                            valid = False
                            break
                    if valid:
                        # Check all edges
                        for (u, v) in tree_edges:
                            hu, hv = assignment[u], assignment[v]
                            edge = (min(hu, hv), max(hu, hv))
                            if edge in used_edges:
                                valid = False
                                break
                        if valid:
                            found = True
                            perm = assignment
                            break
                
                if found:
                    break
            
            if not found:
                success = False
                break
            
            # Record the embedding
            result[idx] = perm[:k]
            # Add edges to used
            for (u, v) in tree_edges:
                hu, hv = perm[u], perm[v]
                edge = (min(hu, hv), max(hu, hv))
                used_edges.add(edge)
            total_edges += k - 1
        
        if success and total_edges > best_edges:
            best_edges = total_edges
            best_result = result[:]
    
    # Handle T1 if it wasn't set
    if best_result[0] is None:
        best_result[0] = [0]
    
    return best_result
# EVOLVE-BLOCK-END
