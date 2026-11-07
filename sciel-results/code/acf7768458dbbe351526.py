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
    deadline = start_time + 115.0  # Safety margin before 150s limit
    
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
            
    # Strategy: CSP Backtracking with Forward Checking and MRV
    # We treat the problem as packing trees one by one.
    # To maximize success, we pack the largest trees first (T20, T19, ...).
    # For each tree, we use a CSP solver to find a valid embedding.
    
    # Order of trees to embed: descending by size
    order = list(range(19, -1, -1))
    
    best_score = 0
    best_mappings = [None] * 20
    
    # We will use a single powerful search with randomized tie-breaking (or just deterministic MRV)
    # Given the time limit, we might need to try a few restarts if the first deterministic pass fails,
    # but MRV is usually very strong for this kind of constraint.
    
    # Let's implement a robust CSP solver for embedding a single tree given current used edges.
    
    def embed_tree_csp(tree_idx, used_edges, rng):
        """
        Attempts to embed tree 'tree_idx' into K20 avoiding 'used_edges'.
        Returns (mapping, new_used_edges) or (None, None).
        mapping is a list of length k where mapping[i] is the host vertex for tree vertex i.
        """
        k = tree_data[tree_idx]['size']
        if k == 1:
            # Find any free vertex
            for v in range(20):
                return [v], used_edges
            return None, None
        
        if k == 2:
            # Find any free edge
            for u in range(20):
                for v in range(u + 1, 20):
                    e_key = (u, v)
                    if e_key not in used_edges:
                        # Mapping: tree vertex 0 -> u, tree vertex 1 -> v (or swapped, doesn't matter for T2)
                        return [u, v], used_edges | {e_key}
            return None, None
            
        # For k > 2:
        # Variables: tree vertices 0..k-1
        # Domain: 0..19
        # Constraints:
        # 1. All assigned host vertices must be distinct.
        # 2. For each edge (u, v) in tree, (mapping[u], mapping[v]) must not be in used_edges.
        
        # Heuristic: MRV (Minimum Remaining Values)
        # We need to track:
        # - assigned: dict tree_vertex -> host_vertex
        # - used_hosts: set of host vertices already assigned
        # - remaining_edges: set of edges in K20 still available (or just check against global used_edges)
        
        # To make MRV effective, we need to know how many valid host vertices remain for each unassigned tree vertex.
        # A tree vertex v can be assigned to host h if:
        #   - h is not in used_hosts
        #   - For all neighbors u of v in the tree that are ALREADY assigned to host h_u, the edge (h, h_u) is NOT in used_edges.
        
        # Initial state
        assigned = {}
        used_hosts = set()
        
        # We will use a recursive backtracking function.
        # To speed up, we can pick the root. Usually a high-degree vertex is a good root.
        # Let's pick the vertex with max degree in the tree as the first to assign.
        degrees = [len(tree_data[tree_idx]['adj'][i]) for i in range(k)]
        root = max(range(k), key=lambda i: degrees[i])
        
        # We'll use a stack or recursion. Recursion depth is at most 20, so it's safe.
        
        def backtrack(assigned, used_hosts):
            if len(assigned) == k:
                # All vertices assigned. Verify all edges are valid (should be guaranteed by forward checking)
                # But just in case, double check.
                for u, v in tree_data[tree_idx]['edges']:
                    hu = assigned[u]
                    hv = assigned[v]
                    e_key = (min(hu, hv), max(hu, hv))
                    if e_key in used_edges:
                        return None
                return assigned.copy()
            
            # Select unassigned vertex with MRV
            unassigned = [v for v in range(k) if v not in assigned]
            
            # Calculate domain size for each unassigned vertex
            best_v = -1
            min_domain_size = 21
            best_domain = []
            
            for v in unassigned:
                # Domain: host vertices h such that:
                # 1. h not in used_hosts
                # 2. For all neighbors u of v in tree where u is assigned:
                #    edge (h, assigned[u]) is not in used_edges
                
                valid_hosts = []
                for h in range(20):
                    if h in used_hosts:
                        continue
                    valid = True
                    for u in tree_data[tree_idx]['adj'][v]:
                        if u in assigned:
                            hu = assigned[u]
                            e_key = (min(h, hu), max(h, hu))
                            if e_key in used_edges:
                                valid = False
                                break
                    if valid:
                        valid_hosts.append(h)
                
                if len(valid_hosts) < min_domain_size:
                    min_domain_size = len(valid_hosts)
                    best_v = v
                    best_domain = valid_hosts
                    if min_domain_size == 0:
                        break # Fail
            
            if best_v == -1 or min_domain_size == 0:
                return None
            
            # Try values in best_domain
            # Randomize order? Or just try all.
            # For determinism and speed, let's try in order. If we want restarts, we can shuffle.
            for h in best_domain:
                new_assigned = assigned.copy()
                new_assigned[best_v] = h
                new_used_hosts = used_hosts | {h}
                
                # Forward Check: check if any unassigned neighbor of best_v is blocked
                # Actually, the MRV calculation in the next step will handle this.
                # But we can prune here if a neighbor has no valid hosts.
                # Let's just recurse. The MRV in the next call will catch immediate failures.
                
                result = backtrack(new_assigned, new_used_hosts)
                if result is not None:
                    return result
                    
            return None
        
        # Start backtracking
        # Assign root first to break symmetry? 
        # Actually, the MRV selection in the first call will pick the vertex with smallest domain.
        # Initially, all vertices have domain size 20 (minus constraints from assigned neighbors, which is 0).
        # So it picks the first one in unassigned list? 
        # To be smarter, we can force the first assignment to be the root (max degree) to constrain others early.
        
        # Let's modify the first step:
        # Assign root to a random (or first) valid host.
        # Then call backtrack.
        
        # However, the backtrack function as written selects the MRV variable.
        # If we start with empty assigned, all domains are size 20.
        # It will pick index 0.
        # Let's explicitly assign the root first to improve pruning.
        
        # Find valid hosts for root
        root_hosts = []
        for h in range(20):
            # No neighbors assigned yet, so only check if h is free (it is)
            root_hosts.append(h)
            
        # Shuffle root hosts for diversity if we were doing restarts. For now, deterministic.
        # Let's try each host for the root.
        for h in root_hosts:
            initial_assigned = {root: h}
            initial_used_hosts = {h}
            result = backtrack(initial_assigned, initial_used_hosts)
            if result is not None:
                # Convert result to list
                mapping = [0] * k
                for tv, hv in result.items():
                    mapping[tv] = hv
                # Update used edges
                new_used_edges = set(used_edges)
                for u, v in tree_data[tree_idx]['edges']:
                    hu = mapping[u]
                    hv = mapping[v]
                    e_key = (min(hu, hv), max(hu, hv))
                    new_used_edges.add(e_key)
                return mapping, new_used_edges
                
        return None, None

    # Main Loop
    # We'll try to pack all trees.
    # If we fail on a tree, we might try a different random seed for tie-breaking in the CSP.
    # But the CSP above is deterministic (iterates hosts 0-19).
    # To add randomness, we can shuffle the host domain in the CSP.
    
    # Let's modify embed_tree_csp to accept a rng and shuffle domains.
    
    def embed_tree_csp_random(tree_idx, used_edges, rng):
        k = tree_data[tree_idx]['size']
        if k == 1:
            for v in range(20):
                return [v], used_edges
            return None, None
        
        if k == 2:
            for u in range(20):
                for v in range(u + 1, 20):
                    e_key = (u, v)
                    if e_key not in used_edges:
                        return [u, v], used_edges | {e_key}
            return None, None
            
        degrees = [len(tree_data[tree_idx]['adj'][i]) for i in range(k)]
        root = max(range(k), key=lambda i: degrees[i])
        
        def backtrack(assigned, used_hosts):
            if len(assigned) == k:
                for u, v in tree_data[tree_idx]['edges']:
                    hu = assigned[u]
                    hv = assigned[v]
                    e_key = (min(hu, hv), max(hu, hv))
                    if e_key in used_edges:
                        return None
                return assigned.copy()
            
            # Time check inside backtracking
            if time.time() > tree_deadline:
                return None
                
            unassigned = [v for v in range(k) if v not in assigned]
            best_v = -1
            min_domain_size = 21
            best_domain = []
            best_deg = 0
            
            for v in unassigned:
                valid_hosts = []
                for h in range(20):
                    if h in used_hosts:
                        continue
                    valid = True
                    for u in tree_data[tree_idx]['adj'][v]:
                        if u in assigned:
                            hu = assigned[u]
                            e_key = (min(h, hu), max(h, hu))
                            if e_key in used_edges:
                                valid = False
                                break
                    if valid:
                        valid_hosts.append(h)
                
                deg_v = len(tree_data[tree_idx]['adj'][v])
                # Tie-break by degree: prefer higher-degree vertices when domain sizes are equal
                if len(valid_hosts) < min_domain_size or (len(valid_hosts) == min_domain_size and deg_v > best_deg):
                    min_domain_size = len(valid_hosts)
                    best_v = v
                    best_domain = valid_hosts
                    best_deg = deg_v
                    if min_domain_size == 0:
                        break
            
            if best_v == -1 or min_domain_size == 0:
                return None
            
            rng.shuffle(best_domain) # Randomize tie-breaking
            
            for h in best_domain:
                new_assigned = assigned.copy()
                new_assigned[best_v] = h
                new_used_hosts = used_hosts | {h}
                result = backtrack(new_assigned, new_used_hosts)
                if result is not None:
                    return result
            return None
        
        # Shuffle root hosts
        root_hosts = list(range(20))
        rng.shuffle(root_hosts)
        
        # Limit time per tree embedding
        tree_start = time.time()
        tree_deadline = tree_start + 30.0  # Max 30s per tree
        
        for h in root_hosts:
            if time.time() > tree_deadline:
                break
            initial_assigned = {root: h}
            initial_used_hosts = {h}
            result = backtrack(initial_assigned, initial_used_hosts)
            if result is not None:
                mapping = [0] * k
                for tv, hv in result.items():
                    mapping[tv] = hv
                new_used_edges = set(used_edges)
                for u, v in tree_data[tree_idx]['edges']:
                    hu = mapping[u]
                    hv = mapping[v]
                    e_key = (min(hu, hv), max(hu, hv))
                    new_used_edges.add(e_key)
                return mapping, new_used_edges
                
        return None, None

    # Run multiple restarts with improved strategy
    for restart in range(2000):
        if time.time() > deadline:
            break
            
        rng = Random(restart * 12345 + 6789)
        current_used_edges = set()
        current_mappings = [None] * 20
        success = True
        
        for tree_idx in order:
            if time.time() > deadline:
                success = False
                break
                
            mapping, new_used_edges = embed_tree_csp_random(tree_idx, current_used_edges, rng)
            if mapping is None:
                # Try to embed smaller trees that we haven't placed yet
                # This allows partial packing if a large tree fails
                success = False
                # Try remaining trees in ascending order (smaller first)
                for small_idx in range(20):
                    if time.time() > deadline:
                        break
                    if current_mappings[small_idx] is None:
                        m2, e2 = embed_tree_csp_random(small_idx, current_used_edges, rng)
                        if m2 is not None:
                            current_mappings[small_idx] = m2
                            current_used_edges = e2
                break
            current_mappings[tree_idx] = mapping
            current_used_edges = new_used_edges
            
        # Calculate score for this attempt
        score = 0
        for i in range(20):
            if current_mappings[i] is not None:
                score += len(tree_data[i]['edges'])
                
        if score > best_score:
            best_score = score
            best_mappings = [m[:] if m else None for m in current_mappings]
            
        if best_score == 190:
            break
                
    # Final check and conversion
    final_result = []
    for i in range(20):
        if best_mappings[i] is None:
            final_result.append(None)
        else:
            final_result.append([int(x) for x in best_mappings[i]])
            
    return final_result

# EVOLVE-BLOCK-END
