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
    deadline = start_time + 115.0
    
    trees = instance_trees(20, 611)
    num_trees = 20
    
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
            
    order = list(range(19, -1, -1))
    best_score = 0
    best_mappings = [None] * 20
    
    def embed_tree_csp_fc(tree_idx, used_edges, rng):
        """
        CSP with Forward Checking.
        """
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
            
        edges = tree_data[tree_idx]['edges']
        adj = tree_data[tree_idx]['adj']
        
        # Initial domains
        initial_domains = {v: set(range(20)) for v in range(k)}
        
        # Precompute edge keys for quick lookup? 
        # used_edges is a set of tuples.
        
        def backtrack(assigned, domains):
            if len(assigned) == k:
                # All assigned. Verify edges just in case (should be guaranteed by FC)
                for u, v in edges:
                    hu = assigned[u]
                    hv = assigned[v]
                    e_key = (min(hu, hv), max(hu, hv))
                    if e_key in used_edges:
                        return None
                return assigned.copy()
            
            # MRV: Pick unassigned variable with smallest domain
            unassigned = [v for v in range(k) if v not in assigned]
            if not unassigned:
                return None # Should not happen if len(assigned) < k
                
            min_size = 21
            best_v = -1
            best_domain = None
            
            for v in unassigned:
                size = len(domains[v])
                if size < min_size:
                    min_size = size
                    best_v = v
                    best_domain = domains[v]
                    if size == 0:
                        break
            
            if best_v == -1 or min_size == 0:
                return None
                
            # Value Ordering: Randomize
            values = list(best_domain)
            rng.shuffle(values)
            
            for h in values:
                # Assign best_v -> h
                new_assigned = assigned.copy()
                new_assigned[best_v] = h
                
                # Forward Check: Update domains for unassigned neighbors and all unassigned
                new_domains = {}
                for v in range(k):
                    if v in new_assigned:
                        continue
                    # Start with current domain
                    new_dom = set(domains[v])
                    
                    # 1. Injectivity: Remove h
                    new_dom.discard(h)
                    
                    # 2. Edge Constraint: If v is neighbor of best_v, check edges
                    if best_v in adj[v]:
                        # For each potential host h' in new_dom, check if (h, h') is used
                        # If (h, h') is in used_edges, h' is invalid
                        to_remove = set()
                        for h_prime in new_dom:
                            e_key = (min(h, h_prime), max(h, h_prime))
                            if e_key in used_edges:
                                to_remove.add(h_prime)
                        new_dom -= to_remove
                        
                    if not new_dom:
                        # Failure
                        new_domains = None
                        break
                    new_domains[v] = new_dom
                
                if new_domains is None:
                    continue
                    
                result = backtrack(new_assigned, new_domains)
                if result is not None:
                    return result
                    
            return None
        
        # Start with root assignment?
        # MRV will pick the first one if all sizes are 20.
        # To break symmetry, we can fix vertex 0 to a specific host?
        # Or just let MRV handle it.
        # However, we can optimize by trying to assign the highest degree vertex first.
        # MRV does this dynamically.
        
        # One optimization: If we fix the first vertex, we reduce the search space by factor 20.
        # Let's pick the vertex with max degree as the first to assign, and try all hosts for it.
        degrees = [len(adj[i]) for i in range(k)]
        root = max(range(k), key=lambda i: degrees[i])
        
        # We can just call backtrack with empty assigned.
        # But to be safe and fast, let's iterate over hosts for the root manually?
        # No, backtrack does MRV. Initially all domains are size 20.
        # It will pick vertex 0 (or whichever comes first in iteration).
        # To ensure we pick the root, we can sort unassigned by degree descending in MRV tie-breaker?
        # Or just force the first assignment.
        
        # Let's force the first assignment to the root to reduce branching.
        for h in range(20):
            initial_assigned = {root: h}
            # Initial domains for others: 
            # All hosts except h are available.
            # Neighbors of root: check edge constraints with h.
            
            init_domains = {}
            valid = True
            for v in range(k):
                if v == root:
                    continue
                dom = set(range(20))
                dom.discard(h)
                
                if root in adj[v]:
                    to_remove = set()
                    for h_prime in dom:
                        e_key = (min(h, h_prime), max(h, h_prime))
                        if e_key in used_edges:
                            to_remove.add(h_prime)
                    dom -= to_remove
                
                if not dom:
                    valid = False
                    break
                init_domains[v] = dom
            
            if not valid:
                continue
                
            result = backtrack(initial_assigned, init_domains)
            if result is not None:
                mapping = [0] * k
                for tv, hv in result.items():
                    mapping[tv] = hv
                new_used_edges = set(used_edges)
                for u, v in edges:
                    hu = mapping[u]
                    hv = mapping[v]
                    e_key = (min(hu, hv), max(hu, hv))
                    new_used_edges.add(e_key)
                return mapping, new_used_edges
                
        return None, None

    # Main Loop
    for restart in range(1000):
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
                
            mapping, new_used_edges = embed_tree_csp_fc(tree_idx, current_used_edges, rng)
            if mapping is None:
                success = False
                break
            current_mappings[tree_idx] = mapping
            current_used_edges = new_used_edges
            
        if success:
            score = 0
            for i in range(20):
                if current_mappings[i] is not None:
                    score += len(tree_data[i]['edges'])
                    
            if score > best_score:
                best_score = score
                best_mappings = [m[:] if m else None for m in current_mappings]
                
            if best_score == 190:
                break
                
    final_result = []
    for i in range(20):
        if best_mappings[i] is None:
            final_result.append(None)
        else:
            final_result.append([int(x) for x in best_mappings[i]])
            
    return final_result

# EVOLVE-BLOCK-END
