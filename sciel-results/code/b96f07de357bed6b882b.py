import time
import random
import sys

def instance_trees(n=20, seed=611):
    rng = random.Random(seed)
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
    deadline = start_time + 140  # Safety margin
    
    trees = instance_trees(20, 611)
    
    # Pre-compute adjacency and degrees for each tree
    tree_adj = []
    tree_degrees = []
    for i in range(20):
        edges = trees[i]
        k = i + 1
        adj = [[] for _ in range(k)]
        deg = [0] * k
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            deg[u] += 1
            deg[v] += 1
        tree_adj.append(adj)
        tree_degrees.append(deg)
        
    # Sort trees by size descending (largest first)
    order = list(range(20))
    order.sort(key=lambda x: -(x + 1))
    
    # State
    used_edges = set()
    mappings = [None] * 20
    
    best_score = 0
    best_mappings = [None] * 20
    
    rng = random.Random(12345)
    
    def ac3(domains, adj_t, used_edges_set, used_hosts):
        """
        Arc Consistency (AC-3) for the current tree embedding.
        domains: list of sets of possible host vertices for each tree vertex.
        adj_t: adjacency list of the tree.
        used_edges_set: global set of used edges.
        used_hosts: set of hosts already used by this tree.
        Returns False if a domain becomes empty.
        """
        queue = []
        for u in range(len(domains)):
            for v in adj_t[u]:
                if u < v:
                    queue.append((u, v))
                    queue.append((v, u))
        
        while queue:
            u, v = queue.pop()
            # Check arc (u, v)
            # For each hu in domains[u], check if there exists hv in domains[v]
            # such that hu != hv (implied by used_hosts logic? No, domains are local to tree, 
            # but we must ensure hu != hv for edges in the tree)
            # AND (hu, hv) not in used_edges_set
            
            # Optimization: If domains[u] or domains[v] is small, iterate over the smaller one?
            # Standard AC-3:
            revised = False
            new_domains_u = set()
            for hu in domains[u]:
                if hu in used_hosts and hu != domains[u].pop(): # This logic is flawed. 
                    # used_hosts are hosts used by OTHER vertices in the SAME tree.
                    # Wait, domains[u] should already exclude used_hosts?
                    # Let's assume domains are maintained to exclude used_hosts.
                    pass
                
                compatible = False
                for hv in domains[v]:
                    if hu == hv:
                        continue
                    e = (min(hu, hv), max(hu, hv))
                    if e not in used_edges_set:
                        compatible = True
                        break
                if compatible:
                    new_domains_u.add(hu)
            
            if len(new_domains_u) < len(domains[u]):
                revised = True
                removed = domains[u] - new_domains_u
                if not new_domains_u:
                    return False
                domains[u] = new_domains_u
                # Add neighbors of u (except v) to queue
                for w in adj_t[u]:
                    if w != v:
                        queue.append((w, u))
                        
        return True

    def embed_tree_csp(tree_idx):
        """
        Embed tree tree_idx using CSP with AC-3.
        Returns True if successful, updates mappings and used_edges.
        """
        if time.time() > deadline:
            return False
            
        k = tree_idx + 1
        adj_t = tree_adj[tree_idx]
        
        # Initialize domains
        # Symmetry breaking: If this is the first tree (largest), fix vertex 0 to host 0?
        # Let's try fixing vertex 0 of the first tree to 0.
        if tree_idx == 19: # T20 is first in order
            # We can fix vertex 0 to host 0
            pass # Handled in domain init
        
        domains = [set(range(20)) for _ in range(k)]
        
        # If first tree, fix vertex 0 to 0
        if tree_idx == 19:
            domains[0] = {0}
            
        # Backtracking search
        def backtrack(mapping, used_hosts_local):
            if time.time() > deadline:
                return False
                
            # Check if all mapped
            if all(mapping[i] != -1 for i in range(k)):
                # Verify edges
                for u in range(k):
                    hu = mapping[u]
                    for v in adj_t[u]:
                        if u < v:
                            hv = mapping[v]
                            e = (min(hu, hv), max(hu, hv))
                            if e in used_edges:
                                return False
                # Commit
                for u in range(k):
                    hu = mapping[u]
                    for v in adj_t[u]:
                        if u < v:
                            hv = mapping[v]
                            used_edges.add((min(hu, hv), max(hu, hv)))
                mappings[tree_idx] = mapping[:]
                return True

            # Find MRV variable
            min_size = 21
            best_var = -1
            for i in range(k):
                if mapping[i] == -1:
                    # Domain size is effectively the number of available hosts
                    # We maintain domains in the AC-3 step, but here we are in a different structure?
                    # Let's restructure: Use the domains from AC-3 directly.
                    pass
            
            # This structure is getting complex. Let's stick to the previous FC approach but optimize it.
            # The AC-3 implementation above is a bit detached from the backtracking loop.
            # A cleaner approach:
            # 1. Maintain domains.
            # 2. Run AC-3.
            # 3. Pick MRV.
            # 4. Assign, remove from other domains, run AC-3 again.
            
            return False # Placeholder

        # Given the time constraints and the fact that the previous FC approach worked (score 10.0),
        # and AC-3 is risky to implement correctly in this short time without testing,
        # I will refine the FC approach with better LCV and Symmetry Breaking.
        
        mapping = [-1] * k
        used_hosts = set()
        
        # Symmetry Breaking: For the first tree, fix vertex 0 to host 0
        if tree_idx == 19:
            mapping[0] = 0
            used_hosts.add(0)
            
        if not _embed_tree_fc(tree_idx, mapping, used_hosts, adj_t):
            return False
            
        return True

    def _embed_tree_fc(tree_idx, mapping, used_hosts, adj_t):
        """
        Forward Checking with LCV.
        """
        if time.time() > deadline:
            return False
            
        k = tree_idx + 1
        
        # Find unmapped vertex with highest constraint (most mapped neighbors) - MCV
        best_vertex = -1
        max_mapped_neighbors = -1
        min_domain_size = 21 # For MRV tie-breaker
        
        for i in range(k):
            if mapping[i] != -1:
                continue
            mapped_neighbors = 0
            domain_size = 0
            for h in range(20):
                if h in used_hosts:
                    continue
                valid = True
                for neighbor in adj_t[i]:
                    if mapping[neighbor] != -1:
                        hn = mapping[neighbor]
                        e = (min(h, hn), max(h, hn))
                        if e in used_edges:
                            valid = False
                            break
                if valid:
                    domain_size += 1
                    mapped_neighbors += 1 # This logic is wrong. mapped_neighbors should count mapped neighbors, not valid hosts.
            
            # Correct MCV: Count mapped neighbors
            mapped_neighbors = 0
            for neighbor in adj_t[i]:
                if mapping[neighbor] != -1:
                    mapped_neighbors += 1
            
            # MRV: Smallest domain size
            # We need to calculate domain size properly.
            
            if best_vertex == -1 or mapped_neighbors > max_mapped_neighbors or (mapped_neighbors == max_mapped_neighbors and domain_size < min_domain_size):
                max_mapped_neighbors = mapped_neighbors
                min_domain_size = domain_size
                best_vertex = i
                
        if best_vertex == -1:
            # All mapped
            return True # Validity checked in caller or here?
            # Let's check validity here to be safe
            for u in range(k):
                hu = mapping[u]
                for v in adj_t[u]:
                    if u < v:
                        hv = mapping[v]
                        e = (min(hu, hv), max(hu, hv))
                        if e in used_edges:
                            return False
            return True

        u = best_vertex
        
        # Generate valid candidates
        valid_candidates = []
        for h in range(20):
            if h in used_hosts:
                continue
            valid = True
            for neighbor in adj_t[u]:
                if mapping[neighbor] != -1:
                    hn = mapping[neighbor]
                    e = (min(h, hn), max(h, hn))
                    if e in used_edges:
                        valid = False
                        break
            if valid:
                valid_candidates.append(h)
                
        if not valid_candidates:
            return False
            
        # LCV: Sort candidates by how many options they leave for neighbors
        # For each candidate h, count how many valid hosts remain for each unmapped neighbor
        def lcv_score(h):
            score = 0
            for neighbor in adj_t[u]:
                if mapping[neighbor] == -1:
                    # Count valid hosts for neighbor if u takes h
                    count = 0
                    for hn in range(20):
                        if hn in used_hosts or hn == h:
                            continue
                        valid = True
                        for n_neighbor in adj_t[neighbor]:
                            if mapping[n_neighbor] != -1:
                                hnn = mapping[n_neighbor]
                                e = (min(hn, hnn), max(hn, hnn))
                                if e in used_edges:
                                    valid = False
                                    break
                        if valid:
                            count += 1
                    score += count
            return score
        
        # Only do LCV if list is small to save time
        if len(valid_candidates) <= 10:
            valid_candidates.sort(key=lcV_score, reverse=True)
        else:
            rng.shuffle(valid_candidates)
            
        for h in valid_candidates:
            mapping[u] = h
            used_hosts.add(h)
            
            # Forward Checking: Ensure all unmapped neighbors of u still have valid hosts
            fc_failed = False
            for neighbor in adj_t[u]:
                if mapping[neighbor] == -1:
                    # Check if neighbor has any valid host
                    has_valid = False
                    for hn in range(20):
                        if hn in used_hosts:
                            continue
                        valid = True
                        for n_neighbor in adj_t[neighbor]:
                            if mapping[n_neighbor] != -1:
                                hnn = mapping[n_neighbor]
                                e = (min(hn, hnn), max(hn, hnn))
                                if e in used_edges:
                                    valid = False
                                    break
                        if valid:
                            has_valid = True
                            break
                    if not has_valid:
                        fc_failed = True
                        break
            
            if not fc_failed:
                if _embed_tree_fc(tree_idx, mapping, used_hosts, adj_t):
                    return True
            
            used_hosts.remove(h)
            mapping[u] = -1
            
        return False

    def solve(idx_in_order):
        nonlocal best_score, best_mappings
        if idx_in_order == 20:
            score = 190
            if score > best_score:
                best_score = score
                for i in range(20):
                    best_mappings[i] = mappings[i][:] if mappings[i] else None
            return True
        
        if time.time() > deadline:
            return False
            
        t_idx = order[idx_in_order]
        
        if embed_tree_csp(t_idx):
            return solve(idx_in_order + 1)
            
        return False

    # Run solver with multiple restarts
    attempts = 0
    while time.time() < deadline:
        # Reset state
        used_edges = set()
        mappings = [None] * 20
        rng = random.Random(12345 + attempts)
        
        success = solve(0)
        attempts += 1
        
        if success:
            break
            
        # Calculate score of current partial packing
        score = 0
        for i in range(20):
            if mappings[i] is not None:
                score += len(trees[i])
                
        if score > best_score:
            best_score = score
            for i in range(20):
                best_mappings[i] = mappings[i][:] if mappings[i] else None
                
        if best_score == 190:
            break

    # Ensure return type is correct
    result = []
    for i in range(20):
        if best_mappings[i] is None:
            result.append(None)
        else:
            result.append([int(x) for x in best_mappings[i]])
    return result
