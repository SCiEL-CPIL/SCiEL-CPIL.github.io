import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Leave 10s margin for safety

    def _find_k5(adj_masks, n):
        """Find a clique of size 5. Returns list of 5 vertices or None."""
        for v in range(n):
            mask_v = adj_masks[v]
            mv = mask_v
            while mv:
                lsb_u = mv & -mv
                u = lsb_u.bit_length() - 1
                mv ^= lsb_u
                
                # C = N(v) ∩ N(u)
                common = mask_v & adj_masks[u]
                
                # Find a triangle in 'common'
                mc = common
                while mc:
                    lsb_w = mc & -mc
                    w = lsb_w.bit_length() - 1
                    mc ^= lsb_w
                    
                    # Find edge (x, y) in common such that x, y are both neighbors of w
                    candidates_x = adj_masks[w] & common
                    cx = candidates_x
                    while cx:
                        lsb_x = cx & -cx
                        x = lsb_x.bit_length() - 1
                        cx ^= lsb_x
                        
                        # Find y in common, adj to w, adj to x
                        candidates_y = (adj_masks[w] & common) & adj_masks[x]
                        if candidates_y:
                            lsb_y = candidates_y & -candidates_y
                            y = lsb_y.bit_length() - 1
                            return [v, u, w, x, y]
        return None

    def _find_ind5(adj_masks, n):
        # Find independent set of size 5.
        # This is a K5 in the complement graph.
        full_mask = (1 << n) - 1
        comp_masks = [0] * n
        for i in range(n):
            comp_masks[i] = full_mask & ~adj_masks[i] & ~(1 << i)
        
        return _find_k5(comp_masks, n)

    def _build_adj_masks(n, edges):
        adj_masks = [0] * n
        for u, v in edges:
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
        return adj_masks

    def _paley_graph(q):
        # q must be a prime p ≡ 1 (mod 4)
        residues = set()
        for x in range(1, q):
            residues.add((x * x) % q)
        
        edges = []
        for i in range(q):
            for j in range(i + 1, q):
                diff = (j - i) % q
                if diff in residues:
                    edges.append((i, j))
        return edges

    def _is_55_free(adj_masks, n):
        """Check if graph is (5,5)-free."""
        if _find_k5(adj_masks, n):
            return False
        if _find_ind5(adj_masks, n):
            return False
        return True

    def _repair_graph(adj_masks, n, max_steps=5000):
        """
        Repair a graph to be (5,5)-free by removing K5s and adding edges to break ind5s.
        Returns (success, adj_masks, edges) or (False, None, None).
        """
        # Build edges from adj_masks
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (adj_masks[i] >> j) & 1:
                    edges.append((i, j))
        
        edges_set = set(edges)
        current_adj = [m for m in adj_masks]
        
        for step in range(max_steps):
            if time.time() > deadline:
                return False, None, None
            
            # Check for K5
            k5 = _find_k5(current_adj, n)
            if k5:
                # Remove a random edge from the clique
                i, j = random.sample(k5, 2)
                if i > j:
                    i, j = j, i
                if (i, j) in edges_set:
                    edges_set.discard((i, j))
                    current_adj[i] &= ~(1 << j)
                    current_adj[j] &= ~(1 << i)
                continue
            
            # Check for Independent Set 5
            ind5 = _find_ind5(current_adj, n)
            if ind5:
                # Add a random edge in the independent set
                i, j = random.sample(ind5, 2)
                if i > j:
                    i, j = j, i
                if (i, j) not in edges_set:
                    edges_set.add((i, j))
                    current_adj[i] |= (1 << j)
                    current_adj[j] |= (1 << i)
                continue
            
            # Valid!
            return True, current_adj, list(edges_set)
        
        return False, None, None

    def _try_extend(adj_masks, n, target_n, max_restarts=100):
        """
        Try to extend a valid graph from n vertices to target_n vertices.
        Returns (success, adj_masks, edges) or (False, None, None).
        """
        if n >= target_n:
            return True, adj_masks, []
        
        # Build edges from adj_masks
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (adj_masks[i] >> j) & 1:
                    edges.append((i, j))
        
        edges_set = set(edges)
        current_adj = [m for m in adj_masks]
        current_n = n
        
        for target in range(current_n + 1, target_n + 1):
            if time.time() > deadline:
                return False, None, None
            
            added = False
            for restart in range(max_restarts):
                if time.time() > deadline:
                    return False, None, None
                
                # Try different strategies for connecting new vertex
                strategy = restart % 3
                
                if strategy == 0:
                    # Random with p=0.5
                    new_adj = 0
                    for i in range(current_n):
                        if random.random() < 0.5:
                            new_adj |= (1 << i)
                elif strategy == 1:
                    # Start with all edges, remove some
                    new_adj = (1 << current_n) - 1
                    for i in range(current_n):
                        if random.random() < 0.3:
                            new_adj &= ~(1 << i)
                else:
                    # Start with no edges, add some
                    new_adj = 0
                    for i in range(current_n):
                        if random.random() < 0.7:
                            new_adj |= (1 << i)
                
                # Build new adj masks
                new_adj_list = list(current_adj)
                new_adj_list.append(new_adj)
                
                # Update adj masks for existing vertices
                for i in range(current_n):
                    if (new_adj >> i) & 1:
                        new_adj_list[i] |= (1 << current_n)
                
                # Check validity
                k5 = _find_k5(new_adj_list, current_n + 1)
                if k5:
                    continue
                ind5 = _find_ind5(new_adj_list, current_n + 1)
                if ind5:
                    continue
                
                # Valid!
                # Add edges to set
                for i in range(current_n):
                    if (new_adj >> i) & 1:
                        edges_set.add((i, current_n))
                
                current_n += 1
                current_adj = new_adj_list
                added = True
                break
            
            if not added:
                return False, None, None
        
        return True, current_adj, list(edges_set)

    best_n = 37
    best_edges = _paley_graph(37)

    # Strategy 1: Try to repair Paley(41)
    for q in [41]:
        if time.time() > deadline:
            break
        
        seed_edges = _paley_graph(q)
        adj_masks = _build_adj_masks(q, seed_edges)
        
        for restart in range(50):
            if time.time() > deadline:
                break
            
            success, repaired_adj, repaired_edges = _repair_graph(adj_masks, q, max_steps=3000)
            if success:
                if q > best_n:
                    best_n = q
                    best_edges = repaired_edges
                
                # Try to extend
                for target_n in range(q + 1, 47):
                    if time.time() > deadline:
                        break
                    
                    ext_success, ext_adj, ext_edges = _try_extend(repaired_adj, q, target_n, max_restarts=30)
                    if ext_success:
                        if target_n > best_n:
                            best_n = target_n
                            best_edges = ext_edges
                        repaired_adj = ext_adj
                    else:
                        break

    # Strategy 2: Try to extend Paley(37) directly
    if best_n == 37 and time.time() < deadline - 60:
        seed_edges = _paley_graph(37)
        adj_masks = _build_adj_masks(37, seed_edges)
        
        for target_n in range(38, 47):
            if time.time() > deadline:
                break
            
            ext_success, ext_adj, ext_edges = _try_extend(adj_masks, 37, target_n, max_restarts=50)
            if ext_success:
                if target_n > best_n:
                    best_n = target_n
                    best_edges = ext_edges
            else:
                break

    # Strategy 3: Random search with repair for n in [38, 46]
    if best_n < 46 and time.time() < deadline - 30:
        for n in range(46, 37, -1):
            if time.time() > deadline:
                break
            
            found = False
            for restart in range(20):
                if time.time() > deadline:
                    break
                
                # Start with random graph
                edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if random.random() < 0.5:
                            edges.append((i, j))
                
                adj_masks = _build_adj_masks(n, edges)
                
                success, repaired_adj, repaired_edges = _repair_graph(adj_masks, n, max_steps=2000)
                if success:
                    if n > best_n:
                        best_n = n
                        best_edges = repaired_edges
                    found = True
                    break
            
            if found:
                break  # Found a valid graph for this n, try larger n next? No, we're going down.
            # If we didn't find for n, we can stop since we're going down from 46

    return best_n, best_edges
