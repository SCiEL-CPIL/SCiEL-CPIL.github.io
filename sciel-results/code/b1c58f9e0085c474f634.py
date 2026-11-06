import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 290.0  # 10s margin
    
    # Seed: Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    seed_edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
    
    def find_clique_fast(adj, n, size):
        """Find a clique of given size using backtracking with bitmasks."""
        if n < size:
            return None
        
        # Precompute: for each vertex, its neighbors as bitmask
        # adj[v] is already a bitmask of neighbors
        
        def dfs(candidates, current_size, current_list):
            if current_size == size:
                return current_list[:]
            
            # Pruning: if not enough candidates left
            if candidates.bit_count() < size - current_size:
                return None
            
            while candidates:
                # Pick a vertex with high degree in candidates for better pruning
                # Try vertices with most connections among remaining candidates
                best_v = -1
                best_deg = -1
                temp = candidates
                while temp:
                    lsb = temp & -temp
                    v = lsb.bit_length() - 1
                    temp ^= lsb
                    deg = (candidates & adj[v]).bit_count()
                    if deg > best_deg:
                        best_deg = deg
                        best_v = v
                
                v = best_v
                candidates ^= (1 << v)
                
                new_candidates = candidates & adj[v]
                current_list.append(v)
                res = dfs(new_candidates, current_size + 1, current_list)
                if res:
                    return res
                current_list.pop()
            return None
        
        return dfs((1 << n) - 1, 0, [])
    
    def count_cliques(adj, n, size):
        """Count cliques of given size (for heuristic)."""
        if n < size:
            return 0
        
        count = 0
        
        def dfs(candidates, current_size):
            nonlocal count
            if current_size == size:
                count += 1
                return
            
            if candidates.bit_count() < size - current_size:
                return
            
            while candidates:
                lsb = candidates & -candidates
                v = lsb.bit_length() - 1
                candidates ^= lsb
                
                new_candidates = candidates & adj[v]
                dfs(new_candidates, current_size + 1)
        
        dfs((1 << n) - 1, 0)
        return count
    
    def is_valid(adj, n):
        """Check if graph has no K5 and no independent set of size 5."""
        # Check for K5
        if find_clique_fast(adj, n, 5):
            return False
        # Check for Independent Set of 5 (Clique of 5 in complement)
        adj_comp = [((1 << n) - 1) ^ (1 << i) ^ adj[i] for i in range(n)]
        if find_clique_fast(adj_comp, n, 5):
            return False
        return True
    
    def repair_graph(adj, n, max_steps=5000):
        """Iteratively repair graph to remove K5 and I5 violations."""
        steps = 0
        while steps < max_steps:
            if time.time() > deadline:
                return False, adj
            
            # Check for K5
            clique = find_clique_fast(adj, n, 5)
            if clique:
                # Find the edge in the clique whose removal is most beneficial
                # Heuristic: remove edge with minimum "impact" - i.e., edge that
                # is part of the fewest other potential K5s
                best_edge = None
                best_score = float('inf')
                
                for u, v in itertools.combinations(clique, 2):
                    # Score: how many other vertices are connected to both u and v
                    # (i.e., how many K5s this edge might be part of)
                    common = adj[u] & adj[v]
                    score = common.bit_count()
                    if score < best_score:
                        best_score = score
                        best_edge = (u, v)
                
                if best_edge:
                    u, v = best_edge
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    steps += 1
                    continue
            
            # Check for I5 (Clique in complement)
            adj_comp = [((1 << n) - 1) ^ (1 << i) ^ adj[i] for i in range(n)]
            indep = find_clique_fast(adj_comp, n, 5)
            if indep:
                # Find the non-edge in the independent set whose addition is most beneficial
                # Heuristic: add edge that connects vertices with most common non-neighbors
                best_edge = None
                best_score = float('inf')
                
                for u, v in itertools.combinations(indep, 2):
                    # In the complement, u and v are connected, meaning they're NOT connected in original
                    # Score: how many other vertices are non-adjacent to both u and v in original
                    # = how many vertices are adjacent to both in complement
                    common = adj_comp[u] & adj_comp[v]
                    score = common.bit_count()
                    if score < best_score:
                        best_score = score
                        best_edge = (u, v)
                
                if best_edge:
                    u, v = best_edge
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    steps += 1
                    continue
            
            # No violations found
            return True, adj
        
        return False, adj
    
    def local_search_polish(adj, n, max_steps=2000):
        """Try to improve the graph by flipping edges that don't create violations."""
        steps = 0
        while steps < max_steps:
            if time.time() > deadline:
                return adj
            
            # Try flipping a random edge
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                steps += 1
                continue
            
            if u > v:
                u, v = v, u
            
            # Flip the edge
            had_edge = bool(adj[u] & (1 << v))
            if had_edge:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            
            # Check if still valid
            if not is_valid(adj, n):
                # Revert
                if had_edge:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
            
            steps += 1
        
        return adj
    
    def try_extend_graph(adj, n, target_n):
        """Try to extend a valid n-vertex graph to target_n vertices."""
        if n >= target_n:
            return adj, n
        
        # Try adding vertices one at a time
        current_n = n
        current_adj = adj[:]
        
        for new_v in range(current_n, target_n):
            if time.time() > deadline:
                break
            
            # Try different connection patterns for the new vertex
            best_pattern = None
            best_score = float('inf')
            
            # Try a few random patterns
            for _ in range(20):
                pattern = 0
                for old_v in range(current_n):
                    if random.random() < 0.5:
                        pattern |= (1 << old_v)
                
                # Create new adj with this pattern
                new_adj = current_adj[:]
                new_adj.append(pattern)
                for old_v in range(current_n):
                    if pattern & (1 << old_v):
                        new_adj[old_v] |= (1 << current_n)
                
                # Check if valid
                if is_valid(new_adj, current_n + 1):
                    # Score: prefer patterns that are "balanced"
                    deg = pattern.bit_count()
                    score = abs(deg - current_n // 2)
                    if score < best_score:
                        best_score = score
                        best_pattern = pattern
            
            if best_pattern is not None:
                current_adj.append(best_pattern)
                for old_v in range(current_n):
                    if best_pattern & (1 << old_v):
                        current_adj[old_v] |= (1 << current_n)
                current_n += 1
            else:
                # Try with repair
                for _ in range(5):
                    pattern = 0
                    for old_v in range(current_n):
                        if random.random() < 0.5:
                            pattern |= (1 << old_v)
                    
                    new_adj = current_adj[:]
                    new_adj.append(pattern)
                    for old_v in range(current_n):
                        if pattern & (1 << old_v):
                            new_adj[old_v] |= (1 << current_n)
                    
                    success, repaired = repair_graph(new_adj, current_n + 1, max_steps=100)
                    if success:
                        current_adj = repaired
                        current_n += 1
                        break
        
        return current_adj, current_n
    
    # Try to find a valid graph on as many vertices as possible
    # Start with target 42, try to go higher
    found_valid = False
    valid_adj = None
    valid_n = 0
    
    # First, try to find a 42-vertex graph
    n_target = 42
    
    for attempt in range(30):
        if time.time() > deadline:
            break
        
        # Initialize random graph
        adj = [0] * n_target
        for i in range(n_target):
            for j in range(i + 1, n_target):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        success, adj = repair_graph(adj, n_target, max_steps=3000)
        if success:
            found_valid = True
            valid_adj = adj
            valid_n = n_target
            break
    
    # If we found a valid graph, try to extend it
    if found_valid:
        # Try to extend to 43, 44, 45, 46
        for target in range(valid_n + 1, 47):
            if time.time() > deadline:
                break
            
            extended_adj, extended_n = try_extend_graph(valid_adj, valid_n, target)
            if extended_n > valid_n:
                valid_adj = extended_adj
                valid_n = extended_n
    
    # Also try direct construction for larger n
    for n_try in [43, 44, 45, 46]:
        if time.time() > deadline:
            break
        if n_try <= valid_n:
            continue
        
        for attempt in range(10):
            if time.time() > deadline:
                break
            
            adj = [0] * n_try
            for i in range(n_try):
                for j in range(i + 1, n_try):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            success, adj = repair_graph(adj, n_try, max_steps=2000)
            if success:
                if n_try > valid_n:
                    valid_adj = adj
                    valid_n = n_try
                break
    
    # Use the best valid graph found
    if found_valid and valid_n > 0:
        edges = []
        for i in range(valid_n):
            for j in range(i + 1, valid_n):
                if valid_adj[i] & (1 << j):
                    edges.append((i, j))
        return valid_n, edges
    
    # Fallback to seed
    return q, seed_edges
