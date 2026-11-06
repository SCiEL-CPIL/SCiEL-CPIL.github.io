import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    def has_independent_set_10(adj, n):
        """Check if graph has independent set of size 10. Returns True if it does."""
        mask_all = (1 << n) - 1
        
        # Greedy lower bound: try to find a large independent set quickly
        candidates = mask_all
        size = 0
        while candidates and size < 10:
            best_v = -1
            min_deg = n + 1
            temp = candidates
            while temp:
                v = (temp & -temp).bit_length() - 1
                temp &= temp - 1
                deg = (adj[v] & candidates).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
            
            if best_v == -1:
                break
                
            size += 1
            candidates &= ~adj[best_v] & ~(1 << best_v)
            
        if size >= 10:
            return True
            
        def backtrack(candidates, size):
            if size == 10:
                return True
            if candidates == 0:
                return False
            
            remaining = candidates.bit_count()
            if size + remaining < 10:
                return False
            
            # Upper bound: even if we pick all remaining, can we reach 10?
            if size + remaining < 10:
                return False
            
            # Pick vertex with max degree in candidates (removes most from neighbors)
            best_v = -1
            max_deg = -1
            temp = candidates
            while temp:
                v = (temp & -temp).bit_length() - 1
                temp &= temp - 1
                deg = (adj[v] & candidates).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
            
            if best_v == -1:
                return False
            
            # Branch 1: include best_v
            if backtrack(candidates & ~adj[best_v] & ~(1 << best_v), size + 1):
                return True
            # Branch 2: exclude best_v
            if backtrack(candidates & ~(1 << best_v), size):
                return True
            return False
        
        return backtrack(mask_all, 0)
    
    def is_triangle_free(adj, n):
        """Check if graph is triangle-free."""
        for u in range(n):
            neighbors = adj[u]
            temp = neighbors
            while temp:
                v = (temp & -temp).bit_length() - 1
                temp &= temp - 1
                if v > u:
                    if adj[u] & adj[v]:
                        return False
        return True

    def independent_set_size(adj, n):
        """Find the maximum independent set size using greedy + local search."""
        mask_all = (1 << n) - 1
        
        # Greedy: repeatedly pick vertex with minimum degree in remaining candidates
        candidates = mask_all
        size = 0
        while candidates and size < 10:
            best_v = -1
            min_deg = n + 1
            temp = candidates
            while temp:
                v = (temp & -temp).bit_length() - 1
                temp &= temp - 1
                deg = (adj[v] & candidates).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
            
            if best_v == -1:
                break
                
            size += 1
            candidates &= ~adj[best_v] & ~(1 << best_v)
            
        return size
    
    def build_edge_list(adj, n):
        """Convert adjacency bitmasks to edge list."""
        edges = []
        for i in range(n):
            temp = adj[i]
            j = 0
            while temp:
                lsb = temp & -temp
                j = lsb.bit_length() - 1
                if j > i:
                    edges.append([i, j])
                temp &= temp - 1
        return edges
    
    def make_andrasfai(k):
        """Andrásfai graph And(k): vertices Z_{3k}, i~j iff (j-i) mod 3k in {1, 3k-1}.
        Triangle-free, alpha = k."""
        n = 3 * k
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                d = (j - i) % n
                if d == 1 or d == n - 1:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj, n
    
    def mycielski(adj, n):
        """Mycielski construction: given triangle-free graph on n vertices,
        produce triangle-free graph on 2n+1 vertices with increased chromatic number."""
        new_n = 2 * n + 1
        new_adj = [0] * new_n
        
        # Copy original graph
        for i in range(n):
            new_adj[i] = adj[i]
        
        # Add mirror vertices n..2n-1: mirror of i is n+i, edges from mirror of i to neighbors of i
        for i in range(n):
            for j in range(n):
                if adj[i] & (1 << j):
                    new_adj[n + i] |= (1 << j)
                    new_adj[j] |= (1 << (n + i))
        
        # Add universal vertex 2n connected to all mirror vertices
        for i in range(n):
            new_adj[2 * n] |= (1 << (n + i))
            new_adj[n + i] |= (1 << (2 * n))
        
        return new_adj, new_n
    
    def blow_up(adj, n, sizes):
        """Blow up a triangle-free graph: each vertex i becomes an independent set of size sizes[i].
        Result is triangle-free. Returns (new_adj, new_n)."""
        new_n = sum(sizes)
        new_adj = [0] * new_n
        offset = [0] * n
        for i in range(n):
            offset[i] = new_n - sum(sizes[i:])  # wrong, let me fix
        # Better: compute offsets
        offsets = [0] * (n + 1)
        for i in range(n):
            offsets[i + 1] = offsets[i] + sizes[i]
        
        for i in range(n):
            for j in range(n):
                if i != j and (adj[i] & (1 << j)):
                    # All vertices in part i connected to all in part j
                    for a in range(offsets[i], offsets[i + 1]):
                        for b in range(offsets[j], offsets[j + 1]):
                            new_adj[a] |= (1 << b)
                            new_adj[b] |= (1 << a)
        
        return new_adj, new_n
    
    def random_bipartite(n, p):
        """Generate random bipartite graph with n/2 in each part."""
        half = n // 2
        adj = [0] * n
        for i in range(half):
            for j in range(half, n):
                if random.random() < p:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    def random_triangle_free_optimized(n, p):
        """Generate random triangle-free graph with controlled edge density."""
        adj = [0] * n
        # Random permutation of edges
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(edges)
        
        for u, v in edges:
            # Check if adding edge creates triangle
            common = adj[u] & adj[v]
            if common == 0 and random.random() < p:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        return adj
    
    def make_random_triangle_free(n):
        """Generate a random triangle-free graph using random edge addition with triangle check."""
        adj = [0] * n
        # Random permutation of edges
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(edges)
        
        for u, v in edges:
            common = adj[u] & adj[v]
            if common == 0:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        return adj
    
    # Try known constructions first, from largest to smallest
    best_result = None
    
    # Andrásfai graphs: And(k) has n=3k, alpha=k. Need k<10 for alpha<10.
    # And(9): n=27, alpha=9 < 10. Valid!
    # And(10): n=30, alpha=10. NOT valid (alpha=10).
    # So And(9) is the largest Andrásfai that works directly.
    
    # Try Andrásfai And(9)
    adj9, n9 = make_andrasfai(9)
    if not has_independent_set_10(adj9, n9):
        best_result = {'n': n9, 'edges': build_edge_list(adj9, n9)}
    
    # Try Mycielski constructions
    # Start with C5 (5-cycle), apply Mycielski to get larger triangle-free graphs
    # C5: n=5, alpha=2
    # Mycielski(C5) = Grötzsch: n=11, alpha=3
    # Mycielski(Grötzsch): n=23, alpha=4
    # Mycielski(23): n=47, too big
    
    # C5
    adj_c5 = [0] * 5
    for i in range(5):
        adj_c5[i] |= (1 << ((i + 1) % 5))
        adj_c5[i] |= (1 << ((i + 4) % 5))
    
    # Mycielski(C5) = Grötzsch graph
    adj_grotzsch, n_grotzsch = mycielski(adj_c5, 5)
    if n_grotzsch <= 41 and not has_independent_set_10(adj_grotzsch, n_grotzsch):
        if best_result is None or n_grotzsch > best_result['n']:
            best_result = {'n': n_grotzsch, 'edges': build_edge_list(adj_grotzsch, n_grotzsch)}
    
    # Mycielski(Grötzsch)
    adj_m2, n_m2 = mycielski(adj_grotzsch, n_grotzsch)
    if n_m2 <= 41 and not has_independent_set_10(adj_m2, n_m2):
        if best_result is None or n_m2 > best_result['n']:
            best_result = {'n': n_m2, 'edges': build_edge_list(adj_m2, n_m2)}
    
    # Try blow-up of Andrásfai graphs
    # And(3): n=9, alpha=3. Blow up to get larger graphs.
    adj_a3, n_a3 = make_andrasfai(3)
    for target_n in [36, 37, 38, 39, 40]:
        if time.time() > deadline:
            break
        # Try different blow-up sizes
        for trial in range(20):
            sizes = [1] * n_a3
            remaining = target_n - n_a3
            for _ in range(remaining):
                sizes[random.randint(0, n_a3 - 1)] += 1
            adj, _ = blow_up(adj_a3, n_a3, sizes)
            if not has_independent_set_10(adj, target_n):
                if best_result is None or target_n > best_result['n']:
                    best_result = {'n': target_n, 'edges': build_edge_list(adj, target_n)}
                break
    
    # Now try random search for larger graphs
    # Strategy: Try various n from 40 down to (best_result['n']+1 if best_result else 28)
    min_target = (best_result['n'] + 1) if best_result else 28
    
    for target_n in range(40, min_target - 1, -1):
        if time.time() > deadline:
            break
        
        found = False
        attempts = 0
        max_attempts = 8000
        
        while attempts < max_attempts and time.time() < deadline:
            attempts += 1
            
            # Try different random generation strategies
            strategy = random.randint(0, 3)
            
            if strategy == 0:
                # Random edge addition with probability control
                p = random.uniform(0.3, 0.7)
                adj = random_triangle_free_optimized(target_n, p)
            elif strategy == 1:
                # Random bipartite with some added edges
                adj = random_bipartite(target_n, random.uniform(0.4, 0.6))
                # Try to add some edges within parts while maintaining triangle-free
                half = target_n // 2
                extra_edges = [(i, j) for i in range(half) for j in range(i + 1, half)]
                extra_edges += [(i, j) for i in range(half, target_n) for j in range(i + 1, target_n)]
                random.shuffle(extra_edges)
                for u, v in extra_edges:
                    common = adj[u] & adj[v]
                    if common == 0:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
            elif strategy == 2:
                # Blow up a small triangle-free graph
                # Use C5 blown up
                base_n = 5
                base_adj = adj_c5[:]
                # Random sizes that sum to target_n
                sizes = [1] * base_n
                remaining = target_n - base_n
                for _ in range(remaining):
                    sizes[random.randint(0, base_n - 1)] += 1
                adj, _ = blow_up(base_adj, base_n, sizes)
            else:
                # Blow up Andrásfai And(3)
                base_n = 9
                base_adj, _ = make_andrasfai(3)
                sizes = [1] * base_n
                remaining = target_n - base_n
                for _ in range(remaining):
                    sizes[random.randint(0, base_n - 1)] += 1
                adj, _ = blow_up(base_adj, base_n, sizes)
            
            if not has_independent_set_10(adj, target_n):
                found = True
                best_result = {'n': target_n, 'edges': build_edge_list(adj, target_n)}
                break
        
        if found:
            break
    
    if best_result is None:
        # Fallback: Andrásfai And(9)
        adj9, n9 = make_andrasfai(9)
        best_result = {'n': n9, 'edges': build_edge_list(adj9, n9)}
    
    # Final verification
    final_adj = [0] * best_result['n']
    for u, v in best_result['edges']:
        final_adj[u] |= (1 << v)
        final_adj[v] |= (1 << u)
    
    # Verify triangle-free
    assert is_triangle_free(final_adj, best_result['n']), "Result is not triangle-free!"
    # Verify no independent set of size 10
    assert not has_independent_set_10(final_adj, best_result['n']), "Result has independent set of size 10!"
    
    return best_result
