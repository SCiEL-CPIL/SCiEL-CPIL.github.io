import time
import random
import sys
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin

    def compute_alpha_fast(adj_mask, n):
        """Compute maximum independent set size using improved Branch and Bound."""
        best = [0]
        
        # Precompute degrees for ordering
        def dfs(candidates, size):
            if size > best[0]:
                best[0] = size
            if candidates == 0:
                return
            
            # Upper bound: current size + number of remaining candidates
            remaining = candidates.bit_count()
            if size + remaining <= best[0]:
                return
            
            # Simple greedy upper bound: try to find a maximal independent set
            # in the candidate subgraph to get a tighter bound
            greedy_candidates = candidates
            greedy_size = 0
            while greedy_candidates:
                # Pick vertex with minimum degree in remaining candidates
                min_deg = n + 1
                chosen = -1
                c = greedy_candidates
                while c:
                    lsb = c & (-c)
                    v = lsb.bit_length() - 1
                    c ^= lsb
                    deg = (adj_mask[v] & greedy_candidates).bit_count()
                    if deg < min_deg:
                        min_deg = deg
                        chosen = v
                if chosen == -1:
                    break
                greedy_size += 1
                greedy_candidates &= ~(adj_mask[chosen] | (1 << chosen))
            
            # If even greedy can't beat best, prune
            if size + greedy_size <= best[0]:
                return
            
            # Pick a vertex to branch on
            # Heuristic: Pick vertex with max degree in the candidate subgraph
            c = candidates
            max_deg = -1
            chosen = -1
            while c:
                lsb = c & (-c)
                v = lsb.bit_length() - 1
                c ^= lsb
                deg = (adj_mask[v] & candidates).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    chosen = v
            
            if chosen == -1:
                return
                
            # Branch 1: Include 'chosen'
            new_candidates = candidates & ~(adj_mask[chosen] | (1 << chosen))
            dfs(new_candidates, size + 1)
            
            # Branch 2: Exclude 'chosen'
            new_candidates = candidates & ~(1 << chosen)
            dfs(new_candidates, size)

        all_candidates = (1 << n) - 1
        dfs(all_candidates, 0)
        return best[0]

    def check_c4_free(adj_mask, n):
        """Check if graph is C4-free using bitmasks - O(n^2) with bit operations."""
        for u in range(n):
            for v in range(u + 1, n):
                # Count common neighbors
                common = adj_mask[u] & adj_mask[v]
                if common.bit_count() >= 2:
                    return False
        return True

    def try_find_graph(n, time_limit):
        """
        Try to find a C4-free graph on n vertices with alpha < 11.
        Uses a smarter greedy construction with local search.
        """
        if time.time() > deadline:
            return None
            
        adj_mask = [0] * n
        
        # Strategy: Build graph vertex by vertex, trying to maximize degree
        # while maintaining C4-free property
        for i in range(n):
            if time.time() > deadline:
                return None
                
            # Candidates: previous vertices j < i
            # Try to connect i to as many as possible
            candidates = list(range(i))
            random.shuffle(candidates)
            
            for j in candidates:
                if time.time() > deadline:
                    return None
                
                # Check if we can add (i, j) without creating C4
                can_add = True
                v_mask = adj_mask[j]
                while v_mask:
                    lsb = v_mask & (-v_mask)
                    v = lsb.bit_length() - 1
                    v_mask ^= lsb
                    
                    # If N(i) and N(v) share a vertex, adding (i,j) creates C4
                    if adj_mask[i] & adj_mask[v]:
                        can_add = False
                        break
                
                if can_add:
                    adj_mask[i] |= (1 << j)
                    adj_mask[j] |= (1 << i)
        
        # After construction, verify C4-free
        if not check_c4_free(adj_mask, n):
            return None
            
        # Check alpha
        if time.time() > deadline:
            return None
            
        alpha = compute_alpha_fast(adj_mask, n)
        if alpha < 11:
            return adj_mask
        else:
            return None

    def try_find_graph_improved(n, time_limit):
        """
        Improved construction: Start with a base structure and add vertices.
        Uses a pool of known good small C4-free graphs with low alpha.
        """
        if time.time() > deadline:
            return None
            
        adj_mask = [0] * n
        
        # Try multiple strategies
        for strategy in range(3):
            if time.time() > deadline:
                return None
                
            adj_mask = [0] * n
            
            if strategy == 0:
                # Random greedy
                for i in range(n):
                    if time.time() > deadline:
                        break
                    candidates = list(range(i))
                    random.shuffle(candidates)
                    for j in candidates:
                        can_add = True
                        v_mask = adj_mask[j]
                        while v_mask:
                            lsb = v_mask & (-v_mask)
                            v = lsb.bit_length() - 1
                            v_mask ^= lsb
                            if adj_mask[i] & adj_mask[v]:
                                can_add = False
                                break
                        if can_add:
                            adj_mask[i] |= (1 << j)
                            adj_mask[j] |= (1 << i)
            elif strategy == 1:
                # Degree-ordered: connect to high-degree vertices first
                for i in range(n):
                    if time.time() > deadline:
                        break
                    candidates = list(range(i))
                    # Sort by degree descending
                    candidates.sort(key=lambda j: adj_mask[j].bit_count(), reverse=True)
                    for j in candidates:
                        can_add = True
                        v_mask = adj_mask[j]
                        while v_mask:
                            lsb = v_mask & (-v_mask)
                            v = lsb.bit_length() - 1
                            v_mask ^= lsb
                            if adj_mask[i] & adj_mask[v]:
                                can_add = False
                                break
                        if can_add:
                            adj_mask[i] |= (1 << j)
                            adj_mask[j] |= (1 << i)
            else:
                # Random with preference for vertices that allow more connections
                for i in range(n):
                    if time.time() > deadline:
                        break
                    candidates = list(range(i))
                    random.shuffle(candidates)
                    # Try to find best j to connect to
                    best_j = -1
                    best_score = -1
                    for j in candidates:
                        can_add = True
                        v_mask = adj_mask[j]
                        while v_mask:
                            lsb = v_mask & (-v_mask)
                            v = lsb.bit_length() - 1
                            v_mask ^= lsb
                            if adj_mask[i] & adj_mask[v]:
                                can_add = False
                                break
                        if can_add:
                            # Score: prefer connecting to vertices with fewer neighbors
                            # (leaves more room for future connections)
                            score = -adj_mask[j].bit_count()
                            if score > best_score:
                                best_score = score
                                best_j = j
                    if best_j >= 0:
                        adj_mask[i] |= (1 << best_j)
                        adj_mask[best_j] |= (1 << i)
        
        # Verify C4-free
        if not check_c4_free(adj_mask, n):
            return None
            
        if time.time() > deadline:
            return None
            
        alpha = compute_alpha_fast(adj_mask, n)
        if alpha < 11:
            return adj_mask
        else:
            return None

    # Fallback: 10 disjoint triangles (n=30, alpha=10)
    fallback_edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        fallback_edges += [[a, b], [a, c], [b, c]]
    
    best_n = 30
    best_edges = fallback_edges
    
    # Try to find a larger graph
    # Focus on n values where we're most likely to succeed
    # Best known is 39, so try 43 down to 31
    
    for n in range(43, 30, -1):
        if time.time() > deadline - 10:
            break
            
        # Try multiple random constructions with different seeds
        # More attempts for higher n values
        num_attempts = 50 if n >= 38 else 30
        
        for attempt in range(num_attempts):
            if time.time() > deadline - 10:
                break
                
            random.seed(attempt * 1000 + n)
            adj_mask = try_find_graph_improved(n, 10)
            if adj_mask:
                edges = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if (adj_mask[u] >> v) & 1:
                            edges.append([u, v])
                return {'n': n, 'edges': edges}
                
    return {'n': best_n, 'edges': best_edges}
