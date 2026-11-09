import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 290  # 10s margin
    
    # Helper to check if a subset is a clique or independent set
    # We will maintain adjacency matrix as a list of sets or bitmasks.
    # Bitmasks are faster for intersection checks.
    
    def check_violations(n, adj_mask):
        """
        Count K5 and Ind5.
        adj_mask[i] is a bitmask of neighbors of i.
        """
        k5_count = 0
        ind5_count = 0
        # Iterate over all 5-subsets
        # This is slow in Python if done naively for every check.
        # But we only need to update incrementally or do a full check at the end.
        # For the local search, we need fast delta calculation.
        pass

    # Fast delta calculation for flipping edge (u, v)
    # A 5-set S is a clique if all pairs in S are edges.
    # If we add edge (u,v), a 5-set S containing u,v becomes a clique if:
    # 1. S \ {u,v} is a clique in the graph with u,v removed? No.
    # 2. The 3 other vertices in S must be connected to u and v, and form a triangle among themselves.
    
    def count_k5_with_edge(u, v, adj_mask, n):
        # Count 5-sets containing u and v that become cliques when edge (u,v) is added.
        # Such a set is {u, v, a, b, c}.
        # Conditions:
        # - a, b, c are distinct, not u or v.
        # - (a,b), (b,c), (c,a) are edges (triangle).
        # - a, b, c are neighbors of u.
        # - a, b, c are neighbors of v.
        # So we need to find triangles in the intersection of N(u) and N(v), excluding u,v.
        
        common_neighbors = adj_mask[u] & adj_mask[v]
        # Remove u and v from common neighbors (they are not neighbors of themselves in simple graph, but mask might have bits? No, standard adjacency matrix has 0 on diagonal).
        # common_neighbors contains vertices w such that w~u and w~v.
        
        count = 0
        # Iterate over all triples a,b,c in common_neighbors
        # This is still O(k^3) where k is degree.
        # For n=42, max degree 41. 41^3 is ~69000.
        # This is acceptable for a delta check.
        
        neighbors_list = []
        mask = common_neighbors
        while mask:
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            neighbors_list.append(idx)
            mask -= lsb
            
        # Find number of triangles in this induced subgraph
        # For each a in neighbors_list:
        #   For each b in neighbors_list after a:
        #     If edge(a,b) exists:
        #       For each c in neighbors_list after b:
        #         If edge(b,c) and edge(c,a) exist: count++
        
        # We can just iterate combinations of 3 from neighbors_list
        for a, b, c in combinations(neighbors_list, 3):
            if (adj_mask[a] & (1 << b)) and (adj_mask[b] & (1 << c)) and (adj_mask[c] & (1 << a)):
                count += 1
        return count

    def count_ind5_with_edge(u, v, adj_mask, n):
        # Count 5-sets containing u and v that become independent sets when edge (u,v) is ADDED?
        # No, adding an edge REMOVES independent sets.
        # We want to know how many independent sets are BROKEN by adding edge (u,v).
        # An independent set S containing u,v is broken if u~v.
        # S is independent if no edges between any pairs in S.
        # Before adding (u,v), S was independent if:
        # - u,v not connected (true, we are adding it).
        # - a,b,c are not connected to u or v.
        # - a,b,c form an independent set among themselves.
        # So we need to find independent triangles (actually independent sets of size 3) in the set of vertices NOT connected to u and NOT connected to v.
        
        non_neighbors_u = ~adj_mask[u] & ((1 << n) - 1)
        non_neighbors_v = ~adj_mask[v] & ((1 << n) - 1)
        # Exclude u and v themselves
        non_neighbors_u &= ~(1 << u)
        non_neighbors_v &= ~(1 << v)
        
        candidates = non_neighbors_u & non_neighbors_v
        
        candidates_list = []
        mask = candidates
        while mask:
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            candidates_list.append(idx)
            mask -= lsb
            
        # Find number of independent sets of size 3 in candidates_list
        count = 0
        for a, b, c in combinations(candidates_list, 3):
            # Check if a,b,c are pairwise non-adjacent
            if not (adj_mask[a] & (1 << b)) and not (adj_mask[b] & (1 << c)) and not (adj_mask[c] & (1 << a)):
                count += 1
        return count

    # Total violations = K5_count + Ind5_count
    # We want to minimize this.
    
    # Initialization
    def build_paley(q):
        residues = {(x * x) % q for x in range(1, q)}
        edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
        return edges

    paley_edges = build_paley(37)
    
    # Try n = 42 down to 38
    for target_n in [42, 41, 40, 39, 38]:
        if time.time() > deadline:
            break
            
        # Initialize graph
        # Start with Paley 37
        adj_mask = [0] * target_n
        for u, v in paley_edges:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
        
        # Add random edges for new vertices (37 to target_n-1)
        # And between new vertices
        for i in range(37, target_n):
            for j in range(i + 1, target_n):
                if random.random() < 0.5:
                    adj_mask[i] |= (1 << j)
                    adj_mask[j] |= (1 << i)
            # Connect to core?
            for k in range(37):
                if random.random() < 0.5:
                    adj_mask[i] |= (1 << k)
                    adj_mask[k] |= (1 << i)
                    
        # Calculate initial violations
        # This full count is expensive.
        # We can just run the local search and calculate delta.
        # But we need a baseline.
        # Let's calculate full violations once.
        
        def full_violations(n, adj_mask):
            k5 = 0
            ind5 = 0
            for combo in combinations(range(n), 5):
                is_clique = True
                is_ind = True
                for u, v in combinations(combo, 2):
                    if adj_mask[u] & (1 << v):
                        is_ind = False
                    else:
                        is_clique = False
                if is_clique: k5 += 1
                if is_ind: ind5 += 1
            return k5 + ind5

        # If full check is too slow, we can skip it and just start flipping.
        # But we need to know if we are at 0.
        # Let's do a quick check.
        
        # Local Search
        current_violations = full_violations(target_n, adj_mask)
        best_violations = current_violations
        best_adj = adj_mask[:]
        
        if current_violations == 0:
            # Convert to edge list
            edges = [(i, j) for i in range(target_n) for j in range(i+1, target_n) if adj_mask[i] & (1 << j)]
            return target_n, edges

        # Simple Local Search
        # Pick random edge, flip it, calculate delta.
        # Accept if delta < 0 or with probability exp(-delta/T)
        
        # To speed up, we precompute the list of all edges
        all_edges = [(i, j) for i in range(target_n) for j in range(i+1, target_n)]
        
        T = 1.0
        no_improve = 0
        max_iter = 50000 # Adjust based on time
        
        while time.time() < deadline - 5:
            u, v = random.choice(all_edges)
            
            # Calculate delta if we flip edge (u,v)
            # If edge exists, we remove it.
            # If edge doesn't exist, we add it.
            
            if adj_mask[u] & (1 << v):
                # Remove edge
                # K5 count DECREASES by number of K5s containing (u,v)
                # Ind5 count INCREASES by number of Ind5s containing (u,v) that were broken?
                # No, removing edge (u,v) BREAKS cliques containing (u,v) and CREATES independent sets containing (u,v).
                
                # K5s containing (u,v):
                # u,v are connected.
                # We need a,b,c such that {u,v,a,b,c} is a clique.
                # This requires a,b,c to be neighbors of u and v, and form a triangle.
                k5_broken = count_k5_with_edge(u, v, adj_mask, target_n)
                
                # Ind5s containing (u,v):
                # u,v are connected.
                # They are NOT in an independent set.
                # After removal, u,v are non-connected.
                # We need a,b,c such that {u,v,a,b,c} is an independent set.
                # This requires a,b,c to be non-neighbors of u and v, and form an independent set.
                ind5_created = count_ind5_with_edge(u, v, adj_mask, target_n)
                
                delta = -k5_broken + ind5_created
                
            else:
                # Add edge
                # K5 count INCREASES by number of K5s containing (u,v) that become valid.
                k5_created = count_k5_with_edge(u, v, adj_mask, target_n)
                
                # Ind5 count DECREASES by number of Ind5s containing (u,v) that become invalid.
                ind5_broken = count_ind5_with_edge(u, v, adj_mask, target_n)
                
                delta = k5_created - ind5_broken
                
            new_violations = current_violations + delta
            
            if new_violations < current_violations:
                # Accept
                if adj_mask[u] & (1 << v):
                    adj_mask[u] &= ~(1 << v)
                    adj_mask[v] &= ~(1 << u)
                else:
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
                
                current_violations = new_violations
                no_improve = 0
                
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_adj = adj_mask[:]
                    
                if best_violations == 0:
                    edges = [(i, j) for i in range(target_n) for j in range(i+1, target_n) if best_adj[i] & (1 << j)]
                    return target_n, edges
                    
            elif random.random() < pow(2.718, -delta / T):
                # Accept with probability
                if adj_mask[u] & (1 << v):
                    adj_mask[u] &= ~(1 << v)
                    adj_mask[v] &= ~(1 << u)
                else:
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
                
                current_violations = new_violations
                no_improve += 1
                
            else:
                no_improve += 1
                
            # Anneal T
            T *= 0.999
            if T < 0.1:
                T = 0.1
                
            if no_improve > 1000:
                break

        # If we didn't find 0, check if best_adj is valid
        if best_violations == 0:
            edges = [(i, j) for i in range(target_n) for j in range(i+1, target_n) if best_adj[i] & (1 << j)]
            return target_n, edges
            
    # Fallback to Paley 37
    edges = build_paley(37)
    return 37, edges
