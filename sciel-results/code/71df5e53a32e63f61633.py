import time
import random
import math

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency bitmask
    adj = [0] * n
    edge_set = set()
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        edge_set.add((u, v) if u < v else (v, u))
        
    def popcount(x):
        return bin(x).count('1')
        
    def greedy_partition(seed):
        random.seed(seed)
        rem_adj = adj[:]
        bicliques = []
        edges_left = len(edges)
        
        while edges_left > 0:
            if time.time() > deadline:
                return None # Timeout
            
            # Pick a random edge (u, v) from remaining edges
            # We pick a random active vertex, then a random neighbor
            active_vertices = [u for u in range(n) if rem_adj[u] != 0]
            if not active_vertices:
                break
                
            u = random.choice(active_vertices)
            mask = rem_adj[u]
            # Get list of neighbors
            neighbors = []
            m = mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                neighbors.append(v)
                m ^= lsb
            
            if not neighbors:
                continue
                
            v = random.choice(neighbors)
            
            # Expand biclique starting with (u, v)
            # L contains u, R contains v. L and R are disjoint.
            L_mask = 1 << u
            R_mask = 1 << v
            
            # Common neighbors of all in L and all in R
            # Actually, we want to add x to L if x is connected to ALL in R.
            # And x must not be in R (disjointness).
            # Similarly for R.
            
            changed = True
            while changed:
                changed = False
                
                # Expand L
                # Candidates: vertices not in L, not in R, connected to all in R
                # Note: if x is in R, we can't add it to L.
                # But wait, if x is in R, it's connected to all in L? No, R is a clique in the bipartite sense? No, R is a set of vertices on the right side.
                # The condition for x to join L is: for all y in R, (x, y) is an edge.
                # And x not in L.
                # Do we require x not in R? Yes, L and R must be disjoint.
                
                # Optimization: 
                # Let common_L = intersection of rem_adj[y] for y in R.
                # Then candidates for L are (common_L & ~L_mask & ~R_mask).
                
                common_L = (1 << n) - 1
                m_r = R_mask
                while m_r:
                    lsb = m_r & -m_r
                    y = lsb.bit_length() - 1
                    m_r ^= lsb
                    common_L &= rem_adj[y]
                    
                new_L_candidates = common_L & ~L_mask & ~R_mask
                if new_L_candidates:
                    L_mask |= new_L_candidates
                    changed = True
                    
                # Expand R
                common_R = (1 << n) - 1
                m_l = L_mask
                while m_l:
                    lsb = m_l & -m_l
                    x = lsb.bit_length() - 1
                    m_l ^= lsb
                    common_R &= rem_adj[x]
                    
                new_R_candidates = common_R & ~R_mask & ~L_mask
                if new_R_candidates:
                    R_mask |= new_R_candidates
                    changed = True
            
            # Remove edges between L and R
            # For each u in L, clear bits in R
            m_l = L_mask
            while m_l:
                lsb = m_l & -m_l
                u_cur = lsb.bit_length() - 1
                m_l ^= lsb
                rem_adj[u_cur] &= ~R_mask
                
            m_r = R_mask
            while m_r:
                lsb = m_r & -m_r
                v_cur = lsb.bit_length() - 1
                m_r ^= lsb
                rem_adj[v_cur] &= ~L_mask
                
            edges_left -= popcount(L_mask) * popcount(R_mask)
            
            # Convert to lists
            L_list = []
            m = L_mask
            while m:
                lsb = m & -m
                u = lsb.bit_length() - 1
                L_list.append(u)
                m ^= lsb
                
            R_list = []
            m = R_mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                R_list.append(v)
                m ^= lsb
                
            bicliques.append([L_list, R_list])
            
        return bicliques

    best_bicliques = None
    best_count = float('inf')
    
    # Try multiple seeds
    # We can afford more seeds if greedy is faster.
    # Let's try 1000 seeds, but stop if time runs out.
    for seed in range(1000):
        if time.time() > deadline:
            break
            
        result = greedy_partition(seed)
        if result is not None:
            count = len(result)
            if count < best_count:
                best_count = count
                best_bicliques = result
                # If we found a very good solution, we might stop early?
                # But let's keep searching for better.
                
    if best_bicliques is None:
        # Fallback: trivial star decomposition
        best_bicliques = []
        for u, v in edges:
            best_bicliques.append([[u], [v]])
            
    # Validate
    for b in best_bicliques:
        if not isinstance(b, list) or len(b) != 2:
            raise ValueError("Invalid biclique structure")
        L, R = b
        if not isinstance(L, list) or not isinstance(R, list):
            raise ValueError("Sides must be lists")
        for x in L:
            if not isinstance(x, int) or isinstance(x, bool):
                raise ValueError("Vertices must be int")
        for x in R:
            if not isinstance(x, int) or isinstance(x, bool):
                raise ValueError("Vertices must be int")
        if not L or not R:
            raise ValueError("Sides must be non-empty")
        if set(L).intersection(set(R)):
            raise ValueError("Sides must be disjoint")
            
    covered_edges = set()
    for L, R in best_bicliques:
        for u in L:
            for v in R:
                e = (u, v) if u < v else (v, u)
                if e not in edge_set:
                    raise ValueError(f"Edge {e} not in graph")
                if e in covered_edges:
                    raise ValueError(f"Edge {e} covered multiple times")
                covered_edges.add(e)
                
    if covered_edges != edge_set:
        raise ValueError("Edges not partitioned correctly")
        
    return best_bicliques
