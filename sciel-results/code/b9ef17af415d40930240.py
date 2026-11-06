import random, time
import numpy as np

def construct():
    deadline = time.time() + 115.0  # safety margin
    rng = random.Random(12345)
    best_n = 0
    best_edges = []

    def has_k4(adj, n):
        """Check for K4 using bitmask."""
        for i in range(n):
            ni = adj[i]
            for j in range(i + 1, n):
                if not (ni >> j) & 1:
                    continue
                nij = ni & adj[j]
                for k in range(j + 1, n):
                    if not (nij >> k) & 1:
                        continue
                    if (nij & adj[k]) >> (k + 1):
                        return True
        return False

    def has_independent_set_k(adj, n, k):
        """Check for independent set of size k using backtracking with bitmasks."""
        if k <= 0:
            return True
        if k == 1:
            return True
        # Use greedy + backtracking
        mask = (1 << n) - 1
        
        def backtrack(candidates, remaining):
            if remaining == 0:
                return True
            if candidates == 0:
                return False
            # Pruning: if popcount(candidates) < remaining, impossible
            if bin(candidates).count('1') < remaining:
                return False
            # Pick a vertex with minimum degree in the induced subgraph (for independence, max non-neighbor count)
            # For independent set, we want vertices with few neighbors in candidates
            # Try each candidate
            while candidates:
                # Pick lowest set bit
                lb = candidates & (-candidates)
                v = lb.bit_length() - 1
                candidates ^= lb
                # New candidates: non-neighbors of v in remaining candidates
                new_cands = candidates & ~adj[v]
                if backtrack(new_cands, remaining - 1):
                    return True
            return False
        
        return backtrack(mask, k)

    def is_valid(adj, n):
        """Check no K4 and no independent set of size 6."""
        if has_k4(adj, n):
            return False
        if has_independent_set_k(adj, n, 6):
            return False
        return True

    def edges_from_adj(adj, n):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    def generate_paley(p):
        """Generate Paley graph of prime order p where p ≡ 1 mod 4."""
        if p % 4 != 1:
            return None
        squares = set()
        for x in range(1, p):
            squares.add((x * x) % p)
        adj = [0] * p
        for i in range(p):
            for j in range(i + 1, p):
                if (j - i) % p in squares:
                    adj[i] |= 1 << j
                    adj[j] |= 1 << i
        return adj

    def generate_conference_matrix(n):
        """Generate a graph from a conference matrix of order n (n ≡ 2 mod 4).
        Conference matrix C: C*C^T = (n-1)*I, C has 0 diagonal, ±1 off-diagonal.
        Graph: edge (i,j) if C[i][j] == 1."""
        if n % 4 != 2:
            return None
        # Use a simple construction: circulant based
        # For n=6,10,14,18,22,26,30,34,38
        # Try to construct using known patterns
        C = np.zeros((n, n), dtype=int)
        C.fill(-1)
        np.fill_diagonal(C, 0)
        
        # Use a simple heuristic construction
        # For small n, try random ±1 and check
        for _ in range(10000):
            if time.time() > deadline:
                break
            C = np.zeros((n, n), dtype=int)
            np.fill_diagonal(C, 0)
            for i in range(n):
                for j in range(i + 1, n):
                    val = 1 if rng.random() < 0.5 else -1
                    C[i][j] = val
                    C[j][i] = -val  # skew-symmetric
            # Check if C*C^T = (n-1)*I
            prod = C @ C.T
            ok = True
            for i in range(n):
                if abs(prod[i][i] - (n - 1)) > 0.1:
                    ok = False
                    break
                for j in range(n):
                    if i != j and abs(prod[i][j]) > 0.1:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                break
        
        if not ok:
            return None
        
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if C[i][j] == 1:
                    adj[i] |= 1 << j
                    adj[j] |= 1 << i
        return adj

    def generate_sr_graph_params():
        """Generate strongly regular graph candidates."""
        candidates = []
        # Known SRG parameters (v, k, lambda, mu)
        # We need K4-free: lambda <= 1 (if two adjacent vertices have 2 common neighbors, with the edge they form K4)
        # Actually K4 means 4 mutually adjacent. In SRG, a K4 exists if lambda >= 2 is possible... 
        # More precisely: if there's a triangle, and two vertices in the triangle have a common neighbor outside, 
        # we might get K4. Let's be careful.
        # For K4-free: no 4-clique. In SRG(v,k,lambda,mu), max clique size can be bounded.
        # Let's just generate and check.
        
        # (16, 6, 2, 2) - Shrikhande or L(2,4)
        # (17, 8, 3, 4) - Paley(17)
        # (25, 12, 5, 6) - Latin square graph
        # (28, 12, 4, 4) - Gewirtz
        # (36, 15, 6, 6) - various
        # (49, 18, 8, 9) - Latin square
        
        # For K4-free we need to be more careful
        # Let's try: (n, k, lambda, mu) where lambda=0 or 1
        # (9, 2, 0, 1) - 9-cycle complement? No
        # (10, 3, 0, 1) - Petersen graph complement? Petersen is (10,3,0,1)
        # Petersen: K4-free? Petersen has girth 5, so no triangles, hence no K4. 
        # Independent set in Petersen: alpha=4, so no independent 6-set. But only 10 vertices.
        
        # Let's focus on constructions we can verify
        return candidates

    def local_search(adj, n, max_iters=50000):
        """Perform local search: try edge additions/removals/swaps to improve validity and size."""
        current_n = n
        current_adj = adj[:]
        
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            # Check if current graph is valid
            if is_valid(current_adj, current_n):
                return current_adj, current_n
            
            # Find violations and try to fix them
            # Try adding a vertex
            if current_n < 40:
                new_v = current_n
                # Try different neighbor sets
                for _ in range(100):
                    if time.time() > deadline - 5.0:
                        break
                    # Random subset of existing vertices
                    neighbor_mask = 0
                    for v in range(current_n):
                        if rng.random() < 0.3:
                            neighbor_mask |= 1 << v
                    
                    # Check if adding this vertex creates K4 or independent 6
                    # K4: for each pair of neighbors of new_v that are adjacent, they'd form K4 with new_v
                    # Actually: new_v + u + v + w where u~v, v~w, w~u, and all adjacent to new_v
                    # So we need: no triangle among neighbors of new_v
                    # Independent 6: new_v + 5 mutually non-adjacent non-neighbors of new_v
                    
                    # Check K4: neighbors of new_v must be triangle-free
                    neigh_mask = neighbor_mask
                    has_tri = False
                    for i in range(current_n):
                        if not (neigh_mask >> i) & 1:
                            continue
                        ni = neigh_mask & current_adj[i]
                        for j in range(i + 1, current_n):
                            if not (ni >> j) & 1:
                                continue
                            if (ni & current_adj[j]) >> (j + 1):
                                has_tri = True
                                break
                        if has_tri:
                            break
                    if has_tri:
                        continue
                    
                    # Check independent 6: non-neighbors of new_v must not have independent 5-set
                    non_neigh = ((1 << current_n) - 1) & ~neighbor_mask
                    if has_independent_set_k(current_adj, current_n, 5):
                        # This is a rough check - we need to check within non_neigh specifically
                        # For now, skip this detailed check and rely on full validation
                        pass
                    
                    # Try adding
                    new_adj = current_adj[:]
                    new_adj.append(neighbor_mask)
                    for v in range(current_n):
                        if (neighbor_mask >> v) & 1:
                            new_adj[v] |= 1 << new_v
                    
                    if is_valid(new_adj, current_n + 1):
                        current_adj = new_adj
                        current_n += 1
                        break
            
            # Try edge swap: remove one edge, add another
            if current_n >= 2:
                # Find a random edge to remove
                all_edges = []
                for i in range(current_n):
                    for j in range(i + 1, current_n):
                        if (current_adj[i] >> j) & 1:
                            all_edges.append((i, j))
                
                if all_edges:
                    # Try removing a random edge
                    for _ in range(10):
                        if time.time() > deadline - 5.0:
                            break
                        u, v = rng.choice(all_edges)
                        new_adj = current_adj[:]
                        new_adj[u] &= ~(1 << v)
                        new_adj[v] &= ~(1 << u)
                        
                        # Try adding a random non-edge
                        non_edges = []
                        for i in range(current_n):
                            for j in range(i + 1, current_n):
                                if not ((i == u and j == v) or (i == v and j == u)):
                                    if not ((current_adj[i] >> j) & 1):
                                        non_edges.append((i, j))
                        
                        if non_edges:
                            a, b = rng.choice(non_edges)
                            new_adj[a] |= 1 << b
                            new_adj[b] |= 1 << a
                            
                            if is_valid(new_adj, current_n):
                                current_adj = new_adj
                                break

        return current_adj, current_n

    def add_vertex_heuristic(adj, n):
        """Try to add a vertex with a smart neighbor selection."""
        if n >= 40:
            return adj, n
        
        new_v = n
        best_neighbor_mask = None
        best_score = -1
        
        # Try several random neighbor configurations
        for _ in range(200):
            if time.time() > deadline - 5.0:
                break
            
            neighbor_mask = 0
            for v in range(n):
                if rng.random() < 0.25:
                    neighbor_mask |= 1 << v
            
            # Score: prefer configurations that are more likely to be valid
            # Penalize triangles in neighbors (K4 risk)
            # Penalize large independent sets in non-neighbors (indep6 risk)
            
            tri_count = 0
            for i in range(n):
                if not (neighbor_mask >> i) & 1:
                    continue
                ni = neighbor_mask & adj[i]
                for j in range(i + 1, n):
                    if not (ni >> j) & 1:
                        continue
                    if (ni & adj[j]) >> (j + 1):
                        tri_count += 1
            
            # Non-neighbor independent set size estimate
            non_neigh = ((1 << n) - 1) & ~neighbor_mask
            # Rough: count vertices with few neighbors in non_neigh
            indep_estimate = 0
            for v in range(n):
                if not (non_neigh >> v) & 1:
                    continue
                deg_in_non = bin(non_neigh & adj[v]).count('1')
                if deg_in_non == 0:
                    indep_estimate += 1
            
            score = -10 * tri_count - indep_estimate
            if score > best_score:
                best_score = score
                best_neighbor_mask = neighbor_mask
        
        if best_neighbor_mask is None:
            return adj, n
        
        # Try the best configuration
        new_adj = adj[:]
        new_adj.append(best_neighbor_mask)
        for v in range(n):
            if (best_neighbor_mask >> v) & 1:
                new_adj[v] |= 1 << new_v
        
        if is_valid(new_adj, n + 1):
            return new_adj, n + 1
        
        return adj, n

    # Phase 1: Generate candidate graphs from spectral constructions
    candidates = []
    
    # Paley graphs
    for p in [5, 13, 17, 29, 37]:
        if p % 4 == 1:
            adj = generate_paley(p)
            if adj is not None:
                candidates.append(adj)
    
    # Conference matrix based graphs
    for n in [6, 10, 14, 18, 22, 26, 30, 34, 38]:
        adj = generate_conference_matrix(n)
        if adj is not None:
            candidates.append(adj)
    
    # Phase 2: For each candidate, try to extend and refine
    for adj in candidates:
        if time.time() > deadline - 10.0:
            break
        n = len(adj)
        
        # Check if already valid
        if is_valid(adj, n):
            if n > best_n:
                best_n = n
                best_edges = edges_from_adj(adj, n)
        
        # Try to extend
        adj, n = add_vertex_heuristic(adj, n)
        if is_valid(adj, n) and n > best_n:
            best_n = n
            best_edges = edges_from_adj(adj, n)
        
        # Local search
        adj, n = local_search(adj, n, max_iters=20000)
        if is_valid(adj, n) and n > best_n:
            best_n = n
            best_edges = edges_from_adj(adj, n)
    
    # Phase 3: Random restarts with heuristic vertex addition
    for attempt in range(200):
        if time.time() > deadline - 10.0:
            break
        
        # Start with a small random valid graph
        n0 = rng.randint(6, 12)
        adj = [0] * n0
        for i in range(n0):
            for j in range(i + 1, n0):
                if rng.random() < 0.35:
                    adj[i] |= 1 << j
                    adj[j] |= 1 << i
        
        if not is_valid(adj, n0):
            continue
        
        n = n0
        # Extend
        for _ in range(30):
            if time.time() > deadline - 10.0:
                break
            adj, n = add_vertex_heuristic(adj, n)
        
        if is_valid(adj, n) and n > best_n:
            best_n = n
            best_edges = edges_from_adj(adj, n)
        
        # Local search
        adj, n = local_search(adj, n, max_iters=10000)
        if is_valid(adj, n) and n > best_n:
            best_n = n
            best_edges = edges_from_adj(adj, n)
    
    # Phase 4: Final push - try to reach 35
    if best_n < 35 and best_n >= 15:
        if time.time() < deadline - 15.0:
            adj = [0] * best_n
            for e in best_edges:
                adj[e[0]] |= 1 << e[1]
                adj[e[1]] |= 1 << e[0]
            
            # Aggressive extension
            for _ in range(100):
                if time.time() > deadline - 10.0:
                    break
                adj, best_n = add_vertex_heuristic(adj, best_n)
            
            if is_valid(adj, best_n):
                best_edges = edges_from_adj(adj, best_n)
    
    # Fallback
    if best_n == 0:
        adj = generate_paley(17)
        if adj is not None:
            best_n = 17
            best_edges = edges_from_adj(adj, 17)
        else:
            best_n = 1
            best_edges = []
    
    return {'n': best_n, 'edges': best_edges}
