import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    def is_prime_power(q):
        if q < 2:
            return False
        # Check if q is a prime power
        # Try small primes
        p = 2
        while p * p <= q:
            if q % p == 0:
                # Check if q is a power of p
                tmp = q
                while tmp % p == 0:
                    tmp //= p
                if tmp == 1:
                    return True
            p += 1
        # q is prime
        return True

    def _polarity_graph_46():
        # Construct polarity graph of projective plane of order 7 (57 vertices)
        # Then take a 46-vertex subgraph with high minimum degree
        # Actually, let's use a simpler approach: use known C4-free graphs
        
        # For n=46, min_deg = 46-39 = 7
        # We need a C4-free graph on 46 vertices with min degree >= 7
        
        # Let's use the incidence graph of a projective plane or similar structure
        # Actually, let's try a random approach with repair, but with better initialization
        
        # Use a bipartite-like structure or known extremal C4-free graphs
        # For now, let's use random initialization with repair
        return None

    def _polarity_graph(n_target):
        """
        Construct a C4-free graph using polarity graph of projective plane.
        For order q, the projective plane has q^2+q+1 points.
        The polarity graph has vertices = points, edges = (p, q) where q lies on polar of p.
        This graph is C4-free and has degree q+1 for each vertex.
        
        For q=7: 57 vertices, degree 8
        For q=5: 31 vertices, degree 6
        For q=4: 21 vertices, degree 5
        For q=3: 13 vertices, degree 4
        For q=2: 7 vertices, degree 3
        """
        # Find largest prime power q such that q^2+q+1 <= some reasonable bound
        # We want to get close to n_target
        # For n=46 or 47, q=7 gives 57 vertices (too many)
        # We can take a subgraph
        
        # Let's construct the polarity graph for q=7 (57 vertices) and take a subgraph
        q = 7
        # Points in PG(2,7) are 1D subspaces of F_7^3
        # Total points: (7^3-1)/(7-1) = (343-1)/6 = 342/6 = 57
        
        # Generate all points (1D subspaces)
        points = []
        seen = set()
        for a in range(8):
            for b in range(8):
                for c in range(8):
                    if a == 0 and b == 0 and c == 0:
                        continue
                    # Normalize: make first non-zero coordinate 1
                    if a != 0:
                        rep = (1, b * pow(a, -1, 7) % 7, c * pow(a, -1, 7) % 7)
                    elif b != 0:
                        rep = (0, 1, c * pow(b, -1, 7) % 7)
                    else:
                        rep = (0, 0, 1)
                    if rep not in seen:
                        seen.add(rep)
                        points.append(rep)
        
        # There should be 57 points
        # Now construct polarity: p is adjacent to q if q lies on polar line of p
        # Polar of point (a,b,c) is line ax + by + cz = 0
        # Point (x,y,z) lies on polar of (a,b,c) if ax+by+cz = 0
        
        n_pts = len(points)
        adj = [0] * n_pts
        
        for i, p in enumerate(points):
            pa, pb, pc = p
            for j, q in enumerate(points):
                if i == j:
                    continue
                qa, qb, qc = q
                # Check if q lies on polar of p: pa*qa + pb*qb + pc*qc = 0 mod 7
                if (pa * qa + pb * qb + pc * qc) % 7 == 0:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # This gives a C4-free graph on 57 vertices with degree 8
        # We need to extract a subgraph of size n_target with min degree >= n_target - 39
        
        if n_target >= 57:
            # Return full graph
            return adj[:n_target] if n_target <= n_pts else None
        
        # Take first n_target vertices
        # But we need to ensure min degree is high enough
        # Better: use local search to find good subgraph
        
        # For now, just return the full polarity graph and let caller handle subgraph
        return adj

    def check_valid(adj, n, min_deg):
        # Check min degree
        for i in range(n):
            if bin(adj[i]).count('1') < min_deg:
                return False
        # Check C4: any two vertices share at most 1 common neighbor
        for i in range(n):
            for j in range(i + 1, n):
                common = adj[i] & adj[j]
                if bin(common).count('1') > 1:
                    return False
        return True

    def get_edges(adj, n):
        edges = []
        for i in range(n):
            mask = adj[i]
            j = 0
            while mask:
                if mask & 1:
                    if i < j:
                        edges.append([i, j])
                mask >>= 1
                j += 1
        return edges

    def solve_for_n(n, min_deg):
        """
        Try to find a C4-free graph on n vertices with min degree >= min_deg.
        Strategy:
        1. Start with polarity graph of PG(2,7) (57 vertices, degree 8)
        2. Take a random subset of n vertices
        3. Local search: add edges between non-adjacent vertices with no common neighbors
        4. If stuck, try different subsets or random restarts
        """
        if time.time() > deadline:
            return None
        
        # Get polarity graph
        polar_adj = _polarity_graph(n)
        if polar_adj is None:
            return None
        
        n_polar = len(polar_adj)
        
        best_adj = None
        best_min_deg = -1
        
        # Try multiple random subsets
        for attempt in range(300):
            if time.time() > deadline:
                break
            
            random.seed(attempt * 104729 + 7)
            
            if n_polar >= n:
                # Choose random subset of n vertices
                subset = random.sample(range(n_polar), n)
                # Map to new indices
                idx_map = {old: new for new, old in enumerate(subset)}
                adj = [0] * n
                for old_i, new_i in idx_map.items():
                    for old_j, new_j in idx_map.items():
                        if old_i < old_j:
                            if polar_adj[old_i] & (1 << old_j):
                                adj[new_i] |= (1 << new_j)
                                adj[new_j] |= (1 << new_i)
            else:
                # Use full graph and pad? Not possible. Use random initialization instead.
                adj = [0] * n
                # Random dense initialization
                p = min_deg / n * 1.2
                if p > 0.8:
                    p = 0.8
                for i in range(n):
                    for j in range(i + 1, n):
                        if random.random() < p:
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
            
            # Local search: improve min degree
            for iteration in range(500):
                if time.time() > deadline:
                    break
                
                # Check current min degree
                min_deg_current = min(bin(adj[i]).count('1') for i in range(n))
                
                if min_deg_current >= min_deg:
                    # Check C4
                    c4_free = True
                    for i in range(n):
                        for j in range(i + 1, n):
                            if bin(adj[i] & adj[j]).count('1') > 1:
                                c4_free = False
                                break
                        if not c4_free:
                            break
                    
                    if c4_free:
                        # Success!
                        if check_valid(adj, n, min_deg):
                            return adj
                    break
                
                # Find vertex with minimum degree
                u = min(range(n), key=lambda i: bin(adj[i]).count('1'))
                
                # Find non-neighbors of u that can be connected without creating C4
                candidates = []
                for v in range(n):
                    if v == u:
                        continue
                    if not (adj[u] & (1 << v)):
                        # Check if adding (u,v) creates C4
                        # It creates C4 if common_neighbors(u, v) >= 1
                        # Wait, no: adding edge (u,v) creates C4 if there exists w
                        # such that u-w and v-w are edges, AND there exists x such that
                        # u-x and v-x are edges, where w != x.
                        # Actually, the condition for C4-free is: any two vertices share
                        # at most 1 common neighbor.
                        # If we add (u,v), then u and v will share all their previous
                        # common neighbors plus potentially more.
                        # The new common neighbors of u and v are:
                        # (N(u) ∩ N(v)) ∪ {w : w adjacent to both u and v after adding}
                        # Before adding: common = N(u) ∩ N(v)
                        # After adding: u is now neighbor of v, v is now neighbor of u
                        # So new common neighbors = (N(u) ∩ N(v)) ∪ {w : w in N(u) and w in N(v)} 
                        # Wait, that's the same. Let me think again.
                        # 
                        # Common neighbors of u and v after adding edge (u,v):
                        # A vertex w is a common neighbor of u and v if w is adjacent to both.
                        # This doesn't change when we add (u,v), unless w=u or w=v.
                        # But w can't be u or v since we're looking at other vertices.
                        # So common neighbors remain the same.
                        # 
                        # But wait, the condition is about pairs of vertices sharing common neighbors.
                        # Adding (u,v) might create new C4s involving u and v.
                        # A C4 is u-a-v-b-u where a,b are distinct.
                        # This exists if there are two distinct vertices a,b both adjacent to u and v.
                        # So the condition is: |N(u) ∩ N(v)| <= 1 after adding.
                        # Before adding, |N(u) ∩ N(v)| = k.
                        # After adding, u is in N(v) and v is in N(u), but u and v are not
                        # considered as "common neighbors" of each other in the usual sense.
                        # Common neighbors are vertices w != u,v such that w adjacent to both.
                        # So |N(u) ∩ N(v) \ {u,v}| = |N(u) ∩ N(v)| since u not in N(u) etc.
                        # So the condition is simply |N(u) ∩ N(v)| <= 1.
                        common = adj[u] & adj[v]
                        if bin(common).count('1') <= 1:
                            candidates.append(v)
                
                if candidates:
                    # Add edge to a random candidate
                    v = random.choice(candidates)
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                else:
                    # Can't improve u, try another approach
                    # Try to remove some edges to allow future additions?
                    # For now, break
                    break
            
            # Track best
            if adj is not None:
                current_min = min(bin(adj[i]).count('1') for i in range(n))
                if current_min > best_min_deg:
                    best_min_deg = current_min
                    best_adj = adj[:]
        
        return best_adj if best_min_deg >= min_deg else None

    # Try n from 47 down to 43
    for n in range(47, 42, -1):
        min_deg = n - 39
        if min_deg < 1:
            continue
        
        result_adj = solve_for_n(n, min_deg)
        
        if result_adj is not None:
            # Final verification
            if check_valid(result_adj, n, min_deg):
                edges = get_edges(result_adj, n)
                return {'n': n, 'edges': edges}
    
    # Fallback: Return a valid small graph
    # n=39, min_deg=0 is trivially valid (empty graph)
    return {'n': 39, 'edges': []}
