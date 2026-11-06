"""K4-free graph on 18 vertices minimizing largest triangle-free induced subgraph."""
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Paley graph on 17 vertices (0..16): edge if difference is quadratic residue mod 17
    # Quadratic residues mod 17: 1, 4, 9, 16, 8, 2, 15, 13
    qr = {1, 4, 9, 16, 8, 2, 15, 13}
    
    def is_paley_edge(u, v):
        if u == v:
            return False
        d = (v - u) % 17
        return d in qr
    
    # Build Paley graph edges on vertices 0..16
    paley_edges = []
    for u in range(17):
        for v in range(u + 1, 17):
            if is_paley_edge(u, v):
                paley_edges.append([u, v])
    
    # Try different connection patterns for vertex 17
    # We want to minimize the largest triangle-free induced subgraph
    # Strategy: connect vertex 17 to a set that breaks up large triangle-free sets
    
    def largest_triangle_free(edges_set, n=18):
        """Find largest triangle-free induced subgraph via branch and bound."""
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges_set:
            adj[u].add(v)
            adj[v].add(u)
        
        best = 0
        best_set = set()
        
        def is_triangle_free(vertex_set):
            for u in vertex_set:
                for v in vertex_set:
                    if u < v and v in adj[u]:
                        # Check if there's a w in vertex_set adjacent to both
                        common = adj[u] & adj[v] & vertex_set
                        if common:
                            return False
            return True
        
        # Branch and bound
        def bnb(candidates, current, depth):
            nonlocal best, best_set
            if time.time() > deadline - 5:
                return
            if len(candidates) + len(current) <= best:
                return
            if not candidates:
                if len(current) > best:
                    best = len(current)
                    best_set = set(current)
                return
            
            # Try adding a vertex
            v = candidates[0]
            remaining = candidates[1:]
            
            # Option 1: include v
            new_current = current | {v}
            if is_triangle_free(new_current):
                # Filter candidates: can only add vertices that don't create triangle with v
                # A vertex w can be added if it's not adjacent to both endpoints of any edge in new_current
                # More precisely: w must not form a triangle with any pair in new_current
                valid_remaining = set()
                for w in remaining:
                    ok = True
                    for u in new_current:
                        if w in adj[u]:
                            # Check if there's an edge from u to some x in new_current, and w-x edge
                            # Actually: w forms triangle with u,x if u-x in E and w-u in E and w-x in E
                            for x in new_current:
                                if x != u and x in adj[u] and w in adj[x]:
                                    ok = False
                                    break
                    if ok:
                        valid_remaining.add(w)
                bnb(list(valid_remaining), new_current, depth + 1)
            
            # Option 2: exclude v
            bnb(remaining, current, depth + 1)
        
        bnb(list(range(n)), set(), 0)
        return best, best_set
    
    def is_k4_free(edges_list):
        """Check if graph is K4-free."""
        adj = [set() for _ in range(18)]
        for u, v in edges_list:
            adj[u].add(v)
            adj[v].add(u)
        for combo in itertools.combinations(range(18), 4):
            a, b, c, d = combo
            if b in adj[a] and c in adj[a] and d in adj[a] and \
               c in adj[b] and d in adj[b] and d in adj[c]:
                return False
        return True
    
    # Try different subsets for vertex 17's connections
    # The Paley graph has independence number 5, so we want to keep it at 5 or reduce it
    # Let's try connecting vertex 17 to various sets
    
    best_score = 18
    best_edges = paley_edges + []  # default: vertex 17 isolated
    
    # Try all subsets of size 0-8 for vertex 17's neighbors (symmetric considerations)
    # But that's 2^17 = 131072 possibilities, too many for exhaustive
    # Instead, try strategic subsets
    
    candidate_neighbor_sets = []
    
    # Empty set
    candidate_neighbor_sets.append(set())
    
    # Single vertices
    for v in range(17):
        candidate_neighbor_sets.append({v})
    
    # Pairs - try all pairs
    for a in range(17):
        for b in range(a + 1, 17):
            candidate_neighbor_sets.append({a, b})
    
    # Triples - sample
    import random
    random.seed(42)
    triples = list(itertools.combinations(range(17), 3))
    random.shuffle(triples)
    for triple in triples[:500]:
        candidate_neighbor_sets.append(set(triple))
    
    # Larger sets - sample
    for size in [4, 5, 6, 7, 8]:
        for _ in range(200):
            s = set(random.sample(range(17), size))
            candidate_neighbor_sets.append(s)
    
    # Also try the complement of small sets
    for small in candidate_neighbor_sets[:50]:
        comp = set(range(17)) - small
        if len(comp) <= 12:
            candidate_neighbor_sets.append(comp)
    
    evaluated = 0
    for neighbors in candidate_neighbor_sets:
        if time.time() > deadline - 10:
            break
        
        # Build edge list
        edges = list(paley_edges)
        for v in neighbors:
            edges.append([17, v])
        
        # Check K4-free
        if not is_k4_free(edges):
            continue
        
        # Find largest triangle-free induced subgraph
        score, _ = largest_triangle_free(edges)
        
        if score < best_score:
            best_score = score
            best_edges = edges
        
        evaluated += 1
    
    # If we haven't found something better, try a more systematic approach
    if best_score > 5 and time.time() < deadline - 20:
        # Try random restarts with simulated annealing-like approach
        import random
        random.seed(123)
        
        for _ in range(1000):
            if time.time() > deadline - 10:
                break
            
            # Random neighbor set for vertex 17
            p = random.random()
            neighbors = set(v for v in range(17) if random.random() < p)
            
            edges = list(paley_edges)
            for v in neighbors:
                edges.append([17, v])
            
            if not is_k4_free(edges):
                continue
            
            score, _ = largest_triangle_free(edges)
            
            if score < best_score:
                best_score = score
                best_edges = edges
    
    return best_edges
