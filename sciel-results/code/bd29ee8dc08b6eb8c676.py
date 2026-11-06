import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    def make_k4_graph(num_copies):
        n = 4 * num_copies
        edges = []
        for b in range(num_copies):
            base = 4 * b
            for i in range(4):
                for j in range(i + 1, 4):
                    edges.append([base + i, base + j])
        return {'n': n, 'edges': edges}

    def build_adj(n, edges):
        adj = [[False] * n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = True
            adj[v][u] = True
        return adj

    def build_adj_list(adj, n):
        adj_list = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j]:
                    adj_list[i].append(j)
                    adj_list[j].append(i)
        return adj_list

    def has_c5_fast(adj, n):
        """Fast C5 detection using adjacency lists.
        A 5-cycle is a-b-c-d-e-a. We enumerate pairs (a,b) with edge,
        then pairs (c,d) with edge where c,d not in {a,b}, c!=d,
        check a-d and b-c edges, and a-c and b-d edges... 
        Actually, for a 5-cycle a-b-c-d-e-a, we need:
        edges: ab, bc, cd, de, ea
        Enumerate ordered pairs (a,b) with edge, then (b,c) with edge, c!=a,
        then (c,d) with edge, d!=a,b, then (d,e) with edge, e!=a,b,c,
        then check (e,a) edge and e!=b.
        This is O(n^5) worst case but with adjacency lists it's much faster in practice.
        """
        adj_list = build_adj_list(adj, n)
        for a in range(n):
            for b in adj_list[a]:
                if b <= a:
                    continue
                for c in adj_list[b]:
                    if c <= b or c == a:
                        continue
                    for d in adj_list[c]:
                        if d <= c or d == a or d == b:
                            continue
                        for e in adj_list[d]:
                            if e <= d or e == a or e == b or e == c:
                                continue
                            if adj[e][a]:
                                return True
        return False

    def max_independent_set_size_fast(adj, n):
        """Compute maximum independent set size.
        For the 8xK4 structure, we know it's 8. For extended graphs,
        use a branch-and-bound with good ordering.
        """
        adj_list = build_adj_list(adj, n)
        # Order vertices by degree (highest first) for better pruning
        order = sorted(range(n), key=lambda v: -len(adj_list[v]))
        
        best = [0]
        
        def bnb(remaining, current_size):
            if time.time() > deadline:
                return
            if not remaining:
                if current_size > best[0]:
                    best[0] = current_size
                return
            
            # Upper bound
            if current_size + len(remaining) <= best[0]:
                return
            
            v = remaining[0]
            rest = remaining[1:]
            
            # Include v: remove v and all its neighbors from remaining
            neighbors_set = set(adj_list[v])
            new_remaining = [x for x in rest if x not in neighbors_set]
            bnb(new_remaining, current_size + 1)
            
            # Exclude v
            bnb(rest, current_size)
        
        bnb(order, 0)
        return best[0]

    def check_valid(adj, n):
        """Check C5-free and alpha < 9."""
        if has_c5_fast(adj, n):
            return False
        alpha = max_independent_set_size_fast(adj, n)
        return alpha < 9

    # Start with 8xK4 (n=32)
    result = make_k4_graph(8)
    n = result['n']
    edges = result['edges']
    adj = build_adj(n, edges)
    
    # Verify base case
    if not check_valid(adj, n):
        # Should never happen, but fallback
        return make_k4_graph(7)
    
    best_n = n
    best_edges = edges[:]
    best_adj = adj
    
    # Try to extend to n+1, n+2, n+3 with structured search
    # Strategy: add vertices one at a time, trying structured connection patterns
    for target_n in range(n + 1, min(n + 4, 36)):
        if time.time() > deadline:
            break
        
        found = False
        
        # Generate candidate connection patterns for the new vertex
        # Pattern types:
        # 1. Connect to 1 vertex in each of k cliques (for various k)
        # 2. Connect to 2 vertices in each of k cliques
        # 3. Random with various densities
        
        # First, try structured patterns based on the 8-clique structure
        # The existing graph has 8 cliques of size 4: vertices 4b, 4b+1, 4b+2, 4b+3
        
        candidates = []
        
        # Pattern A: connect to exactly 1 vertex in each of all 8 cliques
        # Try all 4^8 = 65536 combinations? Too many. Sample.
        random.seed(12345)
        
        # Try: connect to vertex 0 in each clique (all same offset)
        for offset in range(4):
            conn = [4 * b + offset for b in range(8)]
            candidates.append(conn)
        
        # Try: connect to 2 vertices in each of 4 cliques (random which 4, random which 2)
        for _ in range(200):
            cliques_to_use = random.sample(range(8), 4)
            conn = []
            for b in cliques_to_use:
                two = random.sample(range(4), 2)
                conn.extend([4 * b + t for t in two])
            candidates.append(conn)
        
        # Try: connect to 1 vertex in each of 6 cliques
        for _ in range(200):
            cliques_to_use = random.sample(range(8), 6)
            conn = []
            for b in cliques_to_use:
                conn.append(4 * b + random.randint(0, 3))
            candidates.append(conn)
        
        # Try: connect to 1 vertex in each of 7 cliques
        for _ in range(200):
            cliques_to_use = random.sample(range(8), 7)
            conn = []
            for b in cliques_to_use:
                conn.append(4 * b + random.randint(0, 3))
            candidates.append(conn)
        
        # Try: connect to 1 vertex in each of 5 cliques
        for _ in range(200):
            cliques_to_use = random.sample(range(8), 5)
            conn = []
            for b in cliques_to_use:
                conn.append(4 * b + random.randint(0, 3))
            candidates.append(conn)
        
        # Try random densities
        for density in [0.3, 0.4, 0.5, 0.6, 0.7]:
            for _ in range(50):
                conn = [v for v in range(n) if random.random() < density]
                candidates.append(conn)
        
        # Shuffle candidates
        random.shuffle(candidates)
        
        for conn in candidates:
            if time.time() > deadline:
                break
            if found:
                break
            
            # Build new adjacency
            new_adj = [[False] * target_n for _ in range(target_n)]
            for i in range(n):
                for j in range(n):
                    new_adj[i][j] = best_adj[i][j]
            
            for v in conn:
                new_adj[n][v] = True
                new_adj[v][n] = True
            
            new_edges = best_edges[:]
            for v in conn:
                new_edges.append([n, v])
            
            if check_valid(new_adj, target_n):
                found = True
                best_n = target_n
                best_edges = new_edges[:]
                best_adj = new_adj
                break
        
        if not found:
            break
    
    return {'n': best_n, 'edges': best_edges}
