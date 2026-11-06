import random
import time
from collections import deque

def construct():
    n = 18
    deadline = time.time() + 145  # safety margin before 150s limit
    
    # Precompute all triangles (triples of pairwise-adjacent vertices)
    def compute_triangles(adj):
        """Return list of (i,j,k) where all three edges exist."""
        triangles = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i] & (1 << j):
                    for k in range(j+1, n):
                        if (adj[i] & (1 << k)) and (adj[j] & (1 << k)):
                            triangles.append((i, j, k))
        return triangles
    
    def is_k4_free(adj):
        """Check K4-free using adjacency bitmasks."""
        for i in range(n):
            for j in range(i+1, n):
                if adj[i] & (1 << j):
                    for k in range(j+1, n):
                        if (adj[i] & (1 << k)) and (adj[j] & (1 << k)):
                            common = adj[i] & adj[j] & adj[k]
                            common &= ~(1 << i) & ~(1 << j) & ~(1 << k)
                            if common:
                                return False
        return True
    
    def get_edges(adj):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return edges
    
    def max_triangle_free_size(adj):
        """Exact branch-and-bound for largest triangle-free induced subgraph.
        
        Returns (size, set_of_vertices).
        Key idea: build set S incrementally. A vertex v can be added to S iff
        for all u,w in S, not (v-u, v-w, u-w all edges). Equivalently, v must
        not be adjacent to two vertices in S that are themselves adjacent.
        
        We use a tighter bound: for remaining candidates, at most (n - len(S))
        can be added, but also bounded by the structure.
        """
        best = 0
        best_set = frozenset()
        
        # Precompute: for each vertex v, which pairs (u,w) in the remaining
        # would form a triangle with v if both were in S.
        # Actually, simpler: can_add(v, S) checks if v is adjacent to two
        # vertices in S that form an edge.
        
        def can_add(v, S):
            # v can be added to S iff there do NOT exist u, w in S
            # such that v-u, v-w, and u-w are all edges.
            adj_v_in_S = S & adj[v]  # vertices in S adjacent to v
            # Check if any two in adj_v_in_S are adjacent
            for u in adj_v_in_S:
                for w in adj_v_in_S:
                    if u < w and (adj[u] & (1 << w)):
                        return False
            return True
        
        def branch(current_S, available):
            nonlocal best, best_set
            if not available:
                if len(current_S) > best:
                    best = len(current_S)
                    best_set = frozenset(current_S)
                return
            
            # Bound: can't beat best
            if len(current_S) + len(available) <= best:
                return
            
            # Try adding each available vertex
            for v in sorted(available):
                if can_add(v, current_S):
                    new_S = current_S | {v}
                    # New available: vertices that can still be added to new_S
                    new_avail = set()
                    for w in available:
                        if w == v:
                            continue
                        if can_add(w, new_S):
                            new_avail.add(w)
                    branch(new_S, new_avail)
        
        branch(frozenset(), set(range(n)))
        return best, best_set
    
    def try_add_edge(adj, u, v):
        """Return new adj if adding (u,v) keeps K4-free, else None."""
        if adj[u] & (1 << v):
            return None
        common = adj[u] & adj[v]
        common &= ~(1 << u) & ~(1 << v)
        # For any w in common, and any x in common with x>w, if w-x edge exists -> K4
        common_list = [w for w in range(n) if common & (1 << w)]
        for idx, w in enumerate(common_list):
            for x in common_list[idx+1:]:
                if adj[w] & (1 << x):
                    return None
        new_adj = adj[:]
        new_adj[u] |= (1 << v)
        new_adj[v] |= (1 << u)
        return new_adj
    
    def try_remove_edge(adj, u, v):
        if not (adj[u] & (1 << v)):
            return None
        new_adj = adj[:]
        new_adj[u] &= ~(1 << v)
        new_adj[v] &= ~(1 << u)
        return new_adj
    
    # ---- Construction strategies ----
    
    def make_random_k4_free():
        """Build random K4-free graph by adding edges greedily."""
        adj = [0] * n
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(pairs)
        for u, v in pairs:
            new_adj = try_add_edge(adj, u, v)
            if new_adj is not None:
                adj = new_adj
        return adj
    
    def make_complete_k_partite(k):
        """Create complete k-partite graph on 18 vertices, K4-free if k <= 3... 
        Actually complete k-partite has K_{k+1}? No. Complete k-partite contains
        K_{k+1} only if... wait, complete k-partite contains K_{k+1}? 
        In K_{k+1}, we need k+1 vertices all pairwise adjacent. In complete k-partite,
        two vertices in same part are NOT adjacent. So max clique is k.
        So complete 3-partite is K4-free. Complete 4-partite contains K4.
        """
        # For k=3: 6,6,6
        adj = [0] * n
        parts = []
        sizes = [6, 6, 6]
        idx = 0
        for s in sizes:
            parts.append(list(range(idx, idx+s)))
            idx += s
        for i in range(n):
            for j in range(i+1, n):
                # Check if i and j are in different parts
                same_part = False
                for p in parts:
                    if i in p and j in p:
                        same_part = True
                        break
                if not same_part:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj
    
    def make_structured_partite(num_parts):
        """Create a random partition into num_parts parts, then add edges between parts
        with some probability, ensuring K4-free."""
        # For K4-free, we need no K4. If we use complete multipartite with <=3 parts,
        # it's K4-free. For more parts, we need to be careful.
        # Let's try: partition into parts, add edges between parts probabilistically,
        # check K4-free.
        
        # Random partition
        vertices = list(range(n))
        random.shuffle(vertices)
        parts = [[] for _ in range(num_parts)]
        for i, v in enumerate(vertices):
            parts[i % num_parts].append(v)
        
        adj = [0] * n
        # Add edges between different parts with probability 0.8
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)
                 if i // num_parts != j // num_parts]
        random.shuffle(pairs)
        for u, v in pairs:
            if random.random() < 0.8:
                new_adj = try_add_edge(adj, u, v)
                if new_adj is not None:
                    adj = new_adj
        return adj
    
    def make_random_regular_k4_free(degree_range):
        """Build a graph with degrees in degree_range that is K4-free."""
        adj = [0] * n
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(pairs)
        for u, v in pairs:
            deg_u = bin(adj[u]).count('1')
            deg_v = bin(adj[v]).count('1')
            if deg_u < degree_range[1] and deg_v < degree_range[1]:
                new_adj = try_add_edge(adj, u, v)
                if new_adj is not None:
                    adj = new_adj
        # Check degrees
        min_deg = min(bin(adj[i]).count('1') for i in range(n))
        max_deg = max(bin(adj[i]).count('1') for i in range(n))
        if min_deg >= degree_range[0] and max_deg <= degree_range[1]:
            return adj
        return None
    
    # ---- Local search ----
    
    def local_search(adj, time_limit):
        """Improve graph by adding/removing edges to reduce max triangle-free size."""
        start_time = time.time()
        current_score, _ = max_triangle_free_size(adj)
        best_score = current_score
        best_adj = adj[:]
        
        while time.time() - start_time < time_limit:
            if time.time() > deadline:
                break
            
            current_score, _ = max_triangle_free_size(adj)
            if current_score < best_score:
                best_score = current_score
                best_adj = adj[:]
            
            # Try random edge additions
            improved = False
            for _ in range(30):
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v:
                    continue
                new_adj = try_add_edge(adj, u, v)
                if new_adj is not None:
                    new_score, _ = max_triangle_free_size(new_adj)
                    if new_score < current_score:
                        adj = new_adj
                        current_score = new_score
                        improved = True
                        break
                    elif new_score == current_score:
                        # Accept with some probability (simulated annealing-like)
                        if random.random() < 0.3:
                            adj = new_adj
                            improved = True
                            break
            
            if not improved:
                # Try edge removals
                removed = False
                for _ in range(10):
                    # Find a dense edge
                    u = random.randint(0, n-1)
                    v = random.randint(0, n-1)
                    if u == v or not (adj[u] & (1 << v)):
                        continue
                    new_adj = try_remove_edge(adj, u, v)
                    if new_adj is not None:
                        new_score, _ = max_triangle_free_size(new_adj)
                        if new_score < current_score:
                            adj = new_adj
                            current_score = new_score
                            removed = True
                            break
                        elif new_score == current_score and random.random() < 0.2:
                            adj = new_adj
                            removed = True
                            break
                if not removed and random.random() < 0.05:
                    # Occasional restart
                    adj = make_random_k4_free()
                    current_score, _ = max_triangle_free_size(adj)
        
        return best_score, best_adj
    
    # ---- Main search loop ----
    
    best_overall = (99, [0]*n)
    
    # Start with known constructions
    constructions = []
    
    # Complete 3-partite (6,6,6)
    constructions.append(make_complete_k_partite(3))
    
    # Random K4-free
    for _ in range(5):
        constructions.append(make_random_k4_free())
    
    # Structured partitions
    for np_ in [2, 3, 4, 5]:
        for _ in range(3):
            constructions.append(make_structured_partite(np_))
    
    # Random regular
    for dr in [(5,8), (6,9), (7,10), (8,11)]:
        for _ in range(3):
            g = make_random_regular_k4_free(dr)
            if g is not None:
                constructions.append(g)
    
    # Evaluate all initial constructions
    for adj in constructions:
        if not is_k4_free(adj):
            continue
        score, _ = max_triangle_free_size(adj)
        if score < best_overall[0]:
            best_overall = (score, adj[:])
    
    # Local search from best and some other starts
    starts = [best_overall[1]]
    # Add a few other good starts
    for adj in constructions:
        if not is_k4_free(adj):
            continue
        score, _ = max_triangle_free_size(adj)
        if score <= best_overall[0] + 2:
            starts.append(adj)
    
    # Deduplicate starts by their score
    seen_scores = set()
    unique_starts = []
    for adj in starts:
        score, _ = max_triangle_free_size(adj)
        if score not in seen_scores:
            seen_scores.add(score)
            unique_starts.append(adj)
    
    for start_adj in unique_starts:
        if time.time() > deadline - 30:
            break
        time_per_start = min(20, max(5, (deadline - 30 - time.time()) / max(1, len(unique_starts))))
        score, improved_adj = local_search(start_adj, time_per_start)
        if score < best_overall[0]:
            best_overall = (score, improved_adj[:])
    
    # Final local search from the best
    if time.time() < deadline - 10:
        remaining = deadline - 10 - time.time()
        score, improved_adj = local_search(best_overall[1], remaining)
        if score < best_overall[0]:
            best_overall = (score, improved_adj[:])
    
    return get_edges(best_overall[1])
