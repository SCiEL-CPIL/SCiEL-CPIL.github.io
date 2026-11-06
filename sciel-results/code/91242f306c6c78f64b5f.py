import random
import time
import itertools
import sys
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit

    def is_triangle_free(n, adj):
        for u in range(n):
            for v in range(n):
                if adj[u] & (1 << v):
                    common = adj[u] & adj[v]
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    if common:
                        return False
        return True

    def greedy_alpha_lower_bound(n, adj):
        """Return size of a greedy independent set (lower bound on alpha)."""
        remaining = (1 << n) - 1
        size = 0
        while remaining:
            best_v = -1
            best_deg = n + 1
            temp = remaining
            while temp:
                v = (temp & -temp).bit_length() - 1
                temp &= ~(1 << v)
                deg = bin(remaining & adj[v]).count('1')
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
            size += 1
            remaining &= ~(1 << best_v)
            remaining &= ~adj[best_v]
        return size

    def has_independent_set_10(n, adj):
        """Exact branch and bound for independent set of size 10."""
        # Quick greedy check
        if greedy_alpha_lower_bound(n, adj) >= 10:
            return True

        def backtrack(candidates, size):
            if size == 10:
                return True
            if size + bin(candidates).count('1') < 10:
                return False

            # Pick vertex with max degree in candidate subgraph
            best_v = -1
            best_deg = -1
            temp = candidates
            while temp:
                v = (temp & -temp).bit_length() - 1
                temp &= ~(1 << v)
                deg = bin(candidates & adj[v]).count('1')
                if deg > best_deg:
                    best_deg = deg
                    best_v = v

            if best_v == -1:
                return False

            candidates &= ~(1 << best_v)

            # Option 1: Include best_v
            new_candidates = candidates & ~adj[best_v]
            if backtrack(new_candidates, size + 1):
                return True

            # Option 2: Exclude best_v
            if backtrack(candidates, size):
                return True

            return False

        return backtrack((1 << n) - 1, 0)

    def validate(n, adj):
        if not is_triangle_free(n, adj):
            return False
        if has_independent_set_10(n, adj):
            return False
        return True

    def edges_from_adj(n, adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    edges.append([u, v])
        return edges

    # Fallback: known valid graph on 26 vertices
    best_n = 26
    best_adj = [0] * 26
    for i in range(26):
        for j in range(i + 1, 26):
            if (j - i) % 26 % 3 == 1:
                best_adj[i] |= (1 << j)
                best_adj[j] |= (1 << i)
    best_edges = edges_from_adj(26, best_adj)

    def find_triangle(adj, n):
        """Find a triangle in the graph, return (u,v,w) or None."""
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    common = adj[u] & adj[v]
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    if common:
                        w = (common & -common).bit_length() - 1
                        return (u, v, w)
        return None

    def remove_edge_to_break_triangle(adj, n, tri, edge_to_remove):
        """Remove an edge from the triangle and return new adj."""
        u, v = edge_to_remove
        new_adj = adj[:]
        new_adj[u] &= ~(1 << v)
        new_adj[v] &= ~(1 << u)
        return new_adj

    def start_large_and_prune(n, deadline):
        """Start with a dense random graph and iteratively remove edges to eliminate triangles.
        Then check if the resulting graph has no independent set of size 10."""
        # Start with a moderately dense random graph
        adj = [0] * n
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_edges)
        
        # Add edges until we have a reasonable density (~30-40%)
        target_edges = int(n * n * 0.35)
        for u, v in all_edges:
            if sum(bin(adj[i]).count('1') for i in range(n)) // 2 >= target_edges:
                break
            # Only add if it doesn't create a triangle
            if (adj[u] & adj[v]) == 0:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        # Now iteratively remove edges to break all triangles
        iterations = 0
        max_iterations = 5000
        while time.time() < deadline - 2 and iterations < max_iterations:
            tri = find_triangle(adj, n)
            if tri is None:
                break
            iterations += 1
            
            u, v, w = tri
            # Choose which edge to remove - use a heuristic
            # Remove the edge whose removal minimizes potential independent set growth
            candidates = [(u, v), (u, w), (v, w)]
            random.shuffle(candidates)
            
            for edge in candidates:
                new_adj = remove_edge_to_break_triangle(adj, n, edge, edge)
                # Quick check: does this still have a chance?
                if greedy_alpha_lower_bound(n, new_adj) < 10:
                    adj = new_adj
                    break
        
        if find_triangle(adj, n) is not None:
            return None
        
        if greedy_alpha_lower_bound(n, adj) >= 10:
            return None
        
        if validate(n, adj):
            return edges_from_adj(n, adj)
        
        return None

    # Structural change: Systematic circulant graph search with sum-free difference sets
    # This replaces weak random edge-adding with algebraic construction
    # A circulant graph C_n(S) has edge (i,j) iff (j-i) mod n in S
    # Triangle-free condition: S is sum-free mod n (no a,b,c in S with a+b+c ≡ 0 mod n)
    # Actually for triangle-free: no a,b in S with a-b in S (directed) or a+b in S (undirected)
    # For undirected circulant: S is symmetric (if s in S then n-s in S)
    # Triangle-free: no a,b,c in S with a+b ≡ c mod n (since a,b,c edges means a-b, b-c, a-c in S)
    # Actually: vertices 0,a,b form triangle iff a in S, b in S, (b-a) mod n in S
    # So S must be such that for no a,b in S is (a-b) mod n in S (and a != b)
    
    def is_sum_free_circulant(n, S):
        """Check if circulant graph C_n(S) is triangle-free.
        S is a set of differences. Edge (i,j) iff (j-i) mod n in S.
        Triangle 0,a,b exists iff a in S, b in S, (b-a) mod n in S.
        """
        S_set = set(S)
        for a in S:
            for b in S:
                if a != b:
                    if (b - a) % n in S_set:
                        return False
        return True

    def andrassyi_template(n):
        """Generate a graph inspired by Andrássai graph A_9 structure.
        A_9 is triangle-free with alpha=9. We extend this idea.
        Uses a 3-partite-like structure with controlled cross-edges."""
        adj = [0] * n
        
        # Divide vertices into 3 groups
        g1_size = n // 3
        g2_size = n // 3
        g3_size = n - g1_size - g2_size
        
        g1 = list(range(g1_size))
        g2 = list(range(g1_size, g1_size + g2_size))
        g3 = list(range(g1_size + g2_size, n))
        
        # Within each group: no edges (independent sets within groups)
        # Between groups: add edges carefully to avoid triangles
        
        # G1-G2 edges: use a cyclic pattern
        for i, u in enumerate(g1):
            for j, v in enumerate(g2):
                if (i - j) % max(1, g1_size) in [1, 2]:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        # G2-G3 edges: use a cyclic pattern
        for i, u in enumerate(g2):
            for j, v in enumerate(g3):
                if (i - j) % max(1, g2_size) in [1, 2]:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        # G1-G3 edges: use a cyclic pattern
        for i, u in enumerate(g1):
            for j, v in enumerate(g3):
                if (i - j) % max(1, g1_size) in [1, 2]:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        return adj

    def circulant_alpha_lower_bound(n, S):
        """Quick greedy alpha lower bound for circulant graph."""
        adj = [0] * n
        for s in S:
            for i in range(n):
                adj[i] |= (1 << ((i + s) % n))
                adj[i] |= (1 << ((i - s) % n))
        # Remove self-loops
        for i in range(n):
            adj[i] &= ~(1 << i)
        return greedy_alpha_lower_bound(n, adj)

    def build_circulant(n, S):
        """Build adjacency list for circulant graph C_n(S)."""
        adj = [0] * n
        S_set = set(S)
        for i in range(n):
            for s in S:
                adj[i] |= (1 << ((i + s) % n))
                adj[i] |= (1 << ((i - s) % n))
            adj[i] &= ~(1 << i)
        return adj

    # Try Andrássai-inspired constructions for n from 40 down to 27
    for n in range(40, 26, -1):
        if time.time() > deadline - 5:
            break
        
        adj = andrassyi_template(n)
        if not is_triangle_free(n, adj):
            # Try to fix by removing edges
            for _ in range(100):
                tri = find_triangle(adj, n)
                if tri is None:
                    break
                u, v, w = tri
                # Remove one edge randomly
                candidates = [(u, v), (u, w), (v, w)]
                random.shuffle(candidates)
                for edge in candidates:
                    eu, ev = edge
                    adj[eu] &= ~(1 << ev)
                    adj[ev] &= ~(1 << eu)
                    break
        
        if is_triangle_free(n, adj) and greedy_alpha_lower_bound(n, adj) < 10:
            if validate(n, adj):
                edges = edges_from_adj(n, adj)
                return {'n': n, 'edges': edges}

    # Try systematic circulant constructions for n from 40 down to 27
    for n in range(40, 26, -1):
        if time.time() > deadline - 5:
            break

        # Generate candidate sum-free sets S for this n
        # S must be symmetric: if s in S then n-s in S
        # For small n, enumerate subsets of {1, 2, ..., n//2}
        max_s = n // 2
        candidates = list(range(1, max_s + 1))
        
        # Try different sizes of S, starting from larger (denser graphs)
        # But we need triangle-free, so S must be sum-free
        # For n around 38-40, good S sizes are around 5-8
        
        found_for_n = False
        for size in range(max_s // 2, 2, -1):
            if found_for_n:
                break
            if time.time() > deadline - 3:
                break
                
            # Enumerate combinations of 'size' elements from candidates
            # This can be expensive, so limit it
            count = 0
            for S_comb in itertools.combinations(candidates, size):
                if count % 1000 == 0 and time.time() > deadline - 2:
                    break
                count += 1
                
                S = set(S_comb)
                # Make symmetric
                S_sym = set()
                for s in S:
                    S_sym.add(s)
                    S_sym.add(n - s)
                
                # Check triangle-free
                if not is_sum_free_circulant(n, S_sym):
                    continue
                
                # Quick density check
                edge_count = n * len(S_sym) // 2
                if edge_count < n * n // 10:
                    continue
                
                # Build and validate
                adj = build_circulant(n, S_sym)
                if greedy_alpha_lower_bound(n, adj) >= 10:
                    continue
                
                if validate(n, adj):
                    edges = edges_from_adj(n, adj)
                    return {'n': n, 'edges': edges}
        
        # Also try random circulant-like constructions
        for attempt in range(200):
            if time.time() > deadline - 2:
                break
            
            # Random symmetric set
            S = set()
            for s in range(1, n // 2 + 1):
                if random.random() < 0.3:
                    S.add(s)
                    S.add(n - s)
            
            if len(S) < 4:
                continue
            
            if not is_sum_free_circulant(n, S):
                continue
            
            edge_count = n * len(S) // 2
            if edge_count < n * n // 10:
                continue
            
            adj = build_circulant(n, S)
            if greedy_alpha_lower_bound(n, adj) >= 10:
                continue
            
            if validate(n, adj):
                edges = edges_from_adj(n, adj)
                return {'n': n, 'edges': edges}

    # Start-large-and-prune search: begin with dense graphs, remove edges to kill triangles
    for n in range(40, 26, -1):
        if time.time() > deadline - 10:
            break

        remaining_time = deadline - time.time()
        max_attempts = min(200, max(20, int(remaining_time * 5)))

        for attempt in range(max_attempts):
            if time.time() > deadline - 3:
                break

            edges = start_large_and_prune(n, deadline)
            if edges is not None:
                return {'n': n, 'edges': edges}

    # Fallback to random maximal triangle-free search
    for n in range(40, 26, -1):
        if time.time() > deadline - 5:
            break

        remaining_time = deadline - time.time()
        max_attempts = min(1000, max(100, int(remaining_time * 30)))

        for attempt in range(max_attempts):
            if time.time() > deadline - 2:
                break

            adj = [0] * n
            all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
            random.shuffle(all_edges)

            for u, v in all_edges:
                if (adj[u] & adj[v]) == 0:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)

            edge_count = sum(bin(adj[i]).count('1') for i in range(n)) // 2
            if edge_count < n * n // 10:
                continue

            if greedy_alpha_lower_bound(n, adj) >= 10:
                continue

            edges = edges_from_adj(n, adj)

            if validate(n, adj):
                return {'n': n, 'edges': edges}

    return {'n': best_n, 'edges': best_edges}
