import random
import time
import itertools

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s hard limit
    
    def count_edges(G):
        return sum(G[i].bit_count() for i in range(n)) // 2

    def is_k33_free(G):
        """Check if graph is K3,3-free."""
        for i, j, k in itertools.combinations(range(n), 3):
            mask = G[i] & G[j] & G[k]
            if mask.bit_count() >= 3:
                return False
        return True

    def is_add_valid(G, u, v):
        """Check if adding edge (u,v) keeps graph K3,3-free.
        
        After adding (u,v), we need no 3 vertices with 3 common neighbors.
        The new edge can only create a K3,3 if there exist a1,a2 in N(v)\{u}
        and b1,b2 in N(u)\{v} such that {a1,a2,b1,b2} form a K2,2 (i.e.,
        a1,a2 share 2 common neighbors in N(u)\{v}, or equivalently the
        bipartite graph between N(v)\{u} and N(u)\{v} has a K2,2).
        """
        A = G[v] & ~(1 << u)  # neighbors of v except u
        B = G[u] & ~(1 << v)  # neighbors of u except v
        
        # Get list of vertices in A
        A_list = []
        m = A
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            A_list.append(idx)
            m ^= lsb
        
        # For each pair in A, check if they have >= 2 common neighbors in B
        for i in range(len(A_list)):
            a1 = A_list[i]
            for j in range(i + 1, len(A_list)):
                a2 = A_list[j]
                common = G[a1] & G[a2] & B
                if common.bit_count() >= 2:
                    return False
        return True

    def greedy_add_edges(G, rng, deadline):
        """Try to add all possible edges greedily until no more can be added."""
        missing = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (G[i] & (1 << j)):
                    missing.append((i, j))
        
        while missing:
            rng.shuffle(missing)
            added_any = False
            new_missing = []
            for u, v in missing:
                if G[u] & (1 << v):
                    continue
                if is_add_valid(G, u, v):
                    G[u] |= (1 << v)
                    G[v] |= (1 << u)
                    added_any = True
                else:
                    new_missing.append((u, v))
            if not added_any:
                break
            missing = new_missing
            if time.time() > deadline:
                break

    def local_search(G, rng, deadline):
        """Edge swap local search: remove one edge, try to add another."""
        all_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if G[i] & (1 << j):
                    all_edges.append((i, j))
        
        missing = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (G[i] & (1 << j)):
                    missing.append((i, j))
        
        if not all_edges or not missing:
            return
        
        rng.shuffle(all_edges)
        rng.shuffle(missing)
        
        for u, v in all_edges:
            if time.time() > deadline:
                break
            # Remove edge (u,v)
            G[u] &= ~(1 << v)
            G[v] &= ~(1 << u)
            
            # Try to add a missing edge
            found = False
            for x, y in missing:
                if (x == u and y == v) or (x == v and y == u):
                    continue
                if G[x] & (1 << y):
                    continue
                if is_add_valid(G, x, y):
                    G[x] |= (1 << y)
                    G[y] |= (1 << x)
                    found = True
                    break
            
            if not found:
                # Put edge back
                G[u] |= (1 << v)
                G[v] |= (1 << u)

    def two_opt_swap(G, rng, deadline):
        """Try 2-edge swaps: remove two edges, add two different edges."""
        all_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if G[i] & (1 << j):
                    all_edges.append((i, j))
        
        missing = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (G[i] & (1 << j)):
                    missing.append((i, j))
        
        if len(all_edges) < 2 or len(missing) < 2:
            return
        
        rng.shuffle(all_edges)
        rng.shuffle(missing)
        
        # Limit number of attempts
        max_attempts = 50
        attempts = 0
        for idx1 in range(min(len(all_edges), 30)):
            if time.time() > deadline or attempts >= max_attempts:
                break
            e1 = all_edges[idx1]
            for idx2 in range(idx1 + 1, min(len(all_edges), 30)):
                if time.time() > deadline or attempts >= max_attempts:
                    break
                attempts += 1
                e2 = all_edges[idx2]
                
                u1, v1 = e1
                u2, v2 = e2
                
                # Remove both edges
                G[u1] &= ~(1 << v1)
                G[v1] &= ~(1 << u1)
                G[u2] &= ~(1 << v2)
                G[v2] &= ~(1 << u2)
                
                # Try to add two missing edges
                found = False
                for m1_idx in range(min(len(missing), 20)):
                    if time.time() > deadline:
                        break
                    x1, y1 = missing[m1_idx]
                    if G[x1] & (1 << y1):
                        continue
                    if not is_add_valid(G, x1, y1):
                        continue
                    G[x1] |= (1 << y1)
                    G[y1] |= (1 << x1)
                    
                    for m2_idx in range(m1_idx + 1, min(len(missing), 20)):
                        if time.time() > deadline:
                            break
                        x2, y2 = missing[m2_idx]
                        if G[x2] & (1 << y2):
                            continue
                        if not is_add_valid(G, x2, y2):
                            continue
                        G[x2] |= (1 << y2)
                        G[y2] |= (1 << x2)
                        found = True
                        break
                    if found:
                        break
                
                if not found:
                    # Restore edges
                    G[u1] |= (1 << v1)
                    G[v1] |= (1 << u1)
                    G[u2] |= (1 << v2)
                    G[v2] |= (1 << u2)

    def construct_deterministic():
        """Construct a K3,3-free graph using combinatorial design.
        
        Strategy: Use the complete graph on 7 vertices with a 1-factor removed.
        K7 minus a perfect matching (or near-perfect for odd n) is K3,3-free
        because any 3 vertices in the 7-vertex part have at most 4 common neighbors
        within that part, and we can verify the full constraint.
        
        Better approach: Use a known construction. The graph formed by taking
        vertices 0..19 and connecting i to j if (i-j) mod 20 is in a specific set.
        
        Actually, let's use a simpler deterministic construction:
        Partition into groups and use a bipartite-like structure that avoids K3,3.
        
        Key insight: A graph where each vertex has degree <= d and the graph is
        sufficiently "spread out" can be K3,3-free. 
        
        Let's try: Complete 5-partite graph with parts of size 4.
        In K_{4,4,4,4,4}, any 3 vertices from different parts have 12 common neighbors.
        That's too many.
        
        Better: Use a random but structured approach. Start with a cycle-based construction.
        
        Simplest reliable construction: The complete graph K7 is K3,3-free (it has
        only 7 vertices, so no K3,3 which needs 6 vertices from two parts of size 3,
        but K7 contains K3,3 as a subgraph! K7 has 7 vertices, and we can pick 3+3=6
        of them, and all edges exist between them, so K7 contains K3,3).
        
        So K7 is NOT K3,3-free. We need a different approach.
        
        Let me think: K3,3 is bipartite with parts of size 3. A graph is K3,3-free
        if no 6 vertices can be partitioned into two sets of 3 with all 9 edges present.
        
        Known result: The Petersen graph is K3,3-free. It has 10 vertices and 15 edges.
        We can take two copies and connect them in a K3,3-free way.
        
        Actually, let's just use a high-quality random initialization and rely on
        local search to find good solutions.
        """
        rng = random.Random(42)
        G = [0] * n
        
        # Start with a moderately dense random graph
        p = 0.6
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < p:
                    G[i] |= (1 << j)
                    G[j] |= (1 << i)
        
        # Remove edges to make K3,3-free
        max_removals = 200
        removals = 0
        while removals < max_removals:
            if time.time() > deadline:
                break
            # Find a K3,3
            found = False
            for i, j, k in itertools.combinations(range(n), 3):
                mask = G[i] & G[j] & G[k]
                if mask.bit_count() >= 3:
                    # Get 3 common neighbors
                    bits = []
                    m = mask
                    while len(bits) < 3 and m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        bits.append(idx)
                        m ^= lsb
                    # Remove a random edge from the K3,3
                    edges_to_remove = [(i, bits[0]), (j, bits[1]), (k, bits[2])]
                    u, v = rng.choice(edges_to_remove)
                    G[u] &= ~(1 << v)
                    G[v] &= ~(1 << u)
                    found = True
                    break
            if not found:
                break
            removals += 1
        
        return G

    best_edges = 0
    best_graph = None
    
    # Phase 1: Deterministic construction + local search
    G = construct_deterministic()
    greedy_add_edges(G, random.Random(1), deadline)
    local_search(G, random.Random(2), deadline)
    ec = count_edges(G)
    if ec > best_edges:
        best_edges = ec
        best_graph = G[:]
    
    # Phase 2: Multiple random restarts with local search
    seed = 10
    while time.time() < deadline - 5:
        rng = random.Random(seed)
        
        # Random initialization at varying densities
        p = rng.uniform(0.5, 0.85)
        G = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < p:
                    G[i] |= (1 << j)
                    G[j] |= (1 << i)
        
        # Remove edges to make K3,3-free
        max_removals = 200
        removals = 0
        while removals < max_removals:
            if time.time() > deadline:
                break
            found = False
            for i, j, k in itertools.combinations(range(n), 3):
                mask = G[i] & G[j] & G[k]
                if mask.bit_count() >= 3:
                    bits = []
                    m = mask
                    while len(bits) < 3 and m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        bits.append(idx)
                        m ^= lsb
                    edges_to_remove = [(i, bits[0]), (j, bits[1]), (k, bits[2])]
                    u, v = rng.choice(edges_to_remove)
                    G[u] &= ~(1 << v)
                    G[v] &= ~(1 << u)
                    found = True
                    break
            if not found:
                break
            removals += 1
        
        # Greedy add
        greedy_add_edges(G, rng, deadline)
        
        # Local search
        local_search(G, rng, deadline)
        
        # Two-opt swap
        two_opt_swap(G, rng, deadline)
        
        ec = count_edges(G)
        if ec > best_edges:
            best_edges = ec
            best_graph = G[:]
        
        seed += 1
    
    if best_graph is None:
        edges = [[0, i] for i in range(1, n)]
        return edges
    
    edges = []
    for u in range(n):
        mask = best_graph[u]
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            if u < v:
                edges.append([u, v])
            m ^= lsb
    return edges
