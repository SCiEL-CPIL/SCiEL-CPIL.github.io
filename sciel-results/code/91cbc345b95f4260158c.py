import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n = 40
    m = 401

    def build_bitmasks(edge_set):
        adj = [0] * n
        for (u, v) in edge_set:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def book(u, v, adj):
        return bin(adj[u] & adj[v]).count('1')

    def max_book(books):
        return max(books.values()) if books else 0

    def all_books(edge_set, adj):
        return {(u, v): book(u, v, adj) for (u, v) in edge_set}

    def clique_bipartite_init():
        """
        Structural initialization:
        Partition vertices into 4 groups of 10: A={0..9}, B={10..19}, C={20..29}, D={30..39}.
        Add all edges within A (K10) and within B (K10).
        Add all edges between C and D (K10,10).
        This creates a highly structured base with known local properties.
        Then fill remaining edges to reach m=401, preferring edges that connect
        groups with lower current common-neighbor density or balancing degrees.
        """
        edge_set = set()
        
        # Group A clique (0-9)
        for i in range(10):
            for j in range(i+1, 10):
                edge_set.add((i, j))
                
        # Group B clique (10-19)
        for i in range(10, 20):
            for j in range(i+1, 20):
                edge_set.add((i, j))
                
        # Group C-D bipartite (20-29) x (30-39)
        for i in range(20, 30):
            for j in range(30, 40):
                edge_set.add((i, j))
                
        # Current edges: 45 (A) + 45 (B) + 100 (C-D) = 190
        # Need 401 - 190 = 211 more edges.
        
        # We will add edges to balance the graph.
        # Strategy: 
        # 1. Connect A and B? No, that creates huge triangles.
        # 2. Connect A to C/D and B to C/D?
        # 3. Connect within C and D?
        
        # Let's calculate degrees to guide the fill.
        deg = [0]*n
        for u, v in edge_set:
            deg[u] += 1
            deg[v] += 1
            
        # Target degree is 20.1.
        # A and B vertices have degree 9.
        # C and D vertices have degree 10.
        
        # We need to add edges. 
        # To minimize books, we should avoid creating triangles.
        # Edges within C or within D will create triangles with the C-D edges.
        # Edges between A and C will create triangles if A is connected to another C vertex? No, A is a clique.
        # If we add edge (a, c) where a in A, c in C:
        #   Neighbors of a: A\{a} (9 nodes)
        #   Neighbors of c: D (10 nodes) + (newly added neighbors in A/B if any)
        #   Common neighbors: intersection of A\{a} and D is empty.
        #   So book(a,c) is 0 initially, plus any other neighbors added later.
        
        # So, adding edges between {A,B} and {C,D} is triangle-free initially.
        # Adding edges between A and B is triangle-free initially?
        #   Neighbors of a in A, b in B.
        #   Common neighbors: none initially.
        #   So book(a,b) is 0 initially.
        
        # Adding edges within C or D:
        #   Edge (c1, c2) in C.
        #   Neighbors of c1: D (10) + A/B connections.
        #   Neighbors of c2: D (10) + A/B connections.
        #   Common neighbors: All of D (10 nodes).
        #   So book(c1, c2) is at least 10. This is bad.
        
        # Therefore, we should AVOID adding edges within C or within D.
        # We should add edges between:
        #   A and B
        #   A and C
        #   A and D
        #   B and C
        #   B and D
        
        # Total possible edges in these cross-groups:
        # A-B: 100
        # A-C: 100
        # A-D: 100
        # B-C: 100
        # B-D: 100
        # Total 500 available. We need 211.
        
        # To keep books low, we want to distribute these edges such that no vertex gets too many connections to a common neighbor set.
        # A vertex in A is connected to 9 in A.
        # If it connects to k vertices in C, its book with those C vertices is determined by common neighbors in D? No.
        # Let's just pick random edges from the allowed cross-groups, but bias towards balancing degrees.
        
        allowed_pairs = []
        # Generate all allowed pairs
        for u in range(40):
            for v in range(u+1, 40):
                # Check if (u,v) is allowed (not in initial set, and not within C or within D)
                if (u, v) in edge_set:
                    continue
                # Disallow within C (20-29) and within D (30-39)
                if 20 <= u < 30 and 20 <= v < 30:
                    continue
                if 30 <= u < 40 and 30 <= v < 40:
                    continue
                allowed_pairs.append((u, v))
                
        random.shuffle(allowed_pairs)
        
        # Add edges from allowed_pairs until we reach m
        # To be smarter, we can pick pairs that result in lower immediate book increase.
        # But for init, random selection from this "safe" pool is a good structural start.
        
        for u, v in allowed_pairs:
            if len(edge_set) >= m:
                break
            edge_set.add((u, v))
            
        # Fallback if we didn't reach m (shouldn't happen as 190 + 211 = 401 and pool is 500)
        while len(edge_set) < m:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u != v:
                if u > v: u, v = v, u
                if (u, v) not in edge_set:
                    edge_set.add((u, v))
                    
        return edge_set

    def local_search(edge_set, adj, books, current_score, max_steps=4000, stuck_limit=60):
        stuck = 0
        for step in range(max_steps):
            if time.time() > deadline - 2:
                return books, current_score

            # worst edge
            worst_edge = None
            worst_val = -1
            for e, val in books.items():
                if val > worst_val:
                    worst_val = val
                    worst_edge = e

            if worst_val <= 7:
                break

            u, v = worst_edge
            swapped = False

            # candidate new edges: prefer low-triangle
            cands = []
            seen = set()
            for _ in range(400):
                ru = random.randint(0, n - 1)
                rv = random.randint(0, n - 1)
                if ru == rv:
                    continue
                if ru > rv:
                    ru, rv = rv, ru
                if (ru, rv) in edge_set or (ru, rv) == (u, v):
                    continue
                if (ru, rv) in seen:
                    continue
                seen.add((ru, rv))
                tc = book(ru, rv, adj)
                cands.append((tc, ru, rv))

            cands.sort(key=lambda x: x[0])

            for tc, ru, rv in cands[:40]:
                # apply swap
                edge_set.remove((u, v))
                edge_set.add((ru, rv))
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                adj[ru] |= (1 << rv)
                adj[rv] |= (1 << ru)

                affected = {u, v, ru, rv}
                new_books = {}
                for e, val in books.items():
                    eu, ev = e
                    if eu in affected or ev in affected:
                        new_books[e] = book(eu, ev, adj)
                    else:
                        new_books[e] = val
                new_books[(ru, rv)] = book(ru, rv, adj)

                new_max = max_book(new_books)
                if new_max < current_score:
                    current_score = new_max
                    books = new_books
                    swapped = True
                    stuck = 0
                    break
                else:
                    # revert
                    edge_set.remove((ru, rv))
                    edge_set.add((u, v))
                    adj[ru] &= ~(1 << rv)
                    adj[rv] &= ~(1 << ru)
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)

            if not swapped:
                stuck += 1
            else:
                stuck = 0

            if stuck > stuck_limit:
                break

        return books, current_score

    best_score = 999
    best_edge_set = None

    # Use the new structural initialization
    # We might also keep a simpler random init as a fallback or alternative restart
    def simple_random_init():
        edge_set = set()
        while len(edge_set) < m:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u != v:
                if u > v: u, v = v, u
                edge_set.add((u, v))
        return edge_set

    inits = [clique_bipartite_init, simple_random_init]
    
    for restart in range(40):
        if time.time() > deadline - 5:
            break
        init_fn = inits[restart % len(inits)]
        edge_set = init_fn()
        adj = build_bitmasks(edge_set)
        books = all_books(edge_set, adj)
        current_score = max_book(books)

        if current_score < best_score:
            best_score = current_score
            best_edge_set = set(edge_set)

        if current_score <= 7:
            break

        books, current_score = local_search(edge_set, adj, books, current_score)

        if current_score < best_score:
            best_score = current_score
            best_edge_set = set(edge_set)

        if current_score <= 7:
            break

    if best_edge_set is None:
        best_edge_set = clique_bipartite_init()

    return [[u, v] for (u, v) in best_edge_set]
