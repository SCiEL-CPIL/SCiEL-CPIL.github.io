import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n = 40
    m = 401

    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    all_edges_set = set(all_edges)

    def build_adj(edge_set):
        adj = [0] * n
        for (u, v) in edge_set:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def book_size(u, v, adj):
        return bin(adj[u] & adj[v]).count('1')

    def compute_books(edge_set, adj):
        books = {}
        for (u, v) in edge_set:
            books[(u, v)] = book_size(u, v, adj)
        return books

    def max_book(books):
        return max(books.values()) if books else 0

    def circulant_init():
        """
        Initialize with a circulant-like graph.
        We want a regular graph with degree approx 20.1.
        Since 401 is odd, we can't have a perfectly regular graph.
        We start with a 20-regular circulant graph (40 * 20 / 2 = 400 edges)
        and add 1 random edge.
        
        Circulant graph C(40, S) where S is a set of shifts.
        For 20-regular, we need 10 shifts (symmetric).
        Let's use shifts {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}.
        This connects i to i+1, i+2, ..., i+10 (mod 40) and their symmetric counterparts.
        Actually, if we include shift k, we get edges (i, i+k) for all i.
        For k != 20, this gives 40 edges.
        We need 10 such shifts to get 400 edges.
        Shifts: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10.
        """
        edge_set = set()
        shifts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for shift in shifts:
            for i in range(n):
                j = (i + shift) % n
                if i < j:
                    edge_set.add((i, j))
                else:
                    edge_set.add((j, i))
        
        # We have 400 edges. Add 1 more.
        # Pick a random edge not in the set.
        # To be deterministic-ish but varied, pick based on random seed or just random.
        available = [e for e in all_edges if e not in edge_set]
        if available:
            edge_set.add(random.choice(available))
        
        return edge_set

    def near_regular_init():
        """
        Initialize with a random graph that is as regular as possible.
        Start with 20-regular (400 edges) and add 1.
        Use a random permutation approach or just random sampling and fixing degrees.
        Simpler: Start with circulant, then do a few random swaps to break symmetry.
        """
        edge_set = circulant_init()
        # Perform 20 random swaps to break symmetry
        adj = build_adj(edge_set)
        for _ in range(20):
            # Remove a random edge
            e = random.choice(list(edge_set))
            u, v = e
            edge_set.remove(e)
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            
            # Add a random non-edge
            # Try to find a non-edge
            found = False
            for _ in range(100):
                ru = random.randint(0, n - 1)
                rv = random.randint(0, n - 1)
                if ru == rv:
                    continue
                if ru > rv:
                    ru, rv = rv, ru
                if (ru, rv) not in edge_set:
                    edge_set.add((ru, rv))
                    adj[ru] |= (1 << rv)
                    adj[rv] |= (1 << ru)
                    found = True
                    break
            if not found:
                # Revert if we couldn't add
                edge_set.add(e)
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        return edge_set

    def simulated_annealing(edge_set, adj, books, current_score, max_steps=20000):
        """
        Simulated Annealing to minimize the max book size.
        """
        T = 1.0
        cooling = 0.9995
        min_T = 0.001
        
        best_score = current_score
        best_books = books.copy()
        best_edge_set = set(edge_set)
        
        for step in range(max_steps):
            if time.time() > deadline - 2:
                break
            if T < min_T:
                break
                
            # Pick a random edge to remove
            u, v = random.choice(list(edge_set))
            
            # Pick a random non-edge to add
            # Sample until we find one
            added = False
            for _ in range(50):
                ru = random.randint(0, n - 1)
                rv = random.randint(0, n - 1)
                if ru == rv:
                    continue
                if ru > rv:
                    ru, rv = rv, ru
                if (ru, rv) not in edge_set:
                    # Apply move
                    edge_set.remove((u, v))
                    edge_set.add((ru, rv))
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    adj[ru] |= (1 << rv)
                    adj[rv] |= (1 << ru)
                    
                    # Calculate new books for affected edges
                    # Affected vertices: u, v, ru, rv
                    affected_vertices = {u, v, ru, rv}
                    
                    # We need to recalculate books for all edges incident to affected vertices
                    # And the new edge itself
                    new_books = books.copy()
                    
                    # Edges that might change:
                    # 1. Edges incident to u, v, ru, rv
                    # 2. The removed edge (u,v) is gone
                    # 3. The added edge (ru,rv) is new
                    
                    # Let's just recalculate all edges incident to affected vertices
                    # This is O(deg) per vertex, which is small.
                    
                    # First, remove (u,v) from books
                    if (u, v) in new_books:
                        del new_books[(u, v)]
                    
                    # Add (ru, rv)
                    new_books[(ru, rv)] = book_size(ru, rv, adj)
                    
                    # Update other edges
                    # For any edge (a, b) in new_books, if a or b is in affected_vertices, recalculate
                    # We can iterate through all edges in edge_set and check
                    # But that's O(m). 
                    # Better: iterate through neighbors of affected vertices.
                    
                    # Let's create a list of edges to check
                    edges_to_check = set()
                    for vert in affected_vertices:
                        # Get neighbors from adj
                        neighbors = [i for i in range(n) if (adj[vert] >> i) & 1]
                        for nb in neighbors:
                            if vert < nb:
                                edges_to_check.add((vert, nb))
                            else:
                                edges_to_check.add((nb, vert))
                    
                    for e in edges_to_check:
                        if e in new_books:
                            new_books[e] = book_size(e[0], e[1], adj)
                    
                    new_score = max_book(new_books)
                    
                    # Acceptance criterion
                    delta = new_score - current_score
                    if delta < 0 or random.random() < math.exp(-delta / T):
                        # Accept
                        current_score = new_score
                        books = new_books
                        if new_score < best_score:
                            best_score = new_score
                            best_books = new_books.copy()
                            best_edge_set = set(edge_set)
                    else:
                        # Revert
                        edge_set.remove((ru, rv))
                        edge_set.add((u, v))
                        adj[ru] &= ~(1 << rv)
                        adj[rv] &= ~(1 << ru)
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        
                    added = True
                    break
            
            if not added:
                continue
                
            T *= cooling
            
        return best_edge_set, best_score, best_books

    def local_search_refine(edge_set, adj, books, current_score, max_steps=5000):
        """
        Greedy local search: always accept improving moves.
        Focus on reducing the worst edge.
        """
        for step in range(max_steps):
            if time.time() > deadline - 2:
                break
                
            # Find worst edge
            worst_edge = None
            worst_val = -1
            for e, val in books.items():
                if val > worst_val:
                    worst_val = val
                    worst_edge = e
            
            if worst_val <= 6: # Theoretical lower bound is 40/6 = 6.66, so 7 is min integer?
                # If we reach 7, we are at the theoretical limit for n=40?
                # Theorem says >= 40/6 = 6.66. So 7 is the smallest integer.
                # If we can't go below 7, we stop.
                break

            u, v = worst_edge
            improved = False
            
            # Try to replace (u, v) with a random non-edge that has low book size
            # And ensure the new max doesn't increase
            
            candidates = []
            seen = set()
            for _ in range(100):
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
                
                # Check book size of candidate
                # We need to simulate adding it to see its book size?
                # book_size(ru, rv) in current graph is a good proxy.
                # But it might change if we remove (u,v).
                # However, (u,v) removal only affects books of edges incident to u or v.
                # If (ru, rv) is not incident to u or v, its book size is unchanged.
                # If it is, it might change.
                
                # For simplicity, use current book size as heuristic.
                tc = book_size(ru, rv, adj)
                candidates.append((tc, ru, rv))
            
            candidates.sort(key=lambda x: x[0])
            
            for tc, ru, rv in candidates[:10]:
                # Apply
                edge_set.remove((u, v))
                edge_set.add((ru, rv))
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                adj[ru] |= (1 << rv)
                adj[rv] |= (1 << ru)
                
                # Recalculate books for affected edges
                affected = {u, v, ru, rv}
                new_books = {}
                for e, val in books.items():
                    if e[0] in affected or e[1] in affected:
                        new_books[e] = book_size(e[0], e[1], adj)
                    else:
                        new_books[e] = val
                new_books[(ru, rv)] = book_size(ru, rv, adj)
                
                new_max = max_book(new_books)
                
                if new_max < current_score:
                    current_score = new_max
                    books = new_books
                    improved = True
                    break
                else:
                    # Revert
                    edge_set.remove((ru, rv))
                    edge_set.add((u, v))
                    adj[ru] &= ~(1 << rv)
                    adj[rv] &= ~(1 << ru)
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            
            if not improved:
                # Try random moves to escape local minimum?
                # For now, just break if no improvement found in this step
                # Or do a random swap?
                pass
        
        return books, current_score

    best_score = 999
    best_edge_set = None
    best_books = None

    inits = [circulant_init, near_regular_init]

    for restart in range(50):
        if time.time() > deadline - 5:
            break
            
        init_fn = inits[restart % len(inits)]
        edge_set = init_fn()
        adj = build_adj(edge_set)
        books = compute_books(edge_set, adj)
        current_score = max_book(books)

        if current_score < best_score:
            best_score = current_score
            best_edge_set = set(edge_set)
            best_books = books.copy()

        if current_score <= 7:
            break

        # Run Simulated Annealing
        edge_set, current_score, books = simulated_annealing(edge_set, adj, books, current_score)
        
        if current_score < best_score:
            best_score = current_score
            best_edge_set = set(edge_set)
            best_books = books.copy()

        if current_score <= 7:
            break

        # Run Local Search Refinement
        adj = build_adj(edge_set)
        books = compute_books(edge_set, adj)
        current_score = max_book(books)
        
        books, current_score = local_search_refine(edge_set, adj, books, current_score)
        
        if current_score < best_score:
            best_score = current_score
            best_edge_set = set(edge_set)
            best_books = books.copy()

        if current_score <= 7:
            break

    if best_edge_set is None:
        best_edge_set = circulant_init()

    return [[u, v] for (u, v) in best_edge_set]
