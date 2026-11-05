import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    n = 20
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Precompute triples for fast validity check if needed, 
    # but try_add is faster for incremental updates.
    
    def count_edges(adj):
        return sum(bin(adj[i]).count('1') for i in range(n)) // 2

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def try_add(adj, u, v):
        """
        Returns True if adding (u,v) keeps the graph K3,3-free.
        Constraint: No 3 vertices have 3 common neighbours.
        Adding (u,v) affects triples containing u or v.
        For a triple {u, a, b}, the common neighbours are N(u) \cap N(a) \cap N(b).
        After adding (u,v), N(u) includes v.
        So new common neighbours = (N(u) \cap N(a) \cap N(b)) \cup ({v} \cap N(a) \cap N(b)).
        The count increases by 1 if v is in N(a) \cap N(b).
        Violation occurs if new count >= 3.
        So if v in N(a) \cap N(b), we require old count <= 1.
        If v not in N(a) \cap N(b), count is unchanged.
        """
        # Check triples containing u
        # For any a, b != u, v:
        # If v is neighbour of a and b, then (u,a,b) gains a common neighbour v.
        # We need count_common(u,a,b) < 2 in that case.
        # Note: a, b can be any vertices except u.
        # We iterate over pairs (a,b) where v is a common neighbour of a and b.
        # This is equivalent to iterating over common neighbours of u and v? No.
        
        # Efficient check:
        # 1. Find all common neighbours of u and v? No, that's for the edge itself.
        # 2. We need to check if there exist a,b such that v in N(a) and v in N(b) and |N(u) \cap N(a) \cap N(b)| >= 2.
        #    Wait, if v in N(a) \cap N(b), the new count is old_count + 1.
        #    So we need old_count <= 1.
        
        # Let's iterate over all pairs (a,b) from V \ {u,v}.
        # If (adj[a] & (1<<v)) and (adj[b] & (1<<v)):
        #    cm = adj[u] & adj[a] & adj[b]
        #    cm &= ~((1<<u) | (1<<a) | (1<<b)) # Exclude the vertices themselves from common neighbours?
        #    Actually, common neighbours are vertices x such that x~u, x~a, x~b.
        #    u is not a neighbour of u. So u is not in N(u).
        #    So we don't need to mask u, a, b unless the graph has self-loops (it doesn't).
        #    However, v is in N(a) and N(b). Is v in N(u)? No, we are adding it.
        #    So v is not in the old N(u).
        #    So the old common neighbours do not include v.
        #    The new common neighbours include v if v~a and v~b.
        #    So if v~a and v~b, we check if old_count >= 2. If so, adding (u,v) makes it 3.
        
        # Similarly for triples containing v.
        
        # Optimization:
        # Instead of all pairs, note that if v is not a common neighbour of any pair (a,b) 
        # that has >= 2 common neighbours with u, we are safe.
        
        # Let's just do the O(n^2) check. n=20, so 400 pairs. Very fast.
        
        # Mask for u and v to exclude from consideration as 'a' or 'b' if we were iterating all, 
        # but we explicitly skip them.
        
        # Triples with u: {u, a, b} where a,b != u,v
        for a in range(n):
            if a == u or a == v: continue
            if not (adj[a] >> v) & 1: continue # v must be neighbour of a
            for b in range(a + 1, n):
                if b == u or b == v: continue
                if not (adj[b] >> v) & 1: continue # v must be neighbour of b
                # v is common neighbour of a and b.
                # Check common neighbours of u, a, b.
                cm = adj[u] & adj[a] & adj[b]
                # We need to ensure cm does not contain u, a, b?
                # N(u) does not contain u.
                # N(a) does not contain a.
                # N(b) does not contain b.
                # So cm automatically excludes u, a, b.
                if bin(cm).count('1') >= 2:
                    return False
        
        # Triples with v: {v, a, b} where a,b != u,v
        for a in range(n):
            if a == u or a == v: continue
            if not (adj[a] >> u) & 1: continue # u must be neighbour of a
            for b in range(a + 1, n):
                if b == u or b == v: continue
                if not (adj[b] >> u) & 1: continue # u must be neighbour of b
                cm = adj[v] & adj[a] & adj[b]
                if bin(cm).count('1') >= 2:
                    return False
                    
        return True

    def repair(adj):
        """Remove edges until valid."""
        max_iters = 5000
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            found_violation = False
            # Check all triples
            for i in range(n):
                for j in range(i + 1, n):
                    for k in range(j + 1, n):
                        cm = adj[i] & adj[j] & adj[k]
                        if bin(cm).count('1') >= 3:
                            # Remove a random edge incident to one of i,j,k and a common neighbour
                            common_neighbours = [x for x in range(n) if (cm >> x) & 1]
                            c = random.choice(common_neighbours)
                            v = random.choice([i, j, k])
                            remove_edge(adj, v, c)
                            found_violation = True
                            break
                    if found_violation:
                        break
                if found_violation:
                    break
            if not found_violation:
                break

    def greedy_add(adj):
        """Add all possible edges."""
        edges = count_edges(adj)
        changed = True
        while changed:
            if time.time() > deadline - 5:
                break
            changed = False
            for u, v in all_edges:
                if (adj[u] >> v) & 1:
                    continue
                if try_add(adj, u, v):
                    add_edge(adj, u, v)
                    edges += 1
                    changed = True
        return edges

    def local_search(adj):
        """
        Escape local minima by swapping edges.
        """
        edges = count_edges(adj)
        present = [(u, v) for (u, v) in all_edges if (adj[u] >> v) & 1]
        absent = [(u, v) for (u, v) in all_edges if not ((adj[u] >> v) & 1)]
        
        # Try to add edges first
        edges = greedy_add(adj)
        present = [(u, v) for (u, v) in all_edges if (adj[u] >> v) & 1]
        absent = [(u, v) for (u, v) in all_edges if not ((adj[u] >> v) & 1)]
        
        max_swaps = 100
        for _ in range(max_swaps):
            if time.time() > deadline - 5:
                break
            if not present or not absent:
                break
            
            # Pick random edge to remove and random edge to add
            # To be more effective, we can try to remove edges that are "central"
            # But random is a good baseline.
            e_remove = random.choice(present)
            e_add = random.choice(absent)
            
            if e_remove == e_add:
                continue
                
            u1, v1 = e_remove
            u2, v2 = e_add
            
            # Temporarily remove e_remove
            remove_edge(adj, u1, v1)
            
            # Try to add e_add
            if try_add(adj, u2, v2):
                add_edge(adj, u2, v2)
                # Update lists
                present.remove(e_remove)
                present.append(e_add)
                absent.remove(e_add)
                absent.append(e_remove)
                # After a swap, try to add more edges
                edges = greedy_add(adj)
                present = [(u, v) for (u, v) in all_edges if (adj[u] >> v) & 1]
                absent = [(u, v) for (u, v) in all_edges if not ((adj[u] >> v) & 1)]
            else:
                # Restore
                add_edge(adj, u1, v1)
                
        return edges

    def make_paley_start():
        """
        Construct a graph based on Paley graph of order 17, plus 3 vertices.
        """
        adj = [0] * n
        # Paley graph on 17 vertices: i~j if (i-j)^2 is a quadratic residue mod 17
        # Quadratic residues mod 17: 1, 2, 4, 8, 9, 13, 15, 16
        # 1^2=1, 2^2=4, 3^2=9, 4^2=16, 5^2=8, 6^2=2, 7^2=15, 8^2=13
        residues = {1, 2, 4, 8, 9, 13, 15, 16}
        
        for i in range(17):
            for j in range(i + 1, 17):
                d = (j - i) % 17
                if d in residues:
                    add_edge(adj, i, j)
        
        # Add vertices 17, 18, 19
        # Connect them to some vertices in the Paley graph
        # To maintain K3,3-free, we should be careful.
        # Connect 17 to 0, 1, 2
        # Connect 18 to 3, 4, 5
        # Connect 19 to 6, 7, 8
        # And connect 17,18,19 to each other?
        # Let's just add random edges for 17,18,19 and let repair fix it.
        for v in [17, 18, 19]:
            for u in range(17):
                if random.random() < 0.5:
                    add_edge(adj, v, u)
            # Connect the new vertices among themselves
            for w in [17, 18, 19]:
                if v < w and random.random() < 0.5:
                    add_edge(adj, v, w)
                    
        return adj

    def make_random_start(density):
        adj = [0] * n
        for (i, j) in all_edges:
            if random.random() < density:
                add_edge(adj, i, j)
        return adj

    best_edges = 0
    best_adj = None

    # Run multiple iterations
    num_iterations = 0
    while time.time() < deadline - 10:
        num_iterations += 1
        
        # Choose a start strategy
        if num_iterations % 3 == 0:
            adj = make_paley_start()
        else:
            density = 0.5 + random.random() * 0.3 # 0.5 to 0.8
            adj = make_random_start(density)
            
        # Repair
        repair(adj)
        
        # Greedy Add
        ec = greedy_add(adj)
        
        # Local Search
        ec = local_search(adj)
        
        if ec > best_edges:
            best_edges = ec
            best_adj = [a for a in adj]
            # print(f"New best: {best_edges} at iter {num_iterations}")

    if best_adj is None:
        return []

    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])

    return edges

