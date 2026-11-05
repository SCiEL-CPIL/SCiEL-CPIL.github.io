import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def enumerate_maximal_cliques(adj, n):
        """
        Enumerates all maximal cliques using Bron-Kerbosch with pivot.
        Returns a list of bitmasks, each representing a maximal clique.
        """
        cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                cliques.append(R)
                return
            
            # Pivot selection: choose u in P ∪ X that maximizes |P ∩ N(u)|
            PX = P | X
            if PX:
                max_common = -1
                pivot = -1
                temp = PX
                while temp:
                    lsb = temp & -temp
                    u = lsb.bit_length() - 1
                    common = (P & adj[u]).bit_count()
                    if common > max_common:
                        max_common = common
                        pivot = u
                    temp ^= lsb
            else:
                pivot = -1

            if pivot != -1:
                candidates = P & ~(adj[pivot])
            else:
                candidates = P

            temp = candidates
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                temp ^= lsb
                
                bk(R | lsb, P & adj[v], X & adj[v])
                
                P ^= lsb
                X |= lsb

        bk(0, (1 << n) - 1, 0)
        return cliques

    def solve_instance(adj, n):
        """
        Finds a minimum clique transversal.
        Strategy:
        1. Enumerate all maximal cliques of size >= 2.
        2. Filter out cliques that are subsets of other cliques (redundant).
           Actually, maximal cliques are not subsets of each other, but a clique of size 2
           might be contained in a larger clique? No, by definition maximal cliques are not
           contained in any other clique. However, if we have a clique C1 and C2, and C1 is
           not a superset of C2, they are distinct.
           Wait, the condition is "meet every maximal clique".
           If we pick a vertex v, it hits all cliques containing v.
           
           This is a Set Cover problem:
           Universe U = set of maximal cliques (size >= 2)
           Sets S_v = { C in U | v in C }
           Find min |T| such that union_{v in T} S_v = U.
           
           Greedy Set Cover is a good heuristic.
           For small n (<=40) and sparse random graphs, the number of maximal cliques is usually manageable.
        """
        cliques = enumerate_maximal_cliques(adj, n)
        
        # Filter cliques of size >= 2
        cliques = [c for c in cliques if c.bit_count() >= 2]
        
        if not cliques:
            return []

        # Remove redundant cliques?
        # In a set cover context, if clique C1 is a subset of clique C2, then hitting C2 implies hitting C1?
        # No. If C1 ⊆ C2, then any vertex in C1 is in C2. So if we pick a vertex from C2, it might not be in C1.
        # Wait. The requirement is: for EVERY maximal clique C, T ∩ C != empty.
        # If C1 ⊆ C2, and T ∩ C2 != empty, it does NOT imply T ∩ C1 != empty.
        # Example: C1={1,2}, C2={1,2,3}. T={3}. T hits C2, but misses C1.
        # So we MUST hit every maximal clique explicitly.
        
        # However, note that maximal cliques are not subsets of each other.
        # So no redundancy removal based on subset relation is possible between maximal cliques.
        
        # Greedy Set Cover
        # Represent cliques as list of ints (bitmasks)
        # For each vertex v, precompute which cliques it hits.
        # To speed up, we can just iterate.
        
        # Number of cliques might be up to ~2^(n/2) in worst case, but for random graphs n=40, it's usually small.
        # Let's check the count. If it's too large, greedy might be slow, but n is small.
        
        remaining_cliques = set(cliques)
        selected = []
        
        # Precompute for each vertex the set of cliques it covers?
        # Or just loop.
        # Let's try a simple greedy loop.
        
        while remaining_cliques and time.time() < deadline:
            best_vertex = -1
            max_hits = 0
            
            # Try all vertices
            for v in range(n):
                v_mask = 1 << v
                hits = 0
                # Count how many remaining cliques contain v
                # Optimization: if hits <= max_hits, we can break early? No, we need max.
                # But we can skip if potential max is low.
                for c in remaining_cliques:
                    if c & v_mask:
                        hits += 1
                        if hits > max_hits:
                            # Early check: if we already beat best, continue counting?
                            # Actually, we just need to know if it's better.
                            pass
                
                if hits > max_hits:
                    max_hits = hits
                    best_vertex = v
            
            if best_vertex == -1 or max_hits == 0:
                # Should not happen if remaining_cliques is not empty
                break
            
            selected.append(best_vertex)
            best_mask = 1 << best_vertex
            
            # Remove all cliques hit by best_vertex
            new_remaining = set()
            for c in remaining_cliques:
                if not (c & best_mask):
                    new_remaining.add(c)
            remaining_cliques = new_remaining
            
            # Pruning: if remaining_cliques is empty, break
            if not remaining_cliques:
                break
        
        # If we timed out, we might have an incomplete cover.
        # We need a valid cover.
        # If greedy didn't finish, we must add vertices to cover the rest.
        # Simple fallback: add all vertices not yet selected?
        # Or better: for each remaining clique, pick an arbitrary vertex from it.
        
        if remaining_cliques:
            for c in remaining_cliques:
                # Pick lowest index vertex in clique
                lsb = c & -c
                v = lsb.bit_length() - 1
                if v not in selected:
                    selected.append(v)
        
        return sorted(selected)

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline:
            # Fallback: return all vertices (valid but large)
            adj = random_graph(n, p, seed)
            results.append(list(range(n)))
            continue

        adj = random_graph(n, p, seed)
        transversal = solve_instance(adj, n)
        results.append(transversal)

    return results
