import time
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s kill

    def instance(n, p, seed):
        rng = Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def build_adj(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def bron_kerbosch(adj, P, X, C, maximal_cliques):
        if not P and not X:
            if len(C) >= 2:
                maximal_cliques.append(frozenset(C))
            return
        
        P_union_X = P | X
        if P_union_X:
            # Pivot selection: choose vertex u in P union X with max degree in P
            pivot = None
            max_inter = -1
            for v in P_union_X:
                inter = len(P & adj[v])
                if inter > max_inter:
                    max_inter = inter
                    pivot = v
            
            # Branch on P \ N(pivot)
            candidates = P - adj[pivot] if pivot is not None else P
            
            for v in list(candidates):
                new_P = P & adj[v]
                new_X = X & adj[v]
                new_C = C | {v}
                bron_kerbosch(adj, new_P, new_X, new_C, maximal_cliques)
                P.remove(v)
                X.add(v)

    def find_min_transversal(n, edges):
        if not edges:
            return []
        
        adj = build_adj(n, edges)
        maximal_cliques = []
        # Bron-Kerbosch
        bron_kerbosch(adj, set(range(n)), set(), set(), maximal_cliques)
        
        if not maximal_cliques:
            return []

        # Convert to list for faster iteration if needed, but frozensets are fine
        # Pre-compute clique membership for vertices? 
        # n is small (<=40), number of cliques is manageable.
        
        unhit = list(maximal_cliques)
        transversal = set()
        
        # Greedy Hitting Set
        while unhit:
            if time.time() > deadline:
                break
            
            # Count frequency of each vertex in unhit cliques
            freq = {}
            for clique in unhit:
                for v in clique:
                    freq[v] = freq.get(v, 0) + 1
            
            if not freq:
                break
                
            # Pick vertex with max frequency
            # Tie-break: higher degree first, then lower ID
            best_v = None
            best_score = -1
            for v, f in freq.items():
                # Score: frequency is primary. Degree is secondary.
                score = (f, len(adj[v]), -v)
                if best_score == -1 or score > best_score:
                    best_score = score
                    best_v = v
            
            transversal.add(best_v)
            
            # Remove all cliques containing best_v
            new_unhit = []
            for clique in unhit:
                if best_v not in clique:
                    new_unhit.append(clique)
            unhit = new_unhit

        # Post-processing: Remove redundant vertices
        # We want to ensure every clique in maximal_cliques is hit by transversal
        # Let's verify current transversal hits all cliques.
        # Then try to remove vertices.
        
        # Check if current transversal is valid
        def is_valid(trans):
            for c in maximal_cliques:
                if not (c & trans):
                    return False
            return True

        if not is_valid(transversal):
             # Fallback if timeout broke loop mid-way: add all remaining needed
             # Just return what we have if invalid? No, must be valid.
             # If timeout, we should return a valid set.
             # The greedy loop adds vertices. If we break early, we might have unhit cliques.
             # To ensure validity under timeout, we can just pick one vertex from each unhit clique.
             if unhit:
                 for c in unhit:
                     # Pick first vertex in clique
                     transversal.add(next(iter(c)))
             
             # Re-verify? The "pick first" ensures each specific unhit clique is hit.
             # But does it hit ALL cliques? Yes, because we processed all unhit ones.
             # And the previously hit ones remain hit.
             
        # Now try to reduce size
        # Greedy removal:
        # We want to remove v if every clique containing v also contains some other vertex in transversal.
        
        # To make removal efficient, we can just loop.
        changed = True
        while changed:
            changed = False
            for v in list(transversal):
                # Check if v is redundant
                redundant = True
                for c in maximal_cliques:
                    if v in c:
                        # Check if c has another vertex in transversal
                        others = c - {v}
                        if not (others & transversal):
                            redundant = False
                            break
                if redundant:
                    transversal.remove(v)
                    changed = True
                    if time.time() > deadline:
                        return sorted(transversal)

        return sorted(transversal)

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        edges = instance(n, p, seed)
        sol = find_min_transversal(n, edges)
        results.append(sol)
        if time.time() > deadline:
             # If we ran out of time, ensure we return what we have
             # But we need to return 3 lists. 
             # If we didn't finish all 3, we should return best effort or empty?
             # Better to return valid sets for completed ones and try fast heuristic for others?
             # Actually, 120s is plenty for N=40. The loop is very fast.
             pass

    return results

# EVOLVE-BLOCK-START
def construct():
    import time
    from random import Random

    start_time = time.time()
    deadline = start_time + 100  # Safety margin

    def instance(n, p, seed):
        rng = Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def build_adj(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def bron_kerbosch(adj, P, X, C, maximal_cliques):
        if not P and not X:
            if len(C) >= 2:
                maximal_cliques.append(frozenset(C))
            return
        
        P_union_X = P | X
        if P_union_X:
            pivot = None
            max_inter = -1
            for v in P_union_X:
                inter = len(P & adj[v])
                if inter > max_inter:
                    max_inter = inter
                    pivot = v
            
            candidates = P - adj[pivot] if pivot is not None else P
            
            for v in list(candidates):
                new_P = P & adj[v]
                new_X = X & adj[v]
                new_C = C | {v}
                bron_kerbosch(adj, new_P, new_X, new_C, maximal_cliques)
                P.remove(v)
                X.add(v)

    def find_min_transversal(n, edges):
        if not edges:
            return []
        
        adj = build_adj(n, edges)
        maximal_cliques = []
        bron_kerbosch(adj, set(range(n)), set(), set(), maximal_cliques)
        
        if not maximal_cliques:
            return []

        unhit = list(maximal_cliques)
        transversal = set()
        
        while unhit:
            if time.time() > deadline:
                break
            freq = {}
            for clique in unhit:
                for v in clique:
                    freq[v] = freq.get(v, 0) + 1
            
            if not freq:
                break
                
            best_v = None
            best_score = None
            for v, f in freq.items():
                score = (f, len(adj[v]), -v)
                if best_score is None or score > best_score:
                    best_score = score
                    best_v = v
            
            transversal.add(best_v)
            
            new_unhit = []
            for clique in unhit:
                if best_v not in clique:
                    new_unhit.append(clique)
            unhit = new_unhit

        # Ensure validity if broken by time
        if unhit:
            for c in unhit:
                transversal.add(next(iter(c)))

        # Redundancy removal
        changed = True
        while changed:
            changed = False
            for v in list(transversal):
                if time.time() > deadline:
                    return sorted(transversal)
                redundant = True
                for c in maximal_cliques:
                    if v in c:
                        others = c - {v}
                        if not (others & transversal):
                            redundant = False
                            break
                if redundant:
                    transversal.remove(v)
                    changed = True

        return sorted(transversal)

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        edges = instance(n, p, seed)
        sol = find_min_transversal(n, edges)
        results.append(sol)

    return results
# EVOLVE-BLOCK-END
