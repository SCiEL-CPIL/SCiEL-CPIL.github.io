import time
import random
from itertools import combinations

def construct():
    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def bron_kerbosch(R, P, X, graph, cliques):
        if not P and not X:
            if len(R) >= 2:
                cliques.append(frozenset(R))
            return
        
        # Pivot selection: choose u in P union X with max degree in P
        U = P.union(X)
        if U:
            pivot = max(U, key=lambda u: len(graph[u].intersection(P)))
        else:
            pivot = None
        
        # If pivot is in P, we only consider vertices in P that are NOT neighbors of pivot
        # This is an optimization: if pivot is in P, any clique containing pivot must include it,
        # but we are looking for maximal cliques. The standard Bron-Kerbosch with pivot says:
        # Iterate over P \ N(pivot).
        # Why? Because if v is in N(pivot), then any clique containing v and R can be extended with pivot,
        # so it won't be maximal in the current recursion branch if pivot is available?
        # Actually, the standard optimization is:
        # If P is empty, check X.
        # Otherwise, choose pivot u from P U X.
        # For each v in P \ N(u):
        #    BronKerbosch(R U {v}, P \ N(v), X \ N(v))
        #    P = P \ {v}
        #    X = X U {v}
        
        if pivot is not None:
            neighbors_of_pivot = graph[pivot]
            candidates = P.difference(neighbors_of_pivot)
        else:
            candidates = P

        for v in list(candidates):
            bron_kerbosch(R | {v}, P & graph[v], X & graph[v], graph, cliques)
            P.remove(v)
            X.add(v)

    def find_maximal_cliques(n, edges):
        # Build adjacency sets
        graph = {i: set() for i in range(n)}
        for u, v in edges:
            graph[u].add(v)
            graph[v].add(u)
        
        # Remove isolated vertices? They can't be in cliques of size >= 2.
        # But Bron-Kerbosch handles them naturally (they won't be in P if we start with all vertices, 
        # but they will be removed quickly or just not form cliques).
        # Actually, better to start P with vertices that have degree > 0 to save time.
        active_vertices = {v for v in range(n) if len(graph[v]) > 0}
        
        cliques = []
        bron_kerbosch(set(), active_vertices, set(), graph, cliques)
        return cliques

    def greedy_hitting_set(cliques, n):
        if not cliques:
            return set()
        
        # Map vertex to list of cliques it belongs to
        vertex_to_cliques = {v: [] for v in range(n)}
        for i, c in enumerate(cliques):
            for v in c:
                vertex_to_cliques[v].append(i)
        
        uncovered = set(range(len(cliques)))
        S = set()
        
        while uncovered:
            best_v = -1
            max_covered = -1
            for v in range(n):
                if v in S:
                    continue
                covered = sum(1 for i in vertex_to_cliques[v] if i in uncovered)
                if covered > max_covered:
                    max_covered = covered
                    best_v = v
            if best_v == -1:
                break
            S.add(best_v)
            for i in vertex_to_cliques[best_v]:
                if i in uncovered:
                    uncovered.remove(i)
        
        return S

    def local_search(S, cliques, n, time_limit=5.0):
        start_time = time.time()
        # Precompute for each clique, the set of vertices in S that hit it
        # To check validity of removal of v:
        # v is redundant if for all c in cliques, c has some other vertex in S.
        # Equivalent: For all c containing v, |c intersect S| > 1.
        
        # Let's maintain counts: count[c] = number of vertices in S that are in clique c
        counts = [0] * len(cliques)
        vertex_in_cliques = {v: [] for v in range(n)}
        for i, c in enumerate(cliques):
            for v in c:
                vertex_in_cliques[v].append(i)
        
        for v in S:
            for i in vertex_in_cliques[v]:
                counts[i] += 1
                
        # Check if initial S is valid (it should be from greedy)
        # But let's be safe.
        
        current_S = set(S)
        
        def is_valid(S_set):
            # Recount? No, we update incrementally.
            # For now, we rely on the count array.
            # If we remove v, we decrement counts for cliques containing v.
            # If any count becomes 0, invalid.
            pass

        # Helper to try removing v
        def try_remove(v):
            nonlocal counts
            # Check if removing v makes any clique uncovered
            can_remove = True
            for i in vertex_in_cliques[v]:
                if counts[i] == 1:
                    can_remove = False
                    break
            if can_remove:
                for i in vertex_in_cliques[v]:
                    counts[i] -= 1
                current_S.remove(v)
                return True
            return False

        # Helper to try adding v
        def try_add(v):
            nonlocal counts
            for i in vertex_in_cliques[v]:
                counts[i] += 1
            current_S.add(v)
            return True

        # Helper to remove v (undo)
        def undo_remove(v):
            nonlocal counts
            for i in vertex_in_cliques[v]:
                counts[i] += 1
            current_S.add(v)

        # Helper to remove v (undo add)
        def undo_add(v):
            nonlocal counts
            for i in vertex_in_cliques[v]:
                counts[i] -= 1
            current_S.remove(v)

        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            # Phase 1: Removal
            for v in list(current_S):
                if try_remove(v):
                    improved = True
                    # After removal, we might be able to remove more, so restart loop?
                    # Or just continue? Restarting is safer for finding all redundant vertices.
                    break 
            
            if improved:
                continue

            # Phase 2: Swap
            # Try to swap v in S with u not in S
            # This is expensive. We limit this phase.
            # Only do swaps if we are stuck and have time.
            if time.time() - start_time > time_limit * 0.5:
                continue
                
            # Pick random v in S
            if not current_S:
                break
            v = random.choice(list(current_S))
            
            # Try a few random u
            candidates_u = [u for u in range(n) if u not in current_S]
            if not candidates_u:
                break
            
            # Sort candidates by potential benefit? 
            # Benefit: u covers cliques that v covers.
            # Let's just try a few random ones.
            random.shuffle(candidates_u)
            for u in candidates_u[:10]: # Limit attempts
                # Check if swap is valid
                # Temporarily remove v
                # Check if any clique covered ONLY by v is now covered by u?
                # Actually, swap is valid if (S - {v} + {u}) is a valid hitting set.
                # This means for every clique c:
                # If v in c, then we need (c \ {v}) intersect (S \ {v}) != empty OR u in c.
                # i.e., if counts[c] == 1 and v in c, then u MUST be in c.
                
                valid = True
                for i in vertex_in_cliques[v]:
                    if counts[i] == 1:
                        # This clique is only hit by v in S.
                        # So u MUST be in this clique.
                        if u not in cliques[i]:
                            valid = False
                            break
                
                if valid:
                    # Perform swap
                    # Remove v
                    for i in vertex_in_cliques[v]:
                        counts[i] -= 1
                    current_S.remove(v)
                    
                    # Add u
                    for i in vertex_in_cliques[u]:
                        counts[i] += 1
                    current_S.add(u)
                    
                    # Accept swap. It doesn't change size, but might enable removals.
                    improved = True
                    break
        
        return current_S

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        edges = random_graph(n, p, seed)
        cliques = find_maximal_cliques(n, edges)
        
        if not cliques:
            results.append([])
            continue
        
        initial_S = greedy_hitting_set(cliques, n)
        final_S = local_search(initial_S, cliques, n, time_limit=10.0)
        results.append(sorted(final_S))
        
    return results

# Ensure the function is named construct
# The code above defines construct.
