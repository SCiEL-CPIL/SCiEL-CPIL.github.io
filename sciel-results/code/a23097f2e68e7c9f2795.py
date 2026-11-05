import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin from 150s

    def get_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def find_maximal_cliques(n, edges):
        # Build adjacency sets
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        cliques = []
        
        def bronk(R, P, X):
            if time.time() > DEADLINE:
                return
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            
            # Pivot selection
            if P:
                # Choose pivot u in P U X such that |P & N(u)| is maximized
                # For simplicity and speed on small graphs, just pick max degree in P
                pivot = max(P, key=lambda v: len(P & adj[v]))
                candidates = P - adj[pivot]
            else:
                candidates = P
            
            for v in list(candidates):
                bronk(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)
        
        bronk(frozenset(), frozenset(range(n)), frozenset())
        return cliques

    def solve_hitting_set(n, cliques):
        if not cliques:
            return set()
        
        # Map cliques to indices
        # cliques is a list of frozensets
        # We need to know for each vertex, which cliques it belongs to
        # And for the B&B, we need to iterate through cliques
        
        m = len(cliques)
        # vertex_clique_mask[v] will be a bitmask of cliques containing v
        vertex_clique_mask = [0] * n
        
        for i, clique in enumerate(cliques):
            mask_bit = 1 << i
            for v in clique:
                vertex_clique_mask[v] |= mask_bit
        
        global_best_size = m # Upper bound: pick one from each clique
        global_best_sol = set()
        
        # Initial greedy to get a good upper bound
        uncovered_mask = (1 << m) - 1
        current_sol = set()
        while uncovered_mask:
            # Find vertex that covers most uncovered cliques
            best_v = -1
            max_covered = -1
            for v in range(n):
                if v in current_sol:
                    continue
                covered = uncovered_mask & vertex_clique_mask[v]
                if covered:
                    # Count bits
                    count = bin(covered).count('1')
                    if count > max_covered:
                        max_covered = count
                        best_v = v
            if best_v == -1:
                break
            current_sol.add(best_v)
            uncovered_mask &= ~vertex_clique_mask[best_v]
        
        if len(current_sol) < global_best_size:
            global_best_size = len(current_sol)
            global_best_sol = set(current_sol)
            
        # B&B
        def dfs(uncovered_mask, current_sol_size, current_sol_list):
            nonlocal global_best_size, global_best_sol
            
            if time.time() > DEADLINE:
                return
                
            if uncovered_mask == 0:
                if current_sol_size < global_best_size:
                    global_best_size = current_sol_size
                    global_best_sol = set(current_sol_list)
                return
            
            # Pruning
            if current_sol_size + 1 >= global_best_size:
                return
            
            # Choose a clique to branch on
            # Pick the first uncovered clique
            lsb = uncovered_mask & -uncovered_mask
            clique_idx = lsb.bit_length() - 1
            clique = cliques[clique_idx]
            
            # Order vertices in clique by how many other uncovered cliques they cover
            # This helps find good solutions earlier
            verts = list(clique)
            verts.sort(key=lambda v: bin(uncovered_mask & vertex_clique_mask[v]).count('1'), reverse=True)
            
            for v in verts:
                # If v doesn't cover any uncovered cliques, skip? 
                # It must cover the chosen clique, so it covers at least 1.
                new_mask = uncovered_mask & ~vertex_clique_mask[v]
                dfs(new_mask, current_sol_size + 1, current_sol_list + [v])
                
                if time.time() > DEADLINE:
                    return

        dfs((1 << m) - 1, 0, [])
        
        return global_best_sol

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > DEADLINE:
            # Fallback: return all vertices (valid but large)
            edges = get_graph(n, p, seed)
            touched = {x for e in edges for x in e}
            results.append(sorted(touched))
            continue
            
        edges = get_graph(n, p, seed)
        cliques = find_maximal_cliques(n, edges)
        sol = solve_hitting_set(n, cliques)
        results.append(sorted(sol))

    return results
