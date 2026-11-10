import time
import random
from itertools import combinations

def construct():
    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
                    adj[u].add(v)
                    adj[v].add(u)
        return edges, adj

    def bron_kerbosch(adj, R, P, X):
        if not P and not X:
            if len(R) >= 2:
                yield tuple(sorted(R))
            return
        if P or X:
            u = max(P | X, key=lambda v: len(P & adj[v]))
            candidates = P - adj[u]
        else:
            candidates = set()
        for v in candidates:
            yield from bron_kerbosch(adj, R | {v}, P & adj[v], X & adj[v])
            P.remove(v)
            X.add(v)

    def get_maximal_cliques(adj):
        n = len(adj)
        return set(bron_kerbosch(adj, set(), set(range(n)), set()))

    def greedy_set_cover(cliques, n, rng, ordering_seed=0):
        """Greedy set cover with varied vertex ordering for diversity."""
        uncovered = list(cliques)
        selected = set()
        
        # Precompute which vertices appear in which cliques for faster lookup
        # Use a dict: vertex -> set of clique indices
        vertex_to_cliques = {}
        for i, c in enumerate(cliques):
            for v in c:
                if v not in vertex_to_cliques:
                    vertex_to_cliques[v] = set()
                vertex_to_cliques[v].add(i)
        
        # Track which cliques are still uncovered
        uncovered_mask = set(range(len(cliques)))
        
        # Create a list of vertices to consider, shuffled based on ordering_seed
        all_vertices = list(range(n))
        order_rng = random.Random(ordering_seed)
        order_rng.shuffle(all_vertices)
        
        while uncovered_mask:
            best_v = -1
            best_count = -1
            ties = []
            
            for v in all_vertices:
                if v not in vertex_to_cliques:
                    continue
                # Count how many uncovered cliques contain v
                count = len(uncovered_mask & vertex_to_cliques[v])
                if count == 0:
                    continue
                if count > best_count:
                    best_count = count
                    best_v = v
                    ties = [v]
                elif count == best_count:
                    ties.append(v)
            
            if best_count == 0:
                break
                
            if len(ties) > 1:
                best_v = rng.choice(ties)
            
            selected.add(best_v)
            
            # Remove covered cliques
            covered = vertex_to_cliques.get(best_v, set())
            uncovered_mask -= covered
            
        return selected

    def local_search(selected, cliques, n, rng, max_iters=100):
        """Try to improve solution via single-vertex swaps and removals."""
        selected = set(selected)
        
        # Precompute for efficiency
        vertex_to_cliques = {}
        for i, c in enumerate(cliques):
            for v in c:
                if v not in vertex_to_cliques:
                    vertex_to_cliques[v] = set()
                vertex_to_cliques[v].add(i)
        
        def is_valid(sol):
            """Check if sol covers all cliques."""
            for c in cliques:
                if not any(v in sol for v in c):
                    return False
            return True
        
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            
            # Try removals
            for v in list(selected):
                candidate = selected - {v}
                if is_valid(candidate):
                    selected = candidate
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swaps: remove one, add one
            for v_out in list(selected):
                for v_in in range(n):
                    if v_in in selected:
                        continue
                    candidate = (selected - {v_out}) | {v_in}
                    if len(candidate) == len(selected):  # Same size, no improvement
                        continue
                    if is_valid(candidate):
                        selected = candidate
                        improved = True
                        break
                if improved:
                    break
        
        return selected

    def polish(selected, cliques, n):
        """Final polish: try all possible single removals in multiple passes."""
        selected = set(selected)
        
        changed = True
        while changed:
            changed = False
            for v in list(selected):
                candidate = selected - {v}
                valid = True
                for c in cliques:
                    if not any(x in candidate for x in c):
                        valid = False
                        break
                if valid:
                    selected = candidate
                    changed = True
        
        return selected

    out = []
    start_time = time.time()
    total_deadline = 110.0  # Leave 10s margin before 120s kill
    
    for idx, (n, p, seed) in enumerate(((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203))):
        # Allocate time proportional to remaining budget
        remaining = total_deadline - (time.time() - start_time)
        if remaining <= 0:
            out.append(sorted(range(n)))  # Fallback
            continue
        
        # Allocate ~1/3 of remaining for each instance, with minimum 10s
        inst_budget = max(10.0, remaining / (3 - idx))
        inst_start = time.time()
        
        edges, adj = random_graph(n, p, seed)
        cliques = get_maximal_cliques(adj)
        
        if not cliques:
            out.append([])
            continue
        
        cliques_list = list(cliques)
        
        best_set = None
        best_size = n + 1
        
        # Phase 1: Multiple greedy runs with different orderings
        num_greedy = 0
        while time.time() - inst_start < inst_budget * 0.6:
            iter_seed = 1000 + num_greedy
            rng_iter = random.Random(iter_seed)
            order_seed = 5000 + num_greedy
            
            candidate = greedy_set_cover(cliques_list, n, rng_iter, ordering_seed=order_seed)
            candidate = polish(candidate, cliques_list, n)
            
            if len(candidate) < best_size:
                best_size = len(candidate)
                best_set = candidate
                
            num_greedy += 1
        
        # Phase 2: Local search on best solution
        if best_set is not None and time.time() - inst_start < inst_budget * 0.9:
            ls_rng = random.Random(99999)
            improved = local_search(best_set, cliques_list, n, ls_rng, max_iters=50)
            improved = polish(improved, cliques_list, n)
            if len(improved) < best_size:
                best_size = len(improved)
                best_set = improved
        
        # Phase 3: More greedy runs if time remains
        while time.time() - inst_start < inst_budget * 0.95:
            iter_seed = 20000 + num_greedy
            rng_iter = random.Random(iter_seed)
            order_seed = 15000 + num_greedy
            
            candidate = greedy_set_cover(cliques_list, n, rng_iter, ordering_seed=order_seed)
            candidate = polish(candidate, cliques_list, n)
            
            if len(candidate) < best_size:
                best_size = len(candidate)
                best_set = candidate
                # Try local search on new best
                if time.time() - inst_start < inst_budget * 0.95:
                    ls_rng2 = random.Random(88888 + num_greedy)
                    improved = local_search(best_set, cliques_list, n, ls_rng2, max_iters=30)
                    improved = polish(improved, cliques_list, n)
                    if len(improved) < best_size:
                        best_size = len(improved)
                        best_set = improved
                        
            num_greedy += 1
        
        if best_set is None:
            best_set = set(range(n))
        
        out.append(sorted(best_set))
    
    return out
