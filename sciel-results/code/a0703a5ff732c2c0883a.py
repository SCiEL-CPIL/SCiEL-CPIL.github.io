"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

sys.setrecursionlimit(10000)

def construct():
    deadline = time.time() + 140  # safety margin before 150s limit
    
    def build_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj
    
    def bron_kerbosch_maximal_cliques(adj):
        n = len(adj)
        maximal_cliques = []
        
        def expand(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    maximal_cliques.append(frozenset(R))
                return
            # Choose pivot for better pruning
            if P:
                pivot = max(P, key=lambda v: len(P & adj[v]) + len(X & adj[v]))
                candidates = P - adj[pivot]
            else:
                candidates = P
            for v in list(candidates):
                expand(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)
        
        expand(set(), set(range(n)), set())
        return maximal_cliques
    
    def max_independent_set_size(clique_adj, n_cliques):
        """Find maximum independent set in the clique-intersection graph.
        Two cliques are adjacent if they share at least one vertex.
        An independent set is a set of pairwise disjoint cliques."""
        # Build complement graph for maximum clique search
        # In the clique-intersection graph, edge = share vertex
        # Independent set in G = Clique in complement of G
        # Complement: edge = do NOT share vertex (disjoint)
        
        # Use Bron-Kerbosch on complement to find max clique
        best_size = [0]
        
        def bk_complement(R, P, X):
            if not P and not X:
                if len(R) > best_size[0]:
                    best_size[0] = len(R)
                return
            if len(R) + len(P) <= best_size[0]:
                return
            # Pivot: choose vertex in P with max neighbors in P (in complement)
            if P:
                # In complement, neighbors of v are those NOT sharing a vertex with v
                pivot = max(P, key=lambda v: sum(1 for u in P if u != v and not clique_adj[v][u]))
                # Candidates: vertices in P not adjacent to pivot in complement
                # i.e., vertices that DO share a vertex with pivot
                candidates = [v for v in P if clique_adj[v][pivot]]
            else:
                candidates = list(P)
            
            for v in candidates:
                new_P = P - {v}
                new_P = {u for u in new_P if not clique_adj[v][u]}  # neighbors in complement
                new_X = X - {v}
                new_X = {u for u in new_X if not clique_adj[v][u]}
                bk_complement(R | {v}, new_P, new_X)
                P.remove(v)
                X.add(v)
        
        bk_complement(set(), set(range(n_cliques)), set())
        return best_size[0]
    
    def solve_set_cover(n, cliques, time_budget=2.0):
        if not cliques:
            return []
        
        num_cliques = len(cliques)
        
        # For each vertex, which cliques does it belong to?
        vertex_to_cliques = [set() for _ in range(n)]
        for ci, clique in enumerate(cliques):
            for v in clique:
                vertex_to_cliques[v].add(ci)
        
        # Build clique-intersection graph for lower bound
        # clique_adj[i][j] = True if clique i and j share at least one vertex
        clique_adj = [[False] * num_cliques for _ in range(num_cliques)]
        for i in range(num_cliques):
            for j in range(i + 1, num_cliques):
                if cliques[i] & cliques[j]:
                    clique_adj[i][j] = True
                    clique_adj[j][i] = True
        
        # Compute packing lower bound
        lb = max_independent_set_size(clique_adj, num_cliques)
        
        # Greedy set cover with multiple strategies
        def greedy_cover(order_mode='max'):
            uncovered = set(range(num_cliques))
            selected = set()
            
            while uncovered:
                best_v = -1
                best_score = -1
                for v in range(n):
                    if v in selected:
                        continue
                    count = len(vertex_to_cliques[v] & uncovered)
                    if count == 0:
                        continue
                    if order_mode == 'max':
                        score = count
                    elif order_mode == 'ratio':
                        # Prefer vertices that cover many uncovered cliques relative to clique sizes
                        avg_clique_size = sum(len(cliques[ci]) for ci in vertex_to_cliques[v] & uncovered) / max(1, count)
                        score = count / max(1, avg_clique_size)
                    else:
                        score = count
                    if score > best_score:
                        best_score = score
                        best_v = v
                if best_v == -1 or best_score == 0:
                    break
                selected.add(best_v)
                uncovered -= vertex_to_cliques[best_v]
            
            return selected
        
        # Try multiple greedy strategies and keep best
        candidates = []
        for mode in ['max', 'ratio']:
            candidates.append(greedy_cover(mode))
        
        # Also try greedy with random tie-breaking
        for _ in range(5):
            uncovered = set(range(num_cliques))
            selected = set()
            while uncovered:
                best_count = -1
                best_vertices = []
                for v in range(n):
                    if v in selected:
                        continue
                    count = len(vertex_to_cliques[v] & uncovered)
                    if count > best_count:
                        best_count = count
                        best_vertices = [v]
                    elif count == best_count and count > 0:
                        best_vertices.append(v)
                if best_count == 0:
                    break
                best_v = random.choice(best_vertices)
                selected.add(best_v)
                uncovered -= vertex_to_cliques[best_v]
            candidates.append(selected)
        
        # Select best candidate
        best_selected = min(candidates, key=lambda s: len(s))
        
        # Local search: remove redundant vertices
        improved = True
        while improved:
            improved = False
            for v in list(best_selected):
                temp_selected = best_selected - {v}
                all_covered = True
                for ci, clique in enumerate(cliques):
                    if not (clique & temp_selected):
                        all_covered = False
                        break
                if all_covered:
                    best_selected = temp_selected
                    improved = True
        
        # Local search: 1-for-1 swaps
        for _ in range(5):
            improved = False
            for v_out in list(best_selected):
                for v_in in range(n):
                    if v_in in best_selected:
                        continue
                    new_selected = (best_selected - {v_out}) | {v_in}
                    # Check if all cliques are covered
                    all_covered = True
                    for ci, clique in enumerate(cliques):
                        if not (clique & new_selected):
                            all_covered = False
                            break
                    if all_covered:
                        best_selected = new_selected
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        
        # Local search: 2-for-1 exchanges (remove two, add one)
        for _ in range(3):
            improved = False
            v_list = list(best_selected)
            for i in range(len(v_list)):
                for j in range(i + 1, len(v_list)):
                    v_out1, v_out2 = v_list[i], v_list[j]
                    for v_in in range(n):
                        if v_in in best_selected:
                            continue
                        new_selected = (best_selected - {v_out1, v_out2}) | {v_in}
                        if len(new_selected) >= len(best_selected) - 1:
                            continue
                        all_covered = True
                        for ci, clique in enumerate(cliques):
                            if not (clique & new_selected):
                                all_covered = False
                                break
                        if all_covered:
                            best_selected = new_selected
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        
        # Branch and bound refinement if we have time and are above lower bound
        if len(best_selected) > lb and time.time() < deadline - 5:
            # Try to improve via branch and bound
            best = best_selected
            best_size = len(best)
            
            def branch_and_bound(uncovered_cliques, current_selected, depth, max_depth=3):
                nonlocal best, best_size
                if time.time() > deadline - 3:
                    return
                if not uncovered_cliques:
                    if len(current_selected) < best_size:
                        best = set(current_selected)
                        best_size = len(best)
                    return
                if depth >= max_depth:
                    return
                
                # Lower bound: need at least lb more vertices
                # Simple bound: each vertex covers at most max_cover cliques
                max_cover = max(len(vertex_to_cliques[v] & uncovered_cliques) for v in range(n) if v not in current_selected) if uncovered_cliques else 0
                if max_cover == 0:
                    return
                min_needed = (len(uncovered_cliques) + max_cover - 1) // max_cover
                if len(current_selected) + min_needed >= best_size:
                    return
                
                # Choose vertex to branch on: pick vertex that covers most uncovered
                best_v = -1
                best_count = -1
                for v in range(n):
                    if v in current_selected:
                        continue
                    count = len(vertex_to_cliques[v] & uncovered_cliques)
                    if count > best_count:
                        best_count = count
                        best_v = v
                
                if best_v == -1 or best_count == 0:
                    return
                
                # Branch 1: include best_v
                new_selected = current_selected | {best_v}
                new_uncovered = uncovered_cliques - vertex_to_cliques[best_v]
                if len(new_selected) < best_size:
                    branch_and_bound(new_uncovered, new_selected, depth + 1, max_depth)
                
                # Branch 2: exclude best_v (try next best)
                if best_count == 1:
                    # If best_v only covers one uncovered clique, we must include it
                    # (or that clique is uncovered forever)
                    return
                
                # Exclude best_v: remove its coverage from consideration
                # This is tricky; for now just skip
                pass
            
            branch_and_bound(set(range(num_cliques)), set(), 0, max_depth=4)
            best_selected = best
        
        # Final polish: try to remove any vertex
        improved = True
        while improved:
            improved = False
            for v in list(best_selected):
                temp_selected = best_selected - {v}
                all_covered = True
                for ci, clique in enumerate(cliques):
                    if not (clique & temp_selected):
                        all_covered = False
                        break
                if all_covered:
                    best_selected = temp_selected
                    improved = True
        
        return sorted(best_selected)
    
    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline:
            out.append(list(range(n)))
            continue
        adj = build_graph(n, p, seed)
        cliques = bron_kerbosch_maximal_cliques(adj)
        cliques = [c for c in cliques if len(c) >= 2]
        solution = solve_set_cover(n, cliques)
        out.append(solution)
    
    return out
# EVOLVE-BLOCK-END
