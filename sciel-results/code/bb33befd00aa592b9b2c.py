"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

    def instance(n, p, seed):
        import random
        rng = random.Random(seed)
        edges = set()
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
                    adj[u].add(v)
                    adj[v].add(u)
        return edges, adj

    def bron_kerbosch(adj, n):
        """Find all maximal cliques of size >= 2 using Bron-Kerbosch with pivoting."""
        maximal_cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    maximal_cliques.append(frozenset(R))
                return
            
            # Choose pivot u from P ∪ X to minimize branching
            pivot_candidates = P | X
            if pivot_candidates:
                best_pivot = None
                best_count = -1
                for u in pivot_candidates:
                    count = len(P & adj[u])
                    if count > best_count:
                        best_count = count
                        best_pivot = u
                
                candidates = P - adj[best_pivot]
            else:
                candidates = P
            
            for v in list(candidates):
                bk(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)
        
        bk(set(), set(range(n)), set())
        return maximal_cliques

    def greedy_transversal(cliques, adj, n):
        """Greedy set cover: repeatedly pick vertex hitting most uncovered cliques."""
        if not cliques:
            return set()
        
        uncovered = list(cliques)
        vertex_clique_count = {}
        for clique in uncovered:
            for v in clique:
                vertex_clique_count[v] = vertex_clique_count.get(v, 0) + 1
        
        solution = set()
        while uncovered:
            best_v = None
            best_hits = -1
            for v, count in vertex_clique_count.items():
                if count > best_hits:
                    best_hits = count
                    best_v = v
            
            if best_v is None:
                break
            
            solution.add(best_v)
            
            new_uncovered = []
            for clique in uncovered:
                if best_v not in clique:
                    new_uncovered.append(clique)
                else:
                    for u in clique:
                        if u != best_v:
                            vertex_clique_count[u] = vertex_clique_count.get(u, 0) - 1
            
            uncovered = new_uncovered
            if best_v in vertex_clique_count:
                del vertex_clique_count[best_v]
        
        return solution

    def local_search_improve(solution, cliques, adj, n, time_limit=5.0):
        """Improve by removing redundant vertices and trying 1-exchange moves."""
        sol = set(solution)
        deadline_ls = time.time() + time_limit
        
        # Precompute for each vertex which cliques it hits
        vertex_cliques = {}
        for v in range(n):
            vertex_cliques[v] = [i for i, c in enumerate(cliques) if v in c]
        
        # Pass 1: remove redundant vertices
        changed = True
        while changed and time.time() < deadline_ls:
            changed = False
            for v in sorted(sol):
                trial = sol - {v}
                valid = True
                for ci in vertex_cliques[v]:
                    clique = cliques[ci]
                    if not (clique & trial):
                        valid = False
                        break
                if valid:
                    sol = trial
                    changed = True
        
        # Pass 2: 1-exchange moves (remove v, add u)
        if time.time() < deadline_ls:
            improved = True
            while improved and time.time() < deadline_ls:
                improved = False
                for v in sorted(sol):
                    if time.time() >= deadline_ls:
                        break
                    for u in range(n):
                        if u in sol or u == v:
                            continue
                        trial = (sol - {v}) | {u}
                        valid = True
                        for ci in vertex_cliques[u]:
                            clique = cliques[ci]
                            if not (clique & trial):
                                valid = False
                                break
                        if valid:
                            # Verify all cliques still hit
                            for ci in vertex_cliques[v]:
                                clique = cliques[ci]
                                if not (clique & trial):
                                    valid = False
                                    break
                        if valid:
                            sol = trial
                            improved = True
                            break
        
        return sol

    def bitmask_bb(cliques, adj, n, current_best, time_limit=10.0):
        """Bitmask branch-and-bound for minimum clique transversal."""
        deadline_bb = time.time() + time_limit
        m = len(cliques)
        if m == 0:
            return set(current_best)
        
        # Build bitmask for each vertex: which cliques it hits
        vertex_mask = {}
        for v in range(n):
            mask = 0
            for i, c in enumerate(cliques):
                if v in c:
                    mask |= (1 << i)
            if mask:
                vertex_mask[v] = mask
        
        # Candidate vertices
        candidates = sorted(vertex_mask.keys())
        
        # Sort cliques by size for better branching
        sorted_cliques = sorted(cliques, key=len)
        # Rebuild masks with sorted clique order
        sorted_clique_set = [frozenset(c) for c in sorted_cliques]
        vertex_mask2 = {}
        for v in candidates:
            mask = 0
            for i, c in enumerate(sorted_clique_set):
                if v in c:
                    mask |= (1 << i)
            if mask:
                vertex_mask2[v] = mask
        candidates = sorted(vertex_mask2.keys())
        
        full_mask = (1 << m) - 1
        best = set(current_best)
        best_size = len(best)
        
        # Lower bound: greedy on remaining
        def lower_bound(uncovered_mask):
            if uncovered_mask == 0:
                return 0
            remaining = uncovered_mask
            count = 0
            while remaining:
                best_hits = 0
                for v in candidates:
                    hits = remaining & vertex_mask2[v]
                    h = bin(hits).count('1')
                    if h > best_hits:
                        best_hits = h
                if best_hits == 0:
                    return float('inf')
                remaining &= ~(best_hits_mask)
                count += 1
            return count
        
        # Better lower bound: pack
        def lb_greedy(uncovered_mask):
            if uncovered_mask == 0:
                return 0
            remaining = uncovered_mask
            count = 0
            while remaining:
                best_v = None
                best_hits = 0
                for v in candidates:
                    hits = remaining & vertex_mask2[v]
                    h = bin(hits).count('1')
                    if h > best_hits:
                        best_hits = h
                        best_v = v
                if best_hits == 0:
                    return float('inf')
                remaining &= ~vertex_mask2[best_v]
                count += 1
            return count
        
        # Branch and bound
        def bb(uncovered_mask, chosen, depth):
            nonlocal best, best_size
            if time.time() > deadline_bb:
                return
            
            if uncovered_mask == 0:
                if len(chosen) < best_size:
                    best = set(chosen)
                    best_size = len(chosen)
                return
            
            # Lower bound
            lb = lb_greedy(uncovered_mask)
            if len(chosen) + lb >= best_size:
                return
            
            # Pick the most constrained clique
            # Find clique with fewest available candidate vertices
            target_idx = -1
            min_options = float('inf')
            for i in range(m):
                if not (uncovered_mask & (1 << i)):
                    continue
                clique = sorted_clique_set[i]
                options = 0
                for v in clique:
                    if v in vertex_mask2:
                        options += 1
                if options < min_options:
                    min_options = options
                    target_idx = i
                if options == 0:
                    return
        
            if target_idx < 0:
                return
            
            clique = sorted_clique_set[target_idx]
            # Branch on vertices in this clique, ordered by coverage
            verts_in_clique = [v for v in clique if v in vertex_mask2]
            verts_in_clique.sort(key=lambda v: bin(uncovered_mask & vertex_mask2[v]).count('1'), reverse=True)
            
            for v in verts_in_clique:
                new_chosen = chosen | {v}
                new_uncovered = uncovered_mask & ~vertex_mask2[v]
                bb(new_uncovered, new_chosen, depth + 1)
                if time.time() > deadline_bb:
                    return
        
        bb(full_mask, set(), 0)
        return best

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline - 25:
            edges, adj = instance(n, p, seed)
            cliques = bron_kerbosch(adj, n)
            sol = greedy_transversal(cliques, adj, n)
            results.append(sorted(sol))
            continue
        
        edges, adj = instance(n, p, seed)
        cliques = bron_kerbosch(adj, n)
        
        if not cliques:
            results.append([])
            continue
        
        # Start with greedy
        sol = greedy_transversal(cliques, adj, n)
        
        # Local search improvement
        remaining_time = deadline - time.time()
        ls_time = min(4.0, remaining_time * 0.25)
        sol = local_search_improve(sol, cliques, adj, n, time_limit=ls_time)
        
        # Bitmask branch and bound if time permits
        remaining_time = deadline - time.time()
        if remaining_time > 8 and len(cliques) <= 2000:
            bb_time = min(remaining_time * 0.6, 20.0)
            improved = bitmask_bb(cliques, adj, n, sol, time_limit=bb_time)
            if len(improved) < len(sol):
                sol = improved
        
        # Final local search polish
        remaining_time = deadline - time.time()
        if remaining_time > 3:
            polish_time = min(2.0, remaining_time * 0.3)
            sol = local_search_improve(sol, cliques, adj, n, time_limit=polish_time)
        
        results.append(sorted(sol))
    
    return results
# EVOLVE-BLOCK-END
