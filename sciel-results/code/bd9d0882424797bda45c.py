import time
import random

def construct():
    start_time = time.time()
    TOTAL_BUDGET = 145  # safety margin from 150s

    def get_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def iter_bits(x):
        while x:
            lsb = x & -x
            yield lsb.bit_length() - 1
            x ^= lsb

    def solve_instance(n, p, seed, time_budget):
        adj_orig = get_graph(n, p, seed)

        if not any(adj_orig):
            return []

        def edge_clique_potential(u, v, adj):
            """Compute potential for edge (u,v) based on clique density."""
            common = (adj[u] & adj[v]).bit_count()
            deg_u = adj[u].bit_count()
            deg_v = adj[v].bit_count()
            if deg_u == 0 or deg_v == 0:
                return 0
            # Higher potential for edges that are in denser regions
            return common / (deg_u + deg_v - common + 1e-9)

        def find_best_edge(adj, n, rng=None):
            """Find the edge with highest clique potential."""
            best_u, best_v = -1, -1
            best_potential = -1
            
            for u in range(n):
                if adj[u] == 0:
                    continue
                for v in iter_bits(adj[u]):
                    if v > u:
                        pot = edge_clique_potential(u, v, adj)
                        if pot > best_potential:
                            best_potential = pot
                            best_u, best_v = u, v
            
            return best_u, best_v, best_potential

        def expand_clique(seed_u, seed_v, adj, rng=None):
            """Expand a clique starting from edge (seed_u, seed_v)."""
            C = [seed_u, seed_v]
            U = adj[seed_u] & adj[seed_v]
            
            while U:
                # Find vertex in U with maximum connections within U
                best_u = -1
                best_score = -1
                best_score2 = -1
                
                candidates = list(iter_bits(U))
                if rng is not None and len(candidates) > 1:
                    rng.shuffle(candidates)
                
                for u in candidates:
                    score = (adj[u] & U).bit_count()
                    score2 = adj[u].bit_count()
                    if score > best_score or (score == best_score and score2 > best_score2):
                        best_score = score
                        best_score2 = score2
                        best_u = u
                
                if best_u == -1:
                    break
                    
                C.append(best_u)
                U &= adj[best_u]
                
            return C

        def remove_clique_edges(C, adj):
            """Remove all edges within clique C from adj."""
            for i in range(len(C)):
                for j in range(i + 1, len(C)):
                    u, v = C[i], C[j]
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)

        def is_clique_union(C1, C2, adj):
            """Check if union of C1 and C2 forms a clique."""
            for u in C1:
                for v in C2:
                    if not (adj[u] & (1 << v)):
                        return False
            return True

        def try_merge(cliques, adj, n, deadline):
            """Try to merge cliques that form a larger clique together."""
            changed = True
            while changed and (time.time() < deadline):
                changed = False
                new_cliques = []
                i = 0
                while i < len(cliques):
                    C_i = cliques[i]
                    merged = False
                    for j in range(i + 1, len(cliques)):
                        C_j = cliques[j]
                        if is_clique_union(C_i, C_j, adj):
                            new_cliques = [C_i + C_j] + [cliques[k] for k in range(i + 1, len(cliques)) if k != j]
                            i = 0
                            changed = True
                            merged = True
                            break
                    if not merged:
                        new_cliques.append(C_i)
                        i += 1
                cliques = new_cliques
            return cliques

        def greedy_partition(adj_mask_orig, n, rng=None, deadline=None):
            """Greedy partition using clique potential heuristic."""
            if deadline is None:
                deadline = time.time() + 10
            adj = list(adj_mask_orig)
            cliques = []
            
            while True:
                if time.time() > deadline:
                    break
                u, v, potential = find_best_edge(adj, n, rng)
                if u == -1:
                    break
                C = expand_clique(u, v, adj, rng)
                cliques.append(C)
                remove_clique_edges(C, adj)
            
            cliques = try_merge(cliques, adj, n, deadline)
            return cliques

        def validate(cliques, adj_mask_orig, n):
            """Validate that cliques form a proper edge partition."""
            for C in cliques:
                if len(C) < 2:
                    return False
                for i in range(len(C)):
                    for j in range(i + 1, len(C)):
                        u, v = C[i], C[j]
                        if not (adj_mask_orig[u] & (1 << v)):
                            return False
            edge_set = set()
            for C in cliques:
                for i in range(len(C)):
                    for j in range(i + 1, len(C)):
                        u, v = C[i], C[j]
                        key = (min(u, v), max(u, v))
                        if key in edge_set:
                            return False
                        edge_set.add(key)
            for u in range(n):
                for v in iter_bits(adj_mask_orig[u]):
                    if v > u:
                        if (u, v) not in edge_set:
                            return False
            return True

        def local_search(cliques, adj_mask_orig, n, rng, deadline):
            """Try to improve partition by merging compatible cliques."""
            if len(cliques) < 2:
                return cliques
            
            best = cliques[:]
            best_count = len(best)
            
            for _ in range(30):
                if time.time() > deadline:
                    break
                
                # Try merging pairs of cliques
                improved = False
                for i in range(len(best)):
                    for j in range(i + 1, len(best)):
                        if is_clique_union(best[i], best[j], adj_mask_orig):
                            new_cliques = [best[i] + best[j]] + [best[k] for k in range(len(best)) if k != i and k != j]
                            if validate(new_cliques, adj_mask_orig, n):
                                if len(new_cliques) < best_count:
                                    best = new_cliques
                                    best_count = len(best)
                                    improved = True
                            break
                    if improved:
                        break
                if not improved:
                    break
            
            return best

        best_cliques = None
        best_count = float('inf')
        
        # Deterministic run first
        det_deadline = start_time + time_budget * 0.25
        det_cliques = greedy_partition(adj_orig, n, rng=None, deadline=det_deadline)
        if validate(det_cliques, adj_orig, n):
            best_cliques = det_cliques
            best_count = len(det_cliques)
        
        # Randomized restarts with different heuristics
        remaining = time_budget - (time.time() - start_time)
        if remaining > 3:
            num_restarts = max(3, int(remaining / 0.8))
            rng_master = random.Random(seed + 1000)
            
            for r in range(num_restarts):
                if time.time() > start_time + time_budget - 1.5:
                    break
                rng = random.Random(rng_master.randint(0, 2**31))
                restart_deadline = min(start_time + time_budget - 1.5, time.time() + max(0.3, remaining / (num_restarts - r)))
                try:
                    cliques = greedy_partition(adj_orig, n, rng=rng, deadline=restart_deadline)
                    if validate(cliques, adj_orig, n):
                        # Try local search to improve
                        ls_deadline = min(start_time + time_budget - 1.5, time.time() + 0.5)
                        improved = local_search(cliques, adj_orig, n, rng, ls_deadline)
                        if validate(improved, adj_orig, n):
                            if len(improved) < best_count:
                                best_count = len(improved)
                                best_cliques = improved
                        else:
                            if len(cliques) < best_count:
                                best_count = len(cliques)
                                best_cliques = cliques
                except:
                    continue
        
        if best_cliques is None:
            # Fallback: one clique per edge
            best_cliques = []
            for u in range(n):
                for v in iter_bits(adj_orig[u]):
                    if v > u:
                        best_cliques.append([u, v])
        
        return best_cliques

    results = []
    # Instance 1
    t1 = min(50, TOTAL_BUDGET - 60)
    results.append(solve_instance(32, 0.6, 601, t1))
    
    # Instance 2
    elapsed = time.time() - start_time
    t2 = max(15, TOTAL_BUDGET - elapsed - 2)
    results.append(solve_instance(64, 0.55, 602, t2))
    
    return results
