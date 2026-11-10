import time
import random

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
        return list(bron_kerbosch(adj, set(), set(range(n)), set()))

    def solve_instance(n, p, seed, time_budget):
        edges, adj = random_graph(n, p, seed)
        cliques = get_maximal_cliques(adj)
        
        if not cliques:
            return []

        cliques_list = cliques
        m = len(cliques_list)
        
        # Precompute vertex to cliques mapping
        # C[v] is a set of clique indices
        C = [set() for _ in range(n)]
        for i, c in enumerate(cliques_list):
            for v in c:
                C[v].add(i)
                
        # Precompute clique to vertices mapping for greedy init
        # Actually, for greedy, we just need to know which vertices cover which cliques.
        # We can use C for that.
        
        def greedy_init(rng):
            uncovered = set(range(m))
            selected = set()
            
            while uncovered:
                best_v = -1
                best_count = -1
                # Iterate over vertices that appear in any uncovered clique
                # To be fast, we can just iterate all vertices and count
                # n is small (<=40), m is small.
                for v in range(n):
                    if not C[v]:
                        continue
                    count = len(uncovered & C[v])
                    if count > best_count:
                        best_count = count
                        best_v = v
                    elif count == best_count and count > 0:
                        # Random tie break
                        if rng.random() < 0.5:
                            best_v = v
                
                if best_v == -1:
                    break # Should not happen if uncovered is not empty
                    
                selected.add(best_v)
                uncovered -= C[best_v]
                
            return selected

        def local_search(S, rng, time_limit):
            S = set(S)
            # Initialize Cover_Count
            cover_count = [0] * m
            for v in S:
                for c in C[v]:
                    cover_count[c] += 1
            
            improved = True
            start = time.time()
            
            while improved and (time.time() - start) < time_limit:
                improved = False
                
                # 1. Try Removals
                # We need to check if removing v leaves all cliques covered.
                # This is equivalent to: for all c in C[v], cover_count[c] > 1.
                for v in list(S):
                    can_remove = True
                    for c in C[v]:
                        if cover_count[c] == 1:
                            can_remove = False
                            break
                    if can_remove:
                        # Remove v
                        for c in C[v]:
                            cover_count[c] -= 1
                        S.remove(v)
                        improved = True
                        break # Restart loop after a change to keep cover_count consistent
                
                if improved:
                    continue
                    
                # 2. Try Swaps
                # For each v_out in S, find v_in not in S such that swap is valid.
                # Valid if C[v_in] covers all c in C[v_out] where cover_count[c] == 1.
                
                # Precompute "critical" cliques for each v in S?
                # Critical[v] = { c in C[v] | cover_count[c] == 1 }
                
                # To speed up, we can just iterate.
                # n is small.
                
                found_swap = False
                for v_out in list(S):
                    critical = [c for c in C[v_out] if cover_count[c] == 1]
                    if not critical:
                        # This vertex is redundant, should have been removed in step 1
                        # But maybe cover_count changed? No, we break on improvement.
                        # So if we are here, no vertex is redundant.
                        continue
                    
                    # Find a v_in that covers all critical cliques
                    # We can pick the v_in that covers the most cliques overall?
                    # Or just the first one found?
                    # Let's try to pick a "good" v_in.
                    # Score for v_in: number of cliques it covers.
                    # Or number of critical cliques it covers (must be all).
                    
                    best_v_in = -1
                    best_score = -1
                    
                    for v_in in range(n):
                        if v_in in S:
                            continue
                        # Check if v_in covers all critical
                        valid = True
                        for c in critical:
                            if c not in C[v_in]:
                                valid = False
                                break
                        if valid:
                            # Score: prefer v_in that covers more total cliques
                            # This might help in future removals
                            score = len(C[v_in])
                            if score > best_score:
                                best_score = score
                                best_v_in = v_in
                                if score == m: # Covers everything, perfect
                                    break
                    
                    if best_v_in != -1:
                        # Perform swap
                        # Update cover_count for v_out
                        for c in C[v_out]:
                            cover_count[c] -= 1
                        # Update cover_count for v_in
                        for c in C[best_v_in]:
                            cover_count[c] += 1
                        
                        S.remove(v_out)
                        S.add(best_v_in)
                        found_swap = True
                        break # Restart loop after a change
                
                if found_swap:
                    improved = True

            return S

        best_set = set(range(n)) # Fallback
        start_time = time.time()
        
        # Run multiple iterations
        iter_count = 0
        while (time.time() - start_time) < time_budget:
            iter_seed = 1000 + iter_count
            rng = random.Random(iter_seed)
            
            # Greedy Init
            candidate = greedy_init(rng)
            
            # Local Search
            # Allocate some time for local search, e.g., 0.1s per iteration
            # Or just run until time budget for this iteration is up?
            # Better: Run local search with a small time limit, e.g., 0.05s
            # This allows many restarts.
            ls_time = 0.05
            candidate = local_search(candidate, rng, ls_time)
            
            if len(candidate) < len(best_set):
                best_set = candidate
                # If we find a very small set, we might want to spend more time polishing it
                # But for now, keep it simple.
                
            iter_count += 1
            
        return sorted(best_set)

    out = []
    start_time = time.time()
    total_deadline = 115.0 # Leave 5s margin
    
    instances = ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203))
    
    for idx, (n, p, seed) in enumerate(instances):
        remaining = total_deadline - (time.time() - start_time)
        if remaining <= 0:
            out.append(sorted(range(n)))
            continue
        
        # Allocate time proportionally
        # Simple: divide remaining by instances left
        inst_budget = remaining / (len(instances) - idx)
        
        res = solve_instance(n, p, seed, inst_budget)
        out.append(res)
        
    return out
