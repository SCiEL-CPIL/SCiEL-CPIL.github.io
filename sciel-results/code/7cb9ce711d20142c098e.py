import time
import random
from collections import defaultdict

def construct():
    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve(n, k, p, seed, deadline):
        # 1. Generate Graph
        rng = random.Random(seed)
        adj_core = [set() for _ in range(k)]
        adj_cross = [set() for _ in range(n)] # adj_cross[i] contains core neighbors for i >= k
        
        # Core is a clique
        for u in range(k):
            for v in range(u+1, k):
                adj_core[u].add(v)
                adj_core[v].add(u)
        
        # Cross edges
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    adj_cross[v].add(u)
        
        # Total edges to cover
        all_edges = set()
        for u in range(k):
            for v in range(u+1, k):
                all_edges.add((u, v))
        
        for v in range(k, n):
            for u in adj_cross[v]:
                all_edges.add((min(u, v), max(u, v)))
                
        # 2. Generate Candidate Cliques
        candidates = []
        
        # A. Maximal Cliques in Core (K-part)
        # Since K is small (10, 20, 30), we can find maximal cliques efficiently.
        # For K=30, number of maximal cliques can be large, but we limit it.
        # Greedy maximal clique extraction from remaining edges? 
        # Actually, for set cover, we want diverse candidates.
        # Let's add the full core clique.
        candidates.append(list(range(k)))
        
        # Let's add some random maximal cliques in the core? 
        # Or just rely on the fact that the full core clique covers all core edges.
        # But maybe a subset is better if it allows combining with I-vertices?
        # No, a core clique {u, v, w} covers (u,v), (u,w), (v,w).
        # An I-vertex clique {i, u, v} covers (i,u), (i,v), (u,v).
        # So (u,v) can be covered by either.
        
        # B. Cliques involving I-vertices
        # For each i in I, find large cliques in N(i) within Core.
        # A clique {i} U S where S is a clique in Core and S subset N(i).
        # To maximize coverage, we want S to be as large as possible.
        # Let's generate a few candidates for each i.
        # 1. The maximal clique in N(i) within Core.
        # 2. Some random large cliques in N(i).
        
        for i in range(k, n):
            N_i = list(adj_cross[i])
            if len(N_i) < 2:
                continue
                
            # Find a maximal clique in the subgraph induced by N_i in Core.
            # Since Core is a clique, any subset of N_i is a clique in Core!
            # Wait, Core is a clique. So ANY subset of N_i is a clique in the whole graph.
            # So the largest clique containing i is {i} U N_i.
            # This covers all cross edges (i, u) for u in N_i, AND all core edges (u, v) for u,v in N_i.
            
            # So the best candidate for i is {i} U N_i.
            # Is it always optimal to take the full N_i?
            # Taking full N_i covers |N_i| cross edges and |N_i|*(|N_i|-1)/2 core edges.
            # If we take a subset S, we cover fewer.
            # However, maybe we don't want to cover those core edges with i, because we could cover them with another i' or a core-only clique?
            # But covering core edges is "free" in the sense that we have to cover them anyway.
            # The constraint is "every edge lies in EXACTLY one clique".
            # So if we use {i} U N_i, we consume those core edges.
            # If those core edges were needed for a different clique, we can't use them again.
            # But since core edges are abundant and we have many I-vertices, it's likely better to use the large cross cliques to "eat" as many core edges as possible, reducing the need for core-only cliques.
            
            # So, candidate: [i] + N_i
            candidates.append([i] + N_i)
            
            # Also, maybe smaller subsets are useful if N_i is very large and we want to leave some core edges for others?
            # Unlikely to be better than the full set in a greedy set cover, as the full set has the highest "value" (edges covered).
            # Greedy set cover picks the set with max uncovered edges.
            # {i} U N_i will have a huge number of edges if N_i is large.
            
            # What if we split N_i?
            # E.g., {i, u1, u2} and {i, u3, u4}?
            # This covers 2 cross + 1 core vs 2 cross + 1 core.
            # But {i, u1, u2, u3, u4} covers 4 cross + 6 core.
            # Clearly the larger clique is better for coverage per clique.
            # The only reason to split is if we need to save core edges for something else?
            # No, core edges are "waste" if not covered by cross cliques. We want to maximize core edge coverage by cross cliques.
            
        # C. Small cliques for fallback
        # We might need 2-cliques if some edges are left over.
        # We'll add them dynamically if needed, or just rely on the greedy to pick remaining edges if we generate all 2-cliques?
        # Generating all 2-cliques is O(E^2) which is too much.
        # Instead, if greedy set cover leaves edges, we handle them in a post-processing step.
        
        # 3. Greedy Set Cover
        # State: set of uncovered edges
        uncovered = set(all_edges)
        selected_cliques = []
        
        # Pre-compute edges for each candidate for fast lookup?
        # Candidates can be large.
        # Let's just iterate.
        
        # Optimization: Sort candidates by potential size?
        # Or just iterate and pick best.
        
        # To speed up, we can maintain a list of candidates and their "current value" (uncovered edges count).
        # But updating value for all candidates when we remove edges is expensive.
        # Alternative: Just re-scan candidates each time?
        # Number of candidates ~ n + 1.
        # Number of iterations ~ number of cliques ~ O(n^2).
        # Total work ~ O(n^3). For n=90, 90^3 = 729,000. Very fast.
        
        # However, calculating "uncovered edges count" for a candidate requires checking each edge in the candidate.
        # Edges in candidate {i} U S:
        # Cross: (i, u) for u in S
        # Core: (u, v) for u, v in S
        
        # Let's implement a simple loop.
        
        start_time = time.time()
        
        while uncovered:
            if time.time() > deadline:
                break
                
            best_cand = None
            best_count = 0
            best_edges = set()
            
            for cand in candidates:
                # Calculate edges in cand that are in uncovered
                # cand is a list of vertices
                # Generate all pairs
                # This is O(|cand|^2)
                # Sum of |cand|^2 over all candidates?
                # For I-vertices, |cand| ~ p*k. Sum ~ n * (pk)^2.
                # For n=90, k=30, p=0.7, |cand|~21. 90 * 441 ~ 40k.
                # This is fine.
                
                cand_set = set(cand)
                count = 0
                edges_in_cand = set()
                
                # Cross edges (if i >= k)
                if cand[0] >= k:
                    i = cand[0]
                    for u in cand[1:]:
                        e = (min(i, u), max(i, u))
                        if e in uncovered:
                            count += 1
                            edges_in_cand.add(e)
                
                # Core edges
                # Only if cand has at least 2 core vertices
                core_vertices = [v for v in cand if v < k]
                for idx in range(len(core_vertices)):
                    u = core_vertices[idx]
                    for jdx in range(idx + 1, len(core_vertices)):
                        v = core_vertices[jdx]
                        e = (min(u, v), max(u, v))
                        if e in uncovered:
                            count += 1
                            edges_in_cand.add(e)
                
                if count > best_count:
                    best_count = count
                    best_cand = cand
                    best_edges = edges_in_cand
                    if count == 0:
                         # Optimization: if first candidate has 0, likely all do? No.
                         pass
            
            if best_cand is None or best_count == 0:
                # No candidate covers any uncovered edge.
                # This means we have "orphan" edges that are not in any candidate?
                # This can happen if we didn't generate 2-cliques.
                # Fallback: cover remaining edges with 2-cliques.
                for e in uncovered:
                    selected_cliques.append([e[0], e[1]])
                uncovered.clear()
                break
            
            selected_cliques.append(best_cand)
            uncovered.difference_update(best_edges)
            
        return selected_cliques

    results = []
    start_time = time.time()
    # Budget: 120s total. Leave 20s for safety.
    deadline = start_time + 100
    
    specs = [(30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)]
    
    for n, k, p, seed in specs:
        # Adjust deadline per instance? 
        # Just use a global deadline.
        res = solve(n, k, p, seed, deadline)
        results.append(res)
        
        # Check time
        if time.time() > deadline:
             # If we ran out of time, we might have incomplete results.
             # The task says "Any broken rule makes the whole result invalid".
             # If we break the loop, the last instance might be empty or incomplete.
             # We must ensure we finish.
             # With the O(n^3) estimate, we should be fine.
             pass

    return results

