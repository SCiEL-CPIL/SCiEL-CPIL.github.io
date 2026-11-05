# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 145.0 # Safety margin before 150s kill

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def bron_kerbosch_maximal_cliques(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        max_cliques = []

        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    max_cliques.append(frozenset(R))
                return

            if not P:
                return

            # pivot: choose u in P ∪ X maximizing |P ∩ N(u)|
            pivot_candidates = P | X
            best_pivot = None
            max_common = -1
            for u in pivot_candidates:
                common = len(P & adj[u])
                if common > max_common:
                    max_common = common
                    best_pivot = u

            if best_pivot is not None:
                candidates = [v for v in P if v not in adj[best_pivot]]
            else:
                candidates = list(P)

            for v in candidates:
                new_R = R | {v}
                new_P = P & adj[v]
                new_X = X & adj[v]
                bk(new_R, new_P, new_X)
                P.discard(v)
                X.add(v)

        bk(frozenset(), set(range(n)), set())
        return max_cliques

    def solve_mwis_and_repair(n, cliques, time_budget):
        """
        Strategy:
        1. Build Clique Intersection Graph (CIG). Nodes = cliques. Edge if cliques share a vertex.
        2. Find a large Independent Set (IS) in CIG using Greedy + Local Search.
           This IS corresponds to a set of pairwise vertex-disjoint cliques.
        3. The size of this IS is a lower bound on the Hitting Set size.
        4. Construct a Hitting Set by picking one vertex from each clique in the IS.
           If conflicts arise (same vertex picked for two cliques - impossible by definition of IS? 
           Wait: IS means cliques are vertex-disjoint. So we can pick ANY vertex from each.
           However, we must ALSO cover cliques NOT in the IS.
           
           Correction on logic:
           An Independent Set in CIG is a set of cliques that do NOT share vertices.
           Let S be such a set. For each clique C in S, we must pick at least one vertex from C.
           Since they are disjoint, we can pick one vertex from each without conflict between them.
           But we still need to cover all OTHER cliques (those not in S).
           
           Actually, the standard reduction is:
           Min Hitting Set is equivalent to Min Vertex Cover in a hypergraph.
           It is NOT directly equivalent to Max Independent Set in the intersection graph.
           
           However, if we find a large set of disjoint cliques (IS in CIG), say of size K,
           then the hitting set MUST have size at least K (since each clique in the IS needs a unique vertex).
           
           So:
           1. Find Max IS in CIG -> Size K. Lower bound LB = K.
           2. Use this structural info.
           
           Alternative Strategy for this specific prompt:
           The prompt suggests: "Formulate ... as minimum weight independent set in the clique intersection graph ... allowing the use of maximum matching-based approximation algorithms for vertex cover on this derived structure to derive a smaller hitting set".
           
           This interpretation is slightly confused theoretically (Hitting Set on Graph vs Vertex Cover on CIG are different). 
           Vertex Cover on CIG would mean picking Cliques such that every edge (pair of intersecting cliques) is covered. 
           This doesn't directly give a Hitting Set of vertices.
           
           Let's stick to a robust structural approach that uses the CIG for better search:
           
           1. Compute CIG.
           2. Use the CIG to identify "conflict groups".
           3. Solve Hitting Set using a specialized BnB that uses the CIG for pruning and branching.
           
           Actually, let's try the "Disjoint Clique Packing" approach more directly as a heuristic:
           1. Find a maximal set of disjoint cliques (Greedy).
           2. Pick one vertex from each.
           3. Solve the remaining problem (cliques not covered by these picks) with a fast Greedy/Local Search.
           4. Compare with standard Greedy Hitting Set.
           5. Take the best.
           
           Let's refine:
           The "Minimum Weight Independent Set" hint might refer to:
           Min Hitting Set = Min Weight Vertex Cover in the conflict graph of vertices? No.
           
           Let's try a different structural decomposition:
           Decompose the graph into connected components of the Clique Intersection Graph?
           No, the cliques are nodes.
           
           Let's implement:
           1. Greedy Hitting Set (baseline).
           2. "Disjoint Clique Seeding":
              - Find a large set of pairwise disjoint cliques (IS in CIG).
              - For each clique in this set, we MUST pick a vertex.
              - We pick the vertex in each clique that covers the most OTHER (non-IS) cliques.
              - This creates a partial solution.
              - Solve the residual problem (cliques not hit by these seeds) with Greedy.
              - Local Search (Remove/Exchange) to optimize.
           3. Return the best of these approaches.
        """
        
        if not cliques:
            return set()
        
        m = len(cliques)
        clique_vertices = [set(c) for c in cliques]
        
        # Incidence: vertex -> list of clique indices
        vertex_cliques = defaultdict(list)
        for i, c in enumerate(cliques):
            for v in c:
                vertex_cliques[v].append(i)

        # 1. Standard Greedy Hitting Set
        def greedy_hs():
            uncovered = set(range(m))
            selected = set()
            while uncovered:
                best_v = -1
                best_gain = 0
                for v in range(n):
                    gain = sum(1 for i in vertex_cliques[v] if i in uncovered)
                    if gain > best_gain:
                        best_gain = gain
                        best_v = v
                if best_v == -1 or best_gain == 0:
                    break
                selected.add(best_v)
                for i in vertex_cliques[best_v]:
                    uncovered.discard(i)
            return selected

        best_sol = greedy_hs()
        
        # Local Search: Remove redundant vertices
        def local_search_remove(sol):
            sol = set(sol)
            changed = True
            while changed:
                changed = False
                for v in list(sol):
                    test = set(sol)
                    test.discard(v)
                    valid = True
                    for i, c in enumerate(cliques):
                        if not any(u in test for u in c):
                            valid = False
                            break
                    if valid:
                        sol = test
                        changed = True
                        break
            return sol
            
        best_sol = local_search_remove(best_sol)
        best_size = len(best_sol)
        
        # 2. Disjoint Clique Seeding Strategy
        # Build Clique Intersection Graph (CIG)
        # Node i, j connected if clique[i] & clique[j] != empty
        # We want a Max Independent Set in CIG (set of disjoint cliques)
        # Approximation: Greedy picking clique with fewest neighbors in CIG (most isolated)
        
        # Precompute CIG neighbors for efficiency? 
        # m is small (max cliques in n=40 random graph is usually < 100-200).
        
        cig_neighbors = [set() for _ in range(m)]
        for i in range(m):
            for j in range(i+1, m):
                if clique_vertices[i] & clique_vertices[j]:
                    cig_neighbors[i].add(j)
                    cig_neighbors[j].add(i)
        
        # Greedy Max Independent Set on CIG
        # Pick node with min degree in current graph? Or max weight (size)?
        # Let's pick cliques that are "hard" to cover? 
        # Actually, we want disjoint cliques. 
        # Greedy: pick clique with smallest intersection with other UNSELECTED cliques?
        
        def find_disjoint_cliques():
            available = set(range(m))
            selected_cliques = []
            
            # Sort by size descending? Or by "exclusivity" (fewest shared vertices with others)?
            # Let's just do simple greedy: pick clique, remove all intersecting cliques.
            # To maximize count, pick cliques that intersect fewest others?
            
            order = sorted(range(m), key=lambda i: len(cig_neighbors[i]))
            
            for i in order:
                if i not in available:
                    continue
                selected_cliques.append(i)
                # Remove i and all its neighbors from available
                to_remove = {i} | cig_neighbors[i]
                available -= to_remove
                
            return selected_cliques

        disjoint_cliques = find_disjoint_cliques()
        
        if disjoint_cliques:
            # For each disjoint clique, we must pick a vertex.
            # Which vertex? The one that covers the most OTHER cliques (not in disjoint_cliques).
            
            # Let's define the "seed" selection
            seeds = set()
            covered_by_seeds = set()
            
            # Sort disjoint cliques by size? 
            # Process them in order of "least flexibility" (smallest size) first?
            # Or just process all.
            
            # For each clique in disjoint_cliques, find best vertex
            # "Best" = covers most uncovered cliques (outside disjoint set)
            
            # We need to track which cliques are already covered by seeds
            # Initially none.
            
            # We pick a vertex v for clique C.
            # v covers C and all other cliques containing v.
            
            # Let's try to optimize the seed selection.
            # Since they are disjoint, the choice of v for C1 doesn't conflict with choice for C2.
            # So we can independently pick the best v for each C in disjoint_cliques.
            
            for ci in disjoint_cliques:
                best_v = -1
                best_gain = 0
                for v in clique_vertices[ci]:
                    # Gain: number of cliques (not in disjoint_cliques) that contain v
                    # Note: cliques in disjoint_cliques are disjoint, so v is only in ci among them.
                    # So gain is total coverage of v minus 1 (for ci itself) ?
                    # Actually, we just want to cover as many OTHER cliques as possible.
                    
                    gain = 0
                    for cj in vertex_cliques[v]:
                        if cj not in disjoint_cliques: # cj is not in the disjoint set
                            gain += 1
                    # Also, does v cover any other disjoint clique? No, they are disjoint.
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_v = v
                        
                if best_v != -1:
                    seeds.add(best_v)
                    # Mark all cliques containing best_v as covered
                    for cj in vertex_cliques[best_v]:
                        covered_by_seeds.add(cj)
                        
            # Now, solve the residual problem
            # Uncovered cliques: those not in covered_by_seeds
            residual_uncovered = set(range(m)) - covered_by_seeds
            
            residual_sol = set(seeds)
            
            if residual_uncovered:
                # Greedy on residual
                # We can reuse the greedy logic but restricted to residual_uncovered
                # For speed, just run standard greedy and merge?
                # No, standard greedy might pick vertices that are already in seeds (redundant) 
                # or pick vertices that don't help.
                
                # Let's run a greedy that only considers residual_uncovered
                while residual_uncovered:
                    best_v = -1
                    best_gain = 0
                    for v in range(n):
                        # Count how many residual_uncovered cliques v covers
                        gain = 0
                        for cj in vertex_cliques[v]:
                            if cj in residual_uncovered:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_v = v
                    if best_v == -1 or best_gain == 0:
                        break
                    residual_sol.add(best_v)
                    for cj in vertex_cliques[best_v]:
                        residual_uncovered.discard(cj)
                        
            # Local search on residual_sol
            residual_sol = local_search_remove(residual_sol)
            
            if len(residual_sol) < best_size:
                best_sol = residual_sol
                best_size = len(residual_sol)
                
        # 3. Try a few random restarts of the "Disjoint Clique" seeding with different orderings?
        # Given time budget, maybe just one more variant:
        # Sort disjoint cliques by size descending in the selection phase?
        
        # Let's try to improve the disjoint clique selection.
        # Instead of greedy min-degree, try to maximize the sum of sizes?
        # Or just try a few random permutations of the "order" for disjoint clique selection.
        
        for _ in range(5): # 5 restarts
            if time.time() > time_budget:
                break
                
            # Randomize the order for disjoint clique selection
            order = list(range(m))
            random.shuffle(order)
            # Maybe bias towards larger cliques? 
            # Let's just shuffle.
            
            available = set(range(m))
            selected_cliques = []
            
            for i in order:
                if i not in available:
                    continue
                selected_cliques.append(i)
                to_remove = {i} | cig_neighbors[i]
                available -= to_remove
                
            if not selected_cliques:
                continue
                
            seeds = set()
            covered_by_seeds = set()
            
            for ci in selected_cliques:
                best_v = -1
                best_gain = 0
                for v in clique_vertices[ci]:
                    gain = 0
                    for cj in vertex_cliques[v]:
                        if cj not in selected_cliques:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_v = v
                        
                if best_v != -1:
                    seeds.add(best_v)
                    for cj in vertex_cliques[best_v]:
                        covered_by_seeds.add(cj)
                        
            residual_uncovered = set(range(m)) - covered_by_seeds
            residual_sol = set(seeds)
            
            if residual_uncovered:
                while residual_uncovered:
                    best_v = -1
                    best_gain = 0
                    for v in range(n):
                        gain = 0
                        for cj in vertex_cliques[v]:
                            if cj in residual_uncovered:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_v = v
                    if best_v == -1 or best_gain == 0:
                        break
                    residual_sol.add(best_v)
                    for cj in vertex_cliques[best_v]:
                        residual_uncovered.discard(cj)
                        
            residual_sol = local_search_remove(residual_sol)
            
            if len(residual_sol) < best_size:
                best_sol = residual_sol
                best_size = len(residual_sol)
                
        return best_sol

    results = []

    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline - 30:
            results.append(list(range(n)))
            continue

        edges = random_graph(n, p, seed)
        cliques = bron_kerbosch_maximal_cliques(n, edges)
        cliques = [c for c in cliques if len(c) >= 2]

        if not cliques:
            results.append([])
            continue

        elapsed = time.time() - start_time
        remaining_instances = 3 - len(results)
        per_instance = (deadline - 15 - elapsed) / max(1, remaining_instances)
        per_instance = max(2.0, min(per_instance, deadline - 15 - elapsed))

        sol = solve_mwis_and_repair(n, cliques, start_time + elapsed + per_instance)
        results.append(sorted(sol))

    return results
# EVOLVE-BLOCK-END
