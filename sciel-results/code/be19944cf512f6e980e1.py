"""Clique partition of split graphs using Simulated Annealing on cross-edge coloring."""
import time
import random
import math
from collections import defaultdict

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_split_graph(n, k, edges, max_time=45.0):
        # 1. Parse Edges
        c_edges_set = set()
        cross_edges = []
        adj_c = [set() for _ in range(k)] # Neighbors in I for c in C
        adj_i = [set() for _ in range(n)] # Neighbors in C for i in I
        
        for (u, v) in edges:
            if u < k and v < k:
                c_edges_set.add((min(u, v), max(u, v)))
            else:
                if u < k:
                    c, i = u, v
                else:
                    c, i = v, u
                cross_edges.append((c, i))
                adj_c[c].add(i)
                adj_i[i].add(c)
        
        if not cross_edges:
            # No cross edges, just partition C-edges into cliques
            # Greedy maximal cliques on C
            cliques = []
            remaining_c = set(c_edges_set)
            while remaining_c:
                # Pick an edge
                u, v = next(iter(remaining_c))
                clique_set = {u, v}
                remaining_c.discard((u, v))
                
                # Try to add other vertices
                # Candidates are neighbors of u and v in C that are connected to both
                candidates = list(adj_c[u].intersection(adj_c[v]))
                # Filter candidates: must be connected to all in clique_set
                # And edges to clique_set must be in remaining_c (or just valid C-edges)
                # Actually, we just need to form a valid clique.
                # Since C is a clique, any subset is a clique.
                # We want to maximize size to minimize count.
                # Greedy: add any candidate that is connected to all in clique_set.
                # Since C is complete, all are connected.
                # But we must ensure we don't "use up" edges incorrectly?
                # No, we just need to partition edges.
                # If we pick a clique S, it covers all edges within S.
                # We should remove all edges within S from remaining_c.
                
                # Let's just pick a maximal clique in the remaining graph.
                # Since C is complete, the remaining graph is just the set of remaining edges.
                # A clique in the remaining graph is a set of vertices such that all pairs are in remaining_c.
                # This is the same as finding a clique in the graph defined by remaining_c.
                
                # Simple greedy for remaining C-edges:
                # Pick vertex with max degree in remaining_c
                deg = defaultdict(int)
                for (a, b) in remaining_c:
                    deg[a] += 1
                    deg[b] += 1
                
                if not deg: break
                start_v = max(deg, key=lambda x: (deg[x], -x))
                
                current_clique = {start_v}
                # Candidates must be connected to all in current_clique via remaining_c
                candidates = set()
                for (a, b) in remaining_c:
                    if a in current_clique and b not in current_clique:
                        candidates.add(b)
                    elif b in current_clique and a not in current_clique:
                        candidates.add(a)
                
                # Greedily add candidates
                # Sort by degree for determinism and quality
                for v in sorted(candidates, key=lambda x: -deg[x]):
                    if v in current_clique: continue
                    # Check if v is connected to all in current_clique via remaining_c
                    can_add = True
                    for u in current_clique:
                        e = (min(u, v), max(u, v))
                        if e not in remaining_c:
                            can_add = False
                            break
                    if can_add:
                        current_clique.add(v)
                        # Remove all edges from v to current_clique from remaining_c
                        for u in current_clique:
                            if u == v: continue
                            e = (min(u, v), max(u, v))
                            remaining_c.discard(e)
                
                if len(current_clique) >= 2:
                    cliques.append(sorted(list(current_clique)))
            return cliques

        # 2. Simulated Annealing Setup
        # State: Assignment of each cross-edge to a "slot".
        # A slot is defined by (i, group_id).
        # We start with a valid assignment where each cross-edge (c, i) is in its own slot?
        # No, that's too many slots.
        # Better: Start with a random valid coloring.
        # A valid coloring means: For a fixed i, edges incident to i can be in any groups.
        # For a fixed c, edges incident to c must be in distinct groups?
        # Wait, the constraint is:
        # Each clique is {i} U S. S is a subset of N(i).
        # The edges covered are (i, s) for s in S and (u, v) for u, v in S.
        # If we assign edge (c, i) to group G, then c is in S_G for i.
        # If edge (c, i1) is in G1 and (c, i2) is in G2, that's fine.
        # The constraint "every edge in exactly one clique" implies:
        # 1. Cross edge (c, i) is covered by the clique of its group G.
        # 2. C-edge (u, v) is covered by clique G if u, v are in S_G.
        #    It is covered by clique H if u, v are in S_H.
        #    It must be covered EXACTLY ONCE.
        #    This means: For any C-edge (u, v), there is exactly one group G such that u, v in S_G.
        #    This implies that the sets S_G (for all G) form a "linear hypergraph" on C with respect to edges?
        #    No, it means the collection of cliques {S_G} partitions the edge set of C?
        #    Not exactly. If u, v in S_G, the edge (u, v) is covered.
        #    If u, v in S_H, it's covered again.
        #    So we require that for every edge (u, v) in C, exactly one group G contains both u and v.
        #    This is equivalent to saying that the intersection of any two S_G sets is an independent set in C?
        #    No, if S_G and S_H share two vertices u, v, then edge (u, v) is covered twice.
        #    So, any two cliques S_G and S_H can share at most 1 vertex.
        #    This is a strong structural constraint.
        
        # This constraint makes the problem very hard.
        # However, we can relax it slightly by allowing "wasted" C-edges?
        # No, "every edge must lie in exactly one listed clique".
        # So we must strictly satisfy this.
        
        # Let's re-evaluate the "Coloring" model.
        # If we use edge coloring of the bipartite graph B=(C, I, E_cross):
        # Proper edge coloring: adjacent edges have different colors.
        # Adjacent in B means sharing a vertex in C or I.
        # If (c, i1) and (c, i2) have same color, they share c. Invalid for proper coloring.
        # If (c1, i) and (c2, i) have same color, they share i.
        # In the clique model:
        # Clique G is {i} U S_G.
        # S_G contains c if (c, i) has color G.
        # If (c1, i) and (c2, i) have color G, then c1, c2 in S_G.
        # This is allowed in the clique model (S_G is a subset of N(i)).
        # But in proper edge coloring of B, edges sharing i must have different colors.
        # So proper edge coloring of B FORBIDS putting (c1, i) and (c2, i) in same color.
        # This would force S_G to be a matching in I? No, S_G is a set of C-vertices.
        # If proper edge coloring, then for a fixed i, all incident edges have different colors.
        # So each i appears in many cliques, but in each clique, i is paired with a DIFFERENT set of C-vertices?
        # No, if (c1, i) is color 1 and (c2, i) is color 2.
        # Clique 1 for i: {i, c1, ...}
        # Clique 2 for i: {i, c2, ...}
        # This is valid.
        # What about C-edges?
        # (u, v) is covered by clique G if u, v in S_G.
        # S_G contains u if (u, i) is color G.
        # So (u, v) is covered by G if (u, i) is G AND (v, i) is G.
        # In proper edge coloring, (u, i) and (v, i) share i, so they MUST have different colors.
        # Therefore, (u, i) and (v, i) can NEVER have the same color.
        # Therefore, S_G can never contain both u and v if u, v are connected to i.
        # Wait, if u and v are connected to i, and they have different colors, they are in different S_G.
        # So no clique S_G contains both u and v.
        # Thus, NO C-edge (u, v) where u, v are both neighbors of i is covered by any clique centered at i.
        # What if u and v are not neighbors of i? Then they are not in S_G.
        # So if we use proper edge coloring, NO C-edge is covered by any cross-clique.
        # All C-edges must be covered by dedicated C-cliques.
        # This leads to a solution:
        # 1. Color B with Delta colors (Delta = max degree).
        # 2. For each color c, and each i, form clique {i} U S_{i, c}.
        #    S_{i, c} = {u in C | (u, i) has color c}.
        #    Since proper coloring, S_{i, c} has size at most 1?
        #    No, (u1, i) and (u2, i) share i, so different colors.
        #    So for a fixed i and color c, there is at most ONE u such that (u, i) has color c.
        #    So S_{i, c} is either empty or {u}.
        #    So cliques are {i, u} (size 2) or {i} (invalid, size 1).
        #    So we just list all cross edges as cliques of size 2.
        #    And all C-edges as cliques (greedy).
        #    This is the "trivial" solution basically.
        #    Total cliques ~ E_cross + E_C.
        
        # To do better, we MUST allow (u1, i) and (u2, i) to have the SAME color.
        # This means the coloring is NOT a proper edge coloring of B.
        # It is a "relaxed" coloring where edges sharing a vertex in I can have the same color.
        # But edges sharing a vertex in C MUST have different colors?
        # If (c, i1) and (c, i2) have same color G.
        # Then c is in S_{i1, G} and c is in S_{i2, G}.
        # Clique 1: {i1, c, ...}
        # Clique 2: {i2, c, ...}
        # Edge (c, i1) is in Clique 1. Edge (c, i2) is in Clique 2.
        # Are they in the same clique? No.
        # Is any edge covered twice?
        # C-edge (c, x).
        # Covered by Clique 1 if x in S_{i1, G}.
        # Covered by Clique 2 if x in S_{i2, G}.
        # If x is in both S_{i1, G} and S_{i2, G}, then (c, x) is covered twice.
        # So we must ensure that for any c, the sets S_{i, G} for all i, G do not create overlaps on C-edges.
        
        # Let's try a different heuristic:
        # Group cross-edges by I-vertex i.
        # For each i, partition N(i) into subsets S_{i, 1}, S_{i, 2}, ...
        # Each subset S_{i, j} defines a clique {i} U S_{i, j}.
        # Constraint: For any C-edge (u, v), it is covered by clique {i} U S_{i, j} if u, v in S_{i, j}.
        # It must be covered EXACTLY ONCE.
        # So for any (u, v), the number of pairs (i, j) such that u, v in S_{i, j} must be 1.
        
        # This is a constraint satisfaction problem.
        # Let's use Local Search (Simulated Annealing) on the partition of N(i).
        # State: For each i, a list of subsets.
        # Move: Move a vertex u from S_{i, j} to S_{i, k} (or create new S_{i, k+1}).
        # Cost: Number of C-edges covered 0 times + Number of C-edges covered >1 times.
        # We want Cost = 0.
        # Also, we want to minimize the number of subsets (cliques).
        # So Objective = (Number of C-edges with coverage != 1) * 1000 + (Total Number of Cliques).
        
        # Initialization:
        # Start with S_{i, 1} = N(i) for all i.
        # This covers many C-edges multiple times.
        
        # Let's implement this SA.
        
        # Data Structures:
        # neighbors[i] = list of C-vertices connected to i.
        # assignment[i] = dict mapping c_vertex -> subset_index.
        # subsets[i] = list of sets.
        
        # Precompute C-edge adjacency for fast coverage counting?
        # coverage_count[edge] = count.
        
        # To make it fast, we track coverage counts incrementally.
        
        # 1. Init
        assignment = {}
        subsets = {}
        for i in range(k, n):
            nbrs = list(adj_i[i])
            if not nbrs: continue
            assignment[i] = {}
            subsets[i] = [set(nbrs)]
            for c in nbrs:
                assignment[i][c] = 0
        
        # Calculate initial coverage
        coverage = defaultdict(int)
        for i in range(k, n):
            if i not in subsets: continue
            for s_idx, S in enumerate(subsets[i]):
                S_list = list(S)
                for u_idx in range(len(S_list)):
                    u = S_list[u_idx]
                    for v_idx in range(u_idx + 1, len(S_list)):
                        v = S_list[v_idx]
                        e = (min(u, v), max(u, v))
                        if e in c_edges_set:
                            coverage[e] += 1
        
        # Objective function
        def get_objective(coverage, num_cliques):
            # Count bad edges
            bad = 0
            for e, cnt in coverage.items():
                if cnt != 1:
                    bad += 1
            # Also count C-edges with 0 coverage
            for e in c_edges_set:
                if coverage.get(e, 0) == 0:
                    bad += 1
            return bad * 1000 + num_cliques
        
        num_cliques = sum(len(subsets[i]) for i in subsets if subsets[i])
        curr_obj = get_objective(coverage, num_cliques)
        
        # SA Loop
        T = 1.0
        T_min = 0.01
        cooling = 0.99
        max_iters = 50000
        rng = random.Random(42)
        
        # Precompute list of active i's
        active_ies = [i for i in range(k, n) if adj_i[i]]
        
        iter_count = 0
        while time.time() < start_time + max_time:
            # Pick random i
            i = rng.choice(active_ies)
            if i not in assignment: continue
            
            # Pick random c in N(i)
            c = rng.choice(list(adj_i[i]))
            
            # Pick random target subset index
            # Or create new
            if rng.random() < 0.1:
                new_idx = len(subsets[i])
                # Create new subset
                subsets[i].append(set())
                target_idx = new_idx
            else:
                target_idx = rng.randrange(len(subsets[i]))
            
            curr_idx = assignment[i][c]
            if curr_idx == target_idx:
                continue
            
            # Move c from curr_idx to target_idx
            
            # 1. Update coverage counts
            # Remove c from S_curr
            S_curr = subsets[i][curr_idx]
            S_target = subsets[i][target_idx]
            
            # Edges (c, u) in S_curr lose coverage
            for u in S_curr:
                if u == c: continue
                e = (min(c, u), max(c, u))
                if e in c_edges_set:
                    coverage[e] -= 1
                    if coverage[e] == 0:
                        del coverage[e]
            
            S_curr.discard(c)
            
            # Add c to S_target
            for u in S_target:
                if u == c: continue
                e = (min(c, u), max(c, u))
                if e in c_edges_set:
                    coverage[e] = coverage.get(e, 0) + 1
            
            S_target.add(c)
            assignment[i][c] = target_idx
            
            # Update num_cliques
            old_num_cliques = num_cliques
            if len(S_curr) == 0:
                # Remove empty subset?
                # We can keep it but it doesn't count if empty?
                # "Every listed set must be a clique with at least two vertices"
                # So empty subsets are not listed.
                # We only count non-empty subsets.
                # But we need to update the list.
                # Let's just count non-empty subsets in objective.
                pass
            
            # Recalculate num_cliques (expensive? No, just sum lengths)
            # Actually, we can track it.
            # If S_curr became empty, num_cliques decreases by 1 (if it was non-empty before).
            # If S_target was empty, num_cliques increases by 1 (if it was empty before).
            # But we appended to list, so index is valid.
            
            # Let's just recalculate num_cliques for simplicity if it's fast enough.
            # n=90, k=30. Subsets are small.
            new_num_cliques = 0
            for ii in active_ies:
                for S in subsets[ii]:
                    if len(S) >= 1: # Will be checked in final, but for obj we care about count
                        new_num_cliques += 1
            
            new_obj = get_objective(coverage, new_num_cliques)
            
            # Accept?
            delta = new_obj - curr_obj
            if delta < 0 or rng.random() < math.exp(-delta / T):
                curr_obj = new_obj
                num_cliques = new_num_cliques
            else:
                # Revert
                # Move c back
                S_target.discard(c)
                for u in S_target:
                    if u == c: continue
                    e = (min(c, u), max(c, u))
                    if e in c_edges_set:
                        coverage[e] -= 1
                        if coverage[e] == 0:
                            del coverage[e]
                
                S_curr.add(c)
                for u in S_curr:
                    if u == c: continue
                    e = (min(c, u), max(c, u))
                    if e in c_edges_set:
                        coverage[e] = coverage.get(e, 0) + 1
                
                assignment[i][c] = curr_idx
                
            T *= cooling
            if T < T_min:
                T = T_min
            
            iter_count += 1
            if iter_count % 1000 == 0:
                if time.time() > deadline - 5:
                    break

        # 3. Construct Final Cliques
        cliques = []
        for i in active_ies:
            for S in subsets[i]:
                if len(S) >= 1:
                    clique = [i] + sorted(list(S))
                    # Ensure it's a valid clique?
                    # i is connected to all in S (by construction).
                    # S is a subset of C, so it's a clique.
                    # So yes.
                    cliques.append(clique)
        
        # 4. Handle Remaining C-Edges
        # Find C-edges with coverage 0.
        # They must be covered by dedicated C-cliques.
        # Also, if coverage > 1, it's invalid.
        # But we tried to minimize this.
        # If any coverage > 1, we must fix it?
        # The problem requires EXACTLY ONE.
        # If SA didn't reach 0 bad edges, the solution is INVALID.
        # We must ensure validity.
        
        # Let's check validity.
        final_coverage = defaultdict(int)
        for c in cliques:
            if c[0] >= k: # Cross clique
                i = c[0]
                S = c[1:]
                S_set = set(S)
                for u_idx in range(len(S)):
                    u = S[u_idx]
                    for v_idx in range(u_idx + 1, len(S)):
                        v = S[v_idx]
                        e = (min(u, v), max(u, v))
                        if e in c_edges_set:
                            final_coverage[e] += 1
            
        # Check for errors
        invalid = False
        for e in c_edges_set:
            if final_coverage.get(e, 0) != 1:
                invalid = True
                break
        
        if invalid:
            # Fallback: Trivial valid solution
            # All cross edges as size-2 cliques.
            # All C-edges via greedy maximal cliques.
            cliques = []
            for (c, i) in cross_edges:
                cliques.append([c, i])
            
            remaining_c = set(c_edges_set)
            while remaining_c:
                u, v = next(iter(remaining_c))
                clique_set = {u, v}
                remaining_c.discard((u, v))
                
                deg = defaultdict(int)
                for (a, b) in remaining_c:
                    deg[a] += 1
                    deg[b] += 1
                
                if not deg: break
                start_v = max(deg, key=lambda x: (deg[x], -x))
                
                current_clique = {start_v}
                candidates = set()
                for (a, b) in remaining_c:
                    if a in current_clique and b not in current_clique:
                        candidates.add(b)
                    elif b in current_clique and a not in current_clique:
                        candidates.add(a)
                
                for v_cand in sorted(candidates, key=lambda x: -deg[x]):
                    if v_cand in current_clique: continue
                    can_add = True
                    for u_cand in current_clique:
                        e = (min(u_cand, v_cand), max(u_cand, v_cand))
                        if e not in remaining_c:
                            can_add = False
                            break
                    if can_add:
                        current_clique.add(v_cand)
                        for u_cand in current_clique:
                            if u_cand == v_cand: continue
                            e = (min(u_cand, v_cand), max(u_cand, v_cand))
                            remaining_c.discard(e)
                
                if len(current_clique) >= 2:
                    cliques.append(sorted(list(current_clique)))
            
            return cliques
        
        # If valid, we are done.
        # But we must also ensure that we didn't miss any C-edges?
        # The loop above checked all c_edges_set.
        # So if invalid is False, all C-edges are covered exactly once.
        # And all cross edges are covered exactly once (by construction of cliques from assignment).
        
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline - 15:
            # Emergency fallback
            edges = instance(*spec)
            results.append([[int(e[0]), int(e[1])] for e in edges])
            continue
        
        edges = instance(*spec)
        cliques = solve_split_graph(spec[0], spec[1], edges)
        
        # Validate and convert types
        final_cliques = []
        for c in cliques:
            c_ints = list(set(int(x) for x in c))
            if len(c_ints) >= 2:
                final_cliques.append(sorted(c_ints))
        
        results.append(final_cliques)
    
    return results
# EVOLVE-BLOCK-END
