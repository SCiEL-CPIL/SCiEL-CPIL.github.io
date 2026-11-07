import time
import random
import itertools
from collections import defaultdict

_DEADLINE = 140.0
_START = time.time()

def _elapsed():
    return time.time() - _START

def _deadline_hit():
    return _elapsed() > _DEADLINE

def construct():
    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def _norm(e):
        return (min(e), max(e))

    def _min_clique_cover_greedy(vertices, edge_set):
        """
        Greedy minimum clique cover for a graph defined by vertices and edge_set.
        Since the graph is a subgraph of a clique, we can be aggressive.
        Returns a list of cliques (lists of vertices).
        """
        if not edge_set:
            return []
        
        # Map vertex to its neighbors in the current edge_set
        adj = {v: set() for v in vertices}
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        
        remaining_edges = set(edge_set)
        remaining_vertices = set(vertices)
        cliques = []
        
        while remaining_edges:
            if _deadline_hit():
                # Fallback: return 2-cliques for all remaining
                for e in remaining_edges:
                    cliques.append(list(e))
                return cliques
            
            # Find a vertex with max degree
            # Optimization: only consider vertices incident to remaining edges
            active_verts = set()
            for u, v in remaining_edges:
                active_verts.add(u)
                active_verts.add(v)
                
            if not active_verts:
                break

            # Pick start vertex
            # Heuristic: pick vertex with max degree in remaining graph
            max_deg = -1
            start = None
            for v in active_verts:
                deg = len(adj[v] & active_verts) # Approximate, adj might have stale info
                # Better: count neighbors in remaining_edges directly if small, or maintain adj
                # Let's maintain adj dynamically
                pass
            
            # Rebuild adj for remaining edges to be safe, or update it.
            # Updating adj is better.
            # But for simplicity in this inner loop, let's just pick a random active vertex 
            # or the one with highest current degree.
            
            # Calculate degrees properly for active vertices
            degs = defaultdict(int)
            for u, v in remaining_edges:
                degs[u] += 1
                degs[v] += 1
            
            if not degs:
                break
                
            start = max(degs, key=lambda x: degs[x])
            
            # Greedy clique expansion
            current_clique = {start}
            candidates = set(adj[start]) & active_verts
            
            while candidates:
                # Pick candidate that is connected to all in current_clique
                # and has highest degree in candidates
                best_cand = None
                best_score = -1
                for c in candidates:
                    # Check if c is connected to all in current_clique
                    is_conn = all(_norm((c, x)) in remaining_edges for x in current_clique)
                    if is_conn:
                        # Score: degree in remaining graph
                        score = degs.get(c, 0)
                        if score > best_score:
                            best_score = score
                            best_cand = c
                if best_cand is None:
                    break
                current_clique.add(best_cand)
                # Update candidates: must be connected to new best_cand
                new_candidates = set()
                for c in candidates:
                    if c == best_cand:
                        continue
                    if _norm((c, best_cand)) in remaining_edges:
                        # Also must be connected to rest of clique (already checked for best_cand, 
                        # but c must be connected to ALL in clique)
                        # Wait, the `is_conn` check above ensures c is connected to current_clique BEFORE adding best_cand.
                        # So c is connected to old clique. We just need c connected to best_cand.
                        new_candidates.add(c)
                candidates = new_candidates
                
            # Add clique
            clique_list = list(current_clique)
            cliques.append(clique_list)
            
            # Remove edges covered by this clique
            for i in range(len(clique_list)):
                for j in range(i+1, len(clique_list)):
                    remaining_edges.discard(_norm((clique_list[i], clique_list[j])))
            
            # Remove vertices if they have no more edges? 
            # Actually, we just remove edges. Vertices with 0 edges are effectively done.
            
        return cliques

    def _solve_split(n, k, p, seed):
        edges = instance(n, k, p, seed)
        adj = [set() for _ in range(n)]
        edge_set = set()
        for (u, v) in edges:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add(_norm((u, v)))

        clique_part = list(range(k))
        indep_part = list(range(k, n))
        
        # Initial assignment: For each v in I, S_v = N(v)
        # We represent the partition of N(v) as a list of sets.
        # Initially, one set containing all neighbors.
        partitions = {}
        for v in indep_part:
            neighbors = list(adj[v])
            if neighbors:
                partitions[v] = [set(neighbors)]
            else:
                partitions[v] = []

        def _evaluate(partitions):
            """
            Returns (total_cliques, cliques_list)
            1. Form cliques for I-vertices.
            2. Determine remaining edges in C.
            3. Cover remaining edges in C with pure cliques.
            """
            covered_edges = set()
            all_cliques = []
            
            # 1. I-vertex cliques
            for v in indep_part:
                for S in partitions[v]:
                    if len(S) < 1:
                        continue
                    clique = [v] + list(S)
                    # Check validity: S must be a clique in C (it is, since C is a clique)
                    # And edges (v, u) for u in S must exist (they do, S subset N(v))
                    # And edges within S must exist (they do)
                    all_cliques.append(clique)
                    # Mark edges as covered
                    for u in S:
                        covered_edges.add(_norm((v, u)))
                    for i in range(len(S)):
                        a = list(S)[i]
                        for j in range(i+1, len(S)):
                            b = list(S)[j]
                            covered_edges.add(_norm((a, b)))
            
            # 2. Check for conflicts (double coverage)
            # If an edge is in covered_edges more than once, it's invalid.
            # We need to ensure the sets E(S) for all S in all partitions are disjoint.
            # Let's verify disjointness.
            # Actually, _evaluate is used for scoring. If there's a conflict, score is infinity.
            # But we can also just count unique edges covered.
            # If |covered_edges| < sum of edges in cliques, there's a conflict.
            
            num_edges_in_cliques = 0
            for c in all_cliques:
                num_edges_in_cliques += len(c)*(len(c)-1)//2
            
            if len(covered_edges) != num_edges_in_cliques:
                # Conflict exists. Penalty.
                return float('inf'), []
            
            # 3. Remaining edges in C
            # Edges in C are all pairs (u, w) with u, w in [0, k)
            # Remaining = E(C) - covered_edges
            c_edges = set()
            for i in range(k):
                for j in range(i+1, k):
                    e = _norm((i, j))
                    if e in edge_set: # All C edges are in edge_set
                        c_edges.add(e)
            
            remaining_c_edges = c_edges - covered_edges
            
            # Cover remaining C edges
            if remaining_c_edges:
                # Get vertices involved in remaining edges
                rem_verts = set()
                for u, v in remaining_c_edges:
                    rem_verts.add(u)
                    rem_verts.add(v)
                
                pure_cliques = _min_clique_cover_greedy(list(rem_verts), remaining_c_edges)
                all_cliques.extend(pure_cliques)
            
            # Final check: Did we cover everything?
            total_covered = set()
            for c in all_cliques:
                for i in range(len(c)):
                    for j in range(i+1, len(c)):
                        total_covered.add(_norm((c[i], c[j])))
            
            if total_covered != edge_set:
                return float('inf'), []
                
            return len(all_cliques), all_cliques

        # Local Search
        best_cost, best_cliques = _evaluate(partitions)
        
        # If initial is invalid (conflicts), we need to fix it first.
        # Conflicts arise if N(v) and N(w) share edges in C.
        # i.e., if u, w in N(v) and u, w in N(w'), then edge (u,w) is double covered.
        # We must split one of the sets.
        
        # Repair Phase: Resolve conflicts
        # While there are conflicts:
        #   Find a conflicting edge (u, w).
        #   Find v1, v2 such that {u, w} subset S_v1 and {u, w} subset S_v2.
        #   Split S_v1: remove w (or u) and put it in a new set? 
        #   Or move w to a new set for v1.
        
        # Simpler Repair:
        # If conflict, pick one of the sets S, remove a random vertex x from S, 
        # and create a new set {x} for that v.
        
        max_repair_steps = 1000
        for _ in range(max_repair_steps):
            if _deadline_hit():
                break
            cost, cliques = _evaluate(partitions)
            if cost != float('inf'):
                break
            
            # Find a conflict
            # Reconstruct covered edge counts
            edge_count = defaultdict(int)
            for v in indep_part:
                for S in partitions[v]:
                    S_list = list(S)
                    for i in range(len(S_list)):
                        a = S_list[i]
                        for j in range(i+1, len(S_list)):
                            b = S_list[j]
                            edge_count[_norm((a, b))] += 1
                    for u in S_list:
                        edge_count[_norm((v, u))] += 1 # Cross edges never conflict
            
            # Find edge with count > 1
            conflict_edge = None
            for e, cnt in edge_count.items():
                if cnt > 1:
                    conflict_edge = e
                    break
            
            if conflict_edge is None:
                # No C-edge conflict? Maybe cross edge? No, cross edges are unique to v.
                # If still inf, something else is wrong.
                break
            
            u, w = conflict_edge
            # Find which v's contain this edge in their S
            involved_vs = []
            for v in indep_part:
                for S in partitions[v]:
                    if u in S and w in S:
                        involved_vs.append(v)
            
            if not involved_vs:
                break
            
            # Pick one v to split
            v = random.choice(involved_vs)
            S_idx = None
            S_set = None
            for i, S in enumerate(partitions[v]):
                if u in S and w in S:
                    S_idx = i
                    S_set = S
                    break
            
            if S_idx is None:
                continue
                
            # Remove w from S_set
            S_set.discard(w)
            # Create new set with w
            # But {w} is not a clique (size 1). 
            # A clique {v, w} covers edge (v, w) and no C-edges.
            # This is valid.
            partitions[v].append({w})
            
            # Also, if S_set becomes empty, remove it
            if not S_set:
                partitions[v].pop(S_idx)
            # If new set {w} can be merged? No, it's separate.

        # Now we have a valid solution (hopefully).
        best_cost, best_cliques = _evaluate(partitions)
        
        # Hill Climbing / Simulated Annealing
        # Moves:
        # 1. Move a vertex from one set of v to another set of v.
        # 2. Move a vertex from S_v1 to S_v2 (different v)? 
        #    This changes which v covers the cross edges. 
        #    If u in S_v1, edge (v1, u) is covered by K_v1.
        #    If we move u to S_v2, edge (v1, u) is UNCOVERED. 
        #    So we can only move u if v1 == v2 OR we handle cross edges carefully.
        #    Actually, each cross edge (v, u) MUST be covered by a clique containing v.
        #    So u must be in exactly one set in partitions[v].
        #    So moves are only WITHIN the same v's partition, or swapping u between v1 and v2 is INVALID 
        #    unless we also adjust cross edge coverage, which is fixed by v.
        #    Wait, if u is in partitions[v1], it covers (v1, u). 
        #    If we move u to partitions[v2], it covers (v2, u). 
        #    Then (v1, u) is uncovered. Invalid.
        #    So u must stay in partitions[v].
        #    So the only moves are:
        #    - Split a set S in partitions[v] into S1, S2.
        #    - Merge two sets S1, S2 in partitions[v] if their union is a clique in C (always true) 
        #      AND the edges they cover are disjoint (i.e., E(S1) and E(S2) disjoint).
        #      Note: E(S1) and E(S2) are disjoint if S1 and S2 are disjoint sets? No.
        #      E(S1) is edges within S1. E(S2) is edges within S2.
        #      If S1 and S2 are disjoint, E(S1) and E(S2) are disjoint.
        #      So we can always merge disjoint sets?
        #      Yes, if S1 and S2 are disjoint, E(S1) and E(S2) are disjoint.
        #      So merging is always valid locally?
        #      Wait, if we merge S1 and S2 into S, the new edges covered are E(S) = E(S1) U E(S2) U E(S1, S2).
        #      The cross edges are (v, u) for u in S1 U S2. These were already covered.
        #      The new C-edges are between S1 and S2. 
        #      Are these already covered by another clique?
        #      If yes, merging causes a conflict.
        #      So merging is only valid if no edge between S1 and S2 is covered by another clique.
        
        # Let's implement simple moves:
        # 1. Merge two sets in partitions[v] if valid.
        # 2. Split a set in partitions[v] (random split) if it reduces cost? 
        #    Splitting increases clique count by 1, but might allow merging elsewhere? Unlikely.
        #    Splitting is only good to resolve conflicts.
        
        # Better Move: 
        # "Reassign" logic is limited.
        # Let's try to optimize the pure clique cover part by changing the partition to leave "cleaner" subgraphs in C.
        
        # Since k is small, we can try to find a partition that minimizes the number of pure cliques.
        
        # Let's do a few iterations of local search on the partition structure.
        
        current_cost, current_cliques = _evaluate(partitions)
        
        if _deadline_hit():
             return current_cliques if current_cliques else []
             
        # Simple Local Search
        for _ in range(500):
            if _deadline_hit():
                break
                
            # Pick a random v
            v = random.choice(indep_part)
            if len(partitions[v]) < 2:
                continue
                
            # Pick two sets to merge
            i, j = random.sample(range(len(partitions[v])), 2)
            S1 = partitions[v][i]
            S2 = partitions[v][j]
            
            # Check if merge is valid
            # Edges between S1 and S2 must not be covered by others
            # And S1, S2 must be disjoint (they are, by partition definition)
            
            # Check conflicts for edges between S1 and S2
            valid = True
            for u in S1:
                for w in S2:
                    e = _norm((u, w))
                    # Is e covered by any other clique?
                    # We can check by seeing if e is in covered_edges of other v's or pure cliques?
                    # This is expensive to check dynamically.
                    # Instead, just try the merge, evaluate, and revert if worse.
                    pass
            
            # Try merge
            merged = S1 | S2
            new_partitions = {vv: list(pp) for vv, pp in partitions.items()}
            new_partitions[v] = [s for idx, s in enumerate(partitions[v]) if idx not in (i, j)]
            new_partitions[v].append(merged)
            
            new_cost, _ = _evaluate(new_partitions)
            
            if new_cost < current_cost:
                partitions = new_partitions
                current_cost = new_cost
                current_cliques = _evaluate(partitions)[1]
            elif new_cost == current_cost:
                # Accept with some probability? Or just reject to keep simple.
                pass

        # Final Evaluation
        final_cost, final_cliques = _evaluate(partitions)
        
        if final_cost == float('inf') or not final_cliques:
             # Fallback to greedy
             return _greedy_fallback(n, k, p, seed)
             
        # Convert to lists of ints
        result = []
        for c in final_cliques:
            result.append([int(x) for x in c])
        return result

    def _greedy_fallback(n, k, p, seed):
        # Simple greedy: 1 clique per I-vertex (N(v)), then greedy for C.
        # This might have conflicts, so we use the repair logic or just return edge-by-edge if needed.
        # But we already have instance.
        edges = instance(n, k, p, seed)
        # Just return 2-cliques for all edges to be safe? No, that's the seed score.
        # We want better than seed.
        # Let's use the _min_clique_cover_greedy on the whole graph? 
        # No, split structure is key.
        
        # Let's try the initial partition again and hope repair works.
        # If not, return seed-like.
        adj = [set() for _ in range(n)]
        edge_set = set()
        for (u, v) in edges:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add(_norm((u, v)))
            
        partitions = {}
        for v in range(k, n):
            neighbors = list(adj[v])
            if neighbors:
                partitions[v] = [set(neighbors)]
            else:
                partitions[v] = []
        
        # Run repair
        for _ in range(1000):
             if _deadline_hit(): break
             cost, cliques = _evaluate(partitions)
             if cost != float('inf'):
                 return [ [int(x) for x in c] for c in cliques ]
             
             # Find conflict
             edge_count = defaultdict(int)
             for v in range(k, n):
                 for S in partitions[v]:
                     S_list = list(S)
                     for i in range(len(S_list)):
                         a = S_list[i]
                         for j in range(i+1, len(S_list)):
                             b = S_list[j]
                             edge_count[_norm((a, b))] += 1
             
             conflict_edge = None
             for e, cnt in edge_count.items():
                 if cnt > 1:
                     conflict_edge = e
                     break
             
             if not conflict_edge: break
             
             u, w = conflict_edge
             involved_vs = []
             for v in range(k, n):
                 for S in partitions[v]:
                     if u in S and w in S:
                         involved_vs.append(v)
             if not involved_vs: break
             
             v = random.choice(involved_vs)
             S_idx = None
             for i, S in enumerate(partitions[v]):
                 if u in S and w in S:
                     S_idx = i
                     break
             if S_idx is None: continue
             
             S_set = partitions[v][S_idx]
             S_set.discard(w)
             if not S_set:
                 partitions[v].pop(S_idx)
             partitions[v].append({w})
             
        # If still invalid, return 2-cliques
        return [ [int(e[0]), int(e[1])] for e in edges ]

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    out = []
    for spec in specs:
        if _deadline_hit():
            edges = instance(*spec)
            out.append([ [int(e[0]), int(e[1])] for e in edges ])
        else:
            out.append(_solve_split(*spec))
    return out

# Helper for _evaluate used in _solve_split
# Note: _evaluate is defined inside _solve_split in the draft above, 
# but I need to make sure it has access to edge_set, k, n, indep_part.
# In the code above, _evaluate is nested inside _solve_split, so it captures these.
# However, _min_clique_cover_greedy is outside.
# _greedy_fallback is outside.

# Wait, _evaluate uses _min_clique_cover_greedy.
# And _evaluate uses edge_set, k, n, indep_part from the enclosing scope.
# This is correct.

# One issue: _evaluate is called many times. It rebuilds covered_edges.
# For n=90, k=30, this is fast.

# Let's double check the _min_clique_cover_greedy logic.
# It takes vertices and edge_set.
# It returns cliques.
# It uses _norm.
# It uses _deadline_hit.
# All good.

# One potential bug: In _evaluate, `remaining_c_edges` is computed.
# `c_edges` is all edges in C.
# `covered_edges` includes cross edges and C-edges covered by I-cliques.
# So `remaining_c_edges` is correct.

# Final check on types:
# "integers must be Python int"
# "sequences must be list or tuple"
# My code returns lists of ints.

# The `instance` function is kept as is.

