import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
                    adj[u].add(v)
                    adj[v].add(u)
        return adj, edges

    def bron_kerbosch_maximal_cliques(adj):
        """
        Enumerates all maximal cliques in the graph defined by adjacency lists.
        Uses Bron-Kerbosch with pivot.
        """
        n = len(adj)
        cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            
            # Choose pivot u from P union X
            # Heuristic: choose u with max degree in P
            P_X = P.union(X)
            if not P_X:
                return
            u = max(P_X, key=lambda v: len(adj[v].intersection(P)))
            
            # Iterate over P \ N(u)
            for v in list(P - adj[u]):
                R_new = R.union({v})
                P_new = P.intersection(adj[v])
                X_new = X.intersection(adj[v])
                bk(R_new, P_new, X_new)
                P.remove(v)
                X.add(v)
        
        # Initial call
        # P = all vertices, X = empty
        bk(set(), set(range(n)), set())
        return cliques

    def solve_instance(adj, n, deadline):
        cliques = bron_kerbosch_maximal_cliques(adj)
        if not cliques:
            return []
        
        # Convert to list of sets for easier manipulation
        cliques_list = [set(c) for c in cliques]
        num_cliques = len(cliques_list)
        
        # Precompute vertex -> list of clique indices
        vertex_to_cliques = [[] for _ in range(n)]
        for i, c in enumerate(cliques_list):
            for v in c:
                vertex_to_cliques[v].append(i)
        
        # Precompute bitmask for each vertex
        # bit i set means vertex v is in clique i
        vertex_masks = [0] * n
        for v in range(n):
            mask = 0
            for ci in vertex_to_cliques[v]:
                mask |= (1 << ci)
            vertex_masks[v] = mask
        
        def greedy_init():
            S = set()
            cover_mask = 0  # Bitmask of covered cliques
            remaining_cliques = num_cliques
            
            while remaining_cliques > 0:
                best_v = -1
                max_new = -1
                for v in range(n):
                    if v in S:
                        continue
                    # Count new cliques covered by v
                    new_cov = vertex_masks[v] & ~cover_mask
                    # Bit count
                    cnt = new_cov.bit_count()
                    if cnt > max_new:
                        max_new = cnt
                        best_v = v
                if best_v == -1:
                    break
                S.add(best_v)
                cover_mask |= vertex_masks[best_v]
                # Update remaining
                # This is slightly inefficient but N is small
                # remaining_cliques = num_cliques - cover_mask.bit_count()
                # But we need to know if we are done.
                if cover_mask.bit_count() == num_cliques:
                    remaining_cliques = 0
                else:
                    # Just check if any uncovered
                    # We can just loop again
                    pass
            
            # Ensure all covered
            # Greedy might stop early if max_new is 0 but remaining > 0?
            # No, if max_new is 0, no vertex covers any new clique.
            # This implies all cliques are covered or some are unreachable (impossible in graph).
            return S

        def prune(S):
            S = set(S)
            # Use bitmask logic for pruning
            # We want to remove any v that is not critical.
            
            # To check if v is critical:
            # Is there a clique in v_mask[v] that is NOT covered by S \ {v}?
            
            changed = True
            while changed:
                changed = False
                for v in list(S):
                    if len(S) <= 1:
                        break
                    # Compute mask of S \ {v}
                    others_mask = 0
                    for u in S:
                        if u != v:
                            others_mask |= vertex_masks[u]
                    
                    # Critical cliques for v
                    critical = vertex_masks[v] & ~others_mask
                    if critical == 0:
                        # Safe to remove
                        S.remove(v)
                        changed = True
            return S

        best_S = greedy_init()
        best_S = prune(best_S)
        
        current_S = set(best_S)
        current_size = len(current_S)
        best_size = current_size
        best_S_copy = set(current_S)
        
        # Local Search with Bitmasks
        rng = random.Random(42)
        no_improve_count = 0
        max_no_improve = 5000
        
        # Precompute list of vertices in S and not in S for fast iteration?
        # S is small (<= n).
        
        while time.time() < deadline:
            if no_improve_count > max_no_improve:
                break
            
            improved = False
            
            # 1. Try Removals
            # Deterministic pass: try all vertices
            for v_out in list(current_S):
                if len(current_S) <= 1:
                    break
                # Compute others mask
                others_mask = 0
                for u in current_S:
                    if u != v_out:
                        others_mask |= vertex_masks[u]
                
                critical = vertex_masks[v_out] & ~others_mask
                if critical == 0:
                    # Remove
                    current_S.remove(v_out)
                    current_size -= 1
                    no_improve_count = 0
                    improved = True
                    if current_size < best_size:
                        best_size = current_size
                        best_S_copy = set(current_S)
                    break # Restart loop to recompute others_mask efficiently? 
                    # Actually, removing one vertex changes the state. 
                    # Breaking and restarting the while loop is safer and simpler.
                    # The overhead of restarting is small.
            
            if improved:
                continue
                
            # 2. Try 2-for-1 Moves
            # Replace two vertices v1, v2 with one vertex v_new
            # This reduces size by 1.
            
            # Only try if S is large enough
            if len(current_S) >= 2:
                # Sample pairs of vertices in S
                # To avoid O(N^2) overhead, sample a few random pairs
                # Or iterate all pairs if S is small (S is usually < 15)
                
                S_list = list(current_S)
                # If S is small, check all pairs
                if len(S_list) <= 20:
                    pairs_to_check = list(combinations(S_list, 2))
                    rng.shuffle(pairs_to_check)
                else:
                    pairs_to_check = [tuple(rng.sample(S_list, 2)) for _ in range(10)]
                
                for v1, v2 in pairs_to_check:
                    # Compute mask of S \ {v1, v2}
                    others_mask = 0
                    for u in current_S:
                        if u != v1 and u != v2:
                            others_mask |= vertex_masks[u]
                    
                    # Critical cliques covered by v1 or v2 but not others
                    critical_12 = (vertex_masks[v1] | vertex_masks[v2]) & ~others_mask
                    
                    if critical_12 == 0:
                        # We can remove v1 and v2 without adding anyone?
                        # That's a 2-for-0 move, which is even better.
                        # But wait, if critical_12 is 0, it means all cliques covered by v1/v2 are covered by others.
                        # So we can just remove them.
                        # This is handled by the removal step?
                        # Removal step checks one vertex at a time.
                        # If v1 is critical, removal fails. If v2 is critical, removal fails.
                        # But together, their unique coverage might be redundant with others?
                        # No, if v1 is critical, it means there is a clique C only in v1 (among S).
                        # If we remove v1 and v2, C is still only in v1 (removed). So C is uncovered.
                        # So critical_12 == 0 implies no clique is uniquely covered by {v1, v2} w.r.t S\{v1,v2}.
                        # So we can remove both.
                        current_S.remove(v1)
                        current_S.remove(v2)
                        current_size -= 2
                        no_improve_count = 0
                        improved = True
                        if current_size < best_size:
                            best_size = current_size
                            best_S_copy = set(current_S)
                        break
                    
                    # Find v_new not in S that covers critical_12
                    # v_new must cover all bits in critical_12
                    found_v_new = False
                    for v_new in range(n):
                        if v_new in current_S:
                            continue
                        if (vertex_masks[v_new] & critical_12) == critical_12:
                            # Valid 2-for-1
                            current_S.remove(v1)
                            current_S.remove(v2)
                            current_S.add(v_new)
                            current_size -= 1
                            no_improve_count = 0
                            improved = True
                            if current_size < best_size:
                                best_size = current_size
                                best_S_copy = set(current_S)
                            found_v_new = True
                            break
                    
                    if found_v_new:
                        break
            
            if improved:
                continue
            
            # 3. Try Swaps
            # Replace v_out in S with v_in not in S. Size constant.
            # Might enable future removals.
            
            if len(current_S) > 0:
                # Try a few random swaps
                for _ in range(5):
                    v_out = rng.choice(list(current_S))
                    
                    # Compute others mask (S \ {v_out})
                    others_mask = 0
                    for u in current_S:
                        if u != v_out:
                            others_mask |= vertex_masks[u]
                    
                    critical_out = vertex_masks[v_out] & ~others_mask
                    
                    # If critical_out is 0, we could have removed it. 
                    # (Should have been caught by removal step, but maybe S changed?)
                    # If critical_out is 0, swap with anyone is safe, but removal is better.
                    # So we assume critical_out is not 0 here.
                    
                    # Find v_in that covers critical_out
                    valid_v_ins = []
                    for v_in in range(n):
                        if v_in in current_S:
                            continue
                        if (vertex_masks[v_in] & critical_out) == critical_out:
                            valid_v_ins.append(v_in)
                    
                    if valid_v_ins:
                        v_in = rng.choice(valid_v_ins)
                        # Perform swap
                        current_S.remove(v_out)
                        current_S.add(v_in)
                        # Size unchanged
                        # We don't count this as improvement in size, but it changes state.
                        # To avoid infinite loops, we rely on the fact that if a removal becomes possible, we take it.
                        # If no removal is possible, we might swap forever.
                        # We increment no_improve_count for swaps?
                        # If we swap and don't remove, it's "no progress" towards size reduction.
                        no_improve_count += 1
                        break # Only one swap per iteration to keep it simple
                    else:
                        # No valid swap for this v_out
                        no_improve_count += 1
                else:
                    # If we didn't break (i.e., all 5 random swaps failed)
                    # We already incremented no_improve_count 5 times?
                    # Let's just increment once if no swap was made.
                    pass

        return sorted(best_S_copy)

    out = []
    # Allocate time roughly equally, but give more to larger instances?
    # 30, 35, 40 nodes.
    # Let's just run until deadline.
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() - start_time > deadline:
            # Fallback: return empty or best effort? 
            # If we run out of time, we should return what we have or a valid simple set.
            # The greedy init is valid. Let's just run greedy if SA didn't start or if we are late.
            adj, edges = random_graph(n, p, seed)
            # Quick greedy
            cliques = bron_kerbosch_maximal_cliques(adj)
            S = set()
            remaining = [set(c) for c in cliques]
            while remaining:
                best_v = None
                max_cov = -1
                for v in range(n):
                    if v in S: continue
                    cov = sum(1 for c in remaining if v in c)
                    if cov > max_cov:
                        max_cov = cov
                        best_v = v
                if best_v is None: break
                S.add(best_v)
                remaining = [c for c in remaining if best_v not in c]
            out.append(sorted(S))
            continue
            
        adj, edges = random_graph(n, p, seed)
        # Pass remaining time
        res = solve_instance(adj, n, deadline)
        out.append(res)
        
    return out
