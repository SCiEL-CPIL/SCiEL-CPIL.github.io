"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(k):
            for v in range(u + 1, k):
                edges.append((u, v))
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.append((u, v))
        return sorted(edges)

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        if not edges:
            return []
        
        # Build edge sets
        edge_set = set(edges)
        
        # Separate edges
        cc_edges = [(u, v) for u, v in edges if u < k and v < k]
        ci_edges = [(u, v) for u, v in edges if (u < k and v >= k) or (v < k and u >= k)]
        
        # Map CC edges to indices for bitmask DP
        cc_edge_to_idx = {e: i for i, e in enumerate(cc_edges)}
        m_cc = len(cc_edges)
        
        # If k is too large for bitmask DP (2^k is too big), fallback to greedy for all
        # For k <= 15, 2^k = 32768 states. DP is O(2^k * k^2). Very fast.
        # For k > 15, we use a greedy heuristic for C as well.
        
        if k <= 15:
            # Exact DP for C partition
            # dp[mask] = min cliques to cover CC edges in mask
            # We need to precompute valid cliques in C and the mask of edges they cover.
            # A clique in C is any subset of vertices.
            # Since G[C] is a complete graph, ANY subset of C is a clique.
            # So we just need to partition the set of CC edges into minimum number of cliques.
            # This is equivalent to finding the minimum number of cliques that cover all edges in K_k.
            # Wait, the graph induced by C is K_k. The edges are all pairs.
            # We want to partition E(K_k) into minimum number of cliques.
            # This is the "clique partition number" of a complete graph, which is related to edge coloring?
            # No, clique partition of K_k.
            # A clique in K_k is just a subset of vertices.
            # The edges covered by a clique S are all pairs in S.
            # We want to cover all pairs in {0..k-1} with min subsets.
            # This is equivalent to finding the minimum number of cliques to cover all edges.
            # This is a known problem.
            # For k=10, min cliques?
            # Actually, we can just use DP on the bitmask of edges.
            
            # Precompute edge masks for all possible cliques in C?
            # Number of cliques is 2^k.
            # For each clique S, mask_S has bits set for edges (u,v) in S.
            # We want min sum y_S such that OR of masks is all 1s? No, partition means disjoint edges.
            # So AND of masks must be 0 for different cliques?
            # Yes, edge-disjoint.
            
            # DP: dp[mask] = min cliques to cover exactly the edges in mask.
            # Transition: pick a clique S such that mask_S is a subset of mask?
            # No, we want to cover mask.
            # Let's use: dp[mask] = min over all cliques S where mask_S is subset of mask of (1 + dp[mask ^ mask_S]).
            
            # Generate all possible masks for cliques in C.
            # A clique is defined by a subset of vertices V_S.
            # mask_S = OR of edge_idx(u,v) for u,v in V_S.
            
            # Since k is small, we can iterate over all 2^k subsets of vertices.
            # There are 2^k subsets.
            # For each subset, compute edge mask.
            
            vertex_to_mask = {}
            for v_mask in range(1 << k):
                # Compute edge mask for this vertex subset
                e_mask = 0
                # Iterate pairs in v_mask
                verts = [i for i in range(k) if (v_mask >> i) & 1]
                for i in range(len(verts)):
                    for j in range(i+1, len(verts)):
                        u, v = verts[i], verts[j]
                        e_idx = cc_edge_to_idx[(min(u,v), max(u,v))]
                        e_mask |= (1 << e_idx)
                if e_mask != 0:
                    vertex_to_mask[v_mask] = e_mask
            
            # We only care about e_masks.
            # Many vertex subsets might produce the same e_mask? No, different vertex sets produce different edge sets.
            # But we can just use the list of unique e_masks.
            unique_e_masks = list(set(vertex_to_mask.values()))
            
            # Sort masks by popcount? Or just iterate.
            # DP array size 2^m_cc.
            # m_cc = k(k-1)/2.
            # For k=10, m_cc=45. 2^45 is too big for standard array.
            # But we only need to reach the full mask.
            # We can use a dictionary for DP states.
            
            full_mask = (1 << m_cc) - 1
            dp = {0: 0}
            # We want to build up to full_mask.
            # This is like shortest path in hypercube.
            # Or we can do BFS/DP over masks.
            # Since m_cc can be up to 45, 2^45 is huge.
            # However, the number of reachable states might be smaller?
            # Actually, for k=10, m_cc=45.
            # Is there a better way?
            # The problem is partitioning edges of K_k into cliques.
            # This is equivalent to finding the minimum number of cliques.
            # For K_k, the min number of cliques is related to the size of a minimum edge cover by cliques.
            # Wait, if we use cliques of size 2, we need m_cc cliques.
            # If we use larger cliques, we save.
            # A clique of size s covers s(s-1)/2 edges.
            # To minimize number of cliques, we want large cliques.
            # But they must be edge-disjoint.
            
            # Let's stick to the DP but optimize it.
            # If k is small enough that 2^m_cc is manageable, use array.
            # k=6 -> m=15 -> 32k.
            # k=8 -> m=28 -> 268M (too big).
            # So for k>6, we need a heuristic or a smarter DP.
            
            # Let's use a heuristic for C as well if k > 6.
            # Greedy: Pick largest available clique (in terms of edges covered) that doesn't conflict?
            # Or just pick random cliques?
            
            # Let's try a simple greedy for C:
            # While edges remain:
            #   Find a clique S in C that covers the maximum number of remaining edges.
            #   Add S to partition.
            #   Remove those edges.
            
            # This is O(2^k * m_cc) per step?
            # Number of steps <= m_cc.
            # Total O(m_cc * 2^k * m_cc).
            # For k=10, m_cc=45. 45 * 1024 * 45 ~ 2M. Fast.
            
            # Let's implement this greedy for C.
            remaining_cc = set(cc_edges)
            c_cliques = []
            
            # Precompute edge count for all vertex subsets
            # Actually, we can just iterate subsets.
            
            while remaining_cc:
                best_clique = None
                best_count = -1
                # Iterate all non-empty subsets of C
                # Optimization: only consider subsets that are cliques (all are) and have at least 2 vertices
                for v_mask in range(1, 1 << k):
                    # Check if this subset has any remaining edges
                    # And count them
                    verts = [i for i in range(k) if (v_mask >> i) & 1]
                    if len(verts) < 2:
                        continue
                    
                    count = 0
                    valid = True
                    for i in range(len(verts)):
                        for j in range(i+1, len(verts)):
                            e = (min(verts[i], verts[j]), max(verts[i], verts[j]))
                            if e in remaining_cc:
                                count += 1
                            else:
                                # If an edge is not remaining, it's already covered.
                                # We can still use the clique, but it only covers the remaining ones.
                                # However, the clique must be a clique in the graph.
                                # It is.
                                # But we want to maximize coverage of REMAINING edges.
                                pass
                    
                    if count > best_count:
                        best_count = count
                        best_clique = verts
                        
                if best_clique is None:
                    break
                    
                # Add best_clique to partition
                c_cliques.append(best_clique)
                
                # Remove edges covered by best_clique
                for i in range(len(best_clique)):
                    for j in range(i+1, len(best_clique)):
                        e = (min(best_clique[i], best_clique[j]), max(best_clique[i], best_clique[j]))
                        remaining_cc.discard(e)
            
        else:
            # Greedy for C as well
            remaining_cc = set(cc_edges)
            c_cliques = []
            while remaining_cc:
                best_clique = None
                best_count = -1
                # Sample random subsets? Or just try all?
                # If k is large, 2^k is too big.
                # Use a heuristic: find a large clique in the remaining graph?
                # But G[C] is complete. So any subset is a clique.
                # So we just want to pick a large subset of vertices that has many remaining edges.
                # Since G[C] is complete, the number of remaining edges in a subset S is
                # the number of pairs in S that are in remaining_cc.
                
                # Heuristic: Pick a random vertex, then add neighbors?
                # Or just pick a random subset of size k/2?
                
                # Let's try: Pick a random vertex u.
                # Add all other vertices v such that (u,v) is in remaining_cc?
                # No, that's not a clique in the remaining graph sense, but it is a clique in G.
                
                # Better: Start with a random vertex.
                # Greedily add vertices that have the most remaining edges to the current set.
                
                # Actually, for simplicity, let's just use size-2 cliques for C if k is large.
                # It's a valid partition.
                for e in list(remaining_cc):
                    c_cliques.append(list(e))
                    remaining_cc.discard(e)
                break # Exit while loop
            
        # Now handle I
        # We have c_cliques.
        # The edges covered by c_cliques are removed from consideration.
        # The remaining CC edges are "orphans".
        # The CI edges must be covered by I-cliques.
        
        # Build I-cliques
        # For each v in I, N(v) is the set of neighbors in C.
        # We need to cover all (u, v) for u in N(v).
        # We can use cliques S ∪ {v} where S ⊆ N(v).
        # This clique covers CI edges (u, v) for u in S.
        # It also covers CC edges (u, w) for u, w in S.
        # We can only use this if those CC edges are still available (not covered by c_cliques).
        
        # Let's track available CC edges.
        # Initially, all CC edges are available.
        # c_cliques have covered some.
        # Let's mark them as used.
        
        used_cc = set()
        for c in c_cliques:
            for i in range(len(c)):
                for j in range(i+1, len(c)):
                    used_cc.add((min(c[i], c[j]), max(c[i], c[j])))
        
        # Now, for each v in I:
        # We want to partition N(v) into subsets S_1, S_2, ...
        # Such that for each S_j, all CC edges within S_j are in (all_cc_edges \ used_cc).
        # And we want to minimize the number of subsets.
        
        # This is a set partition problem for each v, with a constraint on CC edges.
        # Heuristic:
        # Start with N(v).
        # Try to find a large subset S ⊆ N(v) such that all CC edges in S are available.
        # If found, add S ∪ {v} to partition, remove S from N(v), and mark CC edges in S as used.
        # Repeat until N(v) is empty.
        
        i_cliques = []
        
        # Process each v in I
        for v in range(k, n):
            # Get neighbors
            N_v = [u for u in range(k) if (u, v) in edge_set]
            
            # We need to cover all (u, v) for u in N_v.
            # Current available neighbors
            avail = set(N_v)
            
            while avail:
                # Try to find a subset S of avail that is "CC-clique-available"
                # i.e., all pairs in S have edges not in used_cc.
                
                # Greedy: Start with a random u in avail.
                # Add v to S if (u, v) is available.
                
                # Let's try to build a maximal such set.
                # Start with a random element
                start_u = next(iter(avail))
                S = {start_u}
                
                # Try to add other elements
                # Iterate through avail
                for u in list(avail):
                    if u in S:
                        continue
                    # Check if adding u to S is valid
                    # i.e., for all w in S, (u, w) is not in used_cc
                    valid = True
                    for w in S:
                        e = (min(u, w), max(u, w))
                        if e in used_cc:
                            valid = False
                            break
                    if valid:
                        S.add(u)
                
                # If S has size 1, we might be stuck if we can't add anyone.
                # But we must cover (start_u, v).
                # So S = {start_u} is always valid (no CC edges).
                
                # Add clique S ∪ {v}
                clique = list(S) + [v]
                i_cliques.append(clique)
                
                # Update used_cc
                for i in range(len(S)):
                    for j in range(i+1, len(S)):
                        used_cc.add((min(list(S)[i], list(S)[j]), max(list(S)[i], list(S)[j])))
                        
                # Remove S from avail
                avail.difference_update(S)
        
        # Combine results
        all_cliques = c_cliques + i_cliques
        
        # Validation
        # Check that all edges are covered exactly once.
        covered = set()
        for c in all_cliques:
            if len(c) < 2:
                return [[u, v] for u, v in edges] # Fallback
            for i in range(len(c)):
                for j in range(i+1, len(c)):
                    e = (min(c[i], c[j]), max(c[i], c[j]))
                    if e in covered:
                        return [[u, v] for u, v in edges] # Fallback
                    if e not in edge_set:
                        return [[u, v] for u, v in edges] # Fallback
                    covered.add(e)
                    
        if covered != edge_set:
            return [[u, v] for u, v in edges] # Fallback
            
        return all_cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))

    for spec in specs:
        if time.time() > deadline:
            edges = instance(*spec)
            results.append([[int(u), int(v)] for u, v in edges])
        else:
            res = solve_instance(*spec)
            results.append(res)

    return results
# EVOLVE-BLOCK-END
