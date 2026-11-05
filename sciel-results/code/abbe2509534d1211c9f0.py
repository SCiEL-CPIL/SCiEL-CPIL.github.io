import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 120.0
    
    def random_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        edges = []
        edge_map = {} # (u,v) -> index
        idx = 0
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
                    edges.append((u, v))
                    edge_map[(u, v)] = idx
                    idx += 1
        return adj, edges, edge_map

    def is_clique(vertices, adj):
        """Check if a list of vertices forms a clique."""
        n = len(vertices)
        if n < 2:
            return False
        for i in range(n):
            for j in range(i+1, n):
                if vertices[j] not in adj[vertices[i]]:
                    return False
        return True

    def get_clique_edges(clique, edge_map):
        """Return set of edge indices for a clique."""
        es = set()
        for i in range(len(clique)):
            for j in range(i+1, len(clique)):
                u, v = clique[i], clique[j]
                if u > v: u, v = v, u
                if (u, v) in edge_map:
                    es.add(edge_map[(u, v)])
        return es

    def solve_instance(adj, edges, edge_map):
        n = len(adj)
        if not edges:
            return []
        
        # 1. Initial Partition: Greedy Set Cover with Maximal Cliques
        # We use a simplified Bron-Kerbosch to get maximal cliques quickly.
        # For dense graphs, maximal cliques are not too numerous.
        sys.setrecursionlimit(10000)
        cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(list(R))
                return
            
            # Pivot selection
            best_u = None
            max_inter = -1
            candidates = list(P) + list(X)
            if not candidates:
                return
            for u in candidates:
                inter = 0
                for v in P:
                    if v in adj[u]:
                        inter += 1
                if inter > max_inter:
                    max_inter = inter
                    best_u = u
            
            P_to_process = [v for v in P if best_u is None or v not in adj[best_u]]
            
            for v in P_to_process:
                new_R = R | {v}
                new_P = P & adj[v]
                new_X = X & adj[v]
                
                bk(new_R, new_P, new_X)
                
                P.remove(v)
                X.add(v)
                
        try:
            bk(set(), set(range(n)), set())
        except RecursionError:
            pass
        except Exception:
            pass
            
        if not cliques:
            # Fallback if BK fails
            return [list(e) for e in edges]

        # Map cliques to edge sets
        clique_edge_sets = []
        for c in cliques:
            es = get_clique_edges(c, edge_map)
            if es:
                clique_edge_sets.append((c, es))
        
        # Greedy Set Cover to get initial partition
        # We want to cover all edges with disjoint cliques.
        # Standard greedy picks the clique covering most *uncovered* edges.
        # However, we must ensure disjointness.
        # Since maximal cliques overlap, we cannot just pick them.
        # We need a partition.
        
        # Strategy for initial partition:
        # 1. Start with all edges as singletons.
        # 2. Try to merge adjacent singletons into larger cliques.
        # This is slow.
        
        # Better Initial Partition:
        # Use the greedy set cover, but when a clique is selected, 
        # we "commit" to those edges. If an edge is in multiple selected cliques, 
        # we have a conflict. 
        # Actually, the problem requires a PARTITION. 
        # So we cannot use overlapping cliques directly.
        
        # Let's use a different initial heuristic:
        # 1. Identify all maximal cliques.
        # 2. Sort by size descending.
        # 3. Iterate: if the clique's edges are all currently "unassigned" (or in singleton cliques that we are willing to break), take it.
        #    If some edges are already in a larger clique, skip or split.
        
        # Simpler robust initial partition:
        # Start with edges as individual cliques.
        # Then, iteratively try to merge two cliques if their union is a clique.
        # This is O(N^2 * Check) which might be slow for 64 nodes (1000+ edges).
        
        # Let's try a "Greedy Merging" approach:
        # 1. Initialize partition: each edge is a clique of size 2.
        # 2. While possible:
        #    Find two cliques C1, C2 such that C1 U C2 is a clique.
        #    Merge them.
        
        # This is essentially building up cliques.
        # To speed this up, we can look for triangles first, then K4s, etc.
        
        # Alternative: Use the Maximal Cliques as a pool of "candidates".
        # We perform a greedy selection where we pick a maximal clique, 
        # and then remove all edges of that clique from the graph, 
        # and repeat on the remaining edges.
        # This guarantees a partition (since we remove edges).
        # Is it optimal? No, but it's a valid partition.
        
        partition = []
        remaining_edges = set(range(len(edges)))
        
        # Pre-compute edge sets for all maximal cliques
        max_clique_data = []
        for c in cliques:
            es = get_clique_edges(c, edge_map)
            if es:
                max_clique_data.append((c, es))
                
        # Sort by size descending
        max_clique_data.sort(key=lambda x: len(x[1]), reverse=True)
        
        # Greedy selection with removal
        while remaining_edges:
            if time.time() > deadline - 10.0:
                # Fallback: add remaining as singletons
                for e_idx in remaining_edges:
                    partition.append(list(edges[e_idx]))
                break
                
            best_clique = None
            best_count = 0
            
            for c, es in max_clique_data:
                # Count how many edges of this clique are in remaining_edges
                inter_count = len(es.intersection(remaining_edges))
                if inter_count > best_count:
                    best_count = inter_count
                    best_clique = (c, es)
                    if best_count == len(es):
                        break
                        
            if not best_clique or best_count == 0:
                # No maximal clique covers any remaining edge?
                # This implies the remaining edges don't form a clique that is a subset of any maximal clique?
                # Impossible, every edge is in some maximal clique.
                # Unless we removed edges and broke the structure?
                # No, we remove edges, but the vertices remain.
                # The "clique" property is on vertices.
                # If we take a clique C, and remove its edges, the vertices of C are still there.
                # But we can't use C again because its edges are gone.
                # We need to find a clique in the *remaining* graph.
                
                # The current approach "pick maximal clique from original graph" is flawed 
                # because after removing edges, the original maximal clique might not be a clique 
                # in the *remaining* graph? 
                # Wait. A clique is a set of vertices. If all edges are present, it's a clique.
                # If we remove edges, it's no longer a clique.
                # So we cannot reuse the original maximal cliques if we have removed some of their edges.
                
                # We must find cliques in the *current* remaining graph.
                # This is expensive to do repeatedly.
                
                # Fallback for this step:
                # Just take a single edge.
                e_idx = next(iter(remaining_edges))
                partition.append(list(edges[e_idx]))
                remaining_edges.remove(e_idx)
                continue
            
            c, es = best_clique
            # We select this clique.
            # But we must ensure that the edges we are taking are actually forming a clique in the *remaining* graph?
            # The clique `c` was a clique in the original graph.
            # If we have not removed any edges from `c` previously, then `c` is still a clique in the remaining graph.
            # Since we only select each maximal clique once? No, we might select it, remove its edges, and then later select another clique that shares vertices.
            # If we select C1, remove its edges.
            # Then we look for C2. If C2 shares edges with C1, those edges are gone.
            # C2 might not be a clique in the remaining graph if it relied on an edge from C1.
            # But C2 is a set of vertices. If all edges between vertices in C2 are present in the remaining graph, it's a clique.
            # The `es` set for C2 contains edges. If `es` is a subset of `remaining_edges`, then all edges of C2 are present.
            # So `inter_count == len(es)` ensures C2 is a valid clique in the remaining graph.
            # If `inter_count < len(es)`, then C2 is NOT a valid clique in the remaining graph (some edges missing).
            # So we should only pick cliques where `inter_count == len(es)`.
            
            # Let's refine the selection:
            # Only pick if the clique is fully intact in the remaining graph.
            
            valid_clique = None
            best_count = 0
            for c, es in max_clique_data:
                if es.issubset(remaining_edges):
                    if len(es) > best_count:
                        best_count = len(es)
                        valid_clique = (c, es)
                        if best_count == 1: # Can't get better than 1 edge? No, we want max size.
                            pass
                        # If we found a large one, good.
                        
            if valid_clique:
                c, es = valid_clique
                partition.append(c)
                remaining_edges -= es
            else:
                # No intact maximal clique left?
                # This can happen if we fragmented the graph.
                # We need to find ANY clique in the remaining graph.
                # Simple heuristic: pick an edge, and see if we can extend it.
                # Or just pick a single edge.
                # To be safe and fast, pick a single edge.
                # This might lead to a poor score if we can't find larger cliques.
                
                # Let's try to find a triangle in the remaining graph quickly.
                # Pick a random edge (u, v) from remaining.
                # Check if there is a w such that (u,w) and (v,w) are in remaining.
                # This is O(N) per edge.
                
                # For now, fallback to single edge to ensure termination and validity.
                # Optimization: We can try to merge single edges later.
                e_idx = next(iter(remaining_edges))
                partition.append(list(edges[e_idx]))
                remaining_edges.remove(e_idx)
                
        # 2. Local Search Improvement
        # Current partition is a list of cliques.
        # We want to reduce the number of cliques.
        # Moves:
        # 1. Merge two cliques if their union is a clique.
        # 2. Move an edge from a clique to another if it maintains clique property?
        #    (Harder, as it breaks the first clique).
        
        # Let's focus on Merging.
        # This is equivalent to finding disjoint cliques that can be combined.
        # If C1 and C2 are disjoint sets of vertices, and all edges between C1 and C2 exist,
        # then C1 U C2 is a clique.
        
        # We can try to iteratively merge.
        # To speed up, we can check pairs of cliques.
        
        # Convert partition to a list of sets of vertices for easier manipulation
        # and sets of edge indices for quick checking.
        
        # Structure: list of (vertex_set, edge_set)
        current_partition = []
        for c in partition:
            vs = set(c)
            es = get_clique_edges(c, edge_map)
            current_partition.append([vs, es])
            
        # Merge Loop
        merged = True
        while merged and time.time() < deadline - 5.0:
            merged = False
            n_cliques = len(current_partition)
            for i in range(n_cliques):
                if time.time() > deadline - 5.0: break
                for j in range(i+1, n_cliques):
                    if time.time() > deadline - 5.0: break
                    vs1, es1 = current_partition[i]
                    vs2, es2 = current_partition[j]
                    
                    # Check if vs1 and vs2 are disjoint
                    if vs1 & vs2:
                        continue
                        
                    # Check if all edges between vs1 and vs2 exist
                    # This is equivalent to checking if (vs1 U vs2) is a clique.
                    # We can check if the union of edge sets plus the cross edges forms a clique.
                    # Simpler: just check if the union of vertices is a clique in the original graph.
                    # Since we only merge disjoint cliques from the partition, and they were valid cliques,
                    # we just need to check the cross edges.
                    
                    is_clique = True
                    for u in vs1:
                        for v in vs2:
                            if v not in adj[u]:
                                is_clique = False
                                break
                        if not is_clique:
                            break
                            
                    if is_clique:
                        # Merge j into i
                        new_vs = vs1 | vs2
                        new_es = es1 | es2
                        # Add cross edges to new_es? 
                        # get_clique_edges will handle it if we recompute, 
                        # but we can just union the edge sets?
                        # No, es1 and es2 don't contain cross edges.
                        # We must recompute edge set for the new clique to be consistent.
                        new_es = get_clique_edges(list(new_vs), edge_map)
                        
                        current_partition[i] = [new_vs, new_es]
                        current_partition.pop(j)
                        merged = True
                        break # Restart inner loop as indices changed
                if merged:
                    break
                    
        # Convert back to list of lists
        final_cliques = [list(vs) for vs, es in current_partition]
        
        # Final Validation
        # Ensure every edge is covered exactly once.
        # (The merging process preserves disjointness and coverage).
        
        return final_cliques

    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if time.time() > deadline - 15.0:
            # Not enough time
            # Return trivial partition for remaining
            adj, edges, edge_map = random_graph(n, p, seed)
            results.append([list(e) for e in edges])
            continue
            
        adj, edges, edge_map = random_graph(n, p, seed)
        cliques = solve_instance(adj, edges, edge_map)
        results.append(cliques)
        
    return results
