"""
Strategy: Bitset-Accelerated Greedy Set Cover.
1. Generate graphs.
2. Enumerate maximal cliques using Bron-Kerbosch.
3. Represent edges as indices and cliques as bitmasks (Python integers).
4. Use a greedy set cover algorithm where intersection counts are computed via bit operations (popcount), which is orders of magnitude faster than set intersections in Python.
5. This ensures the greedy algorithm completes within the time limit for n=64, avoiding the fallback to edge-by-edge partitioning which yields a poor score.
"""
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 120.0

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        # Store edges in a list for consistent indexing
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

    def bron_kerbosch(adj, n, edge_map):
        """
        Enumerate all maximal cliques.
        Returns a list of lists of vertex indices.
        """
        cliques = []
        sys.setrecursionlimit(10000)
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(list(R))
                return
            
            # Pivot selection
            best_u = None
            max_inter = -1
            # Union P and X for pivot search
            candidates = list(P) + list(X)
            for u in candidates:
                # Count neighbors in P
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
            
        return cliques

    def solve_instance(adj, edges, edge_map):
        n = len(adj)
        if not edges:
            return []
        
        # 1. Find maximal cliques
        maximal_cliques = bron_kerbosch(adj, n, edge_map)
        
        # 2. Create bitmasks for each clique
        # Edge index is 0 to len(edges)-1
        # We use Python integers as bitsets
        clique_masks = []
        for clique in maximal_cliques:
            mask = 0
            for i in range(len(clique)):
                for j in range(i+1, len(clique)):
                    u, v = clique[i], clique[j]
                    if u > v:
                        u, v = v, u
                    if (u, v) in edge_map:
                        idx = edge_map[(u, v)]
                        mask |= (1 << idx)
            if mask != 0:
                clique_masks.append(mask)
        
        if not clique_masks:
            # Fallback: no cliques found? Just return edges
            return [list(e) for e in edges]
        
        # 3. Greedy Set Cover with Bitsets
        total_edges = len(edges)
        # Initial mask: all edges covered
        # We want to cover all edges. 
        # remaining_mask has 1s where edge is NOT yet covered? 
        # Or 1s where edge IS covered?
        # Let's use `remaining_mask` = 1s for edges that still need covering.
        remaining_mask = (1 << total_edges) - 1
        
        selected_cliques = []
        
        # Optimization: Sort cliques by size (popcount) descending initially?
        # Or just greedy.
        
        while remaining_mask:
            if time.time() > deadline - 10.0:
                # Fallback: cover remaining edges individually
                # Find indices of set bits in remaining_mask
                rem_idx = []
                temp = remaining_mask
                idx = 0
                while temp:
                    if temp & 1:
                        rem_idx.append(idx)
                    temp >>= 1
                    idx += 1
                for i in rem_idx:
                    selected_cliques.append(list(edges[i]))
                break
            
            best_mask = 0
            best_count = 0
            
            for mask in clique_masks:
                # Intersection of clique edges with remaining edges
                inter = mask & remaining_mask
                if inter:
                    # Count bits
                    cnt = inter.bit_count() # Python 3.8+ has bit_count
                    if cnt > best_count:
                        best_count = cnt
                        best_mask = mask
                        if best_count == total_edges: # Can't get better
                            break
            
            if best_mask == 0:
                # No clique covers any remaining edge? 
                # This shouldn't happen if maximal cliques cover all edges.
                # But if BK failed or graph is weird, fallback.
                rem_idx = []
                temp = remaining_mask
                idx = 0
                while temp:
                    if temp & 1:
                        rem_idx.append(idx)
                    temp >>= 1
                    idx += 1
                for i in rem_idx:
                    selected_cliques.append(list(edges[i]))
                break
            
            # Select best clique
            # We need to return the vertex list, not just the mask.
            # We need to map mask back to vertices?
            # No, we stored `clique_masks` but lost the vertex list correspondence.
            # Let's store (mask, vertices) pairs.
            
            # RESTRUCTURE: store (mask, vertices)
            # I'll re-implement the loop with this structure.
            
            # Actually, let's just break and re-do the loop properly below.
            break
            
        # Proper Implementation with (mask, vertices)
        selected_cliques = []
        remaining_mask = (1 << total_edges) - 1
        
        # Pre-compute (mask, vertices)
        clique_data = []
        for clique in maximal_cliques:
            mask = 0
            for i in range(len(clique)):
                for j in range(i+1, len(clique)):
                    u, v = clique[i], clique[j]
                    if u > v:
                        u, v = v, u
                    if (u, v) in edge_map:
                        idx = edge_map[(u, v)]
                        mask |= (1 << idx)
            if mask != 0:
                clique_data.append((mask, clique))
                
        if not clique_data:
            return [list(e) for e in edges]

        while remaining_mask:
            if time.time() > deadline - 5.0:
                rem_idx = []
                temp = remaining_mask
                idx = 0
                while temp:
                    if temp & 1:
                        rem_idx.append(idx)
                    temp >>= 1
                    idx += 1
                for i in rem_idx:
                    selected_cliques.append(list(edges[i]))
                break
            
            best_data = None
            best_count = 0
            
            for mask, verts in clique_data:
                inter = mask & remaining_mask
                if inter:
                    cnt = inter.bit_count()
                    if cnt > best_count:
                        best_count = cnt
                        best_data = (mask, verts)
                        if best_count == total_edges:
                            break
            
            if not best_data:
                rem_idx = []
                temp = remaining_mask
                idx = 0
                while temp:
                    if temp & 1:
                        rem_idx.append(idx)
                    temp >>= 1
                    idx += 1
                for i in rem_idx:
                    selected_cliques.append(list(edges[i]))
                break
            
            mask, verts = best_data
            selected_cliques.append(verts)
            remaining_mask &= ~mask
            
        return selected_cliques

    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if time.time() > deadline - 15.0:
            # Not enough time for second graph? 
            # We must return valid results. 
            # If we haven't finished the first, we are in trouble.
            # But BK is fast.
            pass
            
        adj, edges, edge_map = random_graph(n, p, seed)
        cliques = solve_instance(adj, edges, edge_map)
        results.append(cliques)
        
    return results
