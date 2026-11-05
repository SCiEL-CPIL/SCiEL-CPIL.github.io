"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 120.0
    
    def random_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def bron_kerbosch(adj, n):
        """
        Enumerate all maximal cliques using Bron-Kerbosch with pivot.
        Returns a list of sets of vertices.
        """
        cliques = []
        
        # Recursion limit for deep graphs, though n=64 is small
        sys.setrecursionlimit(10000)
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            
            # Pivot selection: choose u in P ∪ X that maximizes |P ∩ N(u)|
            # This minimizes the number of recursive calls
            best_u = None
            max_inter = -1
            for u in P:
                inter = len(P & adj[u])
                if inter > max_inter:
                    max_inter = inter
                    best_u = u
            for u in X:
                inter = len(P & adj[u])
                if inter > max_inter:
                    max_inter = inter
                    best_u = u
            
            if best_u is None:
                # P is empty? No, P or X is not empty. If P is empty, loop doesn't run.
                # If P is not empty, best_u must be found.
                # Actually if P is empty, we return early. So P is not empty here.
                # But if P is empty, the loop below doesn't run.
                # Wait, if P is empty, the condition `if not P and not X` handles it.
                # If P is not empty, we find a pivot.
                pass
                
            # Iterate over P \ N(best_u)
            # We need to iterate over vertices in P that are NOT neighbors of best_u
            # This ensures we only generate maximal cliques once
            P_to_process = list(P)
            if best_u is not None:
                P_to_process = [v for v in P if v not in adj[best_u]]
            
            for v in P_to_process:
                # Recurse with R ∪ {v}, P ∩ N(v), X ∩ N(v)
                new_R = R | {v}
                new_P = P & adj[v]
                new_X = X & adj[v]
                
                bk(new_R, new_P, new_X)
                
                # Move v from P to X
                P.remove(v)
                X.add(v)
                
                # Safety check for time
                if time.time() > deadline - 10.0:
                    raise TimeoutError("Time limit exceeded in BK")

        # Initial call
        try:
            bk(set(), set(range(n)), set())
        except TimeoutError:
            pass
            
        return cliques

    def solve_instance(adj):
        n = len(adj)
        edges = [(u, v) for u in range(n) for v in adj[u] if u < v]
        if not edges:
            return []
        
        remaining_edges = set(edges)
        
        # 1. Enumerate all maximal cliques
        # For dense graphs (p=0.55, 0.6), the number of maximal cliques is manageable for n=32, 64.
        # Worst case is 3^(n/3), but random graphs are usually much sparser in terms of maximal cliques structure.
        maximal_cliques = bron_kerbosch(adj, n)
        
        # Filter out cliques of size < 2 (though BK with size>=2 check handles it)
        # Sort cliques by size descending to prioritize larger ones
        maximal_cliques.sort(key=len, reverse=True)
        
        # 2. Greedy Set Cover
        # We want to cover all edges with the minimum number of cliques.
        # Greedy: at each step, pick the clique that covers the most *remaining* edges.
        
        cliques_found = []
        
        # Pre-compute edge sets for each clique for faster intersection?
        # Or just compute on the fly.
        # n is small, so computing edges of a clique is O(k^2).
        
        # To speed up, we can keep track of remaining edges as a set of tuples.
        
        while remaining_edges:
            if time.time() > deadline - 5.0:
                # Fallback: cover remaining edges individually
                for e in remaining_edges:
                    cliques_found.append(list(e))
                return cliques_found
            
            best_clique = None
            best_count = 0
            best_edges = set()
            
            for clique in maximal_cliques:
                # Calculate how many edges of this clique are in remaining_edges
                # Optimization: if the clique size is small, it might not cover many.
                # Also, if the clique is already "used" in a way that its edges are mostly gone, skip?
                # We don't remove cliques from the list, just their edges from remaining.
                
                # Check if this clique has any remaining edges
                # We can estimate: if the clique has k vertices, it has k*(k-1)/2 edges.
                # We need to count how many are in remaining_edges.
                
                # To avoid O(E) scan for every clique in every iteration, we can be smarter.
                # But n=64, E ~ 1000. Cliques ~ maybe 100-1000.
                # 1000 * 1000 = 1M ops per iteration. Iterations ~ 100-500.
                # 1M * 500 = 500M ops. Might be too slow in Python.
                
                # Optimization:
                # Only consider cliques that contain at least one vertex from a "high degree" remaining edge?
                # Or, simpler: pre-compute the set of edges for each clique.
                
                # Let's pre-compute edge sets for all maximal cliques first.
                # This is O(C * K^2).
                # We can do this outside the loop.
                
                pass # Placeholder, logic below
                
            # The above loop is inside the while, so we need the pre-computed edge sets.
            # Let's restructure.
            
            break # Break out of placeholder loop to implement proper structure
            
        # Proper Implementation:
        
        # Pre-compute edge sets for maximal cliques
        clique_edge_sets = []
        for clique in maximal_cliques:
            verts = list(clique)
            e_set = set()
            for i in range(len(verts)):
                for j in range(i+1, len(verts)):
                    u, v = verts[i], verts[j]
                    if u < v:
                        e_set.add((u, v))
                    else:
                        e_set.add((v, u))
            if e_set:
                clique_edge_sets.append((clique, e_set))
        
        # Greedy selection
        while remaining_edges:
            if time.time() > deadline - 5.0:
                for e in remaining_edges:
                    cliques_found.append(list(e))
                return cliques_found
            
            best_idx = -1
            best_count = 0
            best_e_set = None
            
            # Iterate through all candidate cliques
            # Optimization: Skip cliques that are subsets of previously selected ones?
            # No, we just remove edges.
            
            for idx, (clique, e_set) in enumerate(clique_edge_sets):
                # Count intersection with remaining_edges
                # len(e_set & remaining_edges)
                # To speed this up, we can check if e_set is a subset of remaining_edges? No.
                # Just intersection.
                
                # Heuristic: if the clique has no vertices in common with any edge in remaining_edges, skip?
                # That's hard to check quickly.
                
                # Simple intersection
                inter = e_set & remaining_edges
                cnt = len(inter)
                
                if cnt > best_count:
                    best_count = cnt
                    best_idx = idx
                    best_e_set = inter
                    
                if best_count == len(e_set):
                    # Can't do better than covering all edges of this clique
                    # But maybe another clique covers even more?
                    # If this clique is fully covered, it's a good candidate.
                    # But we should still check others if they are larger?
                    # No, we just want max count.
                    # If we find a clique that covers ALL its edges, and it's the largest clique, it's likely optimal.
                    # But a smaller clique might cover more *remaining* edges if the larger one has many already covered.
                    # So we must check all.
                    pass
            
            if best_idx == -1 or best_count == 0:
                # No clique covers any remaining edge? Should not happen if graph is connected and we have all edges.
                # But if maximal cliques don't cover all edges?
                # Maximal cliques cover all edges in a graph?
                # Yes, every edge belongs to at least one maximal clique.
                # So this case implies a bug or empty remaining_edges.
                break
            
            # Select the best clique
            clique, _ = clique_edge_sets[best_idx]
            cliques_found.append(list(clique))
            
            # Remove covered edges
            remaining_edges.difference_update(best_e_set)
            
            # Optional: Remove this clique from consideration?
            # We can leave it, but its edges are gone, so it will have 0 intersection next time.
            # To save time, we could mark it as "used" or remove from list.
            # Let's remove it from the list to reduce iteration.
            # But we need to be careful with indices if we modify list.
            # We can just set it to None or skip it.
            # Simpler: create a set of active indices.
            
            # For simplicity and given n is small, we leave the list and just skip if count is 0 in next iter?
            # The intersection check is the bottleneck.
            # Let's try to remove used cliques.
            # We'll use a set of active indices.
            
            # Actually, let's just stick to the simple loop. n=64 is small enough.
            
        return cliques_found

    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if time.time() > deadline - 10.0:
            # Not enough time, return empty? No, must be valid.
            # If we run out of time, we can't compute.
            # We should have computed in previous steps.
            # If we are here, it means we didn't finish.
            # We will return empty lists which will fail validity.
            # Better to have a faster algorithm or better time management.
            # But let's assume BK is fast enough for n=64.
            pass
            
        adj = random_graph(n, p, seed)
        cliques = solve_instance(adj)
        results.append(cliques)
        
    return results

# EVOLVE-BLOCK-END
