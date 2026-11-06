import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    # We are searching for a 5-uniform hypergraph with minimum edges that is not 2-colorable.
    # A hypergraph is 2-colorable if there exists a coloring of vertices such that no edge is monochromatic.
    # This is equivalent to saying that the set of all 2^n colorings is covered by the "forbidden sets" of the edges.
    # Each edge e = {v1, ..., v5} forbids colorings where all vi have the same color (all 0 or all 1).
    # So each edge forbids 2 specific colorings (in the space of 2^n).
    # Wait, that's not quite right.
    # An edge is monochromatic if all its vertices have color 0 OR all its vertices have color 1.
    # So the set of colorings that make edge e monochromatic is:
    # S_e = { c in {0,1}^n : c(v1)=...=c(v5)=0 } union { c in {0,1}^n : c(v1)=...=c(v5)=1 }
    # The size of S_e is 2^(n-5) + 2^(n-5) = 2^(n-4).
    # We want to select a set of edges E such that the union of S_e for e in E is ALL of {0,1}^n.
    # This is a set cover problem.
    # Universe U = {0,1}^n (size 2^n).
    # Sets: S_e for each 5-subset e of {0,...,n-1}.
    # Goal: Minimize |E| such that union(S_e) = U.

    # For n=9, |U| = 512. Number of edges = C(9,5) = 126.
    # For n=10, |U| = 1024. Number of edges = C(10,5) = 252.
    # For n=11, |U| = 2048. Number of edges = C(11,5) = 462.
    # For n=12, |U| = 4096. Number of edges = C(12,5) = 792.
    # For n=13, |U| = 8192. Number of edges = C(13,5) = 1287.

    # The best known construction has 51 edges.
    # Let's try to find a set cover with fewer edges.

    best_n = 9
    best_edges = [list(e) for e in combinations(range(9), 5)]
    best_count = len(best_edges)

    for n in range(9, 14):
        if time.time() > deadline - 10.0:
            break

        total_colorings = 1 << n
        all_ones = (1 << total_colorings) - 1

        # Precompute the "forbidden coloring" mask for each edge.
        # For an edge e, S_e is the set of colorings where all vertices in e are 0 OR all are 1.
        # We can represent S_e as a bitmask of length total_colorings.
        
        all_edges = list(combinations(range(n), 5))
        E_count = len(all_edges)
        
        edge_masks = []
        for e in all_edges:
            mask = 0
            for c in range(total_colorings):
                # Check if c is monochromatic on e
                val = (c >> e[0]) & 1
                mono = True
                for v in e[1:]:
                    if ((c >> v) & 1) != val:
                        mono = False
                        break
                if mono:
                    mask |= (1 << c)
            edge_masks.append(mask)
            
        # Greedy Set Cover
        # We want to select edges to cover all_ones.
        
        # Try multiple random orderings or randomized greedy to escape local optima.
        # Or just standard greedy with deletion.
        
        best_local_count = float('inf')
        best_local_edges = None
        
        # Number of restarts
        if n <= 10:
            restarts = 50
        elif n <= 11:
            restarts = 20
        else:
            restarts = 5
            
        for r in range(restarts):
            if time.time() > deadline - 2.0:
                break
                
            covered = 0
            selected_indices = []
            remaining = set(range(E_count))
            
            # Standard Greedy
            while covered != all_ones:
                best_gain = -1
                best_idx = -1
                # To add randomness, we can pick from the top K gains
                gains = []
                for idx in remaining:
                    gain = (edge_masks[idx] & ~covered).bit_count()
                    if gain > 0:
                        gains.append((gain, idx))
                
                if not gains:
                    break # Failed to cover
                
                # Sort by gain descending
                gains.sort(reverse=True)
                
                # Pick randomly from top 5 to add diversity
                k = min(5, len(gains))
                best_gain, best_idx = random.choice(gains[:k])
                
                selected_indices.append(best_idx)
                covered |= edge_masks[best_idx]
                remaining.remove(best_idx)
                
            if covered == all_ones:
                # Deletion Phase
                improved = True
                while improved:
                    improved = False
                    for i in range(len(selected_indices) - 1, -1, -1):
                        e_idx = selected_indices[i]
                        new_covered = 0
                        for j in range(len(selected_indices)):
                            if j != i:
                                new_covered |= edge_masks[selected_indices[j]]
                                if new_covered == all_ones:
                                    break
                        if new_covered == all_ones:
                            selected_indices.pop(i)
                            covered = new_covered
                            improved = True
                            break
                            
                if len(selected_indices) < best_local_count:
                    best_local_count = len(selected_indices)
                    best_local_edges = [all_edges[i] for i in selected_indices]
                    
        if best_local_edges is not None:
            if best_local_count < best_count:
                best_count = best_local_count
                best_n = n
                best_edges = [list(e) for e in best_local_edges]
                
    return {'n': best_n, 'edges': best_edges}
