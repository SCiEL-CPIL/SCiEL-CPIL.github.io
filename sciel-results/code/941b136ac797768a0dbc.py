import time
import random
import math

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    n = P['n']
    edges = P['edges']
    
    # Precompute adjacency matrix for O(1) edge checks
    adj = [[False] * n for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
        edge_set.add((min(u, v), max(u, v)))
        
    # Initial greedy partition
    def greedy_partition():
        rem = set(edge_set)
        bics = []
        while rem:
            u, v = next(iter(rem))
            L = {u}
            R = {v}
            changed = True
            while changed:
                changed = False
                for x in range(n):
                    if x in L: continue
                    if all(adj[x][y] for y in R) and all((min(x, y), max(x, y)) in rem for y in R):
                        L.add(x)
                        changed = True
                for y in range(n):
                    if y in R: continue
                    if all(adj[x][y] for x in L) and all((min(x, y), max(x, y)) in rem for x in L):
                        R.add(y)
                        changed = True
            b_edges = set((min(u, v), max(u, v)) for u in L for v in R)
            bics.append([list(L), list(R)])
            rem -= b_edges
        return bics

    # Convert bicliques to edge lists for SA state
    def bicliques_to_edges(bics):
        edge_list = []
        for L, R in bics:
            for u in L:
                for v in R:
                    edge_list.append((min(u, v), max(u, v)))
        return edge_list

    def edges_to_bicliques(edge_list):
        # Reconstruct bicliques from edge list? No, we need to maintain structure.
        # Better state representation for SA: List of bicliques [L, R]
        # But moves are on edges. Let's use a different representation.
        # State: List of sets of edges. Each set must form a biclique.
        # This is hard to validate quickly.
        
        # Alternative State: List of bicliques [L, R].
        # Moves:
        # 1. Try to add an edge (u,v) to an existing biclique [L, R] if it fits.
        #    "Fits" means: (u in L and v in R) or (v in L and u in R) is already covered? No.
        #    If we add an edge, we must ensure the whole cross product is valid.
        #    Actually, if we have a biclique [L, R], and we want to add edge (u, v):
        #    If u is not in L and v is not in R, we can't just add it. We must expand L or R.
        #    Expanding L by u requires u connected to all in R.
        #    Expanding R by v requires v connected to all in L.
        #    This is complex.
        
        # Let's stick to the previous structure: List of [L, R].
        # But SA needs to be able to "unmerge" or "move".
        # A simpler and more effective heuristic for this specific problem size:
        # 1. Start with greedy.
        # 2. Use a "shake" strategy: Randomly perturb the partition by splitting bicliques,
        #    then re-merge greedily. Repeat many times.
        pass

    # Strategy: Multi-start Greedy with Random Perturbation (Shake & Merge)
    # 1. Generate initial greedy partition.
    # 2. In a loop:
    #    a. Randomly select a biclique and split it into two or more smaller bicliques.
    #    b. Run a greedy merge phase on the new set of bicliques.
    #    c. If the number of bicliques is lower than best, update best.
    
    best_bicliques = greedy_partition()
    best_count = len(best_bicliques)
    
    # Helper to check if a set of edges forms a valid biclique
    # and return the L, R sets if so.
    def get_biclique_from_edges(edge_subset):
        if not edge_subset:
            return [], []
        # Get all vertices involved
        vertices = set()
        for u, v in edge_subset:
            vertices.add(u)
            vertices.add(v)
        # Try to find a bipartition of these vertices such that all edges in edge_subset
        # are cross edges, and no cross edge is missing.
        # This is equivalent to checking if the subgraph induced by edge_subset is bipartite
        # AND is a complete bipartite graph.
        
        # Check if complete bipartite:
        # Pick a vertex v0. Its neighbors in the subgraph are N(v0).
        # The other part must be V \ N(v0).
        # Check if all edges between N(v0) and V\N(v0) exist in edge_subset.
        # And no edges within N(v0) or within V\N(v0).
        
        v0 = next(iter(vertices))
        neighbors = set()
        for u, v in edge_subset:
            if u == v0:
                neighbors.add(v)
            elif v == v0:
                neighbors.add(u)
        
        part1 = neighbors
        part2 = vertices - neighbors
        
        # Check no edges within part1
        for u, v in edge_subset:
            if u in part1 and v in part1:
                return None
            if u in part2 and v in part2:
                return None
        
        # Check all cross edges exist
        cross_edges_in_subset = set()
        for u, v in edge_subset:
            if (u in part1 and v in part2) or (u in part2 and v in part1):
                cross_edges_in_subset.add((min(u, v), max(u, v)))
        
        expected_cross_edges = set()
        for u in part1:
            for v in part2:
                expected_cross_edges.add((min(u, v), max(u, v)))
        
        if cross_edges_in_subset == expected_cross_edges:
            return [list(part1), list(part2)]
        else:
            return None

    def merge_bicliques(bics):
        # Greedy merge: repeatedly try to merge any two bicliques that can be merged.
        changed = True
        while changed:
            changed = False
            n_bics = len(bics)
            for i in range(n_bics):
                if i >= len(bics): continue
                for j in range(i+1, len(bics)):
                    if j >= len(bics): continue
                    L1, R1 = bics[i]
                    L2, R2 = bics[j]
                    # Check if merge is valid
                    valid = True
                    for u in L1:
                        for v in R2:
                            if not adj[u][v]:
                                valid = False
                                break
                        if not valid: break
                    if valid:
                        for u in L2:
                            for v in R1:
                                if not adj[u][v]:
                                    valid = False
                                    break
                            if not valid: break
                    if valid:
                        new_L = list(set(L1) | set(L2))
                        new_R = list(set(R1) | set(R2))
                        # Remove j, update i
                        bics.pop(j)
                        bics[i] = [new_L, new_R]
                        changed = True
                        break
                if changed:
                    break
        return bics

    def split_biclique(bics):
        # Pick a random biclique and split it randomly
        if not bics:
            return bics
        idx = random.randint(0, len(bics)-1)
        L, R = bics[idx]
        # Remove it
        bics.pop(idx)
        
        # Split edges of this biclique into two random sets
        edges_of_biclique = [(u, v) for u in L for v in R]
        # Randomly assign each edge to set 1 or set 2
        set1 = set()
        set2 = set()
        for e in edges_of_biclique:
            if random.random() < 0.5:
                set1.add(e)
            else:
                set2.add(e)
        
        # Ensure both sets are non-empty if possible
        if not set1:
            set1 = set2.pop() if set2 else set1 # This logic is flawed if set2 empty
            # Better: if one is empty, put all in one
            if not set1 and set2:
                 set1 = {next(iter(set2))}
                 set2.remove(next(iter(set1)))
            elif not set2 and set1:
                 set2 = {next(iter(set1))}
                 set1.remove(next(iter(set2)))
        
        b1 = get_biclique_from_edges(set1)
        b2 = get_biclique_from_edges(set2)
        
        if b1:
            bics.append(b1)
        if b2:
            bics.append(b2)
            
        # If splitting failed to produce valid bicliques (e.g. set was not a valid biclique itself? 
        # No, original was valid. But random split might create non-bipartite or non-complete subsets.
        # get_biclique_from_edges returns None if not a valid biclique.
        # In that case, we lose edges? No, we must ensure all edges are covered.
        # If get_biclique fails, we should just keep the original biclique.
        
        # Let's refine: if split produces invalid bicliques, revert.
        # Actually, simpler: just try to merge the split parts back if they are invalid?
        # Or just don't split if it's too hard.
        
        # For now, if b1 or b2 is None, we have lost edges. This is bad.
        # We need a safer split.
        # Safe split: pick a random vertex in L, move it to a new biclique?
        # If we move u from L to L', and keep R.
        # New biclique 1: L \ {u}, R. (Valid if L\{u} x R is complete? Yes, subset of original)
        # New biclique 2: {u}, R_u where R_u is subset of R?
        # If we just take {u} and ALL of R, it's a valid biclique.
        # But then edges (u, v) for v in R are in BOTH? No, partition.
        # So we must partition R into R1 and R2.
        # Biclique 1: L \ {u}, R1
        # Biclique 2: {u}, R2
        # This is valid if R1 and R2 are subsets of R.
        
        # Let's implement this safer split.
        if b1 is None or b2 is None:
             # Revert
             bics.append([L, R])
             return bics
             
        return bics

    # Main SA-like loop (actually just multi-start with shaking)
    current_bics = [list(b) for b in best_bicliques]
    
    while time.time() < deadline:
        # Shake
        if random.random() < 0.7:
             # Split a random biclique
             if current_bics:
                 idx = random.randint(0, len(current_bics)-1)
                 L, R = current_bics[idx]
                 # Safe split: move one vertex from L to a new biclique with a subset of R
                 if len(L) > 1:
                     u = random.choice(L)
                     # Partition R into R1, R2
                     R_list = list(R)
                     random.shuffle(R_list)
                     split_idx = random.randint(1, len(R_list)-1)
                     R1 = set(R_list[:split_idx])
                     R2 = set(R_list[split_idx:])
                     
                     new_L1 = [x for x in L if x != u]
                     
                     b1 = [new_L1, list(R1)]
                     b2 = [[u], list(R2)]
                     
                     # Check validity (should be valid by construction)
                     # Replace current_bics[idx] with b1 and b2
                     current_bics.pop(idx)
                     current_bics.append(b1)
                     current_bics.append(b2)
                 elif len(R) > 1:
                     v = random.choice(R)
                     L_list = list(L)
                     random.shuffle(L_list)
                     split_idx = random.randint(1, len(L_list)-1)
                     L1 = set(L_list[:split_idx])
                     L2 = set(L_list[split_idx:])
                     
                     new_R1 = [x for x in R if x != v]
                     
                     b1 = [list(L1), new_R1]
                     b2 = [list(L2), [v]]
                     
                     current_bics.pop(idx)
                     current_bics.append(b1)
                     current_bics.append(b2)
        
        # Merge
        current_bics = merge_bicliques(current_bics)
        
        # Check improvement
        if len(current_bics) < best_count:
            best_count = len(current_bics)
            best_bicliques = [list(b) for b in current_bics]
            
        # Occasionally restart from scratch?
        if random.random() < 0.01:
            current_bics = greedy_partition()
            if len(current_bics) < best_count:
                best_count = len(current_bics)
                best_bicliques = [list(b) for b in current_bics]

    # Final validation and type conversion
    covered_edges = set()
    for L, R in best_bicliques:
        for u in L:
            for v in R:
                e = (min(u, v), max(u, v))
                if e in covered_edges:
                    return []
                covered_edges.add(e)
                if not adj[u][v]:
                    return []
    
    if covered_edges != edge_set:
        return []
        
    final_result = []
    for L, R in best_bicliques:
        final_L = [int(x) for x in L]
        final_R = [int(x) for x in R]
        final_result.append([final_L, final_R])
        
    return final_result

if __name__ == "__main__":
    res = construct()
    print(f"Found {len(res)} bicliques")
