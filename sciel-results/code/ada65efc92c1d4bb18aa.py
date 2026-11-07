import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin for 90s limit
    n = P['n']
    edges = P['edges']
    
    # Build adjacency using bitmasks for speed
    adj_mask = [0] * n
    edge_set = set()
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        edge_set.add((min(u, v), max(u, v)))
    
    all_edges = list(edge_set)
    total_edges = len(all_edges)
    
    def get_biclique_edges(L, R, remaining_mask):
        """Get the set of edges in L x R that are in remaining_mask."""
        edge_list = []
        for u in L:
            for v in R:
                if u < v:
                    e = (u, v)
                else:
                    e = (v, u)
                if e in remaining_mask:
                    edge_list.append(e)
        return edge_list
    
    def expand_biclique(u, v, remaining_mask):
        """
        Expand a biclique starting from edge (u,v) using only edges in remaining_mask.
        Returns [L, R] where L and R are lists of vertices.
        Uses bitmask operations for speed.
        """
        L = {u}
        R = {v}
        
        # Build local adjacency masks from remaining edges
        # For each vertex, track which vertices it's connected to in remaining
        local_adj = {}
        for x in range(n):
            local_adj[x] = 0
        
        # Only consider edges in remaining_mask
        for (a, b) in remaining_mask:
            local_adj[a] |= (1 << b)
            local_adj[b] |= (1 << a)
        
        # Iteratively expand
        changed = True
        while changed:
            changed = False
            
            # Try adding to L: vertex x can join L if connected to all in R via remaining edges
            for x in range(n):
                if x not in L and x not in R:
                    # Check if x is connected to all vertices in R
                    can_add = True
                    for r in R:
                        if not (local_adj[x] & (1 << r)):
                            can_add = False
                            break
                    if can_add:
                        L.add(x)
                        changed = True
            
            # Try adding to R: vertex y can join R if connected to all in L via remaining edges
            for y in range(n):
                if y not in L and y not in R:
                    # Check if y is connected to all vertices in L
                    can_add = True
                    for l in L:
                        if not (local_adj[y] & (1 << l)):
                            can_add = False
                            break
                    if can_add:
                        R.add(y)
                        changed = True
        
        return [list(L), list(R)]
    
    def greedy_init():
        """
        Greedy set cover: repeatedly find the biclique covering the most remaining edges.
        Uses bitmask-based expansion for speed.
        """
        remaining = set(all_edges)
        bicliques = []
        
        while remaining:
            best_bic = None
            best_count = 0
            
            # Sample edges to find candidate bicliques
            sample_size = min(25, len(remaining))
            sample_edges = list(remaining)
            random.shuffle(sample_edges)
            sample_edges = sample_edges[:sample_size]
            
            for e in sample_edges:
                u, v = e
                bic = expand_biclique(u, v, remaining)
                L, R = bic
                
                # Count edges covered
                count = 0
                for l in L:
                    for r in R:
                        if (min(l, r), max(l, r)) in remaining:
                            count += 1
                
                if count > best_count:
                    best_count = count
                    best_bic = bic
            
            if best_bic is None:
                u, v = next(iter(remaining))
                best_bic = [[u], [v]]
            
            bicliques.append(best_bic)
            
            # Remove covered edges
            L, R = best_bic
            for l in L:
                for r in R:
                    remaining.discard((min(l, r), max(l, r)))
        
        return bicliques
    
    def is_valid_biclique_partition(bicliques):
        """Verify that bicliques form a valid partition of all edges."""
        covered = set()
        for L, R in bicliques:
            if not L or not R:
                return False
            # Check no vertex appears on both sides
            if set(L) & set(R):
                return False
            for u in L:
                for v in R:
                    e = (min(u, v), max(u, v))
                    if e in covered:
                        return False  # Overlap
                    if e not in edge_set:
                        return False  # Not a real edge
                    covered.add(e)
        return covered == edge_set
    
    def try_merge(bicliques):
        """
        Try to merge two bicliques if their union is also a biclique.
        """
        n_bic = len(bicliques)
        for i in range(n_bic):
            for j in range(i + 1, n_bic):
                L1, R1 = bicliques[i]
                L2, R2 = bicliques[j]
                
                # Get union of edges
                edges_i = set()
                for u in L1:
                    for v in R1:
                        edges_i.add((min(u, v), max(u, v)))
                
                edges_j = set()
                for u in L2:
                    for v in R2:
                        edges_j.add((min(u, v), max(u, v)))
                
                union_edges = edges_i | edges_j
                
                # Check if union forms a biclique
                # First check bipartiteness
                vertices = set()
                for u, v in union_edges:
                    vertices.add(u)
                    vertices.add(v)
                
                color = {}
                valid = True
                for v in vertices:
                    if v not in color:
                        stack = [v]
                        color[v] = 0
                        while stack and valid:
                            curr = stack.pop()
                            for nb in range(n):
                                if nb not in vertices:
                                    continue
                                if (min(curr, nb), max(curr, nb)) in union_edges:
                                    if nb not in color:
                                        color[nb] = 1 - color[curr]
                                        stack.append(nb)
                                    elif color[nb] == color[curr]:
                                        valid = False
                                        break
                
                if not valid:
                    continue
                
                L = [v for v in vertices if color[v] == 0]
                R = [v for v in vertices if color[v] == 1]
                
                # Check completeness
                complete = True
                for u in L:
                    for v in R:
                        if (min(u, v), max(u, v)) not in union_edges:
                            complete = False
                            break
                    if not complete:
                        break
                
                if complete:
                    new_bicliques = [b for k, b in enumerate(bicliques) if k != i and k != j]
                    new_bicliques.append([L, R])
                    return new_bicliques
        return None
    
    def try_move(bicliques):
        """
        Try to move a vertex from one side of a biclique to another, or reassign edges.
        This is a local search move that can help escape local optima.
        """
        n_bic = len(bicliques)
        
        for i in range(n_bic):
            L, R = bicliques[i]
            
            # Try moving a vertex from L to R or vice versa
            for v in list(L):
                # Try moving v from L to R
                L_new = [x for x in L if x != v]
                R_new = R + [v]
                
                if not L_new:
                    continue
                
                # Check if all edges in L_new x R_new are in edge_set
                valid = True
                for u in L_new:
                    for w in R_new:
                        if (min(u, w), max(u, w)) not in edge_set:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    new_bicliques = bicliques.copy()
                    new_bicliques[i] = [L_new, R_new]
                    return new_bicliques
            
            for v in list(R):
                # Try moving v from R to L
                R_new = [x for x in R if x != v]
                L_new = L + [v]
                
                if not R_new:
                    continue
                
                # Check if all edges in L_new x R_new are in edge_set
                valid = True
                for u in L_new:
                    for w in R_new:
                        if (min(u, w), max(u, w)) not in edge_set:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    new_bicliques = bicliques.copy()
                    new_bicliques[i] = [L_new, R_new]
                    return new_bicliques
        
        return None
    
    # Main search loop
    best_bicliques = None
    best_count = float('inf')
    
    max_restarts = 80
    for restart in range(max_restarts):
        if time.time() > deadline - 15:
            break
        
        bicliques = greedy_init()
        
        # Local search: merge and move
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            
            # Try merge
            new_bic = try_merge(bicliques)
            if new_bic is not None:
                bicliques = new_bic
                improved = True
                continue
            
            # Try move (with some randomness to avoid cycles)
            if random.random() < 0.3:
                new_bic = try_move(bicliques)
                if new_bic is not None:
                    # Verify validity
                    if is_valid_biclique_partition(new_bic):
                        bicliques = new_bic
                        improved = True
        
        if len(bicliques) < best_count:
            best_count = len(bicliques)
            best_bicliques = bicliques
            
            # Early termination if we found a very good solution
            if best_count <= 8:
                # Try more aggressive local search on this solution
                for _ in range(50):
                    if time.time() > deadline - 10:
                        break
                    new_bic = try_merge(bicliques)
                    if new_bic is not None:
                        bicliques = new_bic
                        if len(bicliques) < best_count:
                            best_count = len(bicliques)
                            best_bicliques = bicliques
                    else:
                        break
    
    # Fallback
    if best_bicliques is None:
        best_bicliques = [[[u], [v]] for u, v in all_edges]
    
    # Convert to proper format and ensure types
    result = []
    for left, right in best_bicliques:
        left_list = [int(v) for v in left]
        right_list = [int(v) for v in right]
        result.append([left_list, right_list])
    
    return result

# EVOLVE-BLOCK-END
