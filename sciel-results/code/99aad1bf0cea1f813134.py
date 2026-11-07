import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    deadline = time.time() + 55
    n = P['n']
    edges = P['edges']
    
    # Build adjacency as bitmasks for speed
    adj_mask = [0] * n
    edge_set = set()
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        edge_set.add((min(u,v), max(u,v)))
    
    all_edges = list(edge_set)
    total_edges = len(all_edges)
    
    def get_cross_edges(L, R):
        """Get set of all edges between L and R."""
        result = set()
        for u in L:
            for v in R:
                result.add((min(u,v), max(u,v)))
        return result
    
    def is_valid_biclique(L, R, edge_subset):
        """Check if L x R forms exactly edge_subset and all cross pairs are host edges."""
        if not L or not R:
            return False
        # All cross pairs must be in host graph
        for u in L:
            for v in R:
                if (min(u,v), max(u,v)) not in edge_set:
                    return False
        # Cross edges must equal edge_subset
        cross = get_cross_edges(L, R)
        return cross == edge_subset
    
    def find_biclique_from_seed(seed_edges, remaining_edges):
        """Find a maximal biclique containing seed_edges using only edges in remaining_edges."""
        if not seed_edges:
            return None
        
        # Start with vertices from seed edges
        vertices = set()
        for u, v in seed_edges:
            vertices.add(u)
            vertices.add(v)
        
        # Try to find a bipartition of these vertices such that all edges are cross-edges
        # and the biclique can be expanded within remaining_edges
        
        # Use a heuristic: try to grow a biclique from each edge in seed_edges
        best = None
        best_size = 0
        
        for u, v in list(seed_edges)[:5]:  # Limit to avoid explosion
            L = [u]
            R = [v]
            
            # Expand greedily
            changed = True
            while changed:
                changed = False
                # Try adding to L
                for x in range(n):
                    if x not in L and x not in R:
                        # Check if x connects to all in R via remaining edges
                        valid = True
                        for r in R:
                            if (min(x,r), max(x,r)) not in remaining_edges:
                                valid = False
                                break
                        if valid:
                            L.append(x)
                            changed = True
                # Try adding to R
                for y in range(n):
                    if y not in L and y not in R:
                        valid = True
                        for l in L:
                            if (min(l,y), max(l,y)) not in remaining_edges:
                                valid = False
                                break
                        if valid:
                            R.append(y)
                            changed = True
            
            # Compute covered edges
            covered = get_cross_edges(L, R)
            # Only count edges that are in remaining
            valid_covered = covered & remaining_edges
            if len(valid_covered) > best_size:
                best_size = len(valid_covered)
                best = (L, R, valid_covered)
        
        return best
    
    def greedy_init():
        remaining = set(all_edges)
        bicliques = []
        
        while remaining:
            if time.time() > deadline - 5:
                break
            
            # Sample edges to find good bicliques
            edge_list = list(remaining)
            random.shuffle(edge_list)
            
            best_bic = None
            best_cover = 0
            
            # Try multiple seeds
            for seed_edge in edge_list[:8]:
                # Find maximal biclique containing this edge
                L = [seed_edge[0]]
                R = [seed_edge[1]]
                
                # Expand
                changed = True
                while changed:
                    changed = False
                    # Add to L
                    for x in range(n):
                        if x not in L and x not in R:
                            if all((min(x,r), max(x,r)) in remaining for r in R):
                                L.append(x)
                                changed = True
                    # Add to R
                    for y in range(n):
                        if y not in L and y not in R:
                            if all((min(l,y), max(l,y)) in remaining for l in L):
                                R.append(y)
                                changed = True
                
                covered = get_cross_edges(L, R) & remaining
                if len(covered) > best_cover:
                    best_cover = len(covered)
                    best_bic = (L, R, covered)
            
            if best_bic is None:
                # Fallback: single edge
                e = next(iter(remaining))
                best_bic = ([e[0]], [e[1]], {e})
            
            bicliques.append((best_bic[0], best_bic[1]))
            remaining -= best_bic[2]
        
        return bicliques
    
    def try_merge(bicliques):
        """Try to merge two bicliques into one."""
        n_bic = len(bicliques)
        for i in range(n_bic):
            if time.time() > deadline - 3:
                return None
            for j in range(i+1, n_bic):
                L1, R1 = bicliques[i]
                L2, R2 = bicliques[j]
                
                # Union of edges
                edges_i = get_cross_edges(L1, R1)
                edges_j = get_cross_edges(L2, R2)
                union_edges = edges_i | edges_j
                
                # Check if union forms a valid biclique
                # Collect all vertices
                all_verts = set(L1) | set(R1) | set(L2) | set(R2)
                
                # Try to find bipartition
                # The union must be a complete bipartite graph
                # Check: no edges within same side
                
                # Try all possible assignments (too many) - use heuristic
                # If the union graph is bipartite and complete, it works
                
                # Build adjacency within union vertices
                adj_union = {v: set() for v in all_verts}
                for u, v in union_edges:
                    adj_union[u].add(v)
                    adj_union[v].add(u)
                
                # Check bipartiteness
                color = {}
                valid = True
                for start in all_verts:
                    if start in color:
                        continue
                    color[start] = 0
                    stack = [start]
                    while stack and valid:
                        curr = stack.pop()
                        for nb in adj_union[curr]:
                            if nb not in color:
                                color[nb] = 1 - color[curr]
                                stack.append(nb)
                            elif color[nb] == color[curr]:
                                valid = False
                                break
                    if not valid:
                        break
                
                if not valid:
                    continue
                
                L = [v for v in all_verts if color[v] == 0]
                R = [v for v in all_verts if color[v] == 1]
                
                # Check completeness: all L-R pairs must be in union_edges
                complete = True
                for u in L:
                    for v in R:
                        if (min(u,v), max(u,v)) not in union_edges:
                            complete = False
                            break
                    if not complete:
                        break
                
                if complete:
                    new_bicliques = [bicliques[k] for k in range(n_bic) if k != i and k != j]
                    new_bicliques.append((L, R))
                    return new_bicliques
        
        return None
    
    def try_split_improve(bicliques):
        """Try to split a biclique to enable better merges elsewhere."""
        # This is complex; skip for now
        return None
    
    # Main optimization loop
    best_bicliques = None
    best_count = float('inf')
    
    for restart in range(15):
        if time.time() > deadline - 10:
            break
        
        bicliques = greedy_init()
        
        # Local search
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            new_bic = try_merge(bicliques)
            if new_bic is not None:
                bicliques = new_bic
                improved = True
        
        if len(bicliques) < best_count:
            best_count = len(bicliques)
            best_bicliques = bicliques
    
    if best_bicliques is None:
        best_bicliques = [( [u], [v] ) for u, v in all_edges]
    
    # Convert to required format
    result = []
    for left, right in best_bicliques:
        result.append([list(left), list(right)])
    
    return result

# EVOLVE-BLOCK-END
