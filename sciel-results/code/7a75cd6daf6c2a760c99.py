"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
from collections import defaultdict

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10 second safety margin before 90s kill
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency sets for fast lookup
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u,v), max(u,v)))
    
    # Convert to sorted list for consistency
    edge_list = sorted(edge_set)
    total_edges = len(edge_list)
    
    # Track which edges are covered
    covered = [False] * total_edges
    edge_index = {e: i for i, e in enumerate(edge_list)}
    
    def get_uncovered_edges():
        return [e for i, e in enumerate(edge_list) if not covered[i]]
    
    def find_best_biclique(uncovered_edges, adj, n, edge_index, covered):
        """Find the best biclique among uncovered edges using greedy expansion."""
        if not uncovered_edges:
            return None, 0
        
        best_biclique = None
        best_coverage = 0
        
        # Try each uncovered edge as a seed
        for u, v in uncovered_edges:
            # Start with this edge
            left = {u}
            right = {v}
            
            # Greedily expand: try adding vertices to left that connect to all of right
            changed = True
            while changed:
                changed = False
                for w in range(n):
                    if w not in left and w not in right:
                        # Check if w connects to all vertices in right via uncovered edges
                        can_add = True
                        for r in right:
                            key = (min(w, r), max(w, r))
                            if key not in edge_index or covered[edge_index[key]]:
                                can_add = False
                                break
                        if can_add:
                            left.add(w)
                            changed = True
            
            # Greedily expand: try adding vertices to right that connect to all of left
            changed = True
            while changed:
                changed = False
                for w in range(n):
                    if w not in left and w not in right:
                        # Check if w connects to all vertices in left via uncovered edges
                        can_add = True
                        for l in left:
                            key = (min(w, l), max(w, l))
                            if key not in edge_index or covered[edge_index[key]]:
                                can_add = False
                                break
                        if can_add:
                            right.add(w)
                            changed = True
            
            # Count how many uncovered edges this biclique covers
            coverage = 0
            biclique_edges = set()
            for l in left:
                for r in right:
                    key = (min(l, r), max(l, r))
                    if key in edge_index:
                        biclique_edges.add(key)
                        if not covered[edge_index[key]]:
                            coverage += 1
            
            if coverage > best_coverage:
                best_coverage = coverage
                best_biclique = (left, right, biclique_edges)
        
        return best_biclique, best_coverage
    
    bicliques = []
    
    while True:
        if time.time() > deadline:
            break
        
        uncovered = get_uncovered_edges()
        if not uncovered:
            break
        
        best_biclique, best_coverage = find_best_biclique(uncovered, adj, n, edge_index, covered)
        
        if best_biclique is None:
            # Fallback: take a single edge
            u, v = uncovered[0]
            left = {u}
            right = {v}
            key = (min(u,v), max(u,v))
            best_biclique = (left, right, {key})
        
        left, right, biclique_edges = best_biclique
        
        # Mark these edges as covered
        for e in biclique_edges:
            idx = edge_index[e]
            covered[idx] = True
        
        # Store this biclique
        bicliques.append([sorted(left), sorted(right)])
    
    # Try to merge bicliques if their union forms a valid biclique
    merged = True
    while merged and time.time() < deadline:
        merged = False
        new_bicliques = []
        i = 0
        while i < len(bicliques):
            if i + 1 >= len(bicliques):
                new_bicliques.append(bicliques[i])
                break
            
            # Try to merge bicliques[i] and bicliques[i+1]
            left1, right1 = bicliques[i]
            left2, right2 = bicliques[i+1]
            
            # Try all 4 combinations of which side goes where
            candidates = [
                (set(left1) | set(left2), set(right1) | set(right2)),
                (set(left1) | set(right2), set(right1) | set(left2)),
                (set(right1) | set(left2), set(left1) | set(right2)),
                (set(right1) | set(right2), set(left1) | set(left2)),
            ]
            
            for new_left, new_right in candidates:
                if not new_left or not new_right:
                    continue
                # Check if new_left and new_right are disjoint
                if new_left & new_right:
                    continue
                # Check if all cross-edges exist and are uncovered (or covered by these two bicliques)
                valid = True
                for l in new_left:
                    for r in new_right:
                        key = (min(l, r), max(l, r))
                        if key not in edge_index:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    new_bicliques.append([sorted(new_left), sorted(new_right)])
                    i += 2
                    merged = True
                    break
            else:
                new_bicliques.append(bicliques[i])
                i += 1
        
        bicliques = new_bicliques
    
    # Final verification and type conversion
    result = []
    for left, right in bicliques:
        result.append([list(left), list(right)])
    
    return result
# EVOLVE-BLOCK-END
