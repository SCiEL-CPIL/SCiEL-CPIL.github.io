"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
import random
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n = P['n']
    edges = P['edges']
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Heuristic: Sort vertices by degree descending to prioritize "hubs"
    vertices = list(range(n))
    vertices.sort(key=lambda x: len(adj[x]), reverse=True)
    
    best_branches = []
    best_paths = {}
    
    # Try to build a large set of branches.
    # We will greedily add vertices to the 'branches' set.
    # For each new branch added, we must ensure that for ALL pairs in the new set,
    # there exists a path that is disjoint from all other paths in the set.
    
    # To speed up, we'll try a few random orderings or just the degree ordering.
    # Since n=24, we can be a bit more exhaustive if needed, but let's start with greedy.
    
    def try_add_vertex(candidate, current_branches, current_paths):
        # Check if we can add 'candidate' to current_branches
        # We need to find paths from 'candidate' to every v in current_branches
        # such that these new paths are disjoint from each other and from existing paths.
        
        # This is a complex constraint. 
        # Simplification: If we add a vertex, we might need to re-route ALL paths.
        # Re-routing all paths for a set of size K is hard.
        
        # Alternative Strategy: 
        # Start with a small clique (size 3) which is easy.
        # Then try to expand by adding vertices that are "close" to the existing structure.
        
        pass # Placeholder, we will implement a different logic below
    
    # Better Strategy: 
    # 1. Find a large set of vertices that can be connected by disjoint paths.
    # This is equivalent to finding a "Steiner Tree" like structure but for multiple pairs.
    # Actually, the problem asks for paths for EVERY unordered pair of branches.
    # This is a very strong constraint. If branches = {A, B, C}, we need paths A-B, A-C, B-C.
    # These paths must be disjoint in their interiors.
    
    # Let's try to find a large clique first, as the seed did, but allow for "long" paths?
    # No, if they are a clique, the paths are just edges.
    # The seed found a triangle (size 3). Score 1.0 means seed found 3 branches.
    # To improve, we need > 3 branches.
    
    # Can we find 4 branches?
    # If we have 4 branches {A,B,C,D}, we need 6 paths.
    # The interiors must be disjoint.
    
    # Let's use a randomized local search / greedy approach.
    # 1. Start with a random pair (u, v) that has an edge.
    # 2. Try to add a third vertex w such that paths u-v, u-w, v-w exist and are disjoint.
    #    Since u-v is an edge, interior is empty.
    #    We need u-w and v-w to be disjoint from each other and from u-v (trivial).
    #    So we need two disjoint paths from w to {u, v}.
    # 3. Try to add a fourth vertex x.
    
    # Let's implement a function to check if a set of branches is valid.
    # Given a set of branches B, can we find disjoint paths for all pairs?
    # This is NP-hard in general, but for small n we can use backtracking or max-flow.
    
    # Max-Flow approach for a fixed set of branches:
    # This is actually a "Multi-Commodity Flow" problem which is hard.
    # However, if we fix the set of branches, we can try to find the paths.
    
    # Let's try a simpler heuristic:
    # Find a large set of vertices that are "well-connected".
    # Specifically, look for a "Fan" structure.
    
    # Let's try to find a set of vertices of size K such that there exists a "Steiner Tree" 
    # connecting them, and then derive paths? No, paths must be disjoint.
    
    # Let's stick to the "Clique" idea but maybe we missed a larger clique?
    # The seed checked all triples. It returned the first one found.
    # Did it check all quadruples? No.
    # Let's check for a clique of size 4.
    
    from itertools import combinations
    
    # Precompute adjacency matrix for O(1) lookup
    adj_mat = [[False]*n for _ in range(n)]
    for u, v in edges:
        adj_mat[u][v] = True
        adj_mat[v][u] = True
        
    # Find Max Clique
    max_clique_size = 0
    max_clique = []
    
    # Since n=24, checking all subsets of size 4 is C(24,4) = 10626. Very fast.
    # Checking size 5 is C(24,5) = 42504. Still fast.
    # Checking size 6 is C(24,6) = 134596. Fast.
    # Checking size 7 is C(24,7) = 346104. Fast.
    
    # Let's find the maximum clique size.
    # If we find a clique of size K, we can use the edges of the clique as the paths.
    # The interiors are empty, so they are trivially disjoint.
    
    for size in range(7, 2, -1): # Try large cliques first
        found = False
        for combo in combinations(range(n), size):
            if all(adj_mat[u][v] for u, v in combinations(combo, 2)):
                max_clique = list(combo)
                max_clique_size = size
                found = True
                break
        if found:
            break
            
    if max_clique_size >= 2:
        branches = max_clique
        paths = {}
        for u, v in combinations(branches, 2):
            paths[(u, v)] = [u, v]
            paths[(v, u)] = [v, u]
            
        # Convert to required format
        result_paths = []
        for u, v in combinations(branches, 2):
            result_paths.append([u, v])
            
        return {'branches': branches, 'paths': result_paths}
        
    # If no clique of size > 2 found (unlikely given seed found 3), fallback
    # But seed found 3, so we should find at least 3.
    # If we didn't find one, something is wrong with my clique logic or the graph changed?
    # The graph is static.
    
    # Wait, the seed found a triangle. My loop starts at 7 down to 3.
    # So it should find the triangle.
    
    # What if we can do BETTER than a clique?
    # Example: 4 branches where not all pairs are adjacent, but disjoint paths exist.
    # This allows for larger branch sets than the max clique.
    
    # Let's try to expand the clique.
    # Current best is a clique of size K.
    # Can we add a vertex v to the clique such that disjoint paths exist?
    
    # This is getting complicated. Let's assume the Max Clique is a good lower bound.
    # Is it possible to have more branches than the max clique?
    # Yes. Consider a "star" graph with center C and leaves L1, L2, L3.
    # Branches: {L1, L2, L3}. 
    # Paths: L1-L2 (via C), L1-L3 (via C), L2-L3 (via C).
    # Interiors: All use C. Not disjoint. So this fails.
    
    # Consider a "cycle" of length 4: 1-2-3-4-1.
    # Branches: {1, 3}. Paths: 1-2-3, 1-4-3. Disjoint interiors {2} and {4}. Valid. Size 2.
    # Branches: {1, 2, 3, 4}. 
    # Pairs: (1,2) edge, (2,3) edge, (3,4) edge, (4,1) edge.
    # (1,3) needs path 1-2-3 or 1-4-3.
    # (2,4) needs path 2-1-4 or 2-3-4.
    # If we pick 1-2-3 for (1,3), interior is {2}.
    # If we pick 2-1-4 for (2,4), interior is {1}.
    # But 1 is a branch! Interiors must avoid branches.
    # So 2-1-4 is invalid because 1 is a branch.
    # Path 2-3-4 uses interior {3}, which is a branch. Invalid.
    # So we can't have 4 branches on a 4-cycle.
    
    # It seems disjoint paths with interiors avoiding branches is very restrictive.
    # It often forces the branches to be a "dominating" set or similar.
    
    # Let's try a heuristic search for a set of branches larger than the max clique.
    # We will use a randomized greedy approach.
    
    best_val = max_clique_size
    best_branches = max_clique
    best_paths_dict = {}
    if best_branches:
        for u, v in combinations(best_branches, 2):
            best_paths_dict[(u, v)] = [u, v]
            best_paths_dict[(v, u)] = [v, u]

    # Try to find a better configuration by random sampling of branch sets
    # and checking validity via a simple backtracking path finder.
    
    def check_validity(branches):
        # Returns paths dict if valid, else None
        # We need to find paths for all pairs.
        # Order of finding paths matters.
        # Let's try to find paths in a specific order.
        
        pairs = list(combinations(branches, 2))
        # Randomize order to escape local minima?
        random.shuffle(pairs)
        
        # We will maintain a set of "used" internal vertices.
        used_interiors = set()
        found_paths = {}
        
        for u, v in pairs:
            # Find a path from u to v that avoids used_interiors and other branches
            # BFS/DFS
            target = v
            start = u
            forbidden = set(branches) - {start, target}
            forbidden.update(used_interiors)
            
            # BFS
            q = deque([(start, [start])])
            visited = {start}
            path = None
            
            while q:
                node, path_so_far = q.popleft()
                if node == target:
                    path = path_so_far
                    break
                for nb in adj[node]:
                    if nb in visited:
                        continue
                    if nb in forbidden:
                        continue
                    visited.add(nb)
                    q.append((nb, path_so_far + [nb]))
                    
            if not path:
                return None
            
            # Add interiors to used_interiors
            for node in path[1:-1]:
                used_interiors.add(node)
            
            found_paths[(u, v)] = path
            found_paths[(v, u)] = list(reversed(path))
            
        return found_paths

    # Try random subsets of vertices of size K+1, K+2...
    # Start from the max clique and try to add vertices?
    # Or just sample random sets.
    
    # Since n=24, we can sample many sets.
    # Let's try sizes from max_clique_size + 1 up to 8.
    
    for size in range(max_clique_size + 1, 9):
        if time.time() > deadline:
            break
            
        # Try to find a valid set of this size
        # Heuristic: Pick vertices with high degree
        candidates = vertices[:15] # Top 15 by degree
        
        # Random sampling
        found = False
        for _ in range(1000):
            if time.time() > deadline:
                break
            sample = random.sample(candidates, size)
            paths = check_validity(sample)
            if paths:
                best_branches = sample
                best_paths_dict = paths
                best_val = size
                found = True
                break
                
        if found:
            # Try to expand further?
            continue # Try larger sizes
            
    # Construct final result
    if best_branches:
        result_paths = []
        for u, v in combinations(best_branches, 2):
            result_paths.append(best_paths_dict[(u, v)])
        
        return {'branches': best_branches, 'paths': result_paths}
    else:
        # Fallback to first edge
        u, v = edges[0]
        return {'branches': [u, v], 'paths': [[u, v]]}

# EVOLVE-BLOCK-END
