"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Deadline guard: 60 seconds max, use 55s for safety
    deadline = time.time() + 55.0
    
    def find_vertex_disjoint_paths(branches, forbidden):
        """
        Find vertex-disjoint paths between all pairs of branch vertices.
        Paths must be simple, use only vertices not in 'forbidden' (except endpoints),
        and all path interiors must be disjoint.
        Returns dict mapping (u,v) to path list, or None if impossible.
        """
        pairs = list(combinations(branches, 2))
        if not pairs:
            return {}
        
        # We need to find one path for each pair, all vertex-disjoint in their interiors.
        # Strategy: iteratively find shortest paths, avoiding previously used interior vertices.
        # To handle conflicts, we try different orderings of pairs.
        
        best_result = None
        
        # Try a few different orderings of pairs
        import random
        random.seed(42)
        
        # First try the natural order
        orderings = [list(pairs)]
        # Add a few random orderings
        for _ in range(10):
            shuffled = list(pairs)
            random.shuffle(shuffled)
            orderings.append(shuffled)
        
        for ordering in orderings:
            if time.time() > deadline:
                break
            used_interiors = set()
            paths = {}
            success = True
            
            for (u, v) in ordering:
                # BFS to find shortest path from u to v, avoiding used_interiors and other branches
                forbidden_set = used_interiors | set(branches) - {u, v}
                # BFS
                visited = {u}
                parent = {u: None}
                queue = deque([u])
                found = False
                while queue:
                    curr = queue.popleft()
                    if curr == v:
                        found = True
                        break
                    for neighbor in adj[curr]:
                        if neighbor not in visited and neighbor not in forbidden_set:
                            visited.add(neighbor)
                            parent[neighbor] = curr
                            queue.append(neighbor)
                
                if not found:
                    success = False
                    break
                
                # Reconstruct path
                path = []
                curr = v
                while curr is not None:
                    path.append(curr)
                    curr = parent[curr]
                path.reverse()
                
                # Check that path is simple (it should be from BFS)
                # Mark interior vertices as used
                for vertex in path[1:-1]:
                    if vertex in used_interiors:
                        success = False
                        break
                    used_interiors.add(vertex)
                
                if not success:
                    break
                
                paths[(u, v)] = path
            
            if success:
                if best_result is None or len(paths) > len(best_result):
                    best_result = paths
                    if len(paths) == len(pairs):
                        break  # Found a complete solution
        
        return best_result
    
    def try_branch_set(branches):
        """Try to find valid paths for the given branch set. Returns result dict or None."""
        if len(branches) < 2:
            return None
        paths = find_vertex_disjoint_paths(branches, set())
        if paths is None or len(paths) != len(list(combinations(branches, 2))):
            return None
        # Build the result
        branch_list = list(branches)
        path_list = []
        for (u, v) in combinations(branch_list, 2):
            path_list.append(paths[(u, v)])
        return {'branches': branch_list, 'paths': path_list}
    
    # Start with a greedy approach to build a large branch set
    # First, find a maximal set of vertices that can serve as branches
    
    # Strategy: Start with all vertices, try to find a large subset that works.
    # Use a greedy heuristic: start with a small set and try to add vertices one by one.
    
    all_vertices = list(range(n))
    best_result = None
    best_size = 0
    
    # Try starting from different initial pairs
    # First, try to find a large clique or near-clique
    # Actually, branches don't need to form a clique in the original graph,
    # they just need vertex-disjoint paths between all pairs.
    
    # Greedy approach: start with a random vertex, try to add more
    import random
    random.seed(123)
    
    # Try multiple random starting points
    for start_vertex in all_vertices:
        if time.time() > deadline:
            break
        
        # Start with just this vertex, try to build up
        current_branches = [start_vertex]
        
        # Try to add vertices one by one
        for candidate in all_vertices:
            if candidate in current_branches:
                continue
            if time.time() > deadline:
                break
            
            new_branches = current_branches + [candidate]
            result = try_branch_set(new_branches)
            if result is not None:
                current_branches = new_branches
                if len(current_branches) > best_size:
                    best_size = len(current_branches)
                    best_result = result
    
    # Also try a different strategy: start with a large set and try to reduce
    # Actually, let's try a more systematic approach
    # Try all subsets of size k for k = 5, 4, 3, 2
    # But that's expensive. Let's try a smarter approach.
    
    # Try to find the largest possible set using a heuristic
    # Start with a random permutation and greedily add vertices
    
    for _ in range(20):
        if time.time() > deadline:
            break
        
        random.shuffle(all_vertices)
        current_branches = []
        for candidate in all_vertices:
            if time.time() > deadline:
                break
            new_branches = current_branches + [candidate]
            if len(new_branches) < 2:
                current_branches = new_branches
                continue
            result = try_branch_set(new_branches)
            if result is not None:
                current_branches = new_branches
                if len(current_branches) > best_size:
                    best_size = len(current_branches)
                    best_result = result
    
    # Fallback: just use two vertices with an edge between them
    if best_result is None:
        for u, v in edges:
            best_result = {'branches': [u, v], 'paths': [[u, v]]}
            break
    
    # Ensure types are correct
    if best_result is not None:
        best_result['branches'] = list(best_result['branches'])
        best_result['paths'] = [list(p) for p in best_result['paths']]
    
    return best_result
# EVOLVE-BLOCK-END
