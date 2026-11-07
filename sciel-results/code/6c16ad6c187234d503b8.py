"""Rewritten program with a stronger clique-subdivision solver."""
# EVOLVE-BLOCK-START
import time
import itertools
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}


def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s kill

    n = P['n']
    edges = P['edges']

    # Build adjacency sets
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    # Helper: find a simple path between u and v avoiding forbidden vertices (BFS)
    def find_path_avoiding(u, v, forbidden):
        if u == v:
            return [u]
        if v in forbidden or u in forbidden:
            return None
        queue = deque()
        queue.append((u, [u]))
        visited = {u}
        while queue:
            node, path = queue.popleft()
            if node == v:
                return path
            for neighbor in adj[node]:
                if neighbor not in visited and neighbor not in forbidden:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None

    # Helper: Validate a full clique subdivision solution
    def validate_solution(branches, paths_dict):
        if len(branches) != len(set(branches)):
            return False
        for i in range(len(branches)):
            for j in range(i + 1, len(branches)):
                pair = (branches[i], branches[j])
                if pair not in paths_dict and (branches[j], branches[i]) not in paths_dict:
                    return False
        all_interiors = set()
        for pair, path in paths_dict.items():
            u, v = pair
            if path[0] != u and path[0] != v:
                return False
            if path[-1] != u and path[-1] != v:
                return False
            if len(path) != len(set(path)):
                return False
            for k in range(len(path) - 1):
                if path[k + 1] not in adj[path[k]]:
                    return False
            interiors = set(path[1:-1])
            for b in branches:
                if b in interiors:
                    return False
            if all_interiors & interiors:
                return False
            all_interiors.update(interiors)
        return True

    # =====================================================================
    # STRATEGY:
    # 1. Find maximum clique. If size >= 5, return it.
    # 2. Search for K5 subdivisions using backtracking with pruning.
    # 3. Search for K6 subdivisions (less likely, but try if time permits).
    # =====================================================================

    # Step 1: Find max clique
    def find_max_clique():
        best_clique = []
        def branch(candidates, current_clique):
            nonlocal best_clique
            if time.time() - start_time > DEADLINE - 5:
                return
            if not candidates:
                if len(current_clique) > len(best_clique):
                    best_clique = list(current_clique)
                return
            if len(current_clique) + len(candidates) <= len(best_clique):
                return
            for i, v in enumerate(candidates):
                if all(v in adj[c] for c in current_clique):
                    current_clique.append(v)
                    new_candidates = [candidates[j] for j in range(i + 1, len(candidates)) if v in adj[candidates[j]]]
                    branch(new_candidates, current_clique)
                    current_clique.pop()
        branch(list(range(n)), [])
        return best_clique

    max_clique = find_max_clique()
    if len(max_clique) >= 5:
        branches = list(max_clique)
        paths_list = [list(pair) for pair in itertools.combinations(branches, 2)]
        return {'branches': branches, 'paths': paths_list}

    # Step 2: Search for K5 subdivisions
    # We will try all 5-tuples, but filter aggressively.
    # A 5-tuple is a candidate if it has at least 7 edges (so at most 3 paths of length >= 2 are needed).
    # Actually, even with 5 edges, it's possible. Let's start with high edge counts.
    
    def try_find_subdivision_k5(vertices):
        """
        Try to find internally disjoint paths for all pairs in vertices (size 5).
        Returns dict of paths or None.
        """
        branch_list = list(vertices)
        k = len(branch_list)
        branch_set = set(branch_list)
        pairs = list(itertools.combinations(branch_list, 2))
        
        # Pre-compute "direct" edges vs "needs path"
        # We will assign paths one by one.
        # Heuristic: Assign pairs that are NOT edges first? Or edges first?
        # Edges use no internal vertices, so they are "free".
        # Non-edges use internal vertices, so they are "costly".
        # It is generally better to fix the "costly" ones first to constrain the space,
        # OR fix the "free" ones first to simplify the graph?
        # Actually, fixing free ones first doesn't constrain the graph at all (interiors are empty).
        # So we should focus on the non-edges.
        
        # Let's sort pairs by:
        # 1. Are they an edge? (If yes, path is fixed to [u, v], cost 0)
        # 2. If not, how many common neighbors do they have? (More options = less constrained? 
        #    Actually, fewer options = more constrained = should be picked first for backtracking efficiency)
        
        # Let's define a "constriction" score for a pair (u, v) given current used vertices.
        # But used vertices change. So we do dynamic ordering.
        
        assignment = {}
        used_interiors = set()
        
        # Initialize with all pairs
        remaining_pairs = set(pairs)
        
        def backtrack():
            if time.time() - start_time > DEADLINE - 2:
                return False
            
            if not remaining_pairs:
                return True
            
            # Select the most constrained pair
            # Constraint: number of available disjoint paths
            # To estimate, we count how many simple paths of length <= 3 exist avoiding used_interiors.
            # This is expensive. Let's use a simpler heuristic:
            # Pick the pair (u, v) that minimizes the number of common neighbors 
            # (excluding used_interiors and branches).
            
            best_pair = None
            min_score = float('inf')
            
            for u, v in remaining_pairs:
                if u in assignment or v in assignment: # This logic is wrong, assignment is per pair
                    pass
                
                # Count available neighbors for path finding
                # A path of length 2 exists if there is a common neighbor w not in used_interiors and not in branch_set
                common_neighs = (adj[u] & adj[v]) - used_interiors - branch_set
                # A path of length 1 exists if u-v is an edge (always true if in graph, but we check used? No, edges don't use interiors)
                # If u-v is an edge, the path is [u, v]. Interior is empty.
                if u in adj[v]:
                    score = 0 # Very easy
                else:
                    score = len(common_neighs)
                
                if score < min_score:
                    min_score = score
                    best_pair = (u, v)
            
            if best_pair is None:
                return False
                
            u, v = best_pair
            remaining_pairs.discard(best_pair)
            
            # Try to find a path for u, v
            # If u, v is an edge, path is [u, v]
            if u in adj[v]:
                path = [u, v]
                assignment[best_pair] = path
                if backtrack():
                    return True
                else:
                    del assignment[best_pair]
                remaining_pairs.add(best_pair)
                return False
            
            # Else, find a path avoiding used_interiors and branch_set
            # We want the shortest path to minimize resource usage.
            # BFS is good for this.
            path = find_path_avoiding(u, v, used_interiors | branch_set)
            if path is None:
                remaining_pairs.add(best_pair)
                return False
            
            # The path might not be unique. But for now, take the shortest one found by BFS.
            # If this leads to a dead end, we might need to try other paths.
            # For efficiency, we only try the shortest path first. If it fails, we might need to backtrack deeper.
            # To handle this, we could generate multiple candidate paths.
            # Let's stick to the shortest path for now, as it uses the fewest vertices.
            
            interiors = set(path[1:-1])
            # Check if interiors are disjoint from used_interiors (should be by construction of find_path_avoiding)
            # And disjoint from branch_set (also by construction)
            
            assignment[best_pair] = path
            used_interiors.update(interiors)
            
            if backtrack():
                return True
            else:
                del assignment[best_pair]
                used_interiors.difference_update(interiors)
                remaining_pairs.add(best_pair)
                return False
        
        if backtrack():
            return assignment
        return None

    # Iterate through 5-tuples
    # Filter: At least 6 edges? (10 pairs total. 6 edges, 4 paths needed)
    # Let's try all 5-tuples with >= 5 edges.
    for comb in itertools.combinations(range(n), 5):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        # Count edges
        ec = 0
        for i in range(5):
            for j in range(i+1, 5):
                if comb[j] in adj[comb[i]]:
                    ec += 1
        
        if ec < 5:
            continue
            
        result = try_find_subdivision_k5(set(comb))
        if result is not None:
            branches = list(comb)
            paths_list = []
            for pair in itertools.combinations(branches, 2):
                p = result.get(pair) or result.get((pair[1], pair[0]))
                if p is None:
                    result = None
                    break
                paths_list.append(p)
            if result is not None and validate_solution(branches, result):
                return {'branches': branches, 'paths': paths_list}

    # Step 3: Fallback to max clique found (size 4 or less)
    if len(max_clique) >= 2:
        branches = list(max_clique)
        paths_list = [list(pair) for pair in itertools.combinations(branches, 2)]
        return {'branches': branches, 'paths': paths_list}
    
    # Final fallback
    if edges:
        u, v = edges[0]
        return {'branches': [u, v], 'paths': [[u, v]]}
    else:
        return {'branches': [0, 1], 'paths': [[0, 1]]}

# EVOLVE-BLOCK-END
