"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations
from collections import deque, defaultdict

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    DEADLINE = 85.0

    n = P['n']
    edges = P['edges']

    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    def time_up():
        return time.time() - start_time > DEADLINE

    def is_clique(vertices):
        for u, v in combinations(vertices, 2):
            if v not in adj[u]:
                return False
        return True

    def find_max_clique():
        best_clique = []
        def bronk(R, P_set, X):
            nonlocal best_clique
            if time_up():
                return
            if not P_set and not X:
                if len(R) > len(best_clique):
                    best_clique = list(R)
                return
            union = P_set | X
            if not union:
                return
            pivot = max(union, key=lambda v: len(adj[v] & P_set))
            candidates = P_set - adj[pivot]
            for v in list(candidates):
                bronk(R | {v}, P_set & adj[v], X & adj[v])
                P_set.remove(v)
                X.add(v)
        bronk(set(), set(range(n)), set())
        return best_clique

    def find_disjoint_paths_flow(branches):
        """Use max-flow with vertex splitting to find vertex-disjoint paths
        for all pairs of branch vertices simultaneously.
        
        Returns dict mapping (u,v) -> path, or None if impossible.
        """
        branches = sorted(branches)
        k = len(branches)
        if k < 2:
            return None
        branch_set = set(branches)
        pairs = list(combinations(branches, 2))
        num_pairs = len(pairs)
        
        # Vertex splitting: each vertex v becomes v_in -> v_out
        # Branch vertices have capacity infinity (they can be used by multiple paths as endpoints)
        # Non-branch vertices have capacity 1 (used by at most one path as interior)
        # 
        # We build a flow network where:
        # - Source S connects to each branch's in-node
        # - Each branch's out-node connects to sink T
        # - Actually, we need multiple sources/sinks. 
        # 
        # Better approach: For each pair (u,v), we need a path from u to v.
        # We can model this as: create a super-source S and super-sink T.
        # For each pair i=(a_i, b_i), add edge S -> a_i_in with cap 1, 
        # and b_i_out -> T with cap 1.
        # Interior edges: for each original edge (x,y), add x_out -> y_in with cap 1
        # and y_out -> x_in with cap 1.
        # Vertex capacity: x_in -> x_out with cap 1 if x not in branch_set, 
        # or cap infinity if x in branch_set.
        #
        # Wait, this doesn't quite work because a branch vertex can be an endpoint
        # for multiple paths. Let me think again.
        #
        # Actually, the constraint is:
        # - Path interiors are pairwise disjoint
        # - Path interiors avoid all branch vertices
        # - Each path starts at one branch and ends at another
        #
        # So branch vertices can only appear as endpoints, not as interior vertices.
        # This means:
        # - For non-branch vertex v: capacity 1 (can be in at most one path interior)
        # - For branch vertex v: it appears as endpoint for (k-1) paths. 
        #   But it cannot be an interior vertex of any path.
        #
        # So we can model:
        # - Split each vertex v into v_in and v_out
        # - Edge v_in -> v_out: capacity 1 for non-branches, capacity k-1 for branches
        #   (but branches can't be interiors, so actually we should prevent flow
        #    from passing through branch vertices as interiors)
        #
        # Hmm, this is tricky. Let me use a different approach.
        #
        # Alternative: Since branch vertices cannot be interiors, we can simply
        # remove all edges incident to branch vertices EXCEPT those that are
        # part of the path endpoints. But that's complex.
        #
        # Simpler approach: Just use the greedy BFS but with a smarter ordering
        # and backtracking. Let me try a different structural approach.
        #
        # Actually, let me try: for each pair, find the shortest path avoiding
        # used interiors AND all branch vertices except the two endpoints.
        # The key insight is that we should order pairs so that "constrained" 
        # pairs (those with few options) go first.
        
        return None  # Placeholder, will use greedy below

    def find_path_bfs(s, t, blocked):
        """Find shortest simple path from s to t avoiding blocked vertices."""
        if s == t:
            return [s]
        if s in blocked or t in blocked:
            return None
        visited = {s}
        queue = deque([[s]])
        while queue:
            path = queue.popleft()
            u = path[-1]
            for v in sorted(adj[u]):
                if v == t:
                    return path + [v]
                if v not in visited and v not in blocked:
                    visited.add(v)
                    queue.append(path + [v])
        return None

    def verify_subdivision(branches):
        """Try to find vertex-disjoint paths for all pairs using greedy with
        multiple orderings and backtracking-lite."""
        branches = sorted(branches)
        k = len(branches)
        if k < 2:
            return None
        branch_set = set(branches)
        pairs = list(combinations(branches, 2))
        
        # Precompute for each pair: number of common neighbors not in branch_set
        pair_info = {}
        for pair in pairs:
            u, v = pair
            common = adj[u] & adj[v] - branch_set
            pair_info[pair] = len(common)
        
        # Generate multiple orderings
        orderings = []
        # 1. Ascending difficulty
        orderings.append(sorted(pairs, key=lambda p: pair_info[p]))
        # 2. Descending difficulty
        orderings.append(sorted(pairs, key=lambda p: -pair_info[p]))
        # 3. Original order
        orderings.append(list(pairs))
        # 4. Reverse
        orderings.append(list(reversed(pairs)))
        # 5. Sort by min degree of endpoints
        orderings.append(sorted(pairs, key=lambda p: min(len(adj[p[0]]), len(adj[p[1]]))))
        # 6. Sort by max degree of endpoints
        orderings.append(sorted(pairs, key=lambda p: -max(len(adj[p[0]]), len(adj[p[1]]))))
        
        best_result = None
        for ordering in orderings:
            if time_up():
                break
            used_interior = set()
            paths_found = {}
            success = True
            for pair in ordering:
                u, v = pair
                blocked = (branch_set - {u, v}) | used_interior
                path = find_path_bfs(u, v, blocked)
                if path is None:
                    success = False
                    break
                for vert in path[1:-1]:
                    used_interior.add(vert)
                paths_found[pair] = path
            if success:
                return paths_found
        return None

    def greedy_search():
        """Greedy incremental search for maximum branch set."""
        max_clique = find_max_clique()
        if not max_clique:
            return None
        
        best = None
        best_size = 0
        
        # Try max clique first
        paths = verify_subdivision(max_clique)
        if paths:
            best = (list(max_clique), paths)
            best_size = len(max_clique)
        
        # Try adding vertices
        vertices_by_degree = sorted(range(n), key=lambda v: -len(adj[v]))
        current = list(max_clique)
        current_set = set(current)
        
        for v in vertices_by_degree:
            if time_up():
                break
            if v in current_set:
                continue
            new_set = current + [v]
            paths = verify_subdivision(new_set)
            if paths:
                current = new_set
                current_set.add(v)
                if len(current) > best_size:
                    best = (list(current), paths)
                    best_size = len(current)
        
        return best

    def local_search():
        """Start from greedy solution, try swaps to improve."""
        initial = greedy_search()
        if not initial:
            return None
        
        best_branches = list(initial[0])
        best_paths = initial[1]
        best_size = len(best_branches)
        
        improved = True
        while improved and not time_up():
            improved = False
            current_set = set(best_branches)
            
            for v in range(n):
                if time_up():
                    break
                if v in current_set:
                    continue
                # Try adding v
                new_set = list(current_set) + [v]
                paths = verify_subdivision(new_set)
                if paths and len(new_set) > best_size:
                    best_branches = new_set
                    best_paths = paths
                    best_size = len(new_set)
                    improved = True
                    current_set = set(best_branches)
        
        return (best_branches, best_paths)

    def try_all_k_sets():
        """Try all k-combinations for various k."""
        best = None
        best_size = 0
        
        for k in range(2, min(n+1, 10)):
            if time_up():
                break
            for combo in combinations(range(n), k):
                if time_up():
                    break
                paths = verify_subdivision(combo)
                if paths and k > best_size:
                    best = (list(combo), paths)
                    best_size = k
        
        return best

    # Main search
    result = local_search()
    
    if result is None:
        # Fallback
        u, v = edges[0]
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
    
    branches, paths = result
    branches_list = [int(b) for b in branches]
    paths_ordered = []
    for pair in combinations(branches_list, 2):
        key = (min(pair), max(pair))
        if key in paths:
            paths_ordered.append([int(v) for v in paths[key]])
        else:
            paths_ordered.append(list(pair))
    
    return {'branches': branches_list, 'paths': paths_ordered}

# EVOLVE-BLOCK-END
