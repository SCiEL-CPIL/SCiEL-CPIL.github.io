"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 75  # 75s deadline, 15s margin
    
    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Build adjacency list as sorted lists for faster iteration
    adj_list = [sorted(adj[i]) for i in range(n)]
    
    # Find maximal cliques using Bron-Kerbosch
    def bron_kerbosch(R, P_set, X, cliques):
        if time.time() > deadline:
            return
        if not P_set and not X:
            cliques.append(list(R))
            return
        # Pivot selection
        if P_set or X:
            pivot = max(P_set | X, key=lambda v: len(adj[v] & P_set))
            candidates = P_set - adj[pivot]
        else:
            candidates = P_set
        for v in list(candidates):
            bron_kerbosch(R | {v}, P_set & adj[v], X & adj[v], cliques)
            P_set.remove(v)
            X.add(v)
    
    cliques = []
    bron_kerbosch(set(), set(range(n)), set(), cliques)
    cliques.sort(key=len, reverse=True)
    
    best_k = 0
    best_branches = None
    best_paths = None
    
    # Flow-based oracle: check if a set of branch vertices admits a valid clique subdivision
    def flow_check(branches, adj_list, n):
        """Use max-flow to check if branches form a valid clique subdivision.
        Returns (valid, paths) or (False, None)."""
        if time.time() > deadline:
            return False, None
        
        k = len(branches)
        if k < 2:
            return False, None
        
        branch_set = set(branches)
        num_interior = n - k
        if num_interior < 0:
            return False, None
        
        # For k branches, we need k*(k-1)/2 internally vertex-disjoint paths
        # Each interior vertex can be used in at most one path
        # Total interior vertices needed: sum over all pairs of (path_length - 1)
        # Minimum: k*(k-1)/2 interior vertices (one per pair if all paths are length 2)
        # But actually paths can be longer, so we need at least k*(k-1)/2 interior vertices
        # Wait - actually each path from u to v needs at least 1 interior vertex if u,v not adjacent
        # If u,v are adjacent, path can be [u,v] with 0 interior vertices
        
        # Let's count how many pairs are NOT adjacent
        non_adj_pairs = 0
        for i in range(k):
            for j in range(i+1, k):
                if branches[j] not in adj[branches[i]]:
                    non_adj_pairs += 1
        
        # We need at least non_adj_pairs interior vertices
        if num_interior < non_adj_pairs:
            return False, None
        
        # Build flow network:
        # Source -> each branch (capacity 1 per edge to pair)
        # Actually, let's model it differently:
        # Each pair (u,v) needs a path from u to v
        # Interior vertices have capacity 1 (can only be in one path)
        # 
        # Network:
        # - Source S
        # - For each branch b: node b_in, b_out (edge b_in->b_out cap k-1)
        # - For each interior vertex w: node w_in, w_out (edge w_in->w_out cap 1)
        # - For each pair (u,v): we need flow from u to v
        # - Edges: for each graph edge (a,b), add a_out -> b_in with cap 1
        #
        # Actually this is complex. Let me use a simpler approach:
        # Multi-commodity flow is NP-hard, but for this small graph we can try:
        # 1. For each pair, find shortest path avoiding other branches
        # 2. Check if all paths are internally vertex-disjoint
        # 3. If not, try alternative paths
        
        # Simpler approach: greedy path assignment with backtracking
        pairs = list(combinations(sorted(branches), 2))
        
        # Sort pairs by number of available interior vertices (hardest first)
        def pair_difficulty(pair):
            u, v = pair
            count = 0
            for w in range(n):
                if w not in branch_set and w in adj[u] and w in adj[v]:
                    count += 1
            return count
        
        pairs_sorted = sorted(pairs, key=pair_difficulty)
        
        # Try to assign paths greedily
        used_interior = set()
        paths = {}
        
        for u, v in pairs_sorted:
            if u in adj[v]:
                # Direct edge, no interior needed
                paths[(u, v)] = [u, v]
                continue
            
            # Find a path from u to v avoiding branch_set-{u,v} and used_interior
            path = find_path_greedy(u, v, branch_set - {u, v}, used_interior, adj_list, n)
            if path is None:
                return False, None
            paths[(u, v)] = path
            for w in path[1:-1]:
                used_interior.add(w)
        
        # Build result
        result_paths = []
        for u, v in pairs:
            result_paths.append(paths[(u, v)])
        
        return True, result_paths
    
    def find_path_greedy(u, v, forbidden, used, adj_list, n):
        """Find a simple path from u to v avoiding forbidden and used vertices."""
        # BFS with preference for shorter paths
        queue = deque()
        queue.append((u, [u]))
        visited = {u}
        
        while queue:
            node, path = queue.popleft()
            if node == v:
                return path
            for neighbor in adj_list[node]:
                if neighbor in visited:
                    continue
                if neighbor in forbidden:
                    continue
                if neighbor in used:
                    continue
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
        
        return None
    
    # Try each maximal clique
    for clique in cliques:
        if time.time() > deadline:
            break
        k = len(clique)
        if k < 2:
            continue
        
        # A clique is always valid (all paths are length 1)
        branches = sorted(clique)
        paths = []
        for i in range(k):
            for j in range(i+1, k):
                paths.append([branches[i], branches[j]])
        
        if k > best_k:
            best_k = k
            best_branches = branches
            best_paths = paths
    
    # Try to extend cliques by adding vertices
    # For each of the top cliques, try adding vertices one at a time
    for clique in cliques[:20]:
        if time.time() > deadline:
            break
        base = set(clique)
        current_k = len(base)
        
        # Try adding each non-clique vertex
        for v in range(n):
            if time.time() > deadline:
                break
            if v in base:
                continue
            
            candidate = base | {v}
            k = len(candidate)
            if k <= best_k:
                continue
            
            valid, paths = flow_check(sorted(candidate), adj_list, n)
            if valid:
                best_k = k
                best_branches = sorted(candidate)
                best_paths = paths
                
                # Try to extend further from this new base
                new_base = set(candidate)
                for w in range(n):
                    if time.time() > deadline:
                        break
                    if w in new_base:
                        continue
                    candidate2 = new_base | {w}
                    k2 = len(candidate2)
                    if k2 <= best_k:
                        continue
                    valid2, paths2 = flow_check(sorted(candidate2), adj_list, n)
                    if valid2:
                        best_k = k2
                        best_branches = sorted(candidate2)
                        best_paths = paths2
    
    # Also try greedy construction from scratch
    # Start with the largest clique and try to add vertices
    if cliques and len(cliques[0]) >= 2:
        if time.time() < deadline - 5:
            current = set(cliques[0])
            improved = True
            while improved and time.time() < deadline - 2:
                improved = False
                for v in range(n):
                    if time.time() > deadline:
                        break
                    if v in current:
                        continue
                    candidate = current | {v}
                    k = len(candidate)
                    if k <= best_k:
                        continue
                    valid, paths = flow_check(sorted(candidate), adj_list, n)
                    if valid:
                        current = candidate
                        best_k = k
                        best_branches = sorted(current)
                        best_paths = paths
                        improved = True
                        break
    
    # Fallback
    if best_branches is None or len(best_branches) < 2:
        u, v = edges[0]
        best_branches = [u, v]
        best_paths = [[u, v]]
    
    return {'branches': list(best_branches), 'paths': [list(p) for p in best_paths]}
# EVOLVE-BLOCK-END
