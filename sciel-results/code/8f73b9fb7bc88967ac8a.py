import time
import random
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

DEADLINE = time.time() + 85.0

def construct():
    t0 = time.time()
    n = P['n']
    edges = P['edges']
    
    def time_up():
        return time.time() - t0 > DEADLINE - 3.0

    # Build adjacency
    adj = [set() for _ in range(n)]
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    adj_list = [list(s) for s in adj]
    
    def fallback():
        if not edges:
            return {'branches': [0, 1], 'paths': [[0, 1]]}
        u, v = edges[0]
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}

    # Max-flow with vertex splitting to check vertex-disjoint paths
    # For a set of branches B, we need |B|*(|B|-1)/2 vertex-disjoint paths
    # between all pairs. We model this as a flow problem.
    
    def check_disjoint_paths(branches):
        """
        Check if there exist pairwise internally disjoint paths for all pairs in branches.
        Returns the list of paths if feasible, else None.
        Uses max-flow with vertex capacities.
        """
        bs = set(branches)
        k = len(branches)
        if k < 2:
            return None
        
        pairs = list(combinations(sorted(branches), 2))
        num_pairs = len(pairs)
        
        # Vertex splitting: each vertex v becomes v_in -> v_out with capacity 1
        # For branch vertices, capacity is infinite (or large enough)
        # We need to route num_pairs units of flow, each pair (u,v) needs one unit
        # from u_out to v_in
        
        # Build flow network
        # Nodes: v_in = 2*v, v_out = 2*v+1
        # For non-branch vertices: v_in -> v_out cap 1
        # For branch vertices: v_in -> v_out cap INF
        # For each edge (u,v): u_out -> v_in cap INF, v_out -> u_in cap INF
        # Source: connect to each branch's out node? No...
        
        # Actually, we need to send flow from each branch to every other branch.
        # This is a multi-commodity flow problem, which is NP-hard in general.
        # But for vertex-disjoint paths, we can use a different approach.
        
        # Alternative: Use the fact that we need internally disjoint paths.
        # We can try to find paths greedily with backtracking, using the flow check
        # as a pruning mechanism.
        
        # For now, let's use a simpler approach: try to find paths with backtracking
        # and use a heuristic ordering.
        
        # Sort pairs by tightness (fewest common neighbors first)
        def pair_tightness(u, v):
            common = (adj[u] & adj[v]) - bs
            return len(common)
        
        pairs_sorted = sorted(pairs, key=lambda p: pair_tightness(p[0], p[1]))
        
        def find_shortest_path(s, t, blocked):
            """Find shortest path from s to t avoiding blocked vertices (except s,t)."""
            if s == t:
                return [s]
            parent = {s: None}
            q = deque([s])
            while q:
                curr = q.popleft()
                for nb in adj_list[curr]:
                    if nb in blocked and nb != s and nb != t:
                        continue
                    if nb in parent:
                        continue
                    parent[nb] = curr
                    if nb == t:
                        path = []
                        node = t
                        while node is not None:
                            path.append(node)
                            node = parent[node]
                        path.reverse()
                        return path
                    q.append(nb)
            return None
        
        def solve(idx, used_internal, chosen):
            if time_up():
                return False
            if idx == len(pairs_sorted):
                return True
            
            u, v = pairs_sorted[idx]
            blocked = used_internal | bs
            
            # Try to find a path
            path = find_shortest_path(u, v, blocked)
            if path is None:
                return False
            
            internal = set(path[1:-1])
            if internal & used_internal or internal & bs:
                return False
            
            chosen[idx] = path
            for node in internal:
                used_internal.add(node)
            
            if solve(idx + 1, used_internal, chosen):
                return True
            
            for node in internal:
                used_internal.discard(node)
            chosen[idx] = None
            return False
        
        chosen = [None] * len(pairs_sorted)
        if solve(0, set(), chosen):
            # Map back to standard pair order
            std_pairs = list(combinations(sorted(branches), 2))
            path_dict = {}
            for i, pair in enumerate(pairs_sorted):
                path_dict[pair] = chosen[i]
            result = []
            for p in std_pairs:
                result.append(path_dict[p])
            return result
        return None

    # Bron-Kerbosch for max cliques
    def bron_kerbosch(R, P_set, X):
        if not P_set and not X:
            yield list(R)
            return
        pivot_candidates = P_set | X
        if pivot_candidates:
            pivot = max(pivot_candidates, key=lambda v: len(adj[v] & P_set))
            candidates = P_set - adj[pivot]
        else:
            candidates = P_set
            
        for v in list(candidates):
            yield from bron_kerbosch(R | {v}, P_set & adj[v], X & adj[v])
            P_set.remove(v)
            X.add(v)

    # Get all cliques
    all_cliques = []
    for clique in bron_kerbosch(set(), set(range(n)), set()):
        all_cliques.append(clique)
    all_cliques.sort(key=len, reverse=True)
    
    # Generate candidate sets: cliques + greedy + random
    deg_sorted = sorted(range(n), key=lambda x: len(adj[x]), reverse=True)
    
    candidate_sets = []
    seen_sets = set()
    
    # Add cliques
    for c in all_cliques:
        key = frozenset(c)
        if key not in seen_sets:
            seen_sets.add(key)
            candidate_sets.append(c)
    
    # Greedy: start from high-degree nodes, add well-connected vertices
    for start in deg_sorted[:8]:
        current_set = {start}
        for v in deg_sorted:
            if v in current_set:
                continue
            connections = sum(1 for u in current_set if v in adj[u])
            if connections >= max(1, len(current_set) // 2):
                new_set = current_set | {v}
                key = frozenset(new_set)
                if key not in seen_sets:
                    seen_sets.add(key)
                    candidate_sets.append(list(new_set))
                current_set = new_set
                if len(current_set) >= 8:
                    break
    
    # Random subsets from top nodes
    rng = random.Random(42)
    top_nodes = deg_sorted[:15]
    for _ in range(30):
        if time_up():
            break
        k = rng.randint(3, 7)
        subset = rng.sample(top_nodes, min(k, len(top_nodes)))
        key = frozenset(subset)
        if key not in seen_sets:
            seen_sets.add(key)
            candidate_sets.append(subset)
    
    # Sort by size descending
    candidate_sets.sort(key=len, reverse=True)
    
    best = None
    best_size = 0
    
    if edges:
        u, v = edges[0]
        best = {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
        best_size = 2

    for c in candidate_sets:
        if time_up():
            break
        if len(c) <= best_size:
            continue
        
        paths = check_disjoint_paths(c)
        if paths is not None:
            best = {'branches': [int(x) for x in sorted(c)], 'paths': [[int(x) for x in p] for p in paths]}
            best_size = len(c)
            break  # Sorted by size desc, so first valid is best

    if best is None:
        return fallback()
    return best
