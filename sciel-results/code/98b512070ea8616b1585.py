"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import deque
from itertools import combinations

sys.setrecursionlimit(10000)

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    deg = {i: len(adj[i]) for i in range(n)}
    
    start_time = time.time()
    deadline = start_time + 55.0
    
    def max_disjoint_paths(src, targets, blocked_verts):
        """
        Find internally vertex-disjoint paths from src to each target in targets.
        blocked_verts: vertices that cannot be used as interiors.
        Returns dict target->path or None.
        
        Uses flow: split each vertex into in/out with capacity 1 for interiors.
        """
        k = len(targets)
        if k == 0:
            return {}
        
        # Build flow network
        # Each vertex v has nodes v_in, v_out with edge v_in->v_out capacity 1
        # Except src and targets which have infinite capacity on their split edge
        # For each graph edge (u,v), add u_out->v_in and v_out->u_in with capacity 1
        # Source is src_out, sinks are target_in for each target
        
        # Map vertices to flow node ids
        # v_in = 2*v, v_out = 2*v+1
        # Total nodes: 2*n + 2 (super source, super sink)
        SS = 2*n
        TT = 2*n + 1
        num_nodes = 2*n + 2
        
        # Adjacency for flow network
        flow_adj = {i: [] for i in range(num_nodes)}
        flow_cap = {}
        
        def add_edge(u, v, cap):
            flow_adj[u].append(v)
            flow_cap[(u, v)] = cap
            flow_adj[v].append(u)
            flow_cap[(v, u)] = 0
        
        # Split edges
        for v in range(n):
            if v == src or v in targets:
                add_edge(2*v, 2*v+1, 10**9)  # effectively infinite
            else:
                add_edge(2*v, 2*v+1, 1)
        
        # Graph edges
        for u in range(n):
            for v in adj[u]:
                if u < v:  # avoid duplicates
                    # u_out -> v_in
                    add_edge(2*u+1, 2*v, 1)
                    # v_out -> u_in
                    add_edge(2*v+1, 2*u, 1)
        
        # Connect super source to src_out
        add_edge(SS, 2*src+1, k)
        
        # Connect target_in to super sink
        for t in targets:
            add_edge(2*t, TT, 1)
        
        # Run max flow using Edmonds-Karp (BFS)
        # For small graphs this is fine
        flow = 0
        while True:
            # BFS to find augmenting path
            parent = [-1] * num_nodes
            parent[SS] = SS
            q = deque([SS])
            found = False
            while q and not found:
                u = q.popleft()
                for v in flow_adj[u]:
                    if parent[v] == -1 and flow_cap[(u, v)] > 0:
                        parent[v] = u
                        if v == TT:
                            found = True
                            break
                        q.append(v)
            if not found:
                break
            # Find bottleneck
            min_cap = float('inf')
            v = TT
            while v != SS:
                u = parent[v]
                min_cap = min(min_cap, flow_cap[(u, v)])
                v = u
            # Update capacities
            v = TT
            while v != SS:
                u = parent[v]
                flow_cap[(u, v)] -= min_cap
                flow_cap[(v, u)] += min_cap
                v = u
            flow += min_cap
        
        if flow < k:
            return None
        
        # Extract paths from flow
        # For each target, trace back from target_in through the flow
        paths = {}
        # We need to trace the actual paths. The flow gives us which edges carry flow.
        # For each target t, there's 1 unit of flow from SS to t_in to TT.
        # Trace: SS -> src_out -> ... -> t_in -> TT
        
        # Build residual graph edges with positive flow
        # Actually, let's trace by following the flow
        
        for t in targets:
            # Trace from t_in backwards to src_out
            path_nodes = [t]
            curr = 2*t  # t_in
            # We need to find where the flow came from
            # The flow into t_in came from some u_out where flow_cap was reduced
            # Let's trace using the parent pointers from the last augmentation? No, that's complex.
            
            # Alternative: reconstruct from the flow values
            # For each vertex, the flow through it is determined by the flow on its split edge
            # Let's just do a DFS from src following edges with positive flow
            
            # Simpler: for each target, do a DFS in the flow network
            # from src_out to t_in following edges with flow > 0
            
            # Reset: we'll trace by following the flow
            # Build a mapping of which edges have flow
            # An edge (u,v) has flow = original_cap - current_cap for forward edges
            
            # This is getting complex. Let me use a different extraction method.
            # For each target, find a path in the original graph that is consistent with the flow.
            
            pass
        
        # Simpler extraction: for each target, do BFS in original graph
        # but only using edges that have flow > 0 in the flow network
        # Actually, let me just return success/failure and use a different method to get paths
        
        return True  # indicates success, we'll extract paths separately
    
    # Given the complexity of flow extraction, let me use a simpler but effective approach:
    # Incremental construction with careful path finding
    
    def find_all_paths(new_v, branches, used_interiors):
        """
        Find internally disjoint paths from new_v to each branch.
        Returns dict branch->path or None.
        """
        targets = list(branches)
        k = len(targets)
        
        # Use backtracking with multiple path orderings
        # For each ordering of targets, greedily find paths
        
        import random
        random.seed(hash((new_v, tuple(branches))) % 10000)
        
        # Try several orderings
        orderings = [list(targets)]
        for _ in range(5):
            p = list(targets)
            random.shuffle(p)
            if p not in orderings:
                orderings.append(p)
        
        for ordering in orderings:
            current_blocked = set(used_interiors) | set(branches)
            current_blocked.discard(new_v)
            found = {}
            success = True
            for t in ordering:
                t_blocked = current_blocked - {t}
                # BFS for shortest path
                path = _bfs_path(new_v, t, t_blocked)
                if path is None:
                    success = False
                    break
                found[t] = path
                for v in path[1:-1]:
                    current_blocked.add(v)
            if success:
                return found
        
        return None
    
    def _bfs_path(src, dst, blocked):
        if src == dst:
            return [src]
        q = deque()
        q.append((src, [src]))
        visited = {src}
        while q:
            node, path = q.popleft()
            for nb in adj[node]:
                if nb == dst:
                    return path + [nb]
                if nb in visited or nb in blocked:
                    continue
                visited.add(nb)
                q.append((nb, path + [nb]))
        return None
    
    # Main: try to build branch sets from different starting vertices
    sorted_verts = sorted(range(n), key=lambda x: deg[x], reverse=True)
    
    best_branches = []
    best_paths_dict = {}
    best_used = set()
    
    for start_idx in range(min(6, len(sorted_verts))):
        if time.time() > deadline:
            break
        
        sv = sorted_verts[start_idx]
        branches = [sv]
        used_interiors = set()
        paths_dict = {}
        
        for candidate in sorted_verts:
            if candidate == sv:
                continue
            if time.time() > deadline:
                break
            
            new_paths = find_all_paths(candidate, branches, used_interiors)
            if new_paths is not None:
                branches.append(candidate)
                for t, p in new_paths.items():
                    paths_dict[(candidate, t)] = p
                    for v in p[1:-1]:
                        used_interiors.add(v)
        
        if len(branches) > len(best_branches):
            best_branches = list(branches)
            best_paths_dict = dict(paths_dict)
            best_used = set(used_interiors)
    
    # Fallback
    if len(best_branches) < 2:
        es = {tuple(e) for e in edges}
        for row in combinations(range(n), 3):
            if all(e in es for e in combinations(row, 2)):
                best_branches = list(row)
                best_paths_dict = {}
                for a, b in combinations(row, 2):
                    best_paths_dict[(a, b)] = [a, b]
                break
        else:
            u, v = edges[0]
            best_branches = [u, v]
            best_paths_dict = {(u, v): [u, v]}
    
    # Build result
    result_paths = []
    for a, b in combinations(best_branches, 2):
        p = best_paths_dict.get((a, b)) or best_paths_dict.get((b, a))
        if p is None:
            p = [a, b]
        result_paths.append(list(p))
    
    return {
        'branches': list(best_branches),
        'paths': result_paths
    }
# EVOLVE-BLOCK-END
