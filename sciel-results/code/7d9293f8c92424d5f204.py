import time
import itertools
import sys
import random
import collections
from collections import defaultdict, deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    edges = P['edges']
    adj = [[] for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        edge_set.add((u, v))
        edge_set.add((v, u))
    
    # Max Flow Implementation (Dinic's Algorithm for robustness, though n is small)
    class Dinic:
        def __init__(self, n):
            self.n = n
            self.graph = [[] for _ in range(n)]
        
        def add_edge(self, u, v, cap):
            self.graph[u].append([v, cap, len(self.graph[v])])
            self.graph[v].append([u, 0, len(self.graph[u]) - 1])
        
        def bfs(self, s, t):
            self.level = [-1] * self.n
            self.level[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                for v, cap, rev in self.graph[u]:
                    if cap > 0 and self.level[v] < 0:
                        self.level[v] = self.level[u] + 1
                        if v == t:
                            return True
                        q.append(v)
            return self.level[t] >= 0
        
        def dfs(self, v, t, f):
            if v == t:
                return f
            for i in range(len(self.graph[v])):
                e = self.graph[v][i]
                if e[1] > 0 and self.level[v] < self.level[e[0]]:
                    d = self.dfs(e[0], t, min(f, e[1]))
                    if d > 0:
                        e[1] -= d
                        self.graph[e[0]][e[2]][1] += d
                        return d
            return 0
        
        def max_flow(self, s, t):
            flow = 0
            while self.bfs(s, t):
                self.it = [0] * self.n
                f = self.dfs(s, t, float('inf'))
                while f > 0:
                    flow += f
                    f = self.dfs(s, t, float('inf'))
            return flow

    def check_and_extract(v_new, branches, used_internals):
        """
        Checks if v_new can be added to branches.
        Returns (True, list_of_paths) or (False, None).
        Paths are lists of vertices from v_new to each branch in 'branches'.
        """
        m = len(branches)
        if m == 0:
            return True, []
        
        # Identify nodes to include in flow network
        # Nodes: v_new, branches, and all other vertices not in used_internals and not in branches
        # We split all nodes except v_new and branches?
        # v_new is source. Branches are sinks.
        # Internal nodes must be disjoint.
        
        # Mapping from original vertex to flow node indices
        # We need to create a flow graph.
        # Let's assign indices.
        # Source: v_new
        # Sinks: branches
        # Internal: V \ (branches U {v_new} U used_internals)
        
        # Actually, simpler:
        # Create a new graph for flow.
        # Nodes in flow graph:
        # 0: Super Source (connected to v_new)
        # 1: Super Sink (connected from branches)
        # 2..: Split nodes for internal vertices
        # And v_new and branches themselves?
        
        # Let's use a standard vertex splitting approach for all nodes in the "available" set.
        # Available set A = V \ (used_internals)
        # But branches are in A? No, branches are removed from internal availability.
        # v_new is in A? Yes, but it's the source.
        
        # Let's define the set of vertices that can be used as INTERNAL vertices:
        # Internal_Candidates = V \ (branches U {v_new} U used_internals)
        
        # The flow network will consist of:
        # 1. Source node S
        # 2. Sink node T
        # 3. Node for v_new (V_new)
        # 4. Nodes for each branch b in branches (B_b)
        # 5. Split nodes for each x in Internal_Candidates (x_in, x_out)
        
        # Total nodes: 2 + 1 + m + 2*|Internal_Candidates|
        
        internal_cands = [x for x in range(n) if x not in branches and x != v_new and x not in used_internals]
        
        # Indexing
        # 0: S
        # 1: T
        # 2: v_new
        # 3 .. 3+m-1: branches
        # 3+m .. : internal candidates
        
        num_internal = len(internal_cands)
        N_flow = 2 + 1 + m + 2 * num_internal
        
        dinic = Dinic(N_flow)
        
        # Map original vertex to flow node indices
        # v_new -> 2
        # branches[i] -> 3 + i
        # internal_cands[j] -> 3 + m + 2*j (in), 3 + m + 2*j + 1 (out)
        
        def get_in(x):
            if x == v_new:
                return 2
            if x in branches:
                return 3 + branches.index(x)
            # else internal
            if x in used_internals:
                return -1
            if x in internal_cands:
                idx = internal_cands.index(x)
                return 3 + m + 2 * idx
            return -1

        def get_out(x):
            if x == v_new:
                return 2 # v_new is not split, it's the source itself? 
                         # Wait, if v_new is not split, we can send multiple flows from it.
                         # This is correct because v_new is the branch, not an internal vertex.
            if x in branches:
                return 3 + branches.index(x)
            if x in used_internals:
                return -1
            if x in internal_cands:
                idx = internal_cands.index(x)
                return 3 + m + 2 * idx + 1
            return -1

        # Add edges for internal vertices (splitting)
        for x in internal_cands:
            in_idx = get_in(x)
            out_idx = get_out(x)
            if in_idx != -1 and out_idx != -1:
                dinic.add_edge(in_idx, out_idx, 1)
        
        # Add edges for original graph edges
        for u in range(n):
            for v in adj[u]:
                # Edge u -> v
                u_out = get_out(u)
                v_in = get_in(v)
                if u_out != -1 and v_in != -1:
                    # Capacity 1 to ensure vertex disjointness?
                    # Vertex capacity is handled by split.
                    # Edge capacity can be 1 or INF. 1 is sufficient.
                    dinic.add_edge(u_out, v_in, 1)
        
        # Connect Source to v_new
        # v_new is node 2.
        # We want to send m units of flow from v_new.
        # So connect S (0) to v_new (2) with capacity m.
        dinic.add_edge(0, 2, m)
        
        # Connect branches to Sink
        for i, b in enumerate(branches):
            b_idx = 3 + i
            dinic.add_edge(b_idx, 1, 1) # Each branch can receive 1 unit of flow (one path)
        
        # Calculate Max Flow
        max_f = dinic.max_flow(0, 1)
        
        if max_f < m:
            return False, None
        
        # Extract paths
        # We need to trace back from the sink to the source in the residual graph.
        # This is tricky with Dinic's implementation as it doesn't store the final residual graph explicitly in a way that's easy to trace back for multiple paths without re-running or storing parent pointers.
        # However, we can just run a BFS/DFS on the original graph restricted to the used edges?
        # No, the flow algorithm determines *which* edges are used.
        
        # Alternative: Since n is small, we can just use the flow value to confirm existence.
        # But we need the *actual paths* to update `used_internals`.
        # We can find the paths by doing a DFS from each branch back to v_new in the subgraph of edges that carry flow.
        # To do this, we need to know which edges have flow.
        # In the Dinic implementation above, we can modify it to return the flow on edges or store the residual capacities.
        
        # Let's simplify: Instead of extracting from Dinic, we can use a simpler Max-Flow (Edmonds-Karp) 
        # or just use the fact that if max flow is m, we can find the paths by 
        # iteratively finding a path from v_new to any unconnected branch in the residual graph.
        
        # Actually, let's just use a simple BFS-based path finding in the "allowed" graph 
        # knowing that a solution exists? No, that might pick conflicting paths.
        
        # Better: Modify Dinic to store parent pointers or return the flow matrix.
        # Given the complexity, and n=24, we can use a different approach for extraction:
        # Since we know a set of disjoint paths exists, we can find them by 
        # repeatedly finding *any* path from v_new to a branch in the graph 
        # G \ (used_internals_so_far_for_this_step U branches_so_far).
        # This is a greedy extraction. Since the max flow is m, there are m disjoint paths.
        # A greedy approach might fail if it picks a "bad" path first.
        # However, for small graphs, it often works.
        # To be safe, we can use the flow residual graph.
        
        # Let's implement a simpler Max-Flow that allows path extraction.
        # Or, we can just return True and let the caller handle the path assignment?
        # No, we need to update `used_internals`.
        
        # Let's try to extract paths from the residual graph of the Dinic instance.
        # We need to access `dinic.graph`.
        
        paths = []
        # We need to find m disjoint paths.
        # We can do this by running a DFS from T back to S in the residual graph?
        # No, flow goes S -> T.
        # We can trace back from each branch node to v_new.
        
        # To make extraction robust, let's use a different Max-Flow implementation 
        # that stores the flow on each edge.
        
        # For now, let's assume we can extract.
        # If extraction is hard, we can just return True and use a 
        # "find path" routine that respects the current used_internals.
        # Since we know a solution exists, a simple DFS from v_new to branches 
        # avoiding used_internals *should* find them, provided we pick them in a 
        # way that doesn't block others.
        # This is risky.
        
        # Let's use a standard "Disjoint Paths" solver for the small case.
        # Since we already verified with Max Flow, we can just use 
        # a backtracking search to find the specific paths, which will be fast 
        # because the search space is pruned by the flow check.
        
        # Actually, we can just return True and let the main recursion handle 
        # the path assignment? No, the main recursion needs to know which 
        # internal vertices are used.
        
        # Let's implement a simple path extractor using the residual graph.
        # We need to modify Dinic to keep track of the flow.
        
        # Given time constraints, I will use a simpler method:
        # The max flow check is sufficient to prune.
        # For path extraction, we can use a greedy DFS from v_new to each branch,
        # removing used internal vertices as we go.
        # Since the flow check passed, this greedy approach is likely to succeed 
        # if we order branches by "difficulty" (e.g., fewest options).
        
        # Let's try greedy extraction.
        remaining_branches = list(branches)
        current_used = set(used_internals)
        extracted_paths = []
        
        # Sort branches by degree in the available graph?
        # Just try all.
        
        for b in remaining_branches:
            # Find a path from v_new to b avoiding current_used and other branches
            # Note: v_new is not in current_used.
            # Other branches are not allowed as internal vertices.
            
            # DFS
            def find_path(start, target, forbidden):
                visited = {start}
                stack = [(start, [start])]
                while stack:
                    node, path = stack.pop()
                    if node == target:
                        return path
                    for nb in adj[node]:
                        if nb in forbidden:
                            continue
                        if nb in visited:
                            continue
                        visited.add(nb)
                        stack.append((nb, path + [nb]))
                return None
            
            # Forbidden: current_used U (branches \ {b}) U {v_new} ?
            # No, v_new is the start.
            # Forbidden internal vertices: current_used U (branches \ {b})
            forbidden = current_used.union(set(branches) - {b})
            
            path = find_path(v_new, b, forbidden)
            if path is None:
                # Greedy failed, but flow said possible.
                # This can happen if greedy picked a path that blocked another.
                # For now, return False to be safe? Or try backtracking extraction?
                # Given the risk, let's return False to avoid invalid states.
                # This might make the search less effective but safe.
                return False, None
            
            # Update current_used with internal vertices of this path
            internals = set(path[1:-1])
            current_used.update(internals)
            extracted_paths.append(path)
        
        return True, extracted_paths

    # Main Search
    best_result = None
    best_k = 0
    deadline = time.time() + 80
    
    # Initial branches: Start with empty or a single high-degree vertex?
    # Let's start with empty and add the first vertex.
    # Or start with a pair?
    # Starting with empty:
    # B = []
    # U = set()
    # Try to add v.
    # check_and_extract(v, [], U) -> True, []
    # This works.
    
    # To speed up, we can start with a small clique.
    # But let's just do DFS from empty.
    
    # Order vertices by degree
    degrees = [len(adj[i]) for i in range(n)]
    sorted_vertices = sorted(range(n), key=lambda x: -degrees[x])
    
    def search(branches, used_internals):
        nonlocal best_result, best_k
        
        if time.time() > deadline:
            return
            
        k = len(branches)
        if k > best_k:
            best_k = k
            # Store a copy
            # We need to reconstruct paths for the result.
            # Wait, we don't store all paths in the state.
            # We only store `used_internals`.
            # We need to return the paths.
            # This is a problem. The state `used_internals` doesn't tell us the paths.
            # We need to store the paths in the state or re-derive them.
            
            # Let's modify the state to include the paths?
            # Or, when we find a better k, we can just record the branches.
            # At the end, we can try to reconstruct the paths?
            # No, the paths must be disjoint.
            
            # Better: Store the paths in the recursion stack.
            # But that makes the state large.
            
            # Alternative: When we add a vertex, we get the paths.
            # We can store `all_paths` in the state.
            
            pass # Placeholder

    # This approach of storing paths in state is heavy.
    # Let's just use the current program's structure but replace the 
    # `solve_for_branches` with the flow-based check.
    
    # Actually, the current program already finds a valid set.
    # The improvement is in the *search* for the set.
    
    # Let's stick to the plan:
    # 1. Use flow check to prune.
    # 2. Use greedy extraction (with safety check).
    # 3. Backtrack if greedy extraction fails.
    
    # To handle path storage, we will store the list of paths in the recursion.
    
    def search(branches, used_internals, all_paths):
        nonlocal best_result, best_k
        
        if time.time() > deadline:
            return
            
        k = len(branches)
        if k > best_k:
            best_k = k
            best_result = {
                'branches': [int(b) for b in branches],
                'paths': [list(map(int, p)) for p in all_paths]
            }
        
        if k >= n:
            return
            
        # Try to add more vertices
        for v in sorted_vertices:
            if v in branches or v in used_internals:
                continue
                
            valid, new_paths = check_and_extract(v, branches, used_internals)
            if valid:
                # Update state
                new_used = set(used_internals)
                for p in new_paths:
                    new_used.update(p[1:-1])
                
                # Combine paths
                # all_paths is a list of paths.
                # new_paths are paths from v to each branch in branches.
                # We need to add these to all_paths.
                # But all_paths already contains paths for pairs within branches.
                # So we just append new_paths.
                
                search(branches + [v], new_used, all_paths + new_paths)
                
                # Backtrack (implicit in recursion)
            
            if time.time() > deadline:
                return

    # Start search
    # Initial state: Empty branches, empty used, empty paths.
    # But we need at least 2 branches.
    # The search will add 1, then 2.
    
    # To avoid trivial empty start, we can start with the top vertex.
    # Or just let the search handle it.
    # check_and_extract(v, [], set()) returns True, [].
    # So it will add v.
    
    search([], set(), [])
    
    if best_result:
        return best_result
    
    # Fallback: Max Clique
    def max_clique():
        best = []
        vertices = list(range(n))
        
        def expand(clique, candidates):
            nonlocal best
            if len(clique) > len(best):
                best = list(clique)
            if not candidates:
                return
            for i, v in enumerate(candidates):
                new_clique = clique + [v]
                new_candidates = [c for c in candidates[i+1:] if (v, c) in edge_set]
                expand(new_clique, new_candidates)
        
        expand([], vertices)
        return best

    mc = max_clique()
    if len(mc) >= 2:
        branches = mc
        pairs = list(itertools.combinations(branches, 2))
        paths = []
        for u, v in pairs:
            paths.append([u, v])
        return {
            'branches': [int(b) for b in branches],
            'paths': [list(map(int, p)) for p in paths]
        }
    
    # Should not reach here
    return {'branches': [0, 1], 'paths': [[0, 1]]}

# EVOLVE-BLOCK-END
