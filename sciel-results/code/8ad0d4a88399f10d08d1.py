"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from random import Random
from collections import deque

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
    for k in range(1, n + 1):
        if k <= 2: trees.append([] if k == 1 else [(0, 1)]); continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code: degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
    return trees

class MinCostMaxFlow:
    """Min-cost max-flow using successive shortest paths with potentials."""
    def __init__(self, n):
        self.n = n
        self.graph = [[] for _ in range(n)]
    
    def add_edge(self, u, v, cap, cost):
        """Add directed edge u->v with capacity cap and cost cost."""
        self.graph[u].append([v, cap, cost, len(self.graph[v])])
        self.graph[v].append([u, 0, -cost, len(self.graph[u]) - 1])
    
    def min_cost_flow(self, s, t, max_flow):
        """Find min-cost flow of up to max_flow units from s to t."""
        flow = 0
        cost = 0
        potential = [0] * self.n
        parent = [None] * self.n
        
        while flow < max_flow:
            # Dijkstra with potentials
            dist = [float('inf')] * self.n
            dist[s] = 0
            pq = [(0, s)]
            visited = [False] * self.n
            
            while pq:
                d, u = heapq.heappop(pq)
                if visited[u]:
                    continue
                visited[u] = True
                for i, (v, cap, c, _) in enumerate(self.graph[u]):
                    if cap > 0 and not visited[v]:
                        new_dist = d + c + potential[u] - potential[v]
                        if new_dist < dist[v]:
                            dist[v] = new_dist
                            parent[v] = (u, i)
                            heapq.heappush(pq, (new_dist, v))
            
            if dist[t] == float('inf'):
                break
            
            # Update potentials
            for i in range(self.n):
                if dist[i] < float('inf'):
                    potential[i] += dist[i]
            
            # Find bottleneck
            bottleneck = max_flow - flow
            v = t
            while v != s:
                u, i = parent[v]
                bottleneck = min(bottleneck, self.graph[u][i][1])
                v = u
            
            # Augment flow
            v = t
            while v != s:
                u, i = parent[v]
                self.graph[u][i][1] -= bottleneck
                self.graph[v][i - len(self.graph[v]) + len(self.graph[u])][1] += bottleneck
                # Fix reverse edge index
                rev_idx = self.graph[u][i][3]
                self.graph[v][rev_idx][1] += bottleneck
                self.graph[u][i][1] -= bottleneck
                v = u
            
            flow += bottleneck
            cost += bottleneck * potential[t]
        
        return flow, cost

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    trees = instance_trees(20, 611)
    
    # Process trees from largest to smallest
    order = list(range(19, -1, -1))
    used_edges = set()
    result = [None] * 20
    
    for idx in order:
        k = idx + 1
        tree_edges = trees[idx]
        
        if time.time() > deadline:
            break
        
        if k == 1:
            for v in range(20):
                result[idx] = [v]
                break
            continue
        
        if k == 2:
            found = False
            for u in range(20):
                for w in range(u + 1, 20):
                    if (u, w) not in used_edges and (w, u) not in used_edges:
                        result[idx] = [u, w]
                        used_edges.add((u, w))
                        found = True
                        break
                if found:
                    break
            if not found:
                result[idx] = None
            continue
        
        # For k >= 3, use min-cost max-flow to find embedding
        # Build auxiliary graph:
        # Nodes: 0 (source), 1-20 (host vertices), 21-21+k-1 (tree vertices), 22+k-1 (sink)
        # Actually, let's use a simpler formulation:
        # We want to assign each tree vertex to a host vertex such that
        # all tree edges map to distinct unused K20 edges.
        
        # Use a different approach: model as assignment problem with edge constraints
        # For each tree edge (a,b), we need host(a) != host(b) and (host(a),host(b)) not in used_edges
        
        # Let's use backtracking with better heuristics
        tree_n = k
        tree_adj = [[] for _ in range(tree_n)]
        for u, v in tree_edge_list if False else tree_edges:
            tree_adj[u].append(v)
            tree_adj[v].append(u)
        
        # Get BFS order
        visited = [False] * tree_n
        bfs_order = [0]
        visited[0] = True
        q = deque([0])
        while q:
            u = q.popleft()
            for v in tree_adj[u]:
                if not visited[v]:
                    visited[v] = True
                    bfs_order.append(v)
                    q.append(v)
        
        # Backtracking with MRV heuristic
        assignment = [None] * tree_n
        
        def get_candidates(tree_v):
            """Get valid host vertices for tree_v, sorted by constraint tightness."""
            candidates = []
            for host_v in range(20):
                if host_v in assignment:
                    continue
                # Check edges to already-assigned neighbors
                valid = True
                for nb in tree_adj[tree_v]:
                    if assignment[nb] is not None:
                        if (assignment[nb], host_v) in used_edges or (host_v, assignment[nb]) in used_edges:
                            valid = False
                            break
                if valid:
                    # Score: number of unassigned neighbors (prefer more constrained)
                    unassigned_nb = sum(1 for nb in tree_adj[tree_v] if assignment[nb] is None)
                    candidates.append((unassigned_nb, host_v))
            candidates.sort()
            return [c[1] for c in candidates]
        
        def backtrack(pos):
            if time.time() > deadline:
                return False
            if pos == tree_n:
                return True
            
            tree_v = bfs_order[pos]
            candidates = get_candidates(tree_v)
            
            for host_v in candidates:
                assignment[tree_v] = host_v
                temp_edges = set()
                for nb in tree_adj[tree_v]:
                    if assignment[nb] is not None:
                        e1 = (assignment[nb], host_v)
                        e2 = (host_v, assignment[nb])
                        temp_edges.add(e1)
                        temp_edges.add(e2)
                
                if backtrack(pos + 1):
                    for e in temp_edges:
                        used_edges.add(e)
                    return True
                
                assignment[tree_v] = None
        
        if backtrack(0):
            result[idx] = assignment[:]
        else:
            result[idx] = None
    
    return result
# EVOLVE-BLOCK-END
