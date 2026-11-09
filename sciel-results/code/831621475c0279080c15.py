# EVOLVE-BLOCK-START
import math
from collections import deque

def construct():
    """
    Find a maximum subset of [1,256] such that a+b is never a perfect square
    for any a<=b in the set (including a=b).
    
    Approach:
    1. Filter out self-conflicting nodes (where 2*i is a perfect square).
    2. Build the conflict graph on remaining nodes.
    3. Check bipartiteness; if bipartite, solve MWIS exactly via max-flow/min-cut.
    4. If not bipartite, use a strong local-search heuristic.
    """
    N = 256
    
    def is_square(x):
        if x < 0:
            return False
        r = math.isqrt(x)
        return r * r == x

    # Identify self-conflicts (2*i is a square)
    valid_nodes = []
    for i in range(1, N + 1):
        if not is_square(2 * i):
            valid_nodes.append(i)
            
    # Map valid nodes to indices
    node_index = {v: i for i, v in enumerate(valid_nodes)}
    M = len(valid_nodes)
    
    # Build adjacency list for valid nodes
    adj = [[] for _ in range(M)]
    values = valid_nodes
    
    for i in range(M):
        u = values[i]
        for j in range(i + 1, M):
            v = values[j]
            if is_square(u + v):
                adj[i].append(j)
                adj[j].append(i)
                
    # Check if the graph is bipartite via BFS coloring
    color = [-1] * M
    is_bipartite = True
    for start in range(M):
        if color[start] == -1:
            color[start] = 0
            q = deque([start])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if color[v] == -1:
                        color[v] = 1 - color[u]
                        q.append(v)
                    elif color[v] == color[u]:
                        is_bipartite = False
                        break
                if not is_bipartite:
                    break
        if not is_bipartite:
            break

    if is_bipartite:
        # Solve MWIS on Bipartite Graph using Min-Cut (Max-Flow)
        # Standard reduction:
        #   Source S -> a (cap = weight_a) for a in partition A
        #   b -> Sink T (cap = weight_b) for b in partition B
        #   a -> b (cap = INF) for each edge (a,b) with a in A, b in B
        # Min vertex cover size = max flow
        # Max independent set = V \ min vertex cover
        
        INF = 10**9
        num_flow_nodes = M + 2
        S = M
        T = M + 1
        
        # flow_adj[u] = list of [v, cap, rev_index]
        flow_adj = [[] for _ in range(num_flow_nodes)]
        
        def add_edge(u, v, cap):
            flow_adj[u].append([v, cap, len(flow_adj[v])])
            flow_adj[v].append([u, 0, len(flow_adj[u]) - 1])
            
        for i in range(M):
            if color[i] == 0:
                add_edge(S, i, 1)
            else:
                add_edge(i, T, 1)
                
        for i in range(M):
            for j in adj[i]:
                if color[i] == 0 and color[j] == 1:
                    add_edge(i, j, INF)
                elif color[i] == 1 and color[j] == 0:
                    add_edge(j, i, INF)
                    
        # Dinic's Algorithm for Max Flow
        def dinic():
            flow = 0
            level = [-1] * num_flow_nodes
            ptr = [0] * num_flow_nodes
            
            def bfs():
                for i in range(num_flow_nodes):
                    level[i] = -1
                level[S] = 0
                q = deque([S])
                while q:
                    u = q.popleft()
                    for e in flow_adj[u]:
                        v, cap, rev = e
                        if cap > 0 and level[v] == -1:
                            level[v] = level[u] + 1
                            q.append(v)
                return level[T] != -1
                
            def dfs(u, pushed):
                if pushed == 0:
                    return 0
                if u == T:
                    return pushed
                while ptr[u] < len(flow_adj[u]):
                    e = flow_adj[u][ptr[u]]
                    v, cap, rev = e
                    if level[v] == level[u] + 1 and cap > 0:
                        tr = dfs(v, min(pushed, cap))
                        if tr > 0:
                            e[1] -= tr
                            flow_adj[v][rev][1] += tr
                            return tr
                    ptr[u] += 1
                return 0
                
            while bfs():
                for i in range(num_flow_nodes):
                    ptr[i] = 0
                while True:
                    pushed = dfs(S, INF)
                    if pushed == 0:
                        break
                    flow += pushed
            return flow
            
        max_flow = dinic()
        
        # Find min vertex cover from residual graph
        # Source side: nodes reachable from S in residual graph
        reachable = [False] * num_flow_nodes
        q = deque([S])
        reachable[S] = True
        while q:
            u = q.popleft()
            for e in flow_adj[u]:
                v, cap, rev = e
                if cap > 0 and not reachable[v]:
                    reachable[v] = True
                    q.append(v)
                    
        # Min vertex cover:
        #   {v in A (color 0) : v NOT reachable from S}  -- wait, let me reconsider
        # Standard result: In the residual graph after max flow,
        #   Let R = set of nodes reachable from S.
        #   Min vertex cover = ({v in A : v not in R} union {v in B : v in R})
        #   Max independent set = ({v in A : v in R} union {v in B : v not in R})
        
        independent_set_indices = []
        for i in range(M):
            if color[i] == 0:
                if reachable[i]:
                    independent_set_indices.append(i)
            else:
                if not reachable[i]:
                    independent_set_indices.append(i)
                    
        result = [values[i] for i in independent_set_indices]
        return sorted(result)
        
    else:
        # Fallback for non-bipartite: Greedy + Local Search
        # Start with greedy coloring-based independent set, then local search
        
        # Greedy: sort by degree, pick nodes that don't conflict
        degrees = [len(adj[i]) for i in range(M)]
        order = sorted(range(M), key=lambda i: -degrees[i])
        
        selected = set()
        for i in order:
            conflict = False
            for j in adj[i]:
                if j in selected:
                    conflict = True
                    break
            if not conflict:
                selected.add(i)
                
        # Local search: try to swap out one selected node for one unselected
        improved = True
        while improved:
            improved = False
            sel_list = list(selected)
            unselected = [i for i in range(M) if i not in selected]
            
            for out_node in sel_list:
                for in_node in unselected:
                    # Check if swapping improves
                    # Remove out_node, add in_node
                    new_sel = selected - {out_node} | {in_node}
                    # Verify no conflicts
                    valid = True
                    for a in new_sel:
                        for b in adj[a]:
                            if b in new_sel and a < b:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid and len(new_sel) > len(selected):
                        selected = new_sel
                        improved = True
                        break
                if improved:
                    break
                    
        # Also try adding unselected nodes
        improved = True
        while improved:
            improved = False
            for i in range(M):
                if i in selected:
                    continue
                conflict = False
                for j in adj[i]:
                    if j in selected:
                        conflict = True
                        break
                if not conflict:
                    selected.add(i)
                    improved = True
                    
        result = [values[i] for i in selected]
        return sorted(result)

# EVOLVE-BLOCK-END
