import time
import random
import math
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50 seconds safety margin
    
    N = 256
    # Precompute squares
    squares = set()
    k = 2
    while k*k <= 2*N:
        squares.add(k*k)
        k += 1
        
    # Precompute conflicts
    forbidden = set()
    conflicts = [set() for _ in range(N+1)]
    for i in range(1, N+1):
        if 2*i in squares:
            forbidden.add(i)
        for j in range(i+1, N+1):
            if i+j in squares:
                conflicts[i].add(j)
                conflicts[j].add(i)
                
    valid_nodes = [i for i in range(1, N+1) if i not in forbidden]
    
    # Find connected components
    visited = [False] * (N+1)
    components = []
    for i in range(1, N+1):
        if i in forbidden:
            continue
        if not visited[i]:
            queue = deque([i])
            visited[i] = True
            comp = []
            while queue:
                u = queue.popleft()
                comp.append(u)
                for v in conflicts[u]:
                    if not visited[v] and v not in forbidden:
                        visited[v] = True
                        queue.append(v)
            components.append(comp)

    def greedy_coloring(vertices, adj):
        """Greedy coloring to get an upper bound on independent set size."""
        if not vertices:
            return 0
        colors = {}
        for v in vertices:
            used_colors = set()
            for u in adj.get(v, []):
                if u in colors:
                    used_colors.add(colors[u])
            c = 1
            while c in used_colors:
                c += 1
            colors[v] = c
        return max(colors.values()) if colors else 0

    def branch_and_bound(vertices, adj, best_size):
        """Branch and bound with greedy coloring upper bound."""
        if not vertices:
            return 0
        
        # Get upper bound via greedy coloring
        ub = greedy_coloring(vertices, adj)
        if ub <= best_size:
            return ub
        
        # Pick vertex with highest degree in the subgraph
        v = max(vertices, key=lambda x: len(adj.get(x, [])))
        
        # Branch 1: Include v
        remaining_after_include = [u for u in vertices if u != v and u not in adj.get(v, [])]
        size_include = 1 + branch_and_bound(remaining_after_include, adj, best_size)
        
        # Branch 2: Exclude v
        remaining_after_exclude = [u for u in vertices if u != v]
        size_exclude = branch_and_bound(remaining_after_exclude, adj, best_size)
        
        return max(size_include, size_exclude)

    def solve_component(comp):
        # If component is small enough, use exact B&B
        if len(comp) <= 30:
            # Build adjacency for this component
            adj = {v: [u for u in conflicts[v] if u in comp] for v in comp}
            best_size = 0
            result = branch_and_bound(comp, adj, best_size)
            # Need to actually find the set, not just the size
            # Use a different approach: find the actual independent set
            
            # Use greedy + local search as fallback for actual set
            nodes = sorted(comp, key=lambda x: len(conflicts[x]))
            S = set()
            for u in nodes:
                ok = all(u not in conflicts[v] for v in S)
                if ok:
                    S.add(u)
            
            # Try to improve with local search
            improved = True
            while improved and time.time() < deadline - 2:
                improved = False
                for x in comp:
                    if x in S:
                        continue
                    ok = all(x not in conflicts[y] for y in S)
                    if ok:
                        S.add(x)
                        improved = True
                        break
                if improved:
                    continue
                for y in list(S):
                    S_temp = S - {y}
                    for x in comp:
                        if x in S:
                            continue
                        ok = all(x not in conflicts[z] for z in S_temp)
                        if ok:
                            S.remove(y)
                            S.add(x)
                            improved = True
                            break
                    if improved:
                        break
            return S
        else:
            # For larger components, use enhanced local search
            best_S = set()
            restarts = 0
            while time.time() < deadline - 2:
                restarts += 1
                nodes = comp[:]
                random.shuffle(nodes)
                S = set()
                for u in nodes:
                    ok = all(u not in conflicts[v] for v in S)
                    if ok:
                        S.add(u)
                        
                improved = True
                while improved and time.time() < deadline - 2:
                    improved = False
                    # Try add
                    for x in comp:
                        if x in S:
                            continue
                        ok = all(x not in conflicts[y] for y in S)
                        if ok:
                            S.add(x)
                            improved = True
                            break
                    if improved:
                        continue
                        
                    # Try 1-swap
                    for y in list(S):
                        S_temp = S - {y}
                        candidates = [x for x in comp if x not in S and all(x not in conflicts[z] for z in S_temp)]
                        if candidates:
                            x = candidates[0]
                            S.remove(y)
                            S.add(x)
                            improved = True
                            break
                    if improved:
                        continue
                        
                    # Try 2-swap
                    for y in list(S):
                        S_temp = S - {y}
                        candidates = [x for x in comp if x not in S and all(x not in conflicts[z] for z in S_temp)]
                        if len(candidates) >= 2:
                            for i in range(len(candidates)):
                                for j in range(i+1, len(candidates)):
                                    x1 = candidates[i]
                                    x2 = candidates[j]
                                    if x1 not in conflicts[x2]:
                                        S.remove(y)
                                        S.add(x1)
                                        S.add(x2)
                                        improved = True
                                        break
                                if improved:
                                    break
                        if improved:
                            break
                            
                if len(S) > len(best_S):
                    best_S = S
                    
                if restarts > 100:
                    break
            return best_S

    best_set = set()
    
    # Solve each component
    for comp in components:
        best_comp = solve_component(comp)
        best_set.update(best_comp)
        
    return list(best_set)
