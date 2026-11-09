import time
import random
from collections import deque
from itertools import combinations
import networkx as nx
import numpy as np
import sys
sys.setrecursionlimit(10000)

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80.0

    n = P['n']
    edges = P['edges']

    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    degrees = [len(adj[i]) for i in range(n)]

    def build_flow_network(branches_list):
        """Build a flow network where we need k*(k-1)/2 units of flow,
        one for each pair of branches. Each non-branch vertex has capacity 1.
        Branch vertices have capacity k-1 (one path through each other branch).
        We use a super-source and super-sink approach with per-pair flow."""
        k = len(branches_list)
        branches_set = set(branches_list)
        
        # Vertex-splitting: each vertex v becomes v_in -> v_out
        # For edges u-v, we add u_out -> v_in and v_out -> u_in
        # Super source S connects to each branch's v_in
        # Each branch's v_out connects to super sink T
        # But we need exactly one path per pair, so we use a different approach:
        # Connect S to each branch_in with capacity k-1
        # Connect each branch_out to T with capacity k-1
        # Total flow should be k*(k-1) (each branch sends k-1 units)
        
        S = 2 * n
        T = 2 * n + 1
        G = nx.DiGraph()
        INF = 10**9
        
        for v in range(n):
            vin = 2 * v
            vout = 2 * v + 1
            if v in branches_set:
                cap = k - 1
            else:
                cap = 1
            G.add_edge(vin, vout, capacity=cap)
        
        for u, v in edges:
            G.add_edge(2 * u + 1, 2 * v, capacity=INF)
            G.add_edge(2 * v + 1, 2 * u, capacity=INF)
        
        for b in branches_list:
            G.add_edge(S, 2 * b, capacity=k - 1)
            G.add_edge(2 * b + 1, T, capacity=k - 1)
        
        return G, S, T

    def check_and_extract(branches_list):
        """Check feasibility via max-flow and extract paths.
        Returns paths_dict or None."""
        k = len(branches_list)
        if k < 2:
            return None
        branches_set = set(branches_list)
        
        G, S, T = build_flow_network(branches_list)
        
        try:
            flow_value, flow_dict = nx.maximum_flow(G, S, T)
        except Exception:
            return None
        
        if flow_value < k * (k - 1):
            return None
        
        # Extract paths from flow
        # Build directed flow graph on original vertices
        flow_edges = {}  # (u,v) -> count
        for u in range(n):
            uout = 2 * u + 1
            if uout in flow_dict:
                for v in flow_dict[uout]:
                    f = flow_dict[uout][v]
                    if f > 0 and v < 2 * n:
                        flow_edges[(u, v)] = flow_edges.get((u, v), 0) + int(f)
        
        # Also account for S->branch_in and branch_out->T
        # Trace paths from each branch
        remaining = dict(flow_edges)
        paths_dict = {}
        
        for b in branches_list:
            # Find k-1 paths starting from b
            traced = 0
            while traced < k - 1:
                # Find outgoing edge from b with remaining flow
                next_v = None
                for (u, v), cnt in remaining.items():
                    if u == b and cnt > 0:
                        next_v = v
                        break
                if next_v is None:
                    return None
                remaining[(b, next_v)] -= 1
                traced += 1
                
                path = [b, next_v]
                cur = next_v
                while cur not in branches_set:
                    next_w = None
                    for (u, v), cnt in remaining.items():
                        if u == cur and cnt > 0:
                            next_w = v
                            break
                    if next_w is None:
                        return None
                    remaining[(cur, next_w)] -= 1
                    path.append(next_w)
                    cur = next_w
                
                if cur == b:
                    return None
                
                i_idx = branches_list.index(b)
                j_idx = branches_list.index(cur)
                key = (min(i_idx, j_idx), max(i_idx, j_idx))
                if key in paths_dict:
                    return None
                paths_dict[key] = path
        
        if len(paths_dict) != k * (k - 1) // 2:
            return None
        
        return paths_dict

    # ---------- Find good branch sets ----------
    # Strategy: try all subsets of high-degree vertices, check via flow
    
    # Sort vertices by degree
    vertices_by_degree = sorted(range(n), key=lambda x: degrees[x], reverse=True)
    
    best_valid = None
    best_valid_size = 0
    
    # Try sizes from large to small
    max_size_to_try = min(n, 12)
    
    for target_size in range(max_size_to_try, 1, -1):
        if time.time() > deadline - 5:
            break
        
        found = False
        
        # Try subsets of top-degree vertices
        top_candidates = vertices_by_degree[:min(n, target_size + 6)]
        
        for subset in combinations(top_candidates, target_size):
            if time.time() > deadline - 3:
                break
            paths = check_and_extract(list(subset))
            if paths is not None:
                best_valid = (list(subset), paths)
                best_valid_size = target_size
                found = True
                break
        
        if found:
            break
        
        # If not found in top candidates, try random subsets
        if target_size <= 8:
            for trial in range(50):
                if time.time() > deadline - 3:
                    break
                random_subset = random.sample(range(n), target_size)
                paths = check_and_extract(random_subset)
                if paths is not None:
                    best_valid = (random_subset, paths)
                    best_valid_size = target_size
                    found = True
                    break
            if found:
                break
    
    # If nothing found, try growing from a small valid set
    if best_valid is None or best_valid_size < 2:
        # Find any edge
        for u, v in edges:
            best_valid = ([u, v], {(0, 1): [u, v]})
            best_valid_size = 2
            break
    
    # Try to grow the solution
    if best_valid is not None and time.time() < deadline - 10:
        current_branches = list(best_valid[0])
        current_paths = best_valid[1]
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            candidates = [v for v in range(n) if v not in set(current_branches)]
            candidates.sort(key=lambda x: degrees[x], reverse=True)
            for v in candidates:
                if time.time() > deadline - 5:
                    break
                new_branches = current_branches + [v]
                new_paths = check_and_extract(new_branches)
                if new_paths is not None:
                    current_branches = new_branches
                    current_paths = new_paths
                    best_valid = (current_branches, current_paths)
                    best_valid_size = len(current_branches)
                    improved = True
                    break
    
    # ---------- Build output ----------
    branches = best_valid[0]
    paths_dict = best_valid[1]

    output_paths = []
    for i in range(len(branches)):
        for j in range(i + 1, len(branches)):
            p = paths_dict.get((i, j))
            if p is None:
                p = paths_dict.get((j, i))
            if p is None:
                if branches[i] in adj[branches[j]]:
                    p = [branches[i], branches[j]]
                else:
                    # BFS fallback
                    visited = {branches[i]}
                    parent = {branches[i]: None}
                    queue = deque([branches[i]])
                    found = False
                    while queue and not found:
                        cur = queue.popleft()
                        for nb in adj[cur]:
                            if nb == branches[j]:
                                path = [branches[j]]
                                node = cur
                                while node is not None:
                                    path.append(node)
                                    node = parent[node]
                                path.reverse()
                                p = path
                                found = True
                                break
                            if nb in set(branches):
                                continue
                            if nb not in visited:
                                visited.add(nb)
                                parent[nb] = cur
                                queue.append(nb)
                    if p is None:
                        p = [branches[i], branches[j]]
            if p[0] != branches[i]:
                p = list(reversed(p))
            output_paths.append(list(p))

    branches_out = [int(b) for b in branches]
    paths_out = [[int(v) for v in p] for p in output_paths]

    return {
        'branches': branches_out,
        'paths': paths_out
    }
