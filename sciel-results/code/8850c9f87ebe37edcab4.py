import numpy as np
import time

def construct():
    n = 40
    # Safety margin: evaluator kills at 150s, we aim for 105s
    deadline = time.time() + 105.0
    
    # Initialize adjacency matrix with 8 disjoint K5s
    A = np.zeros((n, n), dtype=np.int64)
    adj_list = [[] for _ in range(n)]
    
    def add_edge(u, v):
        A[u, v] = 1
        A[v, u] = 1
        adj_list[u].append(v)
        adj_list[v].append(u)
        
    def remove_edge(u, v):
        A[u, v] = 0
        A[v, u] = 0
        adj_list[u].remove(v)
        adj_list[v].remove(u)

    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                u = 5 * k + a
                v = 5 * k + b
                add_edge(u, v)
                
    m = 80
    best_m = m
    best_A = A.copy()
    best_adj_list = [lst.copy() for lst in adj_list]
    
    def compute_blocked_matrix(adj_list):
        """
        Returns a boolean matrix B where B[u, v] is True if there is a simple path
        of length exactly 5 from u to v in the graph defined by adj_list.
        """
        B = np.zeros((n, n), dtype=bool)
        
        # For each start node u, perform DFS/BFS to find nodes reachable by simple path of length 5
        for u in range(n):
            # visited[u] = True
            # We want to reach v in exactly 5 steps.
            # State: (current_node, depth, visited_mask)
            # n=40, mask is 40 bits. Python int is fine.
            
            # Use a set of (node, depth, mask) or just DFS with backtracking
            # Since depth is small (5), backtracking DFS is efficient.
            
            visited_mask = 1 << u
            
            def dfs(node, depth, mask):
                if depth == 5:
                    return
                for neighbor in adj_list[node]:
                    if mask & (1 << neighbor):
                        continue
                    new_mask = mask | (1 << neighbor)
                    if depth + 1 == 5:
                        B[u, neighbor] = True
                    else:
                        dfs(neighbor, depth + 1, new_mask)
                        
            dfs(u, 0, visited_mask)
            
        return B

    B = compute_blocked_matrix(adj_list)
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    def get_valid_non_edges():
        # Non-edges where B[u, v] is False
        valid_mask = (A[i_idx, j_idx] == 0) & (~B[i_idx, j_idx])
        valid_indices = np.where(valid_mask)[0]
        return [(int(i_idx[k]), int(j_idx[k])) for k in valid_indices]
        
    def get_all_edges():
        edge_mask = A[i_idx, j_idx] == 1
        edge_indices = np.where(edge_mask)[0]
        return [(int(i_idx[k]), int(j_idx[k])) for k in edge_indices]

    steps_since_improvement = 0
    max_steps_without_improvement = 1000
    kick_count = 0
    
    while time.time() < deadline:
        valid_edges = get_valid_non_edges()
        
        if not valid_edges:
            # Stuck: No edges can be added
            # Perform a kick: remove an edge
            current_edges = get_all_edges()
            if not current_edges:
                break
            
            # Sample edges to find one whose removal helps
            # Heuristic: remove an edge with high degree or random
            # To be faster, just pick a random edge or one that is part of many short paths?
            # For now, random kick.
            idx = np.random.randint(len(current_edges))
            u, v = current_edges[idx]
            
            remove_edge(u, v)
            m -= 1
            B = compute_blocked_matrix(adj_list)
            kick_count += 1
            
            if m < best_m - 5:
                # Revert
                A = best_A.copy()
                adj_list = [lst.copy() for lst in best_adj_list]
                m = best_m
                B = compute_blocked_matrix(adj_list)
                steps_since_improvement = 0
                kick_count = 0
            else:
                steps_since_improvement += 1
                if steps_since_improvement > max_steps_without_improvement:
                    A = best_A.copy()
                    adj_list = [lst.copy() for lst in best_adj_list]
                    m = best_m
                    B = compute_blocked_matrix(adj_list)
                    steps_since_improvement = 0
                    kick_count = 0
            continue
            
        # Add a valid edge
        # Prefer edges that connect high-degree vertices? Or just random?
        # Let's try a simple heuristic: pick randomly from valid edges.
        # Or pick the one that minimizes future blocking? (Too expensive)
        # Random is often robust for this size.
        idx = np.random.randint(len(valid_edges))
        u, v = valid_edges[idx]
        
        add_edge(u, v)
        m += 1
        B = compute_blocked_matrix(adj_list)
        kick_count = 0
        
        if m > best_m:
            best_m = m
            best_A = A.copy()
            best_adj_list = [lst.copy() for lst in adj_list]
            steps_since_improvement = 0
        else:
            steps_since_improvement += 1
            if steps_since_improvement > max_steps_without_improvement:
                A = best_A.copy()
                adj_list = [lst.copy() for lst in best_adj_list]
                m = best_m
                B = compute_blocked_matrix(adj_list)
                steps_since_improvement = 0
                kick_count = 0
                
    # Convert best_A to list of edges
    result = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_A[u, v] == 1:
                result.append([u, v])
                
    return result
