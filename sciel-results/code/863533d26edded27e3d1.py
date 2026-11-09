import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    N = 40
    best_edges = []
    best_m = 0
    
    def is_c6_free(A):
        """Check if graph A is C6-free. Returns True if no C6."""
        # For each edge (u, v), check if there is a path of length 5 between u and v in A without (u, v)
        # A is a numpy boolean matrix
        for u in range(N):
            for v in range(u + 1, N):
                if A[u, v]:
                    # Temporarily remove (u, v)
                    A[u, v] = False
                    A[v, u] = False
                    
                    # BFS/DFS from u to depth 5 to see if we reach v
                    # We use a queue: (node, depth, path) - but path is not needed, just visited
                    # To avoid revisiting nodes in the same path, we need to track the path or use backtracking.
                    # Since depth is small (5), we can use a set of visited nodes for each branch?
                    # Or simpler: Just do a BFS that tracks depth and doesn't revisit nodes in the current path.
                    # Actually, a simple BFS for "is there a path of length <= 5" is not enough, we need exactly 5?
                    # A C6 is a cycle of length 6. The path connecting the endpoints of an edge in the cycle has length 5.
                    # So we need to know if there is a path of length 5.
                    
                    # Optimization: If dist(u, v) < 5 or dist(u, v) > 5, it's not a C6 via this edge?
                    # No, a path of length 3 could be part of a larger structure? No, C6 is simple cycle.
                    # So we need a simple path of length 5.
                    
                    # Given small N, we can just do a DFS with pruning.
                    visited = [False] * N
                    visited[u] = True
                    
                    def dfs(current, depth):
                        if depth == 5:
                            return current == v
                        for next_node in np.where(A[current])[0]:
                            if not visited[next_node]:
                                visited[next_node] = True
                                if dfs(next_node, depth + 1):
                                    return True
                                visited[next_node] = False
                        return False
                        
                    if dfs(u, 0):
                        A[u, v] = True
                        A[v, u] = True
                        return False # C6 found
                    
                    A[u, v] = True
                    A[v, u] = True
        return True

    def get_edges_from_A(A):
        edges = []
        for u in range(N):
            for v in range(u + 1, N):
                if A[u, v]:
                    edges.append([u, v])
        return edges

    def try_local_search(A, max_time):
        """Perform local search on A. Returns best A and m found."""
        current_A = A.copy()
        current_m = np.sum(current_A) // 2
        best_A = current_A.copy()
        best_m = current_m
        
        while time.time() < max_time:
            # Try a swap: remove 1 edge, add 2
            if current_m < 2:
                break
                
            # Pick a random edge to remove
            edges = get_edges_from_A(current_A)
            if not edges:
                break
            e_remove = edges[random.randint(0, len(edges) - 1)]
            u_r, v_r = e_remove
            
            # Pick 2 random non-edges to add
            non_edges = []
            for u in range(N):
                for v in range(u + 1, N):
                    if not current_A[u, v]:
                        non_edges.append((u, v))
                        
            if len(non_edges) < 2:
                break
                
            # To make it faster, we can just pick random pairs from non_edges
            # But picking 2 random ones might often conflict or be invalid.
            # Let's pick e1 and e2 randomly.
            e1 = non_edges[random.randint(0, len(non_edges) - 1)]
            idx2 = random.randint(0, len(non_edges) - 1)
            if idx2 == non_edges.index(e1): # simple check, might be slow if list is large, but N=40 so non_edges ~ 700
                idx2 = (idx2 + 1) % len(non_edges)
            e2 = non_edges[idx2]
            
            # Apply change
            u1, v1 = e1
            u2, v2 = e2
            
            # Check validity:
            # 1. Remove e_remove
            current_A[u_r, v_r] = False
            current_A[v_r, u_r] = False
            
            # 2. Add e1, e2
            current_A[u1, v1] = True
            current_A[v1, u1] = True
            current_A[u2, v2] = True
            current_A[v2, u2] = True
            
            # Check C6 validity efficiently
            # Only check C6s involving e1 or e2
            valid = True
            
            # Check if e1 creates a C6
            # Path of length 5 between u1, v1 in current_A without e1
            current_A[u1, v1] = False
            current_A[v1, u1] = False
            
            visited = [False] * N
            visited[u1] = True
            
            def dfs_check(u_start, v_end, depth):
                if depth == 5:
                    return u_start == v_end
                for next_node in np.where(current_A[u_start])[0]:
                    if not visited[next_node]:
                        visited[next_node] = True
                        if dfs_check(next_node, v_end, depth + 1):
                            return True
                        visited[next_node] = False
                return False
                
            if dfs_check(u1, v1, 0):
                valid = False
            else:
                # Check if e2 creates a C6
                current_A[u1, v1] = True
                current_A[v1, u1] = True
                
                current_A[u2, v2] = False
                current_A[v2, u2] = False
                
                visited = [False] * N
                visited[u2] = True
                
                if dfs_check(u2, v2, 0):
                    valid = False
                else:
                    current_A[u2, v2] = True
                    current_A[v2, u2] = True
            
            if valid:
                current_m += 1
                if current_m > best_m:
                    best_m = current_m
                    best_A = current_A.copy()
            else:
                # Revert
                current_A[u1, v1] = False
                current_A[v1, u1] = False
                current_A[u2, v2] = False
                current_A[v2, u2] = False
                current_A[u_r, v_r] = True
                current_A[v_r, u_r] = True
                # Note: current_m stays same
                
        return best_A, best_m

    # Main loop: Multiple restarts
    for restart in range(100):
        if time.time() > deadline - 5:
            break
            
        # Initialize random graph
        A = np.zeros((N, N), dtype=bool)
        # Add random edges with probability p
        # Start with a sparse graph and add greedily?
        # Or just random and then fix?
        # Greedy is better for starting point.
        
        # Greedy construction
        # Order edges randomly
        all_edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if time.time() > deadline - 5:
                break
            A[u, v] = True
            A[v, u] = True
            if not is_c6_free(A):
                A[u, v] = False
                A[v, u] = False
        
        m = np.sum(A) // 2
        if m > best_m:
            best_m = m
            best_edges = get_edges_from_A(A)
            
        # Local search
        # Allow some time for local search
        ls_deadline = min(time.time() + 2.0, deadline - 5)
        A_ls, m_ls = try_local_search(A, ls_deadline)
        
        if m_ls > best_m:
            best_m = m_ls
            best_edges = get_edges_from_A(A_ls)
            
    return best_edges
