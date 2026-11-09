import numpy as np
import time
import random

def construct():
    n = 20
    # A is adjacency matrix, 1 if edge exists
    A = np.zeros((n, n), dtype=np.int8)
    # C is common neighbor counts, C[i][j] = |N(i) n N(j)|
    C = np.zeros((n, n), dtype=np.int32)
    
    def update_C_add(u, v):
        # C[x][u] increases by 1 if x in N(v)
        # C[x][v] increases by 1 if x in N(u)
        # Note: C[u][v] does not change
        C[:, u] += A[:, v]
        C[u, :] += A[v, :]
        C[:, v] += A[:, u]
        C[v, :] += A[u, :]
        
    def update_C_remove(u, v):
        # Reverse of update_C_add
        C[:, u] -= A[:, v]
        C[u, :] -= A[v, :]
        C[:, v] -= A[:, u]
        C[v, :] -= A[u, :]

    def can_add(u, v):
        if A[u, v]:
            return False
        # Check 1: C[u][v] <= 2
        if C[u, v] > 2:
            return False
        
        # Check 2: For x in N(v) \ {u}, C[x][u] <= 1
        # A[:, v] is 1 for neighbors of v.
        # We need to check C[x, u] for all x where A[x, v] == 1 and x != u.
        # Since u is not in N(v) (A[u, v] == 0), A[u, v] is 0, so u is not in the mask.
        # So we can just check all x in N(v).
        mask_v = A[:, v]
        if mask_v.any():
            # Get C values for x in N(v) at column u
            # C[mask_v, u] gives array of C[x, u] for x in N(v)
            if np.any(C[mask_v, u] > 1):
                return False
                
        # Check 3: For y in N(u) \ {v}, C[v][y] <= 1
        mask_u = A[:, u]
        if mask_u.any():
            if np.any(C[v, mask_u] > 1):
                return False
                
        return True

    def add_edge(u, v):
        A[u, v] = 1
        A[v, u] = 1
        update_C_add(u, v)

    def remove_edge(u, v):
        update_C_remove(u, v)
        A[u, v] = 0
        A[v, u] = 0

    def get_edge_count():
        return np.sum(A) // 2

    def get_edges_list():
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if A[i, j]:
                    edges.append([i, j])
        return edges

    def is_valid():
        # Double check validity for safety
        for u in range(n):
            for v in range(u + 1, n):
                if C[u, v] > 2:
                    return False
        return True

    start_time = time.time()
    time_limit = 110.0 # Safety margin

    best_count = 0
    best_edges = []
    
    # Multiple restarts to find a better solution
    max_restarts = 50
    for _ in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Reset
        A[:] = 0
        C[:] = 0
        
        # Initial construction: Greedy random
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(pairs)
        
        for u, v in pairs:
            if can_add(u, v):
                add_edge(u, v)
                
        curr_count = get_edge_count()
        if curr_count > best_count:
            best_count = curr_count
            best_edges = get_edges_list()
            
        # Local Search
        # Try to add random edges. If stuck, try to remove a random edge and then add.
        # Also try 1-for-2 swaps: remove e1, add e2, e3.
        
        while time.time() - start_time < time_limit:
            improved = False
            
            # 1. Try simple additions
            for _ in range(50):
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u != v and can_add(u, v):
                    add_edge(u, v)
                    improved = True
                    break
            
            if not improved:
                # 2. Try 1-for-1 swap or 1-for-2 swap
                edges_list = get_edges_list()
                if edges_list:
                    u_rem, v_rem = random.choice(edges_list)
                    remove_edge(u_rem, v_rem)
                    
                    # Try to add two edges
                    added = 0
                    # Try random additions
                    for _ in range(100):
                        u_add = random.randint(0, n - 1)
                        v_add = random.randint(0, n - 1)
                        if u_add != v_add and can_add(u_add, v_add):
                            add_edge(u_add, v_add)
                            added += 1
                            if added == 2:
                                break
                    
                    if added > 1:
                        curr_count = get_edge_count()
                        if curr_count > best_count:
                            best_count = curr_count
                            best_edges = get_edges_list()
                    elif added == 1:
                        # Count is same, but structure changed. Keep it?
                        pass
                    else:
                        # Could not add any back, revert
                        add_edge(u_rem, v_rem)
                else:
                    # No edges, restart
                    break

    # Final Verification
    if best_edges:
        adj = [set() for _ in range(n)]
        for u, v in best_edges:
            adj[u].add(v)
            adj[v].add(u)
        
        valid = True
        for u in range(n):
            for v in range(u + 1, n):
                if len(adj[u] & adj[v]) > 2:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            return best_edges
        else:
            # Fallback to a known valid graph (e.g., empty or simple)
            return []
    else:
        return []
