import numpy as np
import time
import random

def construct():
    n = 20
    deadline = time.time() + 110  # Safety margin
    
    def is_valid(A):
        """Check if the graph has no K2,3 subgraph."""
        K = A @ A
        np.fill_diagonal(K, 0)
        return np.max(K) <= 2

    def count_edges(A):
        return int(np.sum(A)) // 2

    def try_add_edge(A, u, v):
        """Try to add edge (u, v). Returns True if added, False if not."""
        if A[u, v] == 1:
            return False
        
        # Check if adding (u, v) violates the constraint.
        # Adding (u, v) increases the common neighbor count for pairs (u, w) and (v, w) for all w != u, v.
        for w in range(n):
            if w == u or w == v:
                continue
            
            # Check pair (u, w)
            cn_uw = int(np.dot(A[u, :], A[:, w]))
            if A[v, w] == 1:
                if cn_uw >= 2:
                    return False
            
            # Check pair (v, w)
            cn_vw = int(np.dot(A[v, :], A[:, w]))
            if A[u, w] == 1:
                if cn_vw >= 2:
                    return False
        
        # If we passed all checks, add the edge
        A[u, v] = 1
        A[v, u] = 1
        return True

    def initialize():
        # Start with 5 disjoint K4s (30 edges). This is a valid K2,3-free graph.
        A = np.zeros((n, n), dtype=int)
        for k in range(5):
            for a in range(4):
                for b in range(a + 1, 4):
                    u = 4 * k + a
                    v = 4 * k + b
                    A[u, v] = 1
                    A[v, u] = 1
        return A

    def greedy_fill(A):
        changed = True
        while changed:
            changed = False
            for u in range(n):
                for v in range(u + 1, n):
                    if A[u, v] == 0:
                        if try_add_edge(A, u, v):
                            changed = True
        return A

    def local_search(A, budget):
        best_A = A.copy()
        best_edges = count_edges(A)
        
        while time.time() < budget:
            # Try to add a random non-edge
            non_edges = [(u, v) for u in range(n) for v in range(u + 1, n) if A[u, v] == 0]
            if not non_edges:
                break
            
            u, v = random.choice(non_edges)
            if try_add_edge(A, u, v):
                current_edges = count_edges(A)
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_A = A.copy()
            else:
                # Try a swap: remove a random edge, then try to add (u, v)
                edges = [(i, j) for i in range(n) for j in range(i + 1, n) if A[i, j] == 1]
                if not edges:
                    break
                ru, rv = random.choice(edges)
                
                # Backup state
                A_backup = A.copy()
                
                # Remove edge (ru, rv)
                A[ru, rv] = 0
                A[rv, ru] = 0
                
                # Try to add (u, v)
                success_1 = try_add_edge(A, u, v)
                
                # Try to add another random non-edge to gain +1 net (if success_1) or recover (if not)
                non_edges_now = [(i, j) for i in range(n) for j in range(i + 1, n) if A[i, j] == 0]
                added_more = 0
                for _ in range(5):
                    if not non_edges_now:
                        break
                    x, y = random.choice(non_edges_now)
                    if try_add_edge(A, x, y):
                        added_more += 1
                        # Remove from list to avoid re-trying same one immediately
                        non_edges_now.remove((x, y))
                
                current_edges = count_edges(A)
                
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_A = A.copy()
                elif current_edges < best_edges:
                    # Revert if we lost edges
                    A[:] = A_backup
                # If current_edges == best_edges, keep it (diversity)
        
        return best_A, best_edges

    best_overall_A = None
    best_overall_edges = 0
    
    # Run multiple restarts
    while time.time() < deadline:
        A = initialize()
        
        # Add some random edges to seed
        for _ in range(20):
            u, v = random.sample(range(n), 2)
            try_add_edge(A, u, v)
        
        A = greedy_fill(A)
        
        # Local search with time slice
        slice_end = min(time.time() + 5, deadline)
        A, edges = local_search(A, slice_end)
        
        if edges > best_overall_edges:
            best_overall_edges = edges
            best_overall_A = A.copy()
            
            # Optional: Verify best result occasionally
            if not is_valid(best_overall_A):
                # This should not happen if try_add_edge is correct
                # But if it does, revert to last known valid
                print("Warning: Invalid graph found in local search, reverting.")
                best_overall_A = None
                best_overall_edges = 0

    # Convert best_A to list of edges
    edges_list = []
    if best_overall_A is not None and is_valid(best_overall_A):
        for u in range(n):
            for v in range(u + 1, n):
                if best_overall_A[u, v]:
                    edges_list.append([u, v])
    else:
        # Fallback to seed
        A = initialize()
        for u in range(n):
            for v in range(u + 1, n):
                if A[u, v]:
                    edges_list.append([u, v])
                    
    return edges_list
