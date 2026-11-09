import numpy as np
import time
import random

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    def get_edges_from_adj(A):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if A[u][v]:
                    edges.append([u, v])
        return edges
    
    def compute_common(A):
        # C[u][w] = |N(u) ∩ N(w)|
        return A @ A
    
    def can_add(A, C, u, v):
        if A[u][v]:
            return True
        # Check 1: C[u][v] <= 2
        if C[u][v] > 2:
            return False
        # Check 2: for all w in N(v), C[u][w] <= 1
        # N(v) = {w : A[v][w] == 1}
        nv = np.where(A[v] == 1)[0]
        if len(nv) > 0:
            if np.max(C[u, nv]) > 1:
                return False
        # Check 3: for all w in N(u), C[v][w] <= 1
        nu = np.where(A[u] == 1)[0]
        if len(nu) > 0:
            if np.max(C[v, nu]) > 1:
                return False
        return True
    
    def add_edge(A, C, u, v):
        A[u][v] = 1
        A[v][u] = 1
        # Update C: when adding (u,v), for all w, C[u][w] and C[w][u] may change
        # Actually, C = A @ A, so we can just recompute, but that's O(n^3)
        # Better: update incrementally
        # When we add edge (u,v):
        # For any w, the new common neighbors of u and w include v if v is in N(w)
        # And the new common neighbors of v and w include u if u is in N(w)
        # C[u][w] += A[v][w] for all w (since v becomes a neighbor of u, and if w is neighbor of v, then v is common)
        # Wait, let me think again.
        # Before: C_old[u][w] = sum_x A[u][x] * A[w][x]
        # After adding (u,v): A[u][v] becomes 1, A[v][u] becomes 1
        # C_new[u][w] = sum_x A_new[u][x] * A_new[w][x]
        # = sum_x (A_old[u][x] + delta_uv(x)) * A_new[w][x]
        # where delta_uv(x) = 1 if x == v else 0 (since we're adding edge u-v)
        # Actually A_new[u][x] = A_old[u][x] for x != v, and A_new[u][v] = 1
        # Similarly for other entries.
        # This is getting complicated. Let's just recompute C from A.
        # For n=20, A@A is 20x20 matrix multiply, which is fast.
        C[:] = A @ A
    
    def remove_edge(A, C, u, v):
        A[u][v] = 0
        A[v][u] = 0
        C[:] = A @ A
    
    # Greedy construction with multiple random orders
    best_count = 0
    best_A = None
    
    for trial in range(50):
        if time.time() > deadline:
            break
        
        A = np.zeros((n, n), dtype=np.int32)
        C = np.zeros((n, n), dtype=np.int32)
        
        # Random order of all possible edges
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_pairs)
        
        for u, v in all_pairs:
            if time.time() > deadline:
                break
            if not A[u][v]:
                if can_add(A, C, u, v):
                    add_edge(A, C, u, v)
        
        count = int(np.sum(A)) // 2
        if count > best_count:
            best_count = count
            best_A = A.copy()
    
    # Local search: try edge swaps to improve
    if best_A is not None:
        A = best_A.copy()
        C = A @ A
        current_count = int(np.sum(A)) // 2
        
        # Try multiple rounds of local search
        for iteration in range(200):
            if time.time() > deadline:
                break
            
            # Find a random non-edge to try adding
            non_edges = [(i, j) for i in range(n) for j in range(i + 1, n) if not A[i][j]]
            if not non_edges:
                break
            
            # Try to add a non-edge, possibly removing an edge first
            # Strategy: pick a random non-edge (u,v), check if it can be added.
            # If not, try removing one edge from the "blocking" set and then adding (u,v).
            u, v = random.choice(non_edges)
            
            if can_add(A, C, u, v):
                add_edge(A, C, u, v)
                current_count += 1
                if current_count > best_count:
                    best_count = current_count
                    best_A = A.copy()
                continue
            
            # Try to find an edge to remove that would allow adding (u,v)
            # The blocking condition is either:
            # 1. C[u][v] > 2: need to remove an edge incident to a common neighbor
            # 2. Some w in N(v) has C[u][w] > 1: need to remove an edge
            # 3. Some w in N(u) has C[v][w] > 1: need to remove an edge
            
            # Try removing a random edge and see if (u,v) can be added
            all_edges = [(i, j) for i in range(n) for j in range(i + 1, n) if A[i][j]]
            if not all_edges:
                continue
            
            # Try up to 5 random edge removals
            for _ in range(5):
                if time.time() > deadline:
                    break
                ru, rv = random.choice(all_edges)
                # Save state
                A_save = A.copy()
                C_save = C.copy()
                count_save = current_count
                
                remove_edge(A, C, ru, rv)
                current_count -= 1
                
                if can_add(A, C, u, v):
                    add_edge(A, C, u, v)
                    current_count += 1
                    if current_count > best_count:
                        best_count = current_count
                        best_A = A.copy()
                else:
                    # Restore
                    A[:] = A_save
                    C[:] = C_save
                    current_count = count_save
    
    return get_edges_from_adj(best_A if best_A is not None else np.zeros((n, n), dtype=np.int32))
