"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random
import time

def construct():
    n = 40
    start_time = time.time()
    time_limit = 110  # Leave some buffer
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    def is_c6_free_fast(adj, u, v):
        """
        Check if adding edge (u, v) creates a C6.
        A C6 exists if there's a path of length 5 between u and v.
        Equivalently, there exist nodes a, b, c such that:
        u-a-b-c-v is a path of length 4 (plus edge u-v makes C6)
        Or more precisely: path u -> x1 -> x2 -> x3 -> x4 -> v of length 5.
        
        Faster approach:
        - Get neighbors of u (excluding v since edge not yet added): Nu
        - Get neighbors of v (excluding u): Nv
        - A C6 exists iff there's a path of length 3 between any node in Nu and any node in Nv
        - Path of length 3: a-b-c where a in Nu, c in Nv, and b is adjacent to both a and c
        """
        # Get neighbors of u and v
        Nu = np.where(adj[u] == 1)[0]
        Nv = np.where(adj[v] == 1)[0]
        
        # Remove u from Nv and v from Nu (shouldn't be there, but safe)
        Nu = Nu[Nu != v]
        Nv = Nv[Nv != u]
        
        if len(Nu) == 0 or len(Nv) == 0:
            return True
        
        # For each a in Nu, c in Nv, check if there's a common neighbor b
        # such that a-b-c is a path (a != b, b != c, a != c)
        # And also ensure the path u-a-b-c-v doesn't repeat vertices (for simple C6)
        # Actually for C6 detection in simple graph, we just need a path of length 5
        # which may have repeated vertices in intermediate steps? No, C6 is a simple cycle.
        # But if there's any walk of length 5, there might be a C6. Actually, we need
        # to be careful. Let's use the standard approach: check if (A^5)[u,v] > 0.
        # But that's slow. Let's use a smarter method.
        
        # Alternative: compute A^2, A^4, then check A^4[u, *] dot A[* , v] > 0
        # A^5[u,v] = sum_k A^4[u,k] * A[k,v]
        # A^4[u,k] = sum_j A^2[u,j] * A^2[j,k]
        
        # Actually, let's just do the matrix power but only compute what we need.
        # A^2 = adj @ adj
        # A^4 = A^2 @ A^2
        # A^5[u,v] = (A^4 @ adj)[u,v] = sum_k A^4[u,k] * adj[k,v]
        
        A2 = adj @ adj
        A4 = A2 @ A2
        # A5[u,v] = sum_k A4[u,k] * adj[k,v]
        # adj[v] is the column for v, i.e., adj[:, v]
        A5_uv = np.dot(A4[u, :], adj[:, v])
        
        return A5_uv == 0
    
    def greedy_construct(seed_val):
        """Greedy construction with given seed."""
        random.seed(seed_val)
        A = np.zeros((n, n), dtype=np.int32)
        edges = []
        
        candidates = list(all_edges)
        random.shuffle(candidates)
        
        for u, v in candidates:
            if time.time() - start_time > time_limit:
                break
            if A[u, v] == 0:
                if is_c6_free_fast(A, u, v):
                    A[u, v] = 1
                    A[v, u] = 1
                    edges.append((u, v))
        
        return A, edges
    
    def local_search(A, edges, max_time_per_search=5.0):
        """Local search to escape local optima."""
        search_start = time.time()
        current_count = len(edges)
        best_local = (current_count, edges.copy(), A.copy())
        
        while time.time() - search_start < max_time_per_search:
            # Try to add an edge
            added = False
            # Try all non-edges systematically but in random order
            non_edges = [(u, v) for u, v in all_edges if A[u, v] == 0]
            random.shuffle(non_edges)
            
            for u, v in non_edges:
                if time.time() - search_start > max_time_per_search:
                    break
                if is_c6_free_fast(A, u, v):
                    A[u, v] = 1
                    A[v, u] = 1
                    edges.append((u, v))
                    current_count += 1
                    added = True
                    break
            
            if added:
                if current_count > best_local[0]:
                    best_local = (current_count, edges.copy(), A.copy())
                continue
            
            # No edge can be added, perform a swap
            if len(edges) < 2:
                break
            
            # Remove a random edge
            idx = random.randint(0, len(edges) - 1)
            e_remove = edges[idx]
            u_r, v_r = e_remove
            A[u_r, v_r] = 0
            A[v_r, u_r] = 0
            edges.pop(idx)
            current_count -= 1
            
            # Try to add a different edge
            swap_success = False
            non_edges = [(u, v) for u, v in all_edges if A[u, v] == 0]
            random.shuffle(non_edges)
            
            for u, v in non_edges:
                if time.time() - search_start > max_time_per_search:
                    break
                if (u, v) == e_remove or (v, u) == e_remove:
                    continue
                if is_c6_free_fast(A, u, v):
                    A[u, v] = 1
                    A[v, u] = 1
                    edges.append((u, v))
                    current_count += 1
                    swap_success = True
                    break
            
            if not swap_success:
                # Revert
                A[u_r, v_r] = 1
                A[v_r, u_r] = 1
                edges.append(e_remove)
                current_count += 1
        
        return best_local[1], best_local[2], best_local[0]
    
    best_edges = []
    best_count = 0
    
    # Multiple restarts with different strategies
    for restart in range(30):
        if time.time() - start_time > time_limit:
            break
        
        seed_val = restart * 7919 + 12345
        
        # Phase 1: Greedy
        A, edges = greedy_construct(seed_val)
        current_count = len(edges)
        
        if current_count > best_count:
            best_count = current_count
            best_edges = edges.copy()
        
        # Phase 2: Local search
        if time.time() - start_time < time_limit - 10:
            edges_ls, A_ls, count_ls = local_search(A, edges, max_time_per_search=3.0)
            if count_ls > best_count:
                best_count = count_ls
                best_edges = edges_ls.copy()
    
    return best_edges
# EVOLVE-BLOCK-END
