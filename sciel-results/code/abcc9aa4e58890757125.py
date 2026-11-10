import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    n = 31
    m = 91
    
    def count_c4(A):
        # A is n x n adjacency matrix (0/1)
        # CN = A @ A
        # C4 = 0.5 * sum_{i<j} (CN[i,j] * (CN[i,j]-1))
        CN = A @ A
        # Extract upper triangle
        iu = np.triu_indices(n, 1)
        vals = CN[iu]
        # Sum of k*(k-1)
        c4_count = 0.5 * np.sum(vals * (vals - 1))
        return int(c4_count)

    def get_edges(A):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if A[i, j] == 1:
                    edges.append([i, j])
        return edges

    # Initialize with random graph
    best_edges = None
    best_c4 = float('inf')
    
    # Try a few random restarts
    for restart in range(10):
        # Random graph with 91 edges
        A = np.zeros((n, n), dtype=np.int8)
        edges = []
        # Generate random pairs
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_pairs)
        for u, v in all_pairs[:m]:
            A[u, v] = 1
            A[v, u] = 1
            edges.append([u, v])
        
        current_c4 = count_c4(A)
        
        if current_c4 < best_c4:
            best_c4 = current_c4
            best_edges = [e[:] for e in edges]
            if best_c4 == 0:
                return best_edges
        
        # Simulated Annealing
        T = 10.0
        T_min = 0.1
        alpha = 0.999
        
        while time.time() < deadline:
            # Pick a random edge to remove
            edge_to_remove = random.choice(edges)
            u, v = edge_to_remove[0], edge_to_remove[1]
            
            # Pick a random non-edge to add
            # To speed up, we can precompute non-edges or just try random pairs
            # Given n=31, total pairs = 465. 91 edges. 374 non-edges.
            # We can just pick random i, j and check if not edge.
            
            found_non_edge = False
            x, y = 0, 0
            while not found_non_edge:
                x = random.randint(0, n - 1)
                y = random.randint(0, n - 1)
                if x != y and A[x, y] == 0:
                    found_non_edge = True
            
            # Calculate delta C4
            # Remove (u,v), Add (x,y)
            
            # Update A
            A[u, v] = 0
            A[v, u] = 0
            A[x, y] = 1
            A[y, x] = 1
            
            # Update edges list
            edges.remove(edge_to_remove)
            new_edge = [x, y] if x < y else [y, x]
            edges.append(new_edge)
            
            new_c4 = count_c4(A)
            
            delta = new_c4 - current_c4
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_c4 = new_c4
                if current_c4 < best_c4:
                    best_c4 = current_c4
                    best_edges = [e[:] for e in edges]
                    if best_c4 == 0:
                        return best_edges
            else:
                # Revert
                A[u, v] = 1
                A[v, u] = 1
                A[x, y] = 0
                A[y, x] = 0
                
                edges.remove(new_edge)
                edges.append(edge_to_remove)
            
            T *= alpha
            if T < T_min:
                break

    return best_edges
