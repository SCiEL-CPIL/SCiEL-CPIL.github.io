import numpy as np
import time
import random

def construct():
    n = 40
    m = 401
    deadline = time.time() + 115.0

    def compute_books(A):
        A2 = A @ A
        np.fill_diagonal(A2, 0)
        return A2

    def get_max_book(A2):
        return int(np.max(A2))

    best_edges = None
    best_max_b = float('inf')

    # Pre-compute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Multiple restarts with different initializations
    initializers = [
        lambda: "bipartite_plus_one",
        lambda: "random",
        lambda: "bipartite_plus_random_internal"
    ]

    for restart in range(10):
        if time.time() > deadline - 5:
            break
            
        A = np.zeros((n, n), dtype=np.int32)
        edges = set()
        
        init_type = initializers[restart % len(initializers)]()
        
        if init_type == "bipartite_plus_one":
            left = list(range(20))
            right = list(range(20, 40))
            for u in left:
                for v in right:
                    edges.add((u, v))
                    A[u, v] = 1
                    A[v, u] = 1
            # Add 1 internal edge
            u = random.choice(left)
            v = random.choice(left)
            while u == v:
                v = random.choice(left)
            new_edge = (min(u, v), max(u, v))
            edges.add(new_edge)
            A[u, v] = 1
            A[v, u] = 1
        elif init_type == "random":
            random.shuffle(all_edges)
            for u, v in all_edges[:m]:
                edges.add((u, v))
                A[u, v] = 1
                A[v, u] = 1
        else:
            # Bipartite + a few internal edges, remove some bipartite to keep m=401
            left = list(range(20))
            right = list(range(20, 40))
            # Start with full bipartite
            for u in left:
                for v in right:
                    edges.add((u, v))
                    A[u, v] = 1
                    A[v, u] = 1
            # Remove 3 random bipartite edges
            bip_edges = list(edges)
            random.shuffle(bip_edges)
            for e in bip_edges[:3]:
                edges.discard(e)
                A[e[0], e[1]] = 0
                A[e[1], e[0]] = 0
            # Add 4 random internal edges
            for _ in range(4):
                if random.random() < 0.5:
                    u = random.choice(left)
                    v = random.choice(left)
                    while u == v:
                        v = random.choice(left)
                else:
                    u = random.choice(right)
                    v = random.choice(right)
                    while u == v:
                        v = random.choice(right)
                new_edge = (min(u, v), max(u, v))
                if new_edge not in edges:
                    edges.add(new_edge)
                    A[u, v] = 1
                    A[v, u] = 1

        A2 = compute_books(A)
        max_b = get_max_book(A2)
        
        if max_b < best_max_b:
            best_max_b = max_b
            best_edges = set(edges)
        
        # Simulated Annealing
        T = 5.0
        T_min = 0.01
        cooling = 0.999
        
        # Local search loop
        while time.time() < deadline - 2:
            # Pick a random edge to remove
            rem_u, rem_v = random.choice(list(edges))
            
            # Pick a random non-edge to add
            added = False
            for _ in range(20):
                add_u = random.randint(0, n - 1)
                add_v = random.randint(0, n - 1)
                if add_u == add_v:
                    continue
                add_edge = (min(add_u, add_v), max(add_u, add_v))
                if add_edge not in edges:
                    added = True
                    break
            
            if not added:
                continue

            # Apply swap
            A[rem_u, rem_v] = 0
            A[rem_v, rem_u] = 0
            A[add_u, add_v] = 1
            A[add_v, add_u] = 1
            
            A2_new = compute_books(A)
            max_new = get_max_book(A2_new)
            
            delta = max_new - max_b
            
            if delta < 0 or (T > T_min and random.random() < np.exp(-delta / T)):
                edges.discard((rem_u, rem_v))
                edges.add(add_edge)
                A2 = A2_new
                max_b = max_new
                
                if max_b < best_max_b:
                    best_max_b = max_b
                    best_edges = set(edges)
                    T = max(T_min, T * 0.95)
                else:
                    T = max(T_min, T * cooling)
            else:
                A[rem_u, rem_v] = 1
                A[rem_v, rem_u] = 1
                A[add_u, add_v] = 0
                A[add_v, add_u] = 0
                T = max(T_min, T * cooling)

    return [[u, v] for u, v in best_edges]
