import numpy as np
import time
import random
import heapq

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    n = adjacency.shape[0]
    if n == 0:
        return np.array([])
        
    # Precompute neighbor lists as lists of ints for speed in Python loops
    neighbors = [np.where(adjacency[i])[0].tolist() for i in range(n)]
    
    def greedy_mis_size(order):
        """
        order: list of vertices in the order they should be considered (first is highest priority)
        Returns the size of the independent set.
        """
        selected = [False] * n
        count = 0
        for v in order:
            if not selected[v]:
                selected[v] = True
                count += 1
                for u in neighbors[v]:
                    selected[u] = True
        return count

    def dynamic_greedy_order(seed=0):
        """
        Simulates dynamic min-degree greedy MIS.
        Returns an order where selected vertices come first (in selection order),
        followed by blocked vertices.
        """
        rng = np.random.default_rng(seed)
        available = set(range(n))
        # Calculate initial degrees in available graph
        # Since we remove vertices, degrees change.
        # We need to track degree of each available vertex.
        deg = {i: len(neighbors[i]) for i in range(n)}
        # But deg should be count of neighbors in 'available'
        # Initial deg is just degree in G.
        
        # Use a heap for min-degree selection
        # (degree, vertex)
        heap = [(deg[i], i) for i in range(n)]
        heapq.heapify(heap)
        
        selected_order = []
        blocked = set()
        
        while available:
            # Pop min degree vertex
            d, v = heapq.heappop(heap)
            if v not in available:
                continue # Stale entry
            
            # Select v
            selected_order.append(v)
            available.remove(v)
            
            # Remove neighbors of v
            for u in neighbors[v]:
                if u in available:
                    available.remove(u)
                    blocked.add(u)
                    # Update degree of u? No, u is removed.
                    # We need to update degrees of remaining available vertices that are neighbors of u?
                    # No, the graph shrinks.
                    # When we remove u, we must decrease the degree of its neighbors in 'available'.
                    # But u is removed from available, so its neighbors' degrees in the *remaining* graph decrease by 1 if they are connected to u.
                    # Wait, the standard MIS greedy removes v and N(v).
                    # The "degree" of a vertex w in the remaining graph is the number of its neighbors in the remaining graph.
                    # If we remove u, and w is neighbor of u, w's degree decreases by 1.
                    
            # This implementation is tricky to keep the heap consistent with dynamic degree updates efficiently in Python without complex data structures.
            # For N=200, we can just re-calculate min degree from scratch each step?
            # O(N^2) total. 200^2 = 40,000 ops. Very fast.
            
            # Let's restart with a simpler O(N^2) dynamic greedy for correctness and simplicity.
            pass 

    # Simpler O(N^2) Dynamic Greedy
    def dynamic_greedy_order_simple(seed=0):
        rng = np.random.default_rng(seed)
        remaining = set(range(n))
        order = []
        
        while remaining:
            # Find vertex with min degree in subgraph induced by 'remaining'
            min_deg = float('inf')
            best_v = None
            for v in remaining:
                deg = 0
                for u in neighbors[v]:
                    if u in remaining:
                        deg += 1
                # Tie breaking: random or by id
                if deg < min_deg or (deg == min_deg and (best_v is None or rng.random() < 0.5)):
                    min_deg = deg
                    best_v = v
            
            # Select best_v
            order.append(best_v)
            remaining.remove(best_v)
            # Remove neighbors
            for u in neighbors[best_v]:
                if u in remaining:
                    remaining.remove(u)
        
        # The 'order' contains selected vertices.
        # We need to add blocked vertices to make a full permutation for the decoder.
        # Blocked vertices are those in neighbors of selected vertices.
        # Actually, any vertex not in 'order' is blocked or was never selected.
        # In this greedy, all vertices are either selected or removed as neighbor.
        # So 'order' is the selected set.
        # We need to append the rest to the order list so the decoder has a total order.
        # The decoder skips neighbors of selected.
        # If we put blocked vertices at the end, they will be skipped anyway if they are neighbors of earlier selected vertices.
        # But what if a blocked vertex is NOT a neighbor of the *first* selected vertex, but is a neighbor of the *second*?
        # The decoder processes sequentially.
        # If we put selected vertices first (in the order they were picked), then the decoder will pick them in that order (unless a later one is a neighbor of an earlier one, which is impossible by construction).
        # Then the decoder will encounter the blocked vertices. Since they are neighbors of some selected vertex (which is earlier in the list), they will be skipped.
        # So appending blocked vertices at the end is safe.
        
        remaining_vertices = [v for v in range(n) if v not in set(order)]
        # Shuffle remaining vertices to break ties randomly?
        rng.shuffle(remaining_vertices)
        order.extend(remaining_vertices)
        
        return order

    # Initial order: Dynamic Greedy
    current_order = dynamic_greedy_order_simple(42)
    current_size = greedy_mis_size(current_order)
    
    best_order = current_order[:]
    best_size = current_size
    
    rng = np.random.default_rng(42)
    
    # Increase search budget significantly
    num_restarts = 20
    steps_per_restart = 50000
    
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        if r == 0:
            # Already set current_order
            pass
        elif r == 1:
            order = list(rng.permutation(n))
        elif r == 2:
            # Static Min Degree
            degrees = adjacency.sum(axis=1)
            order = list(np.argsort(degrees))
        elif r == 3:
            # Static Max Degree
            degrees = adjacency.sum(axis=1)
            order = list(np.argsort(-degrees))
        elif r == 4:
            # Dynamic Greedy with different seed
            order = dynamic_greedy_order_simple(100 + r)
        else:
            # Noisy Degree
            degrees = adjacency.sum(axis=1)
            noisy_deg = degrees + rng.normal(0, 1, n)
            order = list(np.argsort(noisy_deg))
            
        current_size = greedy_mis_size(order)
        
        if current_size > best_size:
            best_size = current_size
            best_order = order[:]
            
        # Local Search Phase
        for step in range(steps_per_restart):
            if time.time() > deadline - 5:
                break
                
            # Pick two random indices to swap
            i, j = rng.integers(0, n, 2)
            if i == j:
                continue
                
            order[i], order[j] = order[j], order[i]
            
            new_size = greedy_mis_size(order)
            
            if new_size >= current_size:
                current_size = new_size
                if current_size > best_size:
                    best_size = current_size
                    best_order = order[:]
            else:
                # Revert
                order[i], order[j] = order[j], order[i]

    # Convert best_order to scores
    scores = np.zeros(n)
    for rank, v in enumerate(best_order):
        scores[v] = n - rank
        
    return scores
