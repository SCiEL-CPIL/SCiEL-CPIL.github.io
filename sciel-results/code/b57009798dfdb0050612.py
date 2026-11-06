import time
import random

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    def is_valid(adj):
        # Check if graph is K3,3 free
        # No 3 vertices have 3 common neighbors
        for i in range(n):
            ai = adj[i]
            for j in range(i + 1, n):
                aj = adj[j]
                common_ij = ai & aj
                if common_ij.bit_count() < 2:
                    continue
                for k in range(j + 1, n):
                    if (common_ij & adj[k]).bit_count() >= 3:
                        return False
        return True

    def count_edges(adj):
        return sum(a.bit_count() for a in adj) // 2

    def greedy_fill(adj):
        """
        Try to add edges in a random order until no more can be added.
        Returns the number of edges added.
        """
        pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(pairs)
        added_count = 0
        for u, v in pairs:
            if time.time() > deadline:
                break
            if adj[u] & (1 << v):
                continue
            
            # Try adding edge
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            if is_valid(adj):
                added_count += 1
            else:
                # Rollback
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        return added_count

    best_adj = [0] * n
    best_count = 0
    current_adj = [0] * n
    
    # Initial greedy fill from empty graph
    greedy_fill(current_adj)
    current_count = count_edges(current_adj)
    if current_count > best_count:
        best_count = current_count
        best_adj = current_adj[:]

    # Local search: remove a random edge, then greedy fill again
    while time.time() < deadline:
        # Collect current edges
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if current_adj[u] & (1 << v):
                    edges.append((u, v))
        
        if not edges:
            # Restart if empty
            current_adj = [0] * n
            greedy_fill(current_adj)
            current_count = count_edges(current_adj)
            if current_count > best_count:
                best_count = current_count
                best_adj = current_adj[:]
            continue

        # Remove a random edge
        u, v = random.choice(edges)
        current_adj[u] &= ~(1 << v)
        current_adj[v] &= ~(1 << u)
        
        # Greedy fill again
        greedy_fill(current_adj)
        
        current_count = count_edges(current_adj)
        if current_count > best_count:
            best_count = current_count
            best_adj = current_adj[:]

    # Convert best_adj to list of edges
    result = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u] & (1 << v):
                result.append([u, v])
    return result
