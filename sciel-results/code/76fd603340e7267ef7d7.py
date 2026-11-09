import random
import time

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    # Bitmask representation: adj[i] is an integer where bit j is 1 if edge (i,j) exists.
    # adj[i] & (1 << j) != 0 means edge exists.
    
    def is_k33_free(adj):
        """Check if graph is K3,3 free using bitmasks."""
        # No 3 vertices have 3 common neighbors.
        # For any triple (u, v, w), common neighbors = adj[u] & adj[v] & adj[w].
        # If bit_count(common) >= 3, violation.
        for u in range(n):
            for v in range(u + 1, n):
                uv_common = adj[u] & adj[v]
                # Optimization: if uv_common has < 3 bits, no triple including u,v can violate?
                # Not necessarily, the triple could be (u, v, w).
                # We need to check all triples.
                # But we can prune: if adj[u] & adj[v] has < 3 bits, then for any w,
                # adj[u] & adj[v] & adj[w] is a subset of adj[u] & adj[v], so it has < 3 bits.
                # So we only need to check triples where the first two have >= 3 common neighbors?
                # No, the triple is (u,v,w). The common neighbors of u,v,w is a subset of common neighbors of u,v.
                # So if popcount(adj[u] & adj[v]) < 3, then for any w, popcount(adj[u] & adj[v] & adj[w]) < 3.
                # Thus, we can skip v for this u if popcount(adj[u] & adj[v]) < 3.
                if uv_common.bit_count() < 3:
                    continue
                for w in range(v + 1, n):
                    common = uv_common & adj[w]
                    if common.bit_count() >= 3:
                        return False
        return True

    def count_edges(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2

    def try_add_edge(adj, u, v):
        if adj[u] & (1 << v):
            return False
        # Add edge
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
        # Check validity
        # To be faster, we could only check triples involving u or v, but for n=20, full check is fine.
        # Actually, let's just do full check for simplicity, it's very fast now.
        if is_k33_free(adj):
            return True
        else:
            # Revert
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            return False

    best_adj = [0] * n
    best_count = 0

    # Try multiple random restarts with greedy construction
    for restart in range(2000):
        if time.time() > deadline:
            break

        # Initialize with random sparse graph
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.3:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)

        curr_count = count_edges(adj)

        # Greedy add edges
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Create list of non-edges
            non_edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if not (adj[i] & (1 << j)):
                        non_edges.append((i, j))
            
            random.shuffle(non_edges)
            
            for u, v in non_edges:
                if time.time() > deadline:
                    break
                if try_add_edge(adj, u, v):
                    curr_count += 1
                    improved = True
                    if curr_count > best_count:
                        best_count = curr_count
                        best_adj = adj[:]

        # Optional: Local search with swaps to escape local optima
        # Since greedy is fast, we can do a few swap iterations
        for _ in range(50):
            if time.time() > deadline:
                break
            
            # Pick a random edge to remove
            existing_edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i] & (1 << j):
                        existing_edges.append((i, j))
            
            if not existing_edges:
                break
                
            u1, v1 = random.choice(existing_edges)
            
            # Pick a random non-edge to add
            non_edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if not (adj[i] & (1 << j)):
                        non_edges.append((i, j))
            
            if not non_edges:
                break
                
            u2, v2 = random.choice(non_edges)
            
            # Perform swap
            adj[u1] &= ~(1 << v1)
            adj[v1] &= ~(1 << u1)
            adj[u2] |= (1 << v2)
            adj[v2] |= (1 << u2)
            
            if is_k33_free(adj):
                # Valid swap. Count doesn't change, but structure might allow more additions later.
                # We don't update best_count here as it's same count.
                # But we keep the state for potential future additions.
                pass
            else:
                # Revert
                adj[u1] |= (1 << v1)
                adj[v1] |= (1 << u1)
                adj[u2] &= ~(1 << v2)
                adj[v2] &= ~(1 << u2)
            
            # Try to add more edges after swap
            improved2 = True
            while improved2 and time.time() < deadline:
                improved2 = False
                non_edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if not (adj[i] & (1 << j)):
                            non_edges.append((i, j))
                random.shuffle(non_edges)
                for u, v in non_edges:
                    if time.time() > deadline:
                        break
                    if try_add_edge(adj, u, v):
                        curr_count += 1
                        improved2 = True
                        if curr_count > best_count:
                            best_count = curr_count
                            best_adj = adj[:]

    # Convert best_adj to list of edges
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i] & (1 << j):
                edges.append([i, j])
                
    return edges
