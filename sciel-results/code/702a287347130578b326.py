import random
import time

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 115
    
    # Precompute triples for K3,3 check
    triples = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                triples.append((i, j, k))
                
    def is_valid(adj_mask):
        """Returns True if graph is K3,3 free."""
        for u, v, w in triples:
            if (adj_mask[u] & adj_mask[v] & adj_mask[w]).bit_count() >= 3:
                return False
        return True

    def count_edges(adj_mask):
        return sum(m.bit_count() for m in adj_mask) // 2

    best_edges = []
    best_count = 0
    best_adj = [0] * n

    # Strategy: Multiple Random Restarts with Greedy Construction and Hill Climbing
    for trial in range(100):
        if time.time() > deadline:
            break
            
        # Initialize adj_mask
        adj_mask = [0] * n
        
        # Phase 1: Greedy Construction
        # Shuffle all potential edges and add them if valid
        edges_to_try = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(edges_to_try)
        
        for u, v in edges_to_try:
            if time.time() > deadline:
                break
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            if not is_valid(adj_mask):
                adj_mask[u] &= ~(1 << v)
                adj_mask[v] &= ~(1 << u)
            # else: keep it

        current_count = count_edges(adj_mask)
        
        # Phase 2: Hill Climbing (Add non-edges)
        improved = True
        while improved:
            improved = False
            non_edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if not (adj_mask[i] >> j) & 1:
                        non_edges.append((i, j))
            
            if not non_edges:
                break
                
            random.shuffle(non_edges)
            
            for u, v in non_edges:
                if time.time() > deadline:
                    break
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                if is_valid(adj_mask):
                    current_count += 1
                    improved = True
                else:
                    adj_mask[u] &= ~(1 << v)
                    adj_mask[v] &= ~(1 << u)
                    
        if current_count > best_count:
            best_count = current_count
            best_adj = adj_mask[:]
            best_edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if (adj_mask[i] >> j) & 1:
                        best_edges.append([i, j])

        # Phase 3: Kick and Re-optimize
        # Remove a random edge to escape local optimum
        if time.time() > deadline:
            break
            
        edges_list = [(i, j) for i in range(n) for j in range(i+1, n) if (adj_mask[i] >> j) & 1]
        if edges_list:
            u, v = random.choice(edges_list)
            adj_mask[u] &= ~(1 << v)
            adj_mask[v] &= ~(1 << u)
            current_count -= 1
            
            # Re-run Hill Climbing
            improved = True
            while improved:
                improved = False
                non_edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if not (adj_mask[i] >> j) & 1:
                            non_edges.append((i, j))
                random.shuffle(non_edges)
                
                for u2, v2 in non_edges:
                    if time.time() > deadline:
                        break
                    adj_mask[u2] |= (1 << v2)
                    adj_mask[v2] |= (1 << u2)
                    if is_valid(adj_mask):
                        current_count += 1
                        improved = True
                    else:
                        adj_mask[u2] &= ~(1 << v2)
                        adj_mask[v2] &= ~(1 << u2)
                        
            if current_count > best_count:
                best_count = current_count
                best_adj = adj_mask[:]
                best_edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if (adj_mask[i] >> j) & 1:
                            best_edges.append([i, j])

    if best_count == 0:
        # Fallback: Return a valid sparse graph
        return [[i, j] for i in range(n) for j in range(i+1, n) if (i+j)%2==0]
        
    return best_edges
