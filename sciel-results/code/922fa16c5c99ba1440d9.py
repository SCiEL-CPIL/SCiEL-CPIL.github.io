import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    n = 20
    max_edges = n * (n - 1) // 2
    
    # Precompute all pairs
    pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    def can_add_c4_free(adj, u, v):
        # adj is list of int masks
        # Check if u and v share a neighbor
        return (adj[u] & adj[v]).bit_count() == 0

    def get_edges_from_adj(adj1, adj2):
        edges = []
        for u in range(n):
            mask1 = adj1[u]
            mask2 = adj2[u]
            for v in range(u + 1, n):
                if (mask1 & (1 << v)) or (mask2 & (1 << v)):
                    edges.append([u, v])
        return edges

    best_edge_count = 0
    best_adj1 = [0] * n
    best_adj2 = [0] * n

    def greedy_init():
        adj1 = [0] * n
        adj2 = [0] * n
        shuffled_pairs = pairs[:]
        random.shuffle(shuffled_pairs)
        
        for u, v in shuffled_pairs:
            # Try to add to G1
            if can_add_c4_free(adj1, u, v):
                adj1[u] |= (1 << v)
                adj1[v] |= (1 << u)
            # Try to add to G2
            elif can_add_c4_free(adj2, u, v):
                adj2[u] |= (1 << v)
                adj2[v] |= (1 << u)
        
        count1 = sum(a.bit_count() for a in adj1) // 2
        count2 = sum(a.bit_count() for a in adj2) // 2
        return adj1, adj2, count1, count2

    def random_neighbor(adj, u):
        mask = adj[u]
        if not mask:
            return -1
        # Get list of neighbors
        neighbors = [v for v in range(n) if (mask >> v) & 1]
        return random.choice(neighbors)

    while time.time() < deadline:
        adj1, adj2, count1, count2 = greedy_init()
        current_count = count1 + count2
        
        # Local Search / Simulated Annealing
        T = 1.0
        T_min = 0.01
        cooling = 0.9995
        no_improve = 0
        
        for step in range(50000):
            if time.time() > deadline:
                break
            
            if current_count > best_edge_count:
                best_edge_count = current_count
                best_adj1 = adj1[:]
                best_adj2 = adj2[:]
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve > 2000:
                break
            
            move_type = random.random()
            
            if move_type < 0.3:
                # Add edge to G1
                u, v = random.choice(pairs)
                if not (adj1[u] & (1 << v)):
                    if can_add_c4_free(adj1, u, v):
                        adj1[u] |= (1 << v)
                        adj1[v] |= (1 << u)
                        count1 += 1
                        current_count += 1
            elif move_type < 0.6:
                # Add edge to G2
                u, v = random.choice(pairs)
                if not (adj2[u] & (1 << v)):
                    if can_add_c4_free(adj2, u, v):
                        adj2[u] |= (1 << v)
                        adj2[v] |= (1 << u)
                        count2 += 1
                        current_count += 1
            elif move_type < 0.75:
                # Remove edge from G1
                u = random.randint(0, n - 1)
                v = random_neighbor(adj1, u)
                if v != -1:
                    adj1[u] &= ~(1 << v)
                    adj1[v] &= ~(1 << u)
                    count1 -= 1
                    current_count -= 1
            elif move_type < 0.9:
                # Remove edge from G2
                u = random.randint(0, n - 1)
                v = random_neighbor(adj2, u)
                if v != -1:
                    adj2[u] &= ~(1 << v)
                    adj2[v] &= ~(1 << u)
                    count2 -= 1
                    current_count -= 1
            else:
                # Move edge from G1 to G2
                u = random.randint(0, n - 1)
                v = random_neighbor(adj1, u)
                if v != -1:
                    # Remove from G1
                    adj1[u] &= ~(1 << v)
                    adj1[v] &= ~(1 << u)
                    count1 -= 1
                    current_count -= 1
                    # Try to add to G2
                    if can_add_c4_free(adj2, u, v):
                        adj2[u] |= (1 << v)
                        adj2[v] |= (1 << u)
                        count2 += 1
                        current_count += 1
                    else:
                        # Put back in G1 if we can? Or just leave it out?
                        # Leaving it out is a valid move (removal).
                        # But we wanted to move. If it fails, it's just a removal.
                        # To make it a true move, we could try to put it back if G2 fails.
                        # But for simplicity, treat as removal if G2 fails.
                        pass

            T *= cooling
            if T < T_min:
                T = T_min
        
        # Polish the best found graph
        # Try to add any remaining edges to best_adj1 and best_adj2
        # This is a greedy polish
    
    # Polish the best solution
    adj1 = best_adj1[:]
    adj2 = best_adj2[:]
    improved = True
    while improved:
        improved = False
        for u, v in pairs:
            if not (adj1[u] & (1 << v)) and can_add_c4_free(adj1, u, v):
                adj1[u] |= (1 << v)
                adj1[v] |= (1 << u)
                improved = True
            if not (adj2[u] & (1 << v)) and can_add_c4_free(adj2, u, v):
                adj2[u] |= (1 << v)
                adj2[v] |= (1 << u)
                improved = True
    
    # Combine edges
    edges = get_edges_from_adj(adj1, adj2)
    
    return edges
