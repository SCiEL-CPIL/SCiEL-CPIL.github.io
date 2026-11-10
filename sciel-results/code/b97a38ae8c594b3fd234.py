import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    n = 20
    adj = [0] * n
    # common_counts[u][v] = number of common neighbors of u and v
    # This is symmetric.
    common_counts = [[0] * n for _ in range(n)]

    def popcount(x):
        return x.bit_count()

    def get_neighbors(u):
        """Return list of neighbors of u."""
        result = []
        temp = adj[u]
        while temp:
            lsb = temp & -temp
            w = lsb.bit_length() - 1
            result.append(w)
            temp ^= lsb
        return result

    def update_common_counts(u, v, add):
        """
        Update common_counts when adding or removing edge (u,v).
        If add=True, we are adding edge (u,v).
        If add=False, we are removing edge (u,v).
        """
        delta = 1 if add else -1
        
        # For each w in N(v) \ {u}, C(u,w) changes by delta
        # Because v becomes (or stops being) a common neighbor of u and w.
        # v is in N(u) (after add) and v is in N(w) (since w in N(v)).
        neighbors_v = get_neighbors(v)
        for w in neighbors_v:
            if w == u:
                continue
            common_counts[u][w] += delta
            common_counts[w][u] += delta

        # For each w in N(u) \ {v}, C(v,w) changes by delta
        neighbors_u = get_neighbors(u)
        for w in neighbors_u:
            if w == v:
                continue
            common_counts[v][w] += delta
            common_counts[w][v] += delta

    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        update_common_counts(u, v, True)

    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        update_common_counts(u, v, False)

    def can_add(u, v):
        if (adj[u] >> v) & 1:
            return False
        
        # 1. Pair (u,v) common neighbors must be <= 2
        if common_counts[u][v] > 2:
            return False
            
        # 2. For all w in N(v) \ {u}, pair (u,w) gains v as common neighbor.
        #    So current common(u,w) must be <= 1.
        neighbors_v = get_neighbors(v)
        for w in neighbors_v:
            if w == u:
                continue
            if common_counts[u][w] > 1:
                return False
                
        # 3. For all w in N(u) \ {v}, pair (v,w) gains u as common neighbor.
        #    So current common(v,w) must be <= 1.
        neighbors_u = get_neighbors(u)
        for w in neighbors_u:
            if w == v:
                continue
            if common_counts[v][w] > 1:
                return False
                
        return True

    def edge_count():
        total_deg = 0
        for i in range(n):
            total_deg += popcount(adj[i])
        return total_deg // 2

    def reset_adj_and_counts(adj_init, cc_init):
        nonlocal adj, common_counts
        adj = [x for x in adj_init]
        common_counts = [row[:] for row in cc_init]

    def greedy_init(order):
        nonlocal adj, common_counts
        adj = [0] * n
        common_counts = [[0] * n for _ in range(n)]
        
        # 'order' is a list of edges to try
        for u, v in order:
            if can_add(u, v):
                add_edge(u, v)
        return edge_count()

    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    best_adj = [0] * n
    best_cc = [[0] * n for _ in range(n)]
    best_count = 0

    # Multiple greedy restarts
    # Use different random shuffles
    for _ in range(50):
        shuffled_edges = all_edges[:]
        random.shuffle(shuffled_edges)
        cnt = greedy_init(shuffled_edges)
        if cnt > best_count:
            best_count = cnt
            best_adj = adj[:]
            best_cc = [row[:] for row in common_counts]

    # Simulated Annealing
    T = 10.0
    cooling = 0.998
    min_T = 0.05
    stagnation = 0
    max_stagnation = 2000
    
    # Initialize current state to best found
    reset_adj_and_counts(best_adj, best_cc)
    current_count = best_count

    while time.time() < deadline:
        if T < min_T:
            T = 5.0
            reset_adj_and_counts(best_adj, best_cc)
            current_count = best_count
            stagnation = 0
            continue

        r = random.random()

        if r < 0.50:
            # Try to add an edge
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if (adj[u] >> v) & 1:
                continue
            
            if can_add(u, v):
                add_edge(u, v)
                new_count = current_count + 1
                if new_count > best_count:
                    best_count = new_count
                    best_adj = adj[:]
                    best_cc = [row[:] for row in common_counts]
                    stagnation = 0
                else:
                    stagnation += 1
                current_count = new_count
            else:
                stagnation += 1
                
        elif r < 0.80:
            # Try to remove an edge
            u = random.randint(0, n - 1)
            neighbors = adj[u]
            if neighbors == 0:
                continue
            neigh_list = get_neighbors(u)
            if not neigh_list:
                continue
            v = random.choice(neigh_list)
            
            remove_edge(u, v)
            new_count = current_count - 1
            # Accept with probability exp(-delta/T), delta = 1
            if random.random() < math.exp(-1.0 / T):
                current_count = new_count
                # No need to update best if count decreased, but keep state
                # We don't update best_adj here as it's worse
                stagnation += 1
            else:
                add_edge(u, v)
                current_count = new_count + 1 # Restore
                stagnation += 1
        else:
            # Try a swap: remove (u1, v1), add (u2, v2)
            u1 = random.randint(0, n - 1)
            neighbors1 = adj[u1]
            if neighbors1 == 0:
                continue
            neigh_list1 = get_neighbors(u1)
            if not neigh_list1:
                continue
            v1 = random.choice(neigh_list1)
            
            u2 = random.randint(0, n - 1)
            v2 = random.randint(0, n - 1)
            if u2 == v2:
                continue
            if (adj[u2] >> v2) & 1:
                continue
            
            remove_edge(u1, v1)
            
            if can_add(u2, v2):
                add_edge(u2, v2)
                # Count remains same
                new_count = current_count
                # Check if this swap leads to a better structure? 
                # Since count is same, we only accept if it helps future moves?
                # Standard SA accepts moves with delta=0 with prob 1.
                # So we keep the new state.
                current_count = new_count
                stagnation += 1 # Or reset if we consider it progress? Usually not.
            else:
                add_edge(u1, v1)
                stagnation += 1
                
        T *= cooling
        
        if stagnation > max_stagnation:
            T = 5.0
            reset_adj_and_counts(best_adj, best_cc)
            current_count = best_count
            stagnation = 0

    # Final local-search polish pass
    reset_adj_and_counts(best_adj, best_cc)
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        # Try all single-edge additions
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    continue
                if can_add(u, v):
                    add_edge(u, v)
                    improved = True
                    if edge_count() > best_count:
                        best_count = edge_count()
                        best_adj = adj[:]
                        best_cc = [row[:] for row in common_counts]
        
        # Try all swaps
        if time.time() < deadline - 5:
            for u1 in range(n):
                if time.time() >= deadline - 5:
                    break
                for v1 in range(u1 + 1, n):
                    if not ((adj[u1] >> v1) & 1):
                        continue
                    for u2 in range(n):
                        if u2 == u1:
                            continue
                        for v2 in range(u2 + 1, n):
                            if (adj[u2] >> v2) & 1:
                                continue
                            # Skip if the swap would just re-add the same edge
                            if (u1 == u2 and v1 == v2) or (u1 == v2 and v1 == u2):
                                continue
                            remove_edge(u1, v1)
                            if can_add(u2, v2):
                                add_edge(u2, v2)
                                # Check if this allows further additions?
                                # For now, just accept if valid.
                                # To improve, we should check if this state is "better" in terms of potential.
                                # But simple local search just takes it.
                                # However, we only update best if count increases.
                                # Here count is same.
                                # We might want to keep this state if it's a "dead end" breaker.
                                # But for simplicity, we revert if it doesn't immediately improve best_count.
                                # Actually, let's just try to add more edges after the swap.
                                # If we can add an edge after the swap, it's a win.
                                # This is complex. Let's stick to single additions and direct swaps that don't change count but might unblock.
                                # Given time, we'll just revert if no immediate count gain, as SA handles exploration.
                                add_edge(u1, v1)
                                remove_edge(u2, v2)
                            else:
                                add_edge(u1, v1)
        
        if improved:
            reset_adj_and_counts(best_adj, best_cc)
        else:
            break

    # Convert best_adj back to list of edges
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
                
    return edges
