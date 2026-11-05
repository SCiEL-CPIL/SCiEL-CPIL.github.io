import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    n = 40
    best_edges = []
    best_count = 0

    # Fast C6 check using bitsets
    def path5(u, v, adj):
        # Checks if there is a simple path of length 5 from u to v
        # u-a-b-c-d-v
        if u == v:
            return False
        a_mask = adj[u] & ~(1 << v)
        while a_mask:
            a_bit = a_mask & -a_mask
            a = a_bit.bit_length() - 1
            a_mask ^= a_bit

            b_mask = adj[a] & ~(1 << u) & ~(1 << v)
            while b_mask:
                b_bit = b_mask & -b_mask
                b = b_bit.bit_length() - 1
                b_mask ^= b_bit

                c_mask = adj[b] & ~(1 << u) & ~(1 << a) & ~(1 << v)
                while c_mask:
                    c_bit = c_mask & -c_mask
                    c = c_bit.bit_length() - 1
                    c_mask ^= c_bit

                    # d must be in N(c) and N(v), and not u, a, b, v
                    d_mask = adj[c] & adj[v] & ~(1 << u) & ~(1 << a) & ~(1 << b) & ~(1 << v)
                    if d_mask:
                        return True
        return False

    def get_edges(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (1 << j) & adj[i]:
                    edges.append([i, j])
        return edges

    def count_edges(adj):
        return sum(bin(adj[i]).count('1') for i in range(n)) // 2

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def has_c6(adj):
        # Check if the graph contains a C6
        for u in range(n):
            for v in range(u + 1, n):
                if (1 << v) & adj[u]:
                    if path5(u, v, adj):
                        return True
        return False

    def greedy_add_all(adj):
        """Try to add all possible missing edges in greedy order."""
        changed = True
        while changed:
            changed = False
            for i in range(n):
                for j in range(i + 1, n):
                    if not ((1 << j) & adj[i]):
                        if not path5(i, j, adj):
                            add_edge(adj, i, j)
                            changed = True
                            if time.time() > deadline:
                                return

    # Initialize with 8 disjoint K5s
    def init_adj():
        adj = [0] * n
        for k in range(8):
            base = 5 * k
            for a in range(5):
                for b in range(a + 1, 5):
                    add_edge(adj, base + a, base + b)
        return adj

    # Simulated Annealing
    def simulated_annealing(adj, start_temp, end_temp, steps):
        current_adj = [x for x in adj]
        current_count = count_edges(current_adj)
        best_local_adj = [x for x in current_adj]
        best_local_count = current_count
        
        T = start_temp
        cooling_rate = (end_temp / start_temp) ** (1.0 / steps)
        
        # Precompute all possible edges
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        for step in range(steps):
            if time.time() > deadline:
                break
            
            # Choose a move type
            # 1. Add an edge (if possible)
            # 2. Remove an edge
            # 3. Swap an edge
            
            # Strategy: Try to add first. If no add possible, try swap or remove.
            # But to allow exploration, we should randomize.
            
            r = random.random()
            
            if r < 0.5:
                # Try to add a random missing edge
                missing = [(u, v) for u, v in all_edges if not ((1 << v) & current_adj[u])]
                if missing:
                    u, v = random.choice(missing)
                    if not path5(u, v, current_adj):
                        add_edge(current_adj, u, v)
                        current_count += 1
                        if current_count > best_local_count:
                            best_local_count = current_count
                            best_local_adj = [x for x in current_adj]
                    # If invalid, do nothing (reject)
            elif r < 0.8:
                # Try to remove a random existing edge
                existing = [(u, v) for u, v in all_edges if (1 << v) & current_adj[u]]
                if existing:
                    u, v = random.choice(existing)
                    remove_edge(current_adj, u, v)
                    current_count -= 1
                    # Accept if better or with probability exp(-delta/T)
                    # Delta E = -1 (we lost an edge, so energy increased by 1 in minimization sense, 
                    # but we are maximizing edges, so "cost" is -edges. 
                    # Let's stick to standard SA: minimize Energy = -Edges.
                    # E_new = -(E_old - 1) = E_old + 1.
                    # Delta E = 1.
                    # Prob = exp(-1/T)
                    if random.random() < math.exp(-1.0 / T):
                        # Accept
                        if current_count > best_local_count:
                            best_local_count = current_count
                            best_local_adj = [x for x in current_adj]
                    else:
                        # Reject
                        add_edge(current_adj, u, v)
                        current_count += 1
            else:
                # Try a swap: remove one, add one
                existing = [(u, v) for u, v in all_edges if (1 << v) & current_adj[u]]
                missing = [(u, v) for u, v in all_edges if not ((1 << v) & current_adj[u])]
                
                if existing and missing:
                    u1, v1 = random.choice(existing)
                    u2, v2 = random.choice(missing)
                    
                    # Temporarily remove u1-v1
                    remove_edge(current_adj, u1, v1)
                    
                    # Check if u2-v2 is valid
                    if not path5(u2, v2, current_adj):
                        # Add u2-v2
                        add_edge(current_adj, u2, v2)
                        # Edge count is unchanged
                        # This is a neutral move. We accept it always? 
                        # In SA, neutral moves are usually accepted to explore.
                        # However, it might lead to a dead end. 
                        # Let's accept it.
                        if current_count > best_local_count:
                            best_local_count = current_count
                            best_local_adj = [x for x in current_adj]
                    else:
                        # Reject, restore u1-v1
                        add_edge(current_adj, u1, v1)
            
            T *= cooling_rate
            
        return best_local_adj, best_local_count

    for restart in range(50):
        if time.time() > deadline:
            break

        adj = init_adj()
        
        # Phase 1: Greedy Add to get a good starting point
        greedy_add_all(adj)
        
        # Phase 2: Simulated Annealing
        # Use multiple SA runs with different temperatures or just one long one?
        # Given the time limit, let's do a few SA runs.
        
        # Adjust parameters based on remaining time
        time_left = deadline - time.time()
        
        # If time is short, just do greedy
        if time_left < 5:
            current_edges_list = get_edges(adj)
            if len(current_edges_list) > best_count:
                best_count = len(current_edges_list)
                best_edges = current_edges_list
            continue
        
        # Run SA
        # Start temp: 1.0, End temp: 0.01
        # Steps: proportional to time left
        steps = int(time_left * 1000) # Rough estimate
        
        sa_adj, sa_count = simulated_annealing(adj, 1.0, 0.01, steps)
        
        # After SA, run greedy add again to clean up
        greedy_add_all(sa_adj)
        
        current_edges_list = get_edges(sa_adj)
        if len(current_edges_list) > best_count:
            best_count = len(current_edges_list)
            best_edges = current_edges_list

    return best_edges
