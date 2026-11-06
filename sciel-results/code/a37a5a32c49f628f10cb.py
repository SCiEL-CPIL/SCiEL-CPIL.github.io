import time
import random

def construct():
    """
    Return a C4-free graph with alpha < 11, maximizing n.
    Strategy: 
    1. Use Simulated Annealing to find a C4-free graph with alpha <= 10 for n=39.
    2. If found, try to extend to n=40, 41, etc.
    3. If SA fails, fall back to the PG(2,5) polarity graph (n=31).
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    def popcount(x):
        return x.bit_count()

    def max_independent_set_size_greedy(n, adj, limit=10):
        """
        Greedy heuristic to find a lower bound on the independence number.
        If it finds an independent set of size > limit, it returns that size.
        Otherwise, it returns the size found (which might be < alpha).
        This is used for the SA cost function.
        """
        candidates = (1 << n) - 1
        size = 0
        while candidates:
            # Pick vertex with minimum degree in the candidate subgraph
            best_v = -1
            min_deg = n + 1
            m = candidates
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                deg = popcount(candidates & adj[v])
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
                m ^= lsb
            
            if best_v == -1:
                break
            
            size += 1
            if size > limit:
                return size
            
            candidates &= ~adj[best_v]
            candidates &= ~(1 << best_v)
            
        return size

    def count_c4s(n, adj):
        """
        Count the number of C4s in the graph.
        A C4 is a set of 4 vertices {a,b,c,d} such that edges (a,b), (b,c), (c,d), (d,a) exist.
        We can count pairs of vertices with >= 2 common neighbors.
        """
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                common = adj[i] & adj[j]
                c = popcount(common)
                if c >= 2:
                    # Number of C4s involving this pair is C(c, 2)
                    count += c * (c - 1) // 2
        return count

    def check_valid(n, adj, time_limit):
        """
        Check if the graph is C4-free and alpha < 11.
        Returns True if valid.
        """
        # C4 check
        if count_c4s(n, adj) > 0:
            return False
        
        # Alpha check
        # Use a fast branch and bound for exact alpha
        best = [0]
        
        def branch(candidates_mask, current_size):
            if time.time() > time_limit:
                return
            if candidates_mask == 0:
                if current_size > best[0]:
                    best[0] = current_size
                return
            
            remaining = popcount(candidates_mask)
            if current_size + remaining <= best[0]:
                return
            
            # Pick vertex with highest degree
            best_v = -1
            best_deg = -1
            m = candidates_mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                deg = popcount(candidates_mask & adj[v])
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                m ^= lsb
            
            if best_v == -1:
                return

            v_bit = 1 << best_v
            
            # Include best_v
            new_candidates = candidates_mask & ~adj[best_v] & ~v_bit
            branch(new_candidates, current_size + 1)
            
            # Exclude best_v
            branch(candidates_mask & ~v_bit, current_size)

        branch((1 << n) - 1, 0)
        return best[0] < 11

    # --- Simulated Annealing for n=39 ---
    def try_sa(n, time_limit):
        """
        Try to find a valid graph on n vertices using SA.
        Returns adj list if found, else None.
        """
        if time.time() > time_limit:
            return None
        
        # Initialize with a random graph
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.1: # Sparse init
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        current_c4 = count_c4s(n, adj)
        current_alpha_lb = max_independent_set_size_greedy(n, adj, 10)
        current_cost = current_c4 + 100 * max(0, current_alpha_lb - 10)
        
        best_adj = adj[:]
        best_cost = current_cost
        
        T = 1000.0
        cooling = 0.999
        
        while T > 0.1 and time.time() < time_limit:
            # Pick a random edge to flip
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            
            edge_exists = bool(adj[i] & (1 << j))
            
            # Flip
            if edge_exists:
                adj[i] ^= (1 << j)
                adj[j] ^= (1 << i)
            else:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
            
            # Calculate new cost
            new_c4 = count_c4s(n, adj)
            new_alpha_lb = max_independent_set_size_greedy(n, adj, 10)
            new_cost = new_c4 + 100 * max(0, new_alpha_lb - 10)
            
            delta = new_cost - current_cost
            
            if delta < 0 or random.random() < pow(2.718281828, -delta / T):
                current_c4 = new_c4
                current_alpha_lb = new_alpha_lb
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_adj = adj[:]
                    if best_cost == 0:
                        # Found a candidate, verify exactly
                        if check_valid(n, best_adj, time_limit - 5):
                            return best_adj
            else:
                # Revert
                if edge_exists:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                else:
                    adj[i] ^= (1 << j)
                    adj[j] ^= (1 << i)
            
            T *= cooling
        
        # Final check on best found
        if best_cost == 0:
            if check_valid(n, best_adj, time_limit - 5):
                return best_adj
        
        return None

    # Try to find a 39-vertex graph
    best_adj = None
    best_n = 0
    
    # First, try to get a 31-vertex graph (PG(2,5)) as a fallback
    # This is fast and reliable
    points = []
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    inv_x = pow(x, -1, 5)
                    points.append((1, y * inv_x % 5, z * inv_x % 5))
                elif y != 0:
                    inv_y = pow(y, -1, 5)
                    points.append((0, 1, z * inv_y % 5))
                else:
                    inv_z = pow(z, -1, 5)
                    points.append((0, 0, 1))
    
    points = list(set(points))
    n_base = len(points)
    adj_base = [0] * n_base
    for i in range(n_base):
        for j in range(i + 1, n_base):
            if (points[i][0]*points[j][2] + points[i][1]*points[j][1] + points[i][2]*points[j][0]) % 5 == 0:
                adj_base[i] |= (1 << j)
                adj_base[j] |= (1 << i)
    
    if check_valid(n_base, adj_base, start_time + 5):
        best_adj = adj_base
        best_n = n_base

    # Now try to improve to 39
    if time.time() < deadline - 30:
        adj_39 = try_sa(39, deadline - 10)
        if adj_39:
            best_adj = adj_39
            best_n = 39

    # If we have 39, try 40
    if best_n == 39 and time.time() < deadline - 10:
        adj_40 = try_sa(40, deadline - 5)
        if adj_40:
            best_adj = adj_40
            best_n = 40

    # If we failed SA, stick with 31
    if best_n == 0:
        best_n = 31
        best_adj = adj_base

    # Convert to edge list
    edge_list = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if best_adj[i] & (1 << j):
                edge_list.append([i, j])

    return {'n': best_n, 'edges': edge_list}
