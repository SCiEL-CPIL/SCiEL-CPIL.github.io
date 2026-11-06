"""
K2,3-free graph construction on 20 vertices.
Uses bitmask-based fast common-neighbor tracking with simulated annealing
using remove-and-refill moves to escape local optima.
"""
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    n = 20
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    total_pairs = len(all_pairs)
    
    # Precompute bit masks
    bit = [1 << i for i in range(n)]
    
    def popcount(x):
        return bin(x).count('1')
    
    def count_edges(adj):
        c = 0
        for i in range(n):
            c += popcount(adj[i])
        return c // 2
    
    def common_count(adj, u, v):
        """Count common neighbors of u and v."""
        return popcount(adj[u] & adj[v])
    
    def is_valid_add(adj, u, v):
        """
        Check if adding edge (u,v) maintains K2,3-freeness.
        """
        # Check 1: u and v can't have >2 common neighbors
        if common_count(adj, u, v) > 2:
            return False
        
        # For each w in N(v), w != u: common_count(adj, u, w) <= 1
        nv = adj[v]
        while nv:
            low = nv & (-nv)
            w = low.bit_length() - 1
            if w != u:
                if common_count(adj, u, w) > 1:
                    return False
            nv ^= low
        
        # For each w in N(u), w != v: common_count(adj, v, w) <= 1
        nu = adj[u]
        while nu:
            low = nu & (-nu)
            w = low.bit_length() - 1
            if w != v:
                if common_count(adj, v, w) > 1:
                    return False
            nu ^= low
        
        return True
    
    def add_edge(adj, u, v):
        adj[u] |= bit[v]
        adj[v] |= bit[u]
    
    def remove_edge(adj, u, v):
        adj[u] &= ~bit[v]
        adj[v] &= ~bit[u]
    
    def get_edges_list(adj):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (adj[i] >> j) & 1:
                    edges.append((i, j))
        return edges
    
    def get_missing_edges(adj):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if not ((adj[i] >> j) & 1):
                    edges.append((i, j))
        return edges
    
    def greedy_fill(adj, order=None):
        """Add all valid edges from the given order (or all missing)."""
        if order is None:
            order = get_missing_edges(adj)
        for u, v in order:
            if not ((adj[u] >> v) & 1):
                if is_valid_add(adj, u, v):
                    add_edge(adj, u, v)
    
    def simulated_annealing(adj, max_steps=20000, deadline=deadline):
        """
        Simulated annealing with remove-and-refill moves.
        Move types:
        1. Add a random missing edge if valid
        2. Remove a random edge, then greedily refill
        3. Swap: remove one edge, add another if valid, then refill
        """
        current_adj = adj[:]
        current_count = count_edges(current_adj)
        best_adj = current_adj[:]
        best_count = current_count
        
        T_start = 5.0
        T_end = 0.01
        total_steps = max_steps
        
        for step in range(total_steps):
            if time.time() > deadline - 3:
                break
            
            # Temperature schedule (geometric cooling)
            progress = step / total_steps
            T = T_start * (T_end / T_start) ** progress
            
            cur_edges = get_edges_list(current_adj)
            missing = get_missing_edges(current_adj)
            
            if not cur_edges:
                break
            
            move_type = random.random()
            
            if move_type < 0.4:
                # Type 1: Try to add a random missing edge
                if not missing:
                    continue
                u, v = missing[random.randint(0, len(missing) - 1)]
                if is_valid_add(current_adj, u, v):
                    add_edge(current_adj, u, v)
                    new_count = current_count + 1
                    # Always accept improving moves
                    current_count = new_count
                    if new_count > best_count:
                        best_count = new_count
                        best_adj = current_adj[:]
                # If not valid, skip (no penalty)
                
            elif move_type < 0.7:
                # Type 2: Remove a random edge, then greedily refill
                u_rem, v_rem = cur_edges[random.randint(0, len(cur_edges) - 1)]
                backup = current_adj[:]
                remove_edge(current_adj, u_rem, v_rem)
                old_count = current_count - 1
                
                # Greedy refill
                refill_missing = get_missing_edges(current_adj)
                random.shuffle(refill_missing)
                for u, v in refill_missing:
                    if is_valid_add(current_adj, u, v):
                        add_edge(current_adj, u, v)
                
                new_count = count_edges(current_adj)
                delta = new_count - current_count
                
                if delta >= 0 or random.random() < math.exp(delta / max(T, 1e-10)):
                    current_count = new_count
                    if new_count > best_count:
                        best_count = new_count
                        best_adj = current_adj[:]
                else:
                    current_adj = backup[:]
                    
            else:
                # Type 3: Swap - remove one edge, add another, then refill
                if not missing:
                    continue
                u_rem, v_rem = cur_edges[random.randint(0, len(cur_edges) - 1)]
                u_add, v_add = missing[random.randint(0, len(missing) - 1)]
                
                backup = current_adj[:]
                remove_edge(current_adj, u_rem, v_rem)
                
                if is_valid_add(current_adj, u_add, v_add):
                    add_edge(current_adj, u_add, v_add)
                    
                    # Greedy refill
                    refill_missing = get_missing_edges(current_adj)
                    random.shuffle(refill_missing)
                    for u, v in refill_missing:
                        if is_valid_add(current_adj, u, v):
                            add_edge(current_adj, u, v)
                    
                    new_count = count_edges(current_adj)
                    delta = new_count - current_count
                    
                    if delta >= 0 or random.random() < math.exp(delta / max(T, 1e-10)):
                        current_count = new_count
                        if new_count > best_count:
                            best_count = new_count
                            best_adj = current_adj[:]
                    else:
                        current_adj = backup[:]
                else:
                    # Can't add the edge, just keep the removal
                    new_count = current_count - 1
                    delta = new_count - current_count
                    if delta >= 0 or random.random() < math.exp(delta / max(T, 1e-10)):
                        current_count = new_count
                    else:
                        current_adj = backup[:]
        
        return best_adj, best_count
    
    # Main strategy:
    # 1. Run multiple random restarts with greedy + simulated annealing
    # 2. Also try some structured initializations
    # 3. Use SA for refinement
    
    best_adj_overall = None
    best_count_overall = 0
    
    # Strategy 1: Random restarts with varied greedy orderings
    max_restarts = 40
    for r in range(max_restarts):
        if time.time() > deadline - 10:
            break
        
        adj = [0] * n
        
        # Different ordering strategies
        strategy = r % 5
        order = all_pairs[:]
        
        if strategy == 0:
            random.shuffle(order)
        elif strategy == 1:
            # Sort by sum of indices
            order.sort(key=lambda x: x[0] + x[1])
            random.shuffle(order)
        elif strategy == 2:
            # Sort by difference
            order.sort(key=lambda x: abs(x[1] - x[0]))
            random.shuffle(order)
        elif strategy == 3:
            # Random with bias toward smaller indices
            order.sort(key=lambda x: x[0])
            random.shuffle(order)
        else:
            # Fully random
            random.shuffle(order)
        
        # Greedy construct
        greedy_fill(adj, order)
        
        # Simulated annealing
        adj, cnt = simulated_annealing(adj, max_steps=15000)
        
        if cnt > best_count_overall:
            best_count_overall = cnt
            best_adj_overall = adj[:]
        
        # Early exit if we hit a high count
        if best_count_overall >= 65:
            break
    
    # Strategy 2: Try structured initialization with circulant graphs
    if time.time() < deadline - 10:
        step_sets = [
            [1], [2], [3], [4], [5], [6], [7],
            [1,2], [1,3], [1,4], [2,3], [2,4], [3,4],
            [1,2,3], [1,2,4], [2,3,4],
        ]
        
        for steps in step_sets:
            if time.time() > deadline - 8:
                break
            
            adj = [0] * n
            for i in range(n):
                for j in range(i+1, n):
                    diff = (j - i) % n
                    if diff in steps or (n - diff) in steps:
                        if is_valid_add(adj, i, j):
                            add_edge(adj, i, j)
            
            adj, cnt = simulated_annealing(adj, max_steps=10000)
            if cnt > best_count_overall:
                best_count_overall = cnt
                best_adj_overall = adj[:]
    
    # Strategy 3: Try complete bipartite-like structures
    if time.time() < deadline - 8:
        for split in range(5, 15):
            if time.time() > deadline - 6:
                break
            
            adj = [0] * n
            # First try adding all cross edges
            for i in range(split):
                for j in range(split, n):
                    if is_valid_add(adj, i, j):
                        add_edge(adj, i, j)
            
            # Then try adding intra-part edges
            for part in [range(split), range(split, n)]:
                for i in part:
                    for j in part:
                        if i < j:
                            if is_valid_add(adj, i, j):
                                add_edge(adj, i, j)
            
            adj, cnt = simulated_annealing(adj, max_steps=10000)
            if cnt > best_count_overall:
                best_count_overall = cnt
                best_adj_overall = adj[:]
    
    # Strategy 4: Final intensive SA on best result
    if best_adj_overall is not None and time.time() < deadline - 5:
        adj_final, cnt_final = simulated_annealing(best_adj_overall, max_steps=30000)
        if cnt_final > best_count_overall:
            best_count_overall = cnt_final
            best_adj_overall = adj_final[:]
    
    # Convert best_adj to edge list
    if best_adj_overall is None:
        return []
    
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj_overall[i] >> j) & 1:
                edges.append([i, j])
    
    return edges

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
