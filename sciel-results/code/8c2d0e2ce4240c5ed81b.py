import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin

    def count_regular_6_subsets(adj, n):
        """
        Counts the number of induced subgraphs of size 6 that are regular.
        Returns 0 if none found.
        """
        count = 0
        # Iterate over all combinations of 6 vertices
        # n is at most 22, C(22, 6) = 74613. This is fast.
        for comb in combinations(range(n), 6):
            mask = 0
            for v in comb:
                mask |= (1 << v)
            
            # Check if regular
            # We need degrees of vertices in comb within the subgraph induced by comb
            # deg_v = (adj[v] & mask).bit_count()
            
            first_deg = (adj[comb[0]] & mask).bit_count()
            is_reg = True
            for v in comb[1:]:
                if (adj[v] & mask).bit_count() != first_deg:
                    is_reg = False
                    break
            
            if is_reg:
                count += 1
                
        return count

    def is_fully_valid(adj, n):
        """
        Checks if there are NO regular induced subgraphs of size >= 6.
        """
        # Check sizes 6 to n
        for k in range(6, n + 1):
            for comb in combinations(range(n), k):
                mask = 0
                for v in comb:
                    mask |= (1 << v)
                
                first_deg = (adj[comb[0]] & mask).bit_count()
                is_reg = True
                for v in comb[1:]:
                    if (adj[v] & mask).bit_count() != first_deg:
                        is_reg = False
                        break
                
                if is_reg:
                    return False
        return True

    def local_search(n, max_iters=5000):
        # Initialize random graph
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        current_cost = count_regular_6_subsets(adj, n)
        best_adj = adj[:]
        best_cost = current_cost
        
        for i in range(max_iters):
            if time.time() > deadline:
                break
            
            if current_cost == 0:
                # Check if fully valid
                if is_fully_valid(adj, n):
                    return adj, 0
                # If not fully valid, we are stuck in a local minimum for size 6
                # but have larger regular subgraphs. 
                # We can try to escape by accepting some size 6 violations temporarily?
                # Or just restart.
                # Let's try a few more steps with a higher tolerance or just break and restart.
                # For now, break to allow restart.
                break

            # Pick random edge to flip
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v:
                continue
            
            # Flip edge (u, v)
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            
            new_cost = count_regular_6_subsets(adj, n)
            
            if new_cost <= current_cost:
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_adj = adj[:]
            else:
                # Revert
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                
        return best_adj, best_cost

    # Try n=20
    best_n = 5
    best_edges = []
    
    # Try to find a valid graph for n=20
    found_20 = False
    for attempt in range(50): # More restarts since each is faster
        if time.time() > deadline - 5:
            break
        adj, cost = local_search(20, max_iters=1000)
        if cost == 0:
            # Verify fully valid before returning
            if is_fully_valid(adj, 20):
                edges = []
                for i in range(20):
                    for j in range(i+1, 20):
                        if (adj[i] >> j) & 1:
                            edges.append([i, j])
                return {'n': 20, 'edges': edges}
            
    # If 20 failed, try n=19
    for attempt in range(50):
        if time.time() > deadline - 5:
            break
        adj, cost = local_search(19, max_iters=1000)
        if cost == 0:
            if is_fully_valid(adj, 19):
                edges = []
                for i in range(19):
                    for j in range(i+1, 19):
                        if (adj[i] >> j) & 1:
                            edges.append([i, j])
                return {'n': 19, 'edges': edges}

    # Fallback: return a valid graph on 5 vertices (empty graph)
    return {'n': 5, 'edges': []}
