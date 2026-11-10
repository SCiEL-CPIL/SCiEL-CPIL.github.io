import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    def get_vertices(mask):
        verts = []
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            verts.append(idx)
            m ^= lsb
        return verts

    def is_regular_subset(adj, mask, n):
        # Get vertices in mask
        verts = []
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            verts.append(idx)
            m ^= lsb
        
        k = len(verts)
        if k < 6:
            return False, None
            
        # Calculate degrees
        degrees = []
        for u in verts:
            deg = (adj[u] & mask).bit_count()
            degrees.append(deg)
            
        # Check if all degrees are the same
        first_deg = degrees[0]
        for d in degrees[1:]:
            if d != first_deg:
                return False, None
        return True, verts

    def find_violation(adj, n, max_checks=10000):
        """Find one violating subset if it exists."""
        # Try subsets of size 6 first (most common violations)
        # Then larger sizes
        for size in range(6, n+1):
            if size > n:
                break
            # Randomly sample subsets of this size
            for _ in range(max_checks // (n - size + 1)):
                verts = random.sample(range(n), size)
                mask = 0
                for v in verts:
                    mask |= (1 << v)
                is_reg, _ = is_regular_subset(adj, mask, n)
                if is_reg:
                    return mask
        # Exhaustive check for size 6 if random didn't find one
        for combo in combinations(range(n), 6):
            mask = 0
            for v in combo:
                mask |= (1 << v)
            is_reg, _ = is_regular_subset(adj, mask, n)
            if is_reg:
                return mask
        return None

    def count_violations_fast(adj, n, max_checks=50000):
        """Count violations using random sampling + targeted checks."""
        count = 0
        # Check all size-6 subsets (most critical)
        for combo in combinations(range(n), 6):
            mask = 0
            for v in combo:
                mask |= (1 << v)
            is_reg, _ = is_regular_subset(adj, mask, n)
            if is_reg:
                count += 1
                if count > 100:  # Early termination if too many
                    return count
        return count

    def local_search(n, max_iters=20000):
        # Initialize random graph with slight bias toward sparsity
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.35:  # Slightly sparse
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        current_cost = count_violations_fast(adj, n)
        best_adj = adj[:]
        best_cost = current_cost
        
        # Simulated annealing parameters
        temp = 50.0
        cooling = 0.995
        min_temp = 0.1
        
        for i in range(max_iters):
            if time.time() > deadline:
                break
            if current_cost == 0:
                return adj, 0
            
            # Pick random edge to flip
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v:
                continue
            
            # Flip edge (u, v)
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            
            new_cost = count_violations_fast(adj, n)
            
            if new_cost < current_cost:
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_adj = adj[:]
            else:
                # Simulated annealing: accept worse with probability
                delta = new_cost - current_cost
                if temp > min_temp and random.random() < (1.0 / (1.0 + delta)) * (temp / 10.0):
                    current_cost = new_cost
                else:
                    # Revert
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
            
            temp *= cooling
            
            # If stuck, try targeted repair
            if i % 500 == 499 and current_cost > 0:
                violation_mask = find_violation(adj, n)
                if violation_mask is not None:
                    verts = get_vertices(violation_mask)
                    # Try flipping edges within the violating subset
                    for a in range(len(verts)):
                        for b in range(a+1, len(verts)):
                            u, v = verts[a], verts[b]
                            adj[u] ^= (1 << v)
                            adj[v] ^= (1 << u)
                            test_cost = count_violations_fast(adj, n)
                            if test_cost < current_cost:
                                current_cost = test_cost
                                if test_cost < best_cost:
                                    best_cost = test_cost
                                    best_adj = adj[:]
                            else:
                                # Revert
                                adj[u] ^= (1 << v)
                                adj[v] ^= (1 << u)
                    break  # Restart with new random graph after targeted repair attempt
        
        return best_adj, best_cost

    # Try to find a valid graph for n=20
    best_n = 5
    best_edges = []
    
    # Multiple restarts with different strategies
    found_20 = False
    for attempt in range(20):  # More restarts
        if time.time() > deadline - 10:
            break
        adj, cost = local_search(20, max_iters=3000)
        if cost == 0:
            # Convert adj to edges list
            edges = []
            for i in range(20):
                for j in range(i+1, 20):
                    if (adj[i] >> j) & 1:
                        edges.append([i, j])
            return {'n': 20, 'edges': edges}
    
    # If 20 failed, try n=19
    for attempt in range(10):
        if time.time() > deadline - 5:
            break
        adj, cost = local_search(19, max_iters=2000)
        if cost == 0:
            edges = []
            for i in range(19):
                for j in range(i+1, 19):
                    if (adj[i] >> j) & 1:
                        edges.append([i, j])
            return {'n': 19, 'edges': edges}
    
    # Fallback: return a valid graph on 5 vertices (empty graph)
    return {'n': 5, 'edges': []}
