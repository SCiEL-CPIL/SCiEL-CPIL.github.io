"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # Known valid graph: 8 disjoint K4's -> n=32, alpha=8 < 9, no C5
    known_32 = {
        'n': 32,
        'edges': [[4 * b + i, 4 * b + j] for b in range(8) for i in range(4) for j in range(i + 1, 4)]
    }
    
    def build_adj(n, edges):
        """Build adjacency bitmasks from edge list."""
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj
        
    def has_c5(adj, n):
        """Check if graph contains a 5-cycle using bitsets. Returns True if found."""
        for u in range(n):
            nu = adj[u]
            v_mask = nu
            while v_mask:
                v_bit = v_mask & (-v_mask)
                v = v_bit.bit_length() - 1
                v_mask ^= v_bit
                
                # A = neighbors of u excluding v
                # B = neighbors of v excluding u
                A = adj[u] & ~(1 << v)
                B = adj[v] & ~(1 << u)
                
                # We look for a path u-a-b-c-v
                # a in A, c in B, b not in {u,v}
                # a-b and b-c are edges.
                # a != c.
                
                # Iterate a in A
                a_mask = A
                while a_mask:
                    a_bit = a_mask & (-a_mask)
                    a = a_bit.bit_length() - 1
                    a_mask ^= a_bit
                    
                    # b must be neighbor of a, not u, not v
                    b_mask = adj[a] & ~(1 << u) & ~(1 << v)
                    while b_mask:
                        b_bit = b_mask & (-b_mask)
                        b = b_bit.bit_length() - 1
                        b_mask ^= b_bit
                        
                        # c must be neighbor of b, in B
                        # Also c != a
                        c_mask = adj[b] & B
                        # Exclude a if a is in B (i.e., a is common neighbor of u and v)
                        if c_mask & ~(1 << a):
                            return True
                            
        return False

    def has_independent_set_9(adj, n):
        """Check if graph has an independent set of size 9 using bitset backtracking."""
        
        def can_find(available, count):
            if count == 0:
                return True
            if available == 0:
                return False
            if bin(available).count('1') < count:
                return False
            
            # Pick vertex with minimum degree in available
            best_v = -1
            best_excluded = 0
            avail = available
            while avail:
                v_bit = avail & (-avail)
                v = v_bit.bit_length() - 1
                avail ^= v_bit
                adj_in_avail = available & adj[v]
                excluded = bin(adj_in_avail).count('1')
                if best_v == -1 or excluded < best_excluded:
                    best_excluded = excluded
                    best_v = v
                if excluded == 0:
                    break
            
            if best_v == -1:
                return False
            
            v = best_v
            v_bit = 1 << v
            
            # Option 1: include v
            new_avail = available & ~adj[v] & ~v_bit
            if can_find(new_avail, count - 1):
                return True
            
            # Option 2: exclude v
            if can_find(available & ~v_bit, count):
                return True
            
            return False
        
        return can_find((1 << n) - 1, 9)
    
    def adj_to_edges(adj, n):
        """Convert adjacency bitmasks to edge list."""
        edges = []
        for i in range(n):
            mask = adj[i]
            for j in range(i + 1, n):
                if mask & (1 << j):
                    edges.append([i, j])
        return edges

    def local_search(target_n, time_budget):
        """Local search for a valid graph of size target_n."""
        # Initialize: 8xK4 + extra vertices
        adj = [0] * target_n
        for b in range(8):
            for i in range(4):
                for j in range(i + 1, 4):
                    u = 4 * b + i
                    v = 4 * b + j
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        # Connect extra vertices
        for ex in range(32, target_n):
            # Connect to 3 random vertices in distinct K4 blocks
            blocks = random.sample(range(8), 3)
            for b in blocks:
                v = 4 * b + random.randint(0, 3)
                adj[ex] |= (1 << v)
                adj[v] |= (1 << ex)
            
            # Connect extra vertices to each other if multiple
            for prev_ex in range(32, ex):
                if random.random() < 0.5:
                    adj[ex] |= (1 << prev_ex)
                    adj[prev_ex] |= (1 << ex)

        # Check initial
        if not has_c5(adj, target_n) and not has_independent_set_9(adj, target_n):
            return {'n': target_n, 'edges': adj_to_edges(adj, target_n)}

        T = 10.0
        best_adj = adj[:]
        best_cost = float('inf')
        
        # Calculate initial cost
        # Cost: 1 if C5, 100 if Ind9
        c5 = has_c5(adj, target_n)
        ind9 = has_independent_set_9(adj, target_n)
        current_cost = (1 if c5 else 0) + (100 if ind9 else 0)
        best_cost = current_cost

        while time.time() < time_budget:
            # Pick random edge to flip
            u = random.randint(0, target_n - 1)
            v = random.randint(0, target_n - 1)
            if u == v:
                continue
            
            # Flip
            if adj[u] & (1 << v):
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            
            # Evaluate
            new_c5 = has_c5(adj, target_n)
            new_ind9 = has_independent_set_9(adj, target_n)
            
            if not new_c5 and not new_ind9:
                return {'n': target_n, 'edges': adj_to_edges(adj, target_n)}
            
            new_cost = (1 if new_c5 else 0) + (100 if new_ind9 else 0)
            
            # Acceptance
            if new_cost < current_cost:
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_adj = adj[:]
            else:
                # Simulated Annealing
                if random.random() < pow(2.71828, (current_cost - new_cost) / T):
                    current_cost = new_cost
                else:
                    # Revert
                    if adj[u] & (1 << v):
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                    else:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
            
            # Cool
            T *= 0.999
            if T < 0.1:
                T = 0.1
                # Random restart? Or just keep going.
                
        return None

    best_result = known_32
    
    # Try n=33
    if time.time() < deadline:
        budget = min(30.0, deadline - time.time())
        res = local_search(33, start_time + budget)
        if res:
            best_result = res
            
    # Try n=34
    if best_result['n'] < 34 and time.time() < deadline:
        budget = min(30.0, deadline - time.time())
        res = local_search(34, start_time + budget)
        if res:
            best_result = res
            
    # Try n=35
    if best_result['n'] < 35 and time.time() < deadline:
        budget = min(30.0, deadline - time.time())
        res = local_search(35, start_time + budget)
        if res:
            best_result = res

    return best_result

# EVOLVE-BLOCK-END
