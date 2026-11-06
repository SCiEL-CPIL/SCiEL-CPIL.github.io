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
        # For each edge (u,v), check if there's a path of length 4 from u to v
        # that doesn't use edge (u,v) directly.
        # Path: u-a-b-c-v where a,b,c are distinct from u,v and each other.
        # 
        # Efficient approach:
        # For each edge (u,v):
        #   a_mask = N(u) \ {v}
        #   For each a in a_mask:
        #     b_mask = N(a) \ {u, v}
        #     For each b in b_mask:
        #       c_mask = N(b) \ {u, a, v}
        #       if c_mask & N(v): found C5
        
        for u in range(n):
            nu = adj[u]
            v_mask = nu
            while v_mask:
                v_bit = v_mask & (-v_mask)
                v = v_bit.bit_length() - 1
                v_mask ^= v_bit
                
                a_mask = nu & ~(1 << v)
                while a_mask:
                    a_bit = a_mask & (-a_mask)
                    a = a_bit.bit_length() - 1
                    a_mask ^= a_bit
                    
                    b_mask = adj[a] & ~(1 << u) & ~(1 << v)
                    while b_mask:
                        b_bit = b_mask & (-b_mask)
                        b = b_bit.bit_length() - 1
                        b_mask ^= b_bit
                        
                        c_mask = adj[b] & ~(1 << u) & ~(1 << a) & ~(1 << v)
                        if c_mask & adj[v]:
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
            
            # Pick vertex with minimum degree in available (excludes fewest)
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
    
    def is_valid(n, adj):
        """Check if graph is C5-free and has no independent set of 9."""
        if has_c5(adj, n):
            return False
        if has_independent_set_9(adj, n):
            return False
        return True
    
    def adj_to_edges(adj, n):
        """Convert adjacency bitmasks to edge list."""
        edges = []
        for i in range(n):
            mask = adj[i]
            for j in range(i + 1, n):
                if mask & (1 << j):
                    edges.append([i, j])
        return edges
    
    best_result = known_32
    
    # Strategy 1: Extend 8xK4 by adding vertices with careful connections
    for target_n in [33, 34, 35]:
        if time.time() > deadline:
            break
        
        extra = target_n - 32
        num_attempts = 2000
        for attempt in range(num_attempts):
            if time.time() > deadline:
                break
            
            # Initialize: 8xK4 structure
            adj = [0] * target_n
            for b in range(8):
                for i in range(4):
                    for j in range(i + 1, 4):
                        u = 4 * b + i
                        v = 4 * b + j
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
            
            # Add edges from extra vertices to existing vertices
            for ex in range(32, target_n):
                # Connect to vertices from 1-2 blocks to maintain C5-freeness
                block = random.randint(0, 7)
                base = 4 * block
                num_conn = random.randint(2, 4)
                targets = random.sample(range(4), num_conn)
                for t in targets:
                    v = base + t
                    adj[ex] |= (1 << v)
                    adj[v] |= (1 << ex)
                
                # Optionally connect to a second block
                if random.random() < 0.4:
                    block2 = random.randint(0, 7)
                    if block2 != block:
                        base2 = 4 * block2
                        num_conn2 = random.randint(1, 2)
                        targets2 = random.sample(range(4), num_conn2)
                        for t in targets2:
                            v = base2 + t
                            adj[ex] |= (1 << v)
                            adj[v] |= (1 << ex)
            
            # Check validity
            if has_c5(adj, target_n):
                continue
            if has_independent_set_9(adj, target_n):
                continue
            
            edges = adj_to_edges(adj, target_n)
            best_result = {'n': target_n, 'edges': edges}
            break
    
    # Strategy 2: Local search - start from valid n=32 graph and modify
    if best_result['n'] == 32 and time.time() < deadline:
        for target_n in [33, 34, 35]:
            if time.time() > deadline:
                break
            
            found = False
            for attempt in range(3000):
                if time.time() > deadline:
                    break
                
                # Start with 8xK4 + extra isolated vertices
                adj = [0] * target_n
                for b in range(8):
                    for i in range(4):
                        for j in range(i + 1, 4):
                            u = 4 * b + i
                            v = 4 * b + j
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                
                # Add random edges from extra vertices
                for ex in range(32, target_n):
                    # Connect to 3-5 random vertices
                    num_conn = random.randint(3, 5)
                    targets = random.sample(range(32), num_conn)
                    for t in targets:
                        adj[ex] |= (1 << t)
                        adj[t] |= (1 << ex)
                
                # Try to fix C5s by removing edges
                # First check if valid
                if has_c5(adj, target_n):
                    # Try removing some edges to fix
                    adj_copy = adj[:]
                    # Find and remove edges in C5s (simplified: just try random removals)
                    removed = 0
                    for _ in range(20):
                        if time.time() > deadline:
                            break
                        if not has_c5(adj_copy, target_n):
                            break
                        # Remove a random edge
                        all_edges = []
                        for i in range(target_n):
                            for j in range(i + 1, target_n):
                                if adj_copy[i] & (1 << j):
                                    all_edges.append((i, j))
                        if not all_edges:
                            break
                        u, v = random.choice(all_edges)
                        adj_copy[u] &= ~(1 << v)
                        adj_copy[v] &= ~(1 << u)
                        removed += 1
                    
                    adj = adj_copy
                
                if has_c5(adj, target_n):
                    continue
                if has_independent_set_9(adj, target_n):
                    continue
                
                edges = adj_to_edges(adj, target_n)
                best_result = {'n': target_n, 'edges': edges}
                found = True
                break
            
            if found:
                break
    
    # Strategy 3: Pure random search with low density
    if best_result['n'] == 32 and time.time() < deadline:
        for target_n in [33, 34, 35]:
            if time.time() > deadline:
                break
            
            found = False
            for attempt in range(5000):
                if time.time() > deadline:
                    break
                
                # Random graph with low density
                p = random.uniform(0.08, 0.20)
                
                adj = [0] * target_n
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        if random.random() < p:
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
                
                if has_c5(adj, target_n):
                    continue
                if has_independent_set_9(adj, target_n):
                    continue
                
                edges = adj_to_edges(adj, target_n)
                best_result = {'n': target_n, 'edges': edges}
                found = True
                break
            
            if found:
                break
    
    return best_result

# EVOLVE-BLOCK-END
