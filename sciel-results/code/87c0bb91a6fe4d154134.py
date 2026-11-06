import time
import random
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    known_32 = {
        'n': 32,
        'edges': [[4 * b + i, 4 * b + j] for b in range(8) for i in range(4) for j in range(i + 1, 4)]
    }

    def build_adj(n, edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def has_c5_fast(adj, n):
        # Faster C5 detection using bit operations
        # A C5 exists iff there exist vertices a,b,c,d,e with edges ab,bc,cd,de,ea
        # and no chords. We check: for each pair (a,c) at distance 2 (share neighbor b),
        # check if N(a)\{b} and N(c)\{b} have an edge between them with endpoints not in {a,b,c}.
        # But simpler: iterate over all 5-tuples with pruning.
        
        # Alternative: for each edge (a,b), for each common neighbor c of a and b... no.
        # Let's use: a C5 is a-b-c-d-e-a. 
        # For each unordered pair {a,c} that are NOT adjacent:
        #   common = N(a) & N(c)
        #   For each b in common:
        #     remaining_a = N(a) & ~{b,c} & ~N(c)  -- neighbors of a not in {b,c} and not neighbors of c
        #     remaining_c = N(c) & ~{a,b} & ~N(a)  -- neighbors of c not in {a,b} and not neighbors of a
        #     If there's an edge between remaining_a and remaining_c, we have a C5
        # Actually this is getting complex. Let me use a cleaner approach:
        
        # For each vertex a, for each pair of neighbors b,d of a (b<d):
        #   If b and d are not adjacent, then check if there's a path b-x-y-d of length 2
        #   where x is neighbor of b (not a), y is neighbor of d (not a), and x-y is an edge,
        #   and x,y not in {a,b,d}
        # This is O(n * deg^2 * ...) which might be better for sparse graphs.
        
        # Let me try the most straightforward but optimized version:
        for a in range(n):
            na = adj[a]
            # b is a neighbor of a
            b_mask = na
            while b_mask:
                b_bit = b_mask & (-b_mask)
                b = b_bit.bit_length() - 1
                b_mask ^= b_bit
                
                nb = adj[b]
                # c is a neighbor of b, c != a
                c_mask = nb & ~(1 << a)
                while c_mask:
                    c_bit = c_mask & (-c_mask)
                    c = c_bit.bit_length() - 1
                    c_mask ^= c_bit
                    
                    # a and c must NOT be adjacent for a proper C5 (no chord)
                    if na & (1 << c):
                        continue
                    
                    nc = adj[c]
                    # d is a neighbor of c, d != b, d != a
                    d_mask = nc & ~(1 << b) & ~(1 << a)
                    while d_mask:
                        d_bit = d_mask & (-d_mask)
                        d = d_bit.bit_length() - 1
                        d_mask ^= d_bit
                        
                        # c and a already checked not adjacent
                        # b and d must not be adjacent (no chord)
                        if nb & (1 << d):
                            continue
                        
                        nd = adj[d]
                        # e is a neighbor of d, e != c, e != a, e != b, e != d
                        # e must be a neighbor of a
                        e_mask = nd & na & ~(1 << a) & ~(1 << b) & ~(1 << c) & ~(1 << d)
                        if e_mask:
                            return True
        return False

    def has_independent_set_9(adj, n):
        # Branch and bound
        # Use a greedy coloring bound? Or just simple backtracking with pruning.
        # For n=35, depth 9 is small.
        
        def can_find(available, count):
            if count == 0:
                return True
            if available == 0:
                return False
            if bin(available).count('1') < count:
                return False
            
            # Pick vertex with max degree in available?
            # Or min degree?
            # Min degree in available is good for independent set (excludes fewest).
            best_v = -1
            best_excluded = n + 1
            avail = available
            while avail:
                v_bit = avail & (-avail)
                v = v_bit.bit_length() - 1
                avail ^= v_bit
                adj_in_avail = available & adj[v]
                excluded = bin(adj_in_avail).count('1')
                if excluded < best_excluded:
                    best_excluded = excluded
                    best_v = v
                if excluded == 0:
                    break

            if best_v == -1:
                return False

            v = best_v
            v_bit = 1 << v

            # Include v
            new_avail = available & ~adj[v] & ~v_bit
            if can_find(new_avail, count - 1):
                return True

            # Exclude v
            if can_find(available & ~v_bit, count):
                return True

            return False

        return can_find((1 << n) - 1, 9)

    def is_valid(n, adj):
        if has_c5_fast(adj, n):
            return False
        if has_independent_set_9(adj, n):
            return False
        return True

    def adj_to_edges(adj, n):
        edges = []
        for i in range(n):
            mask = adj[i]
            for j in range(i + 1, n):
                if mask & (1 << j):
                    edges.append([i, j])
        return edges

    best_result = known_32

    # Try to find n=35, 34, 33
    for target_n in [35, 34, 33]:
        if time.time() > deadline:
            break
            
        found = False
        # Random restarts with structured initializations
        max_attempts = 200 if target_n == 35 else 300 if target_n == 34 else 500
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
                
            adj = [0] * target_n
            
            # Strategy 1: Start with 8 K4s and add extras with careful connections
            # Strategy 2: Use a more structured approach with independent sets
            
            if attempt % 3 == 0:
                # Start with 8 K4s
                for b in range(8):
                    for i in range(4):
                        for j in range(i + 1, 4):
                            u = 4 * b + i
                            v = 4 * b + j
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                
                # Connect extra vertices more carefully
                for ex in range(32, target_n):
                    for b in range(8):
                        # Connect to 2-3 random vertices in each block
                        num_conn = random.randint(2, 3)
                        chosen = random.sample(range(4), num_conn)
                        for t in chosen:
                            v = 4 * b + t
                            if random.random() < 0.7:
                                adj[ex] |= (1 << v)
                                adj[v] |= (1 << ex)
                
                # Random edges between extras (sparse)
                for i in range(32, target_n):
                    for j in range(i + 1, target_n):
                        if random.random() < 0.3:
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
            
            elif attempt % 3 == 1:
                # Turan-like: partition into 8 groups, make each group a clique of size 4 or 5
                # 35 = 5+5+5+5+4+4+4+4 or similar
                group_sizes = [4]*8
                if target_n > 32:
                    # Distribute extra vertices
                    extra = target_n - 32
                    for i in range(extra):
                        group_sizes[i] += 1
                
                vertex_idx = 0
                for b in range(8):
                    size = group_sizes[b]
                    for i in range(size):
                        for j in range(i+1, size):
                            u = vertex_idx + i
                            v = vertex_idx + j
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                    vertex_idx += size
                
                # Add some inter-group edges to reduce independence number
                # Each vertex should connect to vertices in at least 2-3 other groups
                for v in range(target_n):
                    # Find which group v is in
                    g = 0
                    idx = 0
                    for b in range(8):
                        if idx <= v < idx + group_sizes[b]:
                            g = b
                            break
                        idx += group_sizes[b]
                    
                    # Connect to random vertices in 2-3 other groups
                    other_groups = [x for x in range(8) if x != g]
                    chosen_groups = random.sample(other_groups, min(3, len(other_groups)))
                    for og in chosen_groups:
                        # Find vertices in group og
                        start = sum(group_sizes[:og])
                        end = start + group_sizes[og]
                        candidates = list(range(start, end))
                        num_conn = random.randint(1, 2)
                        for _ in range(num_conn):
                            if candidates:
                                u = random.choice(candidates)
                                if random.random() < 0.6:
                                    adj[v] |= (1 << u)
                                    adj[u] |= (1 << v)
            
            else:
                # Pure random with bias toward denser graph
                for i in range(target_n):
                    for j in range(i+1, target_n):
                        if random.random() < 0.4:
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
            
            # Check and fix C5
            if has_c5_fast(adj, target_n):
                adj_copy = adj[:]
                for _ in range(30):
                    if not has_c5_fast(adj_copy, target_n):
                        break
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
                adj = adj_copy
                
            if has_c5_fast(adj, target_n):
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
