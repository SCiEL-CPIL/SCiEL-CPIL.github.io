import time
import random
import sys

def construct():
    n = 15
    start_time = time.time()
    deadline = 50.0  # Safety margin before 90s kill

    # Bitmask adjacency representation for speed
    # adj[i] is a bitmask where bit j is set iff edge (i,j) exists
    
    def build_adj_from_edges(edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def get_degree(adj, i):
        return bin(adj[i]).count('1')

    def is_valid_adj(adj):
        # Precompute degrees
        degrees = [bin(adj[i]).count('1') for i in range(n)]
        
        # For each edge (u, v)
        for u in range(n):
            # Get neighbors of u excluding v (v is implicit in the edge)
            # Actually we iterate over all edges
            au = adj[u]
            for v in range(n):
                if not (au & (1 << v)):
                    continue
                if u > v:
                    continue
                # Edge (u, v) exists. Check paths s-u-v-t
                # s is neighbor of u, s != v
                # t is neighbor of v, t != u, t != s
                # Condition: degrees[s] != degrees[t] for all such s, t
                
                au_minus_v = au & ~(1 << v)
                av_minus_u = adj[v] & ~(1 << u)
                
                # Iterate over all s in N(u) \ {v}
                s_mask = au_minus_v
                while s_mask:
                    s_bit = s_mask & -s_mask
                    s = s_bit.bit_length() - 1
                    s_mask ^= s_bit
                    
                    # Iterate over all t in N(v) \ {u, s}
                    t_mask = av_minus_u & ~(1 << s)
                    while t_mask:
                        t_bit = t_mask & -t_mask
                        t = t_bit.bit_length() - 1
                        t_mask ^= t_bit
                        
                        if degrees[s] == degrees[t]:
                            return False
        return True

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def count_edges(adj):
        total = 0
        for i in range(n):
            total += bin(adj[i]).count('1')
        return total // 2

    # Initialize with 5 disjoint triangles
    # (0,1,2), (3,4,5), (6,7,8), (9,10,11), (12,13,14)
    initial_edges = []
    for i in range(0, n, 3):
        initial_edges.append([i, i+1])
        initial_edges.append([i+1, i+2])
        initial_edges.append([i, i+2])
    
    best_adj = build_adj_from_edges(initial_edges)
    best_count = count_edges(best_adj)
    
    # Verify initial validity
    if not is_valid_adj(best_adj):
        # Fallback: star
        initial_edges = [[0, v] for v in range(1, n)]
        best_adj = build_adj_from_edges(initial_edges)
        best_count = count_edges(best_adj)

    random.seed(42)
    
    current_adj = best_adj[:]
    current_count = best_count
    
    max_iters = 200000
    iter_count = 0
    stall_count = 0
    max_stall = 5000
    
    while time.time() < deadline and iter_count < max_iters:
        iter_count += 1
        
        # Strategy 1: Try to add a random edge
        added = False
        # Pick random non-edge
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u != v:
            if not (current_adj[u] & (1 << v)):
                new_adj = current_adj[:]
                add_edge(new_adj, u, v)
                if is_valid_adj(new_adj):
                    current_adj = new_adj
                    current_count += 1
                    if current_count > best_count:
                        best_count = current_count
                        best_adj = current_adj[:]
                    added = True
                    stall_count = 0
                    continue
        
        # Strategy 2: Swap (remove one, add one)
        # Pick a random edge to remove
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v or not (current_adj[u] & (1 << v)):
            # Try another random edge
            found_edge = False
            for _ in range(10):
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u != v and (current_adj[u] & (1 << v)):
                    found_edge = True
                    break
            if not found_edge:
                # No edges? Break
                if current_count == 0:
                    break
                else:
                    # Should not happen, but safeguard
                    continue
            
        # Pick a random non-edge to add
        swap_done = False
        for _ in range(10):
            new_u = random.randint(0, n - 1)
            new_v = random.randint(0, n - 1)
            if new_u == new_v:
                continue
            if (new_u, new_v) == (u, v) or (new_u, new_v) == (v, u):
                continue
            if not (current_adj[new_u] & (1 << new_v)):
                new_adj = current_adj[:]
                remove_edge(new_adj, u, v)
                add_edge(new_adj, new_u, new_v)
                if is_valid_adj(new_adj):
                    current_adj = new_adj
                    # Count stays same
                    swap_done = True
                    break
        
        if not added and not swap_done:
            stall_count += 1
        else:
            stall_count = 0
            
        if stall_count > max_stall:
            # Restart from best
            current_adj = best_adj[:]
            current_count = best_count
            stall_count = 0
            max_stall *= 1.5 # Increase patience slightly

    # Convert best_adj back to edge list
    final_edges = []
    for i in range(n):
        ai = best_adj[i]
        for j in range(i + 1, n):
            if ai & (1 << j):
                final_edges.append([i, j])
                
    # Ensure types are Python int
    final_edges = [[int(e[0]), int(e[1])] for e in final_edges]
    
    return final_edges
