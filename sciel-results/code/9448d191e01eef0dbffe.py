import time
import random
import sys

def construct():
    n = 15
    start_time = time.time()
    deadline = 50.0

    def is_valid_adj(adj):
        degrees = [bin(adj[i]).count('1') for i in range(n)]
        
        for u in range(n):
            au = adj[u]
            for v in range(u + 1, n):
                if not (au & (1 << v)):
                    continue
                # Edge (u, v) exists
                au_minus_v = au & ~(1 << v)
                av_minus_u = adj[v] & ~(1 << u)
                
                # We need to check if there exist s != t such that:
                # 1. s is a neighbor of u (s != v)
                # 2. t is a neighbor of v (t != u)
                # 3. s != t (automatically if we check sets, but s could equal t only if s==t, but s!=v and t!=u, and s!=t is required)
                # Actually the condition is s != t.
                # The path is s-u-v-t.
                # s is in N(u)\{v}
                # t is in N(v)\{u}
                # We need to check if there is a pair (s, t) with s != t such that degrees[s] == degrees[t].
                
                # Optimization:
                # Let Set_S = { s in N(u)\{v} }
                # Let Set_T = { t in N(v)\{u} }
                # If there exists s in Set_S and t in Set_T such that s != t and deg[s] == deg[t], invalid.
                
                # We can group degrees in Set_S and Set_T.
                # For each degree d:
                #   count_S[d] = number of s in Set_S with deg[s] == d
                #   count_T[d] = number of t in Set_T with deg[t] == d
                # If we pick s with deg d and t with deg d:
                #   Total pairs = count_S[d] * count_T[d]
                #   Pairs with s == t:
                #      If s == t, then s must be in N(u)\{v} AND in N(v)\{u}.
                #      So s is a common neighbor of u and v (excluding u,v themselves which are not in sets).
                #      Let Common_d = number of vertices x such that x in Set_S AND x in Set_T AND deg[x] == d.
                #   Valid bad pairs = count_S[d] * count_T[d] - Common_d
                #   If any valid bad pair > 0, then there exists s != t.
                
                # So for each edge (u,v), we check:
                # For all d, count_S[d] * count_T[d] > Common_d ?
                
                # Precompute degrees list is already done.
                
                # Let's build the counts for this edge.
                # Since degrees are small (0..14), we can use arrays of size 15.
                
                count_S = [0] * 15
                count_T = [0] * 15
                common_count = [0] * 15
                
                s_mask = au_minus_v
                while s_mask:
                    s_bit = s_mask & -s_mask
                    s = s_bit.bit_length() - 1
                    s_mask ^= s_bit
                    count_S[degrees[s]] += 1
                    # Check if s is also in N(v)\{u}
                    # s is in N(v) if adj[v] has bit s.
                    # s != u is guaranteed because s in N(u)\{v} and u is not in N(u).
                    # So we just need to check if adj[v] has bit s.
                    if adj[v] & (1 << s):
                        common_count[degrees[s]] += 1

                t_mask = av_minus_u
                while t_mask:
                    t_bit = t_mask & -t_mask
                    t = t_bit.bit_length() - 1
                    t_mask ^= t_bit
                    count_T[degrees[t]] += 1

                # Check condition
                for d in range(15):
                    if count_S[d] * count_T[d] > common_count[d]:
                        return False
        return True

    def count_edges(adj):
        total = 0
        for i in range(n):
            total += bin(adj[i]).count('1')
        return total // 2

    # Initialize with K_7,8
    # Partitions: A = 0..6, B = 7..14
    adj = [0] * n
    for u in range(7):
        for v in range(7, 15):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
    
    best_adj = adj[:]
    best_count = count_edges(best_adj)
    
    if not is_valid_adj(best_adj):
        # Fallback: Empty graph or triangles
        # Triangles are safe?
        # Path s-u-v-t.
        # In K3: 0-1-2-0? No, simple path.
        # 0-1-2. Length 2.
        # Path length 3: s-u-v-t.
        # In K3, max path length is 2 edges (3 vertices).
        # So K3 has no paths of length 3.
        # Disjoint triangles are valid.
        adj = [0] * n
        for i in range(0, n, 3):
            if i+2 < n:
                adj[i] |= (1 << (i+1)) | (1 << (i+2))
                adj[i+1] |= (1 << i) | (1 << (i+2))
                adj[i+2] |= (1 << i) | (1 << (i+1))
        best_adj = adj[:]
        best_count = count_edges(best_adj)

    random.seed(42)
    current_adj = best_adj[:]
    current_count = best_count
    
    max_iters = 100000
    iter_count = 0
    stall_count = 0
    max_stall = 5000
    
    # Precomputed list of all pairs for random selection?
    # Or just random generation.
    
    while time.time() < deadline and iter_count < max_iters:
        iter_count += 1
        improved = False
        
        # Try to add a random non-edge
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u != v and not (current_adj[u] & (1 << v)):
            new_adj = current_adj[:]
            new_adj[u] |= (1 << v)
            new_adj[v] |= (1 << u)
            if is_valid_adj(new_adj):
                current_adj = new_adj
                current_count += 1
                if current_count > best_count:
                    best_count = current_count
                    best_adj = current_adj[:]
                improved = True
                stall_count = 0
                continue
        
        # Try swap: Remove edge (u1, v1), add edge (u2, v2)
        # Choose a random edge to remove
        # To be faster, maintain a list of edges?
        # For now, random pick.
        
        # Pick random existing edge
        found_edge = False
        while not found_edge:
            u1 = random.randint(0, n - 1)
            v1 = random.randint(0, n - 1)
            if u1 != v1 and (current_adj[u1] & (1 << v1)):
                found_edge = True
        
        # Pick random non-edge to add
        found_non_edge = False
        while not found_non_edge:
            u2 = random.randint(0, n - 1)
            v2 = random.randint(0, n - 1)
            if u2 != v2 and not (current_adj[u2] & (1 << v2)):
                # Ensure it's not the same edge
                if (u2, v2) != (u1, v1) and (u2, v2) != (v1, u1):
                    found_non_edge = True
        
        if found_non_edge:
            new_adj = current_adj[:]
            new_adj[u1] &= ~(1 << v1)
            new_adj[v1] &= ~(1 << u1)
            new_adj[u2] |= (1 << v2)
            new_adj[v2] |= (1 << u2)
            if is_valid_adj(new_adj):
                current_adj = new_adj
                # Count unchanged
                improved = True
                stall_count = 0
                continue

        if not improved:
            stall_count += 1
        else:
            stall_count = 0
            
        if stall_count > max_stall:
            current_adj = best_adj[:]
            current_count = best_count
            stall_count = 0
            # Try different seed? Or just restart
            # Maybe increase aggressiveness?
            # For now, just reset.

    final_edges = []
    for i in range(n):
        ai = best_adj[i]
        for j in range(i + 1, n):
            if ai & (1 << j):
                final_edges.append([i, j])
                
    final_edges = [[int(e[0]), int(e[1])] for e in final_edges]
    return final_edges
