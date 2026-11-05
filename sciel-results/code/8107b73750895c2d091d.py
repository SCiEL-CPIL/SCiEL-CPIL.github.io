"""R_4(C6) - 4-colouring of K_n without monochromatic C6, using bitset-accelerated search."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = 145.0  # Leave 5s margin for safety

    def get_edges(n):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))
        return edges

    def solve_for_n(n, time_limit):
        if n <= 1:
            return {'n': n, 'colors': []}
        
        edges = get_edges(n)
        num_edges = len(edges)
        
        # Map from edge index to (u, v)
        edge_to_uv = {}
        for idx, (u, v) in enumerate(edges):
            edge_to_uv[idx] = (u, v)
        
        # Bitset representation
        # adj[c] is a list of n integers, where adj[c][i] is a bitmask of neighbors of i in color c
        
        def build_adj(colors):
            adj = [ [0] * n for _ in range(4) ]
            for idx, (u, v) in enumerate(edges):
                c = colors[idx]
                adj[c][u] |= (1 << v)
                adj[c][v] |= (1 << u)
            return adj

        def count_c6s_containing_edge_bitset(adj_c, u, v):
            """
            Count C6s containing edge (u,v) in graph adj_c.
            Cycle: u-a-b-c-d-v-u
            a in N(u), d in N(v), a!=v, d!=u, a!=d
            Path a-b-c-d of length 3 in V \ {u,v}
            """
            mask_all = (1 << n) - 1
            mask_uv = (1 << u) | (1 << v)
            
            nu = adj_c[u] & ~mask_uv
            nv = adj_c[v] & ~mask_uv
            
            # We need sum_{a in nu, d in nv} (number of paths of length 3 from a to d in G[~mask_uv])
            # Let M be adjacency matrix of G[~mask_uv].
            # Number of paths of length 3 from a to d is (M^3)[a,d].
            
            # For small n, we can just iterate?
            # n=18. nu, nv size ~ 16.
            # Iterating a in nu, d in nv is 16*16=256 pairs.
            # For each pair, count paths of length 3.
            # Path a-b-c-d.
            # b in N(a) \ {u,v,d}, c in N(b) \ {u,v,a,d}, c in N(d) \ {u,v,a,b}?
            # This is still heavy in Python if done naively for every edge check.
            
            # Optimized:
            # Precompute M^2? No, adj changes.
            
            # Let's stick to the set logic but with bitsets for intersection checks.
            # Actually, for n=18, the number of C6s is small if valid.
            # We just need to check if count is 0.
            
            count = 0
            # Iterate a in neighbors of u (excluding v)
            # Iterate d in neighbors of v (excluding u)
            # Check if there is a path a-b-c-d
            
            # Get lists of neighbors
            # Using bitsets to get neighbors is fast
            # But iterating over set bits in Python is slow.
            # However, n=18, so max 16 neighbors.
            
            neighbors_u = []
            mask = nu
            while mask:
                lsb = mask & -mask
                idx = lsb.bit_length() - 1
                neighbors_u.append(idx)
                mask ^= lsb
                
            neighbors_v = []
            mask = nv
            while mask:
                lsb = mask & -mask
                idx = lsb.bit_length() - 1
                neighbors_v.append(idx)
                mask ^= lsb
                
            # For each a in neighbors_u, d in neighbors_v
            # Check if a-b-c-d exists
            # b must be in N(a) \ {u,v}
            # c must be in N(b) \ {u,v,a}
            # and c must be in N(d) \ {u,v,b}
            
            for a in neighbors_u:
                na = adj_c[a] & ~mask_uv
                for d in neighbors_v:
                    if a == d:
                        continue
                    nd = adj_c[d] & ~mask_uv
                    # We need b such that b in na, and there exists c in N(b) intersect N(d)
                    # c in N(b) \ {u,v,a}
                    # c in N(d) \ {u,v,b}
                    
                    # Iterate b in na
                    mask_b = na
                    while mask_b:
                        lsb_b = mask_b & -mask_b
                        b = lsb_b.bit_length() - 1
                        mask_b ^= lsb_b
                        
                        if b == d: continue # b != d is implied by path structure? 
                                             # Path a-b-c-d. If b=d, then a-d-c-d? No, simple cycle.
                                             # b cannot be d.
                        
                        # Check if N(b) and N(d) have common neighbor c
                        # c in N(b) \ {u,v,a}
                        # c in N(d) \ {u,v,b}
                        
                        nb = adj_c[b] & ~mask_uv
                        # Exclude a and b from nb? 
                        # c != a, c != b, c != u, c != v
                        # nb already excludes u,v.
                        # Exclude a: nb &= ~(1 << a)
                        # Exclude b: nb &= ~(1 << b) (b is not in N(b) anyway)
                        
                        # We want intersection of (nb \ {a}) and (nd \ {b})
                        # nd already excludes u,v.
                        # Exclude b from nd: nd &= ~(1 << b)
                        
                        common = (nb & ~(1 << a)) & (nd & ~(1 << b))
                        if common:
                            count += 1
                            
            return count

        def total_c6_count_bitset(colors, adj):
            total = 0
            for idx, (u, v) in enumerate(edges):
                c = colors[idx]
                total += count_c6s_containing_edge_bitset(adj[c], u, v)
            return total // 6

        best_result = None
        
        while time.time() < time_limit:
            # Random restart
            colors = [random.randint(0, 3) for _ in range(num_edges)]
            adj = build_adj(colors)
            
            current_sum = total_c6_count_bitset(colors, adj)
            
            if current_sum == 0:
                return {'n': n, 'colors': colors}
            
            # Simulated Annealing
            T = 100.0
            no_improvement = 0
            max_no_improve = 2000
            
            while no_improvement < max_no_improve and time.time() < time_limit:
                # Pick random edge
                idx = random.randint(0, num_edges - 1)
                u, v = edge_to_uv[idx]
                old_color = colors[idx]
                
                # Try all new colors
                best_new_color = old_color
                best_new_sum = current_sum
                best_delta = 0
                
                for new_color in range(4):
                    if new_color == old_color:
                        continue
                    
                    old_count = count_c6s_containing_edge_bitset(adj[old_color], u, v)
                    
                    # Temporarily add edge to new_color
                    adj[new_color][u] |= (1 << v)
                    adj[new_color][v] |= (1 << u)
                    new_count = count_c6s_containing_edge_bitset(adj[new_color], u, v)
                    adj[new_color][u] &= ~(1 << v)
                    adj[new_color][v] &= ~(1 << u)
                    
                    delta = new_count - old_count
                    new_sum = current_sum + delta
                    
                    if new_sum < best_new_sum:
                        best_new_sum = new_sum
                        best_new_color = new_color
                        best_delta = delta
                        
                if best_new_color != old_color:
                    # Accept if better, or with probability exp(-delta/T)
                    if best_delta < 0:
                        accept = True
                    else:
                        accept = random.random() < math.exp(-best_delta / T)
                    
                    if accept:
                        adj[old_color][u] &= ~(1 << v)
                        adj[old_color][v] &= ~(1 << u)
                        adj[best_new_color][u] |= (1 << v)
                        adj[best_new_color][v] |= (1 << u)
                        colors[idx] = best_new_color
                        current_sum = best_new_sum
                        no_improvement = 0
                        
                        if current_sum == 0:
                            return {'n': n, 'colors': colors}
                    else:
                        no_improvement += 1
                else:
                    no_improvement += 1
                    
                T *= 0.995

        return None

    # Try to find 18 first
    for n in [18, 17]:
        if time.time() > deadline:
            break
        res = solve_for_n(n, deadline)
        if res:
            return res
            
    # Fallback: Try to find 17 with more time if 18 failed
    # But we already tried 17 in the loop.
    # If we failed 17, we return a known valid 17?
    # We don't have a hardcoded one in this snippet.
    # We should ensure 17 is found.
    
    # If loop finished without finding 17, do a final desperate attempt for 17
    if time.time() < deadline:
         res = solve_for_n(17, deadline)
         if res:
             return res

    # Absolute fallback: return n=1? Or a known small one.
    # The task requires valid output.
    # Let's return a trivial valid coloring for n=1
    return {'n': 1, 'colors': []}

# EVOLVE-BLOCK-END
