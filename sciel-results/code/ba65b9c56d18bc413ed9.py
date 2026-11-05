import time
import random
import numpy as np

def construct():
    deadline = time.time() + 118  # Safety margin

    def build_adj_for_color(colors, n, color):
        adj = [0] * n
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if colors[idx] == color:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                idx += 1
        return adj

    def count_violations(adj, n):
        """Count K_{2,2,2} subgraphs in graph with adjacency bitmask list adj."""
        count = 0
        for u1 in range(n):
            for u2 in range(u1 + 1, n):
                s_mask = adj[u1] & adj[u2]
                s_mask &= ~(1 << u1)
                s_mask &= ~(1 << u2)
                l = s_mask.bit_count()
                if l < 4:
                    continue
                # For each pair (v1,v2) in S, count common neighbors in S
                # K2,2 exists iff there's a pair with >= 2 common neighbors in S
                # Total K2,2,2 count: sum over pairs (u1,u2) of C(common_in_S(v1,v2), 2)
                # where common_in_S(v1,v2) = |N(v1) & N(v2) & S|
                # But we need to be careful: each K2,2,2 is counted multiple times
                # Let's just count: for fixed {u1,u2}, a K2,2,2 is determined by
                # choosing 2 more vertices from S that form a K2,2 with u1,u2
                # Actually the structure is: partition into 3 pairs of size 2.
                # For fixed pair (u1,u2), we need 4 more vertices v1,v2,v3,v4 in S
                # such that (v1,v2) is a pair and (v3,v4) is a pair, and all cross
                # edges exist. That's K2,2 between {v1,v2} and {v3,v4}.
                # So we need: in G[S], count K2,2 subgraphs.
                # A K2,2 in G[S] is determined by two disjoint pairs {a,b} and {c,d}
                # in S with all 4 cross-edges present.
                # Equivalently: for each pair (a,b) in S, let C(a,b) = |N(a)&N(b)&S|
                # Then K2,2 count = sum over pairs (a,b) in S of C(C(a,b), 2) / 2
                # (divide by 2 because each K2,2 is counted once per pair)
                # Wait, let me think again. A K2,2 has 4 vertices a,b,c,d with
                # edges a-c, a-d, b-c, b-d. It is determined by the pair {a,b}
                # (the "left" side) and the pair {c,d} (the "right" side).
                # For fixed {a,b}, the number of valid {c,d} is C(C(a,b), 2)
                # where C(a,b) = |N(a) & N(b) & S \ {a,b}|.
                # Each K2,2 is counted exactly once per choice of which pair is
                # "left", so total = sum_{pairs (a,b) in S} C(C(a,b), 2).
                # But each K2,2 has 3 ways to partition its 4 vertices into 2 pairs
                # that are the two sides of the K2,2... no wait.
                # In a K2,2 with vertices a,b,c,d (a,b on one side, c,d on other),
                # the bipartition is unique up to swapping sides. So {a,b}|{c,d}
                # and {c,d}|{a,b} are the same K2,2.
                # When we sum over all pairs (a,b) in S, we count:
                # - pair (a,b): contributes C(|N(a)&N(b)&S\{a,b}|, 2) which includes
                #   the pair (c,d) if all cross edges exist
                # - pair (c,d): contributes C(|N(c)&N(d)&S\{c,d}|, 2) which includes
                #   the pair (a,b) if all cross edges exist
                # So each K2,2 is counted twice.
                # Total K2,2 in G[S] = (1/2) * sum_{pairs (a,b) in S} C(C(a,b), 2)
                # And total K2,2,2 = sum_{pairs (u1,u2)} [K2,2 in G[S(u1,u2)]]
                # But each K2,2,2 is counted how many times in this sum?
                # A K2,2,2 has 6 vertices partitioned into 3 pairs.
                # For each choice of which pair is (u1,u2), we get one contribution.
                # There are C(3,1) = 3 ways to choose the "first" pair.
                # So total K2,2,2 count = (1/3) * sum_{pairs (u1,u2)} [K2,2 in G[S]]
                # Hmm, this is getting complex. Let me just use a simpler counting.
                
                # Simpler: just count the number of K2,2 subgraphs in G[S]
                # For each pair (v1, v2) in S:
                #   common = N(v1) & N(v2) & S & ~{(1<<v1)|(1<<v2)}
                #   contrib = C(common.bit_count(), 2)
                # Sum all contribs, divide by 2 (each K2,2 counted twice)
                verts = []
                m = s_mask
                while m:
                    lsb = m & -m
                    verts.append(lsb.bit_length() - 1)
                    m ^= lsb
                l = len(verts)
                k22_count = 0
                for i in range(l):
                    v1 = verts[i]
                    for j in range(i + 1, l):
                        v2 = verts[j]
                        common = adj[v1] & adj[v2] & s_mask
                        common &= ~(1 << v1)
                        common &= ~(1 << v2)
                        c = common.bit_count()
                        if c >= 2:
                            k22_count += c * (c - 1) // 2
                # Each K2,2 counted twice (once per side pair)
                count += k22_count // 2
        return count

    def total_violations_fast(colors, n):
        adj0 = build_adj_for_color(colors, n, 0)
        adj1 = build_adj_for_color(colors, n, 1)
        return count_violations(adj0, n) + count_violations(adj1, n)

    # 1. Paley 29
    q = 29
    squares = {x * x % q for x in range(1, q)}
    colors_29 = []
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in squares:
                colors_29.append(0)
            else:
                colors_29.append(1)
    
    # Verify
    if total_violations_fast(colors_29, 29) == 0:
        best_n = 29
        best_colors = colors_29
    else:
        best_n = 1
        best_colors = [0]

    # 2. Try n=30 with incremental violation counting
    n = 30
    n_edges = n * (n - 1) // 2
    
    # Initialize with Paley 29 + random 29 edges
    init_colors = colors_29.copy()
    for _ in range(29):
        init_colors.append(random.randint(0, 1))
    
    current_colors = init_colors
    adj0 = build_adj_for_color(current_colors, n, 0)
    adj1 = build_adj_for_color(current_colors, n, 1)
    current_viol = count_violations(adj0, n) + count_violations(adj1, n)
    
    if current_viol == 0:
        return {'n': n, 'colors': current_colors}
    
    # Precompute edge endpoints for fast lookup
    edge_u = []
    edge_v = []
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_u.append(i)
            edge_v.append(j)
            idx += 1

    # Simulated Annealing / Hill Climbing with incremental updates
    T = 1.0
    T_min = 0.01
    alpha = 0.999
    
    while time.time() < deadline - 5 and T > T_min:
        edge_idx = random.randint(0, n_edges - 1)
        old_val = current_colors[edge_idx]
        new_val = 1 - old_val
        u = edge_u[edge_idx]
        v = edge_v[edge_idx]
        
        # Flip the edge in both adjacency structures
        bit_u = 1 << u
        bit_v = 1 << v
        
        if new_val == 0:
            # Edge becomes color 0: add to adj0, remove from adj1
            adj0[u] |= bit_v
            adj0[v] |= bit_u
            adj1[u] &= ~bit_v
            adj1[v] &= ~bit_u
        else:
            # Edge becomes color 1: add to adj1, remove from adj0
            adj1[u] |= bit_v
            adj1[v] |= bit_u
            adj0[u] &= ~bit_v
            adj0[v] &= ~bit_u
        
        current_colors[edge_idx] = new_val
        new_viol = count_violations(adj0, n) + count_violations(adj1, n)
        
        delta = new_viol - current_viol
        if delta < 0 or random.random() < pow(2.718, -delta / T):
            current_viol = new_viol
            if current_viol == 0:
                return {'n': n, 'colors': current_colors}
        else:
            # Revert
            current_colors[edge_idx] = old_val
            if old_val == 0:
                adj0[u] |= bit_v
                adj0[v] |= bit_u
                adj1[u] &= ~bit_v
                adj1[v] &= ~bit_u
            else:
                adj1[u] |= bit_v
                adj1[v] |= bit_u
                adj0[u] &= ~bit_v
                adj0[v] &= ~bit_u
            
        T *= alpha
        
    return {'n': best_n, 'colors': best_colors}
