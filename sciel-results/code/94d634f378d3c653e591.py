"""Improved program: local search to maximize n for K2,2,2-free two-colouring."""
import time
import numpy as np

def _edge_index(i, j, n):
    # index of edge (i,j) in the lexicographic list for K_n
    return i * n - i * (i + 1) // 2 + (j - i - 1)

def _count_octahedrons(colors, n):
    """Count monochromatic K2,2,2 subgraphs.
    K2,2,2 on vertices {a,b}x{c,d}x{e,f} means:
      all 4 edges between {a,b} and {c,d} same colour,
      all 4 edges between {c,d} and {e,f} same colour,
      all 4 edges between {e,f} and {a,b} same colour.
    Equivalently, partition 6 vertices into 3 pairs such that all cross-pair edges are same colour.
    We count over all 6-sets and all 15 pairings into 3 pairs.
    """
    count = 0
    # Precompute adjacency for each colour for speed
    # adj0[u] = set of v such that edge(u,v) is colour 0
    # But for counting we need: for a triple of pairs (A,B,C), all edges between A and B, B and C, A and C must be same colour.
    # Let's just iterate over all 6-sets and check.
    # For n up to ~29, C(29,6) = 475020, times 15 pairings = ~7M checks. Each check is 12 edge lookups.
    # That's ~84M operations, might be slow in pure Python. Let's optimize.
    
    # Better approach: for each pair of disjoint pairs (a,b) and (c,d), check if all 4 cross edges are same colour.
    # If yes, then for each pair (e,f) disjoint from {a,b,c,d}, check if all 4 edges between (e,f) and (a,b) AND all 4 between (e,f) and (c,d) are same colour.
    
    # Precompute for each pair (i,j) the colour of the 4 edges from {i,j} to a given vertex k? No.
    # Let's precompute: for each pair (i,j), and for each other vertex k, the colours of edges (i,k) and (j,k).
    # Actually, let's just be careful with the iteration.
    
    # For n <= 29, let's use a somewhat optimized pure Python approach with early termination.
    colors = np.array(colors)
    
    # Precompute edge lookup table
    # edge_color[u][v] for u < v
    edge_color = {}
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            edge_color[(i,j)] = colors[idx]
            idx += 1
    
    # For a set of 4 vertices {a,b,c,d}, check if all 4 edges between {a,b} and {c,d} are same colour.
    # i.e., edge(a,c)==edge(a,d)==edge(b,c)==edge(b,d)
    
    # Iterate over all ways to pick 6 distinct vertices and partition into 3 pairs.
    # To avoid overcounting, we fix an ordering: a < b < c < d < e < f
    # Then the 15 pairings of {a,b,c,d,e,f} into 3 unordered pairs.
    
    # This is getting complex. Let me use a different counting method.
    # Count = sum over all 6-subsets S of (number of perfect matchings of K6[S] that are monochromatic)
    # A perfect matching is monochromatic if all 3 edges have the same colour.
    
    # For n=29, C(29,6)=475020. For each, 15 matchings. Total ~7.1M iterations.
    # In Python, this might take ~30-60s. Too slow for repeated calls.
    
    # Let me use a numpy-based approach or just accept the slowness and limit the search.
    # Actually, for the search, I don't need exact counts, I just need to know if count > 0.
    # And for the search, I can use a faster heuristic check.
    
    # For now, let's implement a reasonably fast version and optimize later if needed.
    
    count = 0
    from itertools import combinations, permutations
    
    # Precompute all 15 pairings of 6 elements
    # A pairing is a set of 3 pairs. Represent as a frozenset of frozensets.
    # For vertices 0,1,2,3,4,5, the pairings are:
    # (0,1)(2,3)(4,5), (0,1)(2,4)(3,5), ..., etc.
    
    # Actually, let's just iterate over all 6-subsets and for each, check all 15 pairings.
    # To speed up, precompute for each pair of pairs whether they are "compatible" (all 4 cross edges same colour).
    
    # Let's precompute a table: for each unordered pair of disjoint pairs, the "compatibility" 
    # i.e., whether all 4 edges between the two pairs are the same colour, and what that colour is.
    
    # pair_compatible[(a,b),(c,d)] = colour if all 4 cross edges are that colour, else -1
    
    # This table has C(n,2) * C(n-2,2) / 2 entries... no, ordered pairs of disjoint pairs.
    # Number of ways to choose two disjoint pairs: C(n,2) * C(n-2,2) / 2? No.
    # Choose 4 vertices: C(n,4). For each, 3 ways to partition into 2 pairs. So 3*C(n,4) entries.
    # For n=29, 3*C(29,4) = 3*23751 = 71253 entries. That's manageable.
    
    # Let's build this table.
    pair_pair_color = {}
    for i in range(n):
        for j in range(i+1, n):
            for k in range(n):
                if k == i or k == j:
                    continue
                for l in range(k+1, n):
                    if l == i or l == j:
                        continue
                    # pairs (i,j) and (k,l), with i<j, k<l, and {i,j} != {k,l}
                    # To avoid duplicates, require min(i,j) < min(k,l) or something.
                    # Actually, let's just store for all ordered pairs of disjoint pairs.
                    c1 = edge_color[(min(i,j), max(i,j))]
                    c2 = edge_color[(min(i,k), max(i,k))]
                    c3 = edge_color[(min(i,l), max(i,l))]
                    c4 = edge_color[(min(j,k), max(j,k))]
                    c5 = edge_color[(min(j,l), max(j,l))]
                    # Wait, I need edges between {i,j} and {k,l}:
                    # (i,k), (i,l), (j,k), (j,l)
                    c_ik = edge_color[(min(i,k), max(i,k))]
                    c_il = edge_color[(min(i,l), max(i,l))]
                    c_jk = edge_color[(min(j,k), max(j,k))]
                    c_jl = edge_color[(min(j,l), max(j,l))]
                    if c_ik == c_il == c_jk == c_jl:
                        pair_pair_color[((i,j),(k,l))] = c_ik
                        pair_pair_color[((k,l),(i,j))] = c_ik
                    else:
                        pair_pair_color[((i,j),(k,l))] = -1
                        pair_pair_color[((k,l),(i,j))] = -1
    
    # Now count: for each 6-subset {a,b,c,d,e,f}, for each of the 15 pairings,
    # check if all 3 pairs of pairs are compatible with the same colour.
    
    # Precompute the 15 pairings of 6 elements
    # Represent a pairing as a tuple of 3 tuples, each of length 2, sorted.
    pairings_6 = []
    # All ways to partition {0,1,2,3,4,5} into 3 pairs
    # Fix 0's partner: 1,2,3,4,5
    for p0 in range(1, 6):
        remaining = [x for x in range(6) if x != 0 and x != p0]
        for p1 in remaining:
            rest = [x for x in remaining if x != p1]
            p2 = rest[0]
            p3 = rest[1]
            # pairs: (0,p0), (p1,p2), (p1,p3)? No, (p1, rest[0]) and (rest[1], rest[2])? 
            # remaining has 4 elements. Pick first as p1, then the other 3: pick one as partner of p1.
            # Let me redo:
            pass
    
    # Let me just hardcode the 15 pairings for 6 elements.
    # Elements: 0,1,2,3,4,5
    pairings_6 = [
        ((0,1),(2,3),(4,5)),
        ((0,1),(2,4),(3,5)),
        ((0,1),(2,5),(3,4)),
        ((0,2),(1,3),(4,5)),
        ((0,2),(1,4),(3,5)),
        ((0,2),(1,5),(3,4)),
        ((0,3),(1,2),(4,5)),
        ((0,3),(1,4),(2,5)),
        ((0,3),(1,5),(2,4)),
        ((0,4),(1,2),(3,5)),
        ((0,4),(1,3),(2,5)),
        ((0,4),(1,5),(2,3)),
        ((0,5),(1,2),(3,4)),
        ((0,5),(1,3),(2,4)),
        ((0,5),(1,4),(2,3)),
    ]
    
    # Now iterate over all 6-subsets
    from itertools import combinations
    for six in combinations(range(n), 6):
        # For each pairing, check if it's a monochromatic K2,2,2
        for pairing in pairings_6:
            # pairing is 3 pairs: (a,b), (c,d), (e,f)
            # Check if all 3 cross-pair connections are same colour
            c1 = pair_pair_color.get(((six[0], six[1]), (six[2], six[3])), -1)
            # Wait, this is wrong. The pairing pairs specific elements of the 6-set.
            # Let me map the pairing indices to the actual vertices.
            pass
    
    # This is getting too complex and slow. Let me use a different approach.
    # For the search, I'll use a faster validity check that may be approximate,
    # and only do exact counting when needed.
    
    # Actually, let me just implement a direct check for a given 6-set and pairing.
    # For speed, I'll precompute edge colors in a 2D array.
    
    edge_arr = np.zeros((n, n), dtype=np.int8)
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            edge_arr[i, j] = colors[idx]
            edge_arr[j, i] = colors[idx]
            idx += 1
    
    count = 0
    for six in combinations(range(n), 6):
        for pairing in pairings_6:
            # pairing: 3 pairs of indices into the 6-set
            # (p0a, p0b), (p1a, p1b), (p2a, p2b)
            pa = six[pairing[0][0]]; pb = six[pairing[0][1]]
            ca = six[pairing[1][0]]; cb = six[pairing[1][1]]
            ea = six[pairing[2][0]]; eb = six[pairing[2][1]]
            
            # Check all 4 edges between {pa,pb} and {ca,cb}
            c1 = edge_arr[pa, ca]; c2 = edge_arr[pa, cb]; c3 = edge_arr[pb, ca]; c4 = edge_arr[pb, cb]
            if not (c1 == c2 == c3 == c4):
                continue
            col = c1
            # Check all 4 edges between {ca,cb} and {ea,eb}
            c5 = edge_arr[ca, ea]; c6 = edge_arr[ca, eb]; c7 = edge_arr[cb, ea]; c8 = edge_arr[cb, eb]
            if not (c5 == c6 == c7 == c8 == col):
                continue
            # Check all 4 edges between {ea,eb} and {pa,pb}
            c9 = edge_arr[ea, pa]; c10 = edge_arr[ea, pb]; c11 = edge_arr[eb, pa]; c12 = edge_arr[eb, pb]
            if not (c9 == c10 == c11 == c12 == col):
                continue
            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120s limit, keep margin
    
    # Start with Paley 17
    n0 = 17
    squares = {x * x % 17 for x in range(1, 17)}
    colors0 = [0 if (j - i) % 17 in squares else 1 for i in range(n0) for j in range(i + 1, n0)]
    
    best_n = n0
    best_colors = colors0[:]
    
    # Try to extend by adding vertices one at a time
    # For n from 18 to 31, try to find a valid colouring
    for target_n in range(n0 + 1, 32):
        if time.time() > deadline - 30:
            break
        
        # Try random initialisation for the new edges
        for attempt in range(200):
            if time.time() > deadline - 30:
                break
            
            # Start from best_colors and add edges for the new vertex
            colors = best_colors[:]
            new_vertex = target_n - 1
            # Add edges from new_vertex to all existing vertices
            new_edges = []
            for i in range(target_n - 1):
                # Random initial colour
                new_edges.append(1 if np.random.random() < 0.5 else 0)
            colors.extend(new_edges)
            
            # Local search: flip edges to reduce octahedron count
            for it in range(500):
                if time.time() > deadline - 30:
                    break
                
                # Pick a random edge to flip
                # Focus on edges involving the new vertex for efficiency
                if np.random.random() < 0.7:
                    # Flip an edge involving the new vertex
                    u = np.random.randint(0, target_n - 1)
                    v = new_vertex
                else:
                    u = np.random.randint(0, target_n)
                    v = np.random.randint(0, target_n)
                    if u == v:
                        continue
                    if u > v:
                        u, v = v, u
                
                idx = _edge_index(u, v, target_n)
                old_color = colors[idx]
                colors[idx] = 1 - old_color
                
                # Check if this creates an octahedron
                # For speed, only check octahedrons involving this edge
                # An octahedron containing edge (u,v) means u and v are in the same pair of the partition
                # So we need to find 4 other vertices that form two pairs, such that all cross edges are same colour.
                
                # For now, do a full check (slow but correct)
                count = _count_octahedrons(colors, target_n)
                
                if count > 0:
                    # Revert
                    colors[idx] = old_color
                else:
                    # Accept
                    best_n = target_n
                    best_colors = colors[:]
                    break
            
            if best_n == target_n:
                break
    
    return {'n': best_n, 'colors': best_colors}
