"""Search-based program for K4-free graphs on 18 vertices minimizing max triangle-free induced subgraph."""
import time
import random
import sys

DEADLINE = 145.0  # safety margin before 150s kill
_start_time = time.time()

def _time_left():
    return DEADLINE - (time.time() - _start_time)

def _is_k4_free_bitset(adj_bits, n=18):
    """
    Check if graph represented by list of bitmasks is K4-free.
    adj_bits[i] is an integer where bit j is 1 if edge (i,j) exists.
    """
    for a in range(n):
        # Get neighbors of a
        neighbors_a = adj_bits[a]
        # Iterate through neighbors b of a
        b = (neighbors_a & -neighbors_a).bit_length() - 1
        while b < n:
            # Actually, iterating set bits is tricky with simple bit ops in Python without popcount.
            # Let's stick to list of neighbors for K4 check but optimize it.
            break
    # The bitmask approach for K4 check:
    # A K4 is a set of 4 vertices {a,b,c,d} such that all 6 edges exist.
    # For each edge (a,b), let C = N(a) & N(b). If C contains an edge (c,d), then {a,b,c,d} is a K4.
    # So, for each pair (a,b) with edge, check if N(a) & N(b) contains any edge.
    
    # Precompute adjacency lists from bitmasks? No, let's just use the bitmask logic directly.
    for a in range(n):
        for b in range(a+1, n):
            if not (adj_bits[a] >> b) & 1:
                continue
            # Common neighbors of a and b
            common = adj_bits[a] & adj_bits[b]
            # Remove a and b from common (though they shouldn't be there if simple graph)
            common &= ~(1 << a)
            common &= ~(1 << b)
            
            # Check if there is an edge between any two vertices in common
            # Iterate through vertices c in common
            c_mask = common
            while c_mask:
                lsb = c_mask & -c_mask
                c = lsb.bit_length() - 1
                c_mask ^= lsb
                # Check if c is connected to any other vertex in common
                # Neighbors of c within common
                neighbors_c_in_common = adj_bits[c] & common
                if neighbors_c_in_common:
                    return False
    return True

def _max_triangle_free_induced_fast(adj_bits, n=18):
    """
    Estimate or exactly find the size of the largest induced triangle-free subgraph.
    Uses a greedy heuristic first, then a limited B&B if time permits.
    For the purpose of this task, a fast heuristic that is often tight is better than a slow exact one.
    However, the score must be exact? "exact branch and bound".
    Okay, I must keep it exact but make it fast.
    
    Strategy:
    1. Compute triangles.
    2. Use B&B with strong pruning:
       - Maintain a set of 'forbidden' triangles (triangles where at least 2 vertices are in the current set).
       - A vertex can be added only if it doesn't complete a triangle with 2 other vertices in the set.
       - Upper bound: |Current| + |Remaining vertices that are not 'blocked'|.
       - A vertex is blocked if it forms a triangle with any pair in Current? No, it forms a triangle with *two* vertices in Current.
    """
    # Precompute triangles as list of (a,b,c)
    triangles = []
    for a in range(n):
        for b in range(a+1, n):
            if (adj_bits[a] >> b) & 1:
                for c in range(b+1, n):
                    if (adj_bits[a] >> c) & 1 and (adj_bits[b] >> c) & 1:
                        triangles.append((a, b, c))
                        
    # For each vertex, list of triangles containing it
    vert_tris = [[] for _ in range(n)]
    for t in triangles:
        vert_tris[t[0]].append(t)
        vert_tris[t[1]].append(t)
        vert_tris[t[2]].append(t)
        
    best = [0]
    
    # Order vertices by degree in triangle graph? Or just fixed order.
    # Let's use a recursive B&B with bitmasks for 'chosen' and 'remaining'.
    
    def can_add(v, chosen_mask):
        # Check if v forms a triangle with any pair in chosen
        for t in vert_tris[v]:
            a, b, c = t
            # The other two vertices in the triangle
            others = [x for x in (a, b, c) if x != v]
            # If both others are in chosen, we cannot add v
            if (chosen_mask >> others[0]) & 1 and (chosen_mask >> others[1]) & 1:
                return False
        return True

    def search(chosen_mask, remaining_mask):
        if _time_left() < 0.5:
            return
        if remaining_mask == 0:
            if chosen_mask.bit_count() > best[0]:
                best[0] = chosen_mask.bit_count()
            return
            
        # Upper bound: chosen + remaining
        if chosen_mask.bit_count() + remaining_mask.bit_count() <= best[0]:
            return
            
        # Pick a vertex from remaining
        # Heuristic: pick vertex with most triangles in remaining?
        # For speed, just pick the lowest index in remaining
        lsb = remaining_mask & -remaining_mask
        v = lsb.bit_length() - 1
        new_remaining = remaining_mask ^ lsb
        
        # Option 1: Exclude v
        search(chosen_mask, new_remaining)
        
        # Option 2: Include v
        if can_add(v, chosen_mask):
            search(chosen_mask | (1 << v), new_remaining)
            
    search(0, (1 << n) - 1)
    return best[0]

def _turan_graph(n=18, k=3):
    """Complete k-partite graph with as equal parts as possible."""
    sizes = [n // k] * k
    for i in range(n % k):
        sizes[i] += 1
    parts = []
    idx = 0
    for s in sizes:
        parts.append(list(range(idx, idx + s)))
        idx += s
    edges = []
    for i in range(k):
        for j in range(i + 1, k):
            for a in parts[i]:
                for b in parts[j]:
                    edges.append([a, b])
    return edges

def _edges_to_bits(edges, n=18):
    adj_bits = [0] * n
    for u, v in edges:
        adj_bits[u] |= (1 << v)
        adj_bits[v] |= (1 << u)
    return adj_bits

def _bits_to_edges(adj_bits, n=18):
    edges = []
    for i in range(n):
        mask = adj_bits[i]
        while mask:
            lsb = mask & -mask
            j = lsb.bit_length() - 1
            if j > i:
                edges.append([i, j])
            mask ^= lsb
    return edges

def _random_k4_free_bits(n=18, density=0.5, rng=None):
    if rng is None:
        rng = random.Random()
    adj_bits = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < density:
                adj_bits[i] |= (1 << j)
                adj_bits[j] |= (1 << i)
    
    # Remove edges to break K4s
    changed = True
    while changed and _time_left() > 1:
        changed = False
        # Find a K4
        # We can reuse the check logic but return the K4 found
        for a in range(n):
            for b in range(a+1, n):
                if not (adj_bits[a] >> b) & 1:
                    continue
                common = adj_bits[a] & adj_bits[b]
                common &= ~(1 << a)
                common &= ~(1 << b)
                c_mask = common
                while c_mask:
                    lsb_c = c_mask & -c_mask
                    c = lsb_c.bit_length() - 1
                    c_mask ^= lsb_c
                    neighbors_c_in_common = adj_bits[c] & common
                    if neighbors_c_in_common:
                        lsb_d = neighbors_c_in_common & -neighbors_c_in_common
                        d = lsb_d.bit_length() - 1
                        # Found K4: a, b, c, d
                        # Remove a random edge from this K4
                        e = rng.choice([(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)])
                        adj_bits[e[0]] &= ~(1 << e[1])
                        adj_bits[e[1]] &= ~(1 << e[0])
                        changed = True
                        break
                if changed: break
            if changed: break
            if changed: break
    return adj_bits

def construct():
    n = 18
    best_bits = _edges_to_bits(_turan_graph(n, 3), n)
    best_score = _max_triangle_free_induced_fast(best_bits, n)
    
    rng = random.Random(42)
    
    while _time_left() > 2.0:
        # Generate candidate
        r = rng.random()
        if r < 0.3:
            # Turan
            k = rng.choice([3, 4, 5])
            cand_bits = _edges_to_bits(_turan_graph(n, k), n)
        elif r < 0.6:
            # Random
            cand_bits = _random_k4_free_bits(n, 0.6, rng)
        else:
            # Perturb best
            cand_bits = best_bits[:]
            for _ in range(rng.randint(1, 3)):
                u, v = rng.sample(range(n), 2)
                if (cand_bits[u] >> v) & 1:
                    cand_bits[u] &= ~(1 << v)
                    cand_bits[v] &= ~(1 << u)
                else:
                    cand_bits[u] |= (1 << v)
                    cand_bits[v] |= (1 << u)
        
        if not _is_k4_free_bitset(cand_bits, n):
            continue
            
        score = _max_triangle_free_induced_fast(cand_bits, n)
        if score < best_score:
            best_score = score
            best_bits = cand_bits
            if best_score <= 8:
                break
                
    return _bits_to_edges(best_bits, n)
