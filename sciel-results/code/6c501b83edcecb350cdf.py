import time
import sys

# EVOLVE-BLOCK-START

def construct():
    """
    Finds a maximum cardinality subset of [1, 256] such that for all distinct a, b in the subset,
    (a * b) % (a + b) != 0.
    Uses Tomita Max Clique algorithm with improved coloring bound.
    """
    N = 256
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    # 1. Build the graph of "good" pairs
    good_adj = [0] * (N + 1)
    
    for u in range(1, N + 1):
        mask = 0
        for v in range(1, N + 1):
            if u == v:
                continue
            if (u * v) % (u + v) != 0:
                mask |= (1 << (v - 1))
        good_adj[u] = mask

    # Precompute degree for each vertex
    degrees = [good_adj[i].bit_count() for i in range(1, N + 1)]

    best_clique_mask = 0
    best_size = 0

    # Greedy Max Clique to get initial lower bound
    P = (1 << N) - 1
    R = 0
    while P:
        max_deg = -1
        best_v = -1
        p_temp = P
        while p_temp:
            lsb = p_temp & -p_temp
            v = lsb.bit_length() - 1
            deg = (good_adj[v + 1] & P).bit_count()
            if deg > max_deg:
                max_deg = deg
                best_v = v
            p_temp ^= lsb
        if best_v == -1:
            break
        R |= (1 << best_v)
        P &= good_adj[best_v + 1]
        P &= ~(1 << best_v)
        
    if R.bit_count() > best_size:
        best_size = R.bit_count()
        best_clique_mask = R

    def expand(P, R):
        nonlocal best_size, best_clique_mask
        
        if time.time() > deadline:
            return

        if not P:
            if R.bit_count() > best_size:
                best_size = R.bit_count()
                best_clique_mask = R
            return

        # Improved greedy coloring: order vertices by degree within P for tighter bound
        def color_bound(P):
            # Get vertices in P sorted by degree in P (descending)
            verts_in_P = []
            temp = P
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                verts_in_P.append(v)
                temp ^= lsb
            
            # Sort by degree in P descending
            verts_in_P.sort(key=lambda v: (good_adj[v + 1] & P).bit_count(), reverse=True)
            
            colors = []
            current_P = P
            color_idx = 0
            while current_P:
                color_class = []
                avail = current_P
                while avail:
                    # Pick vertex with max degree in avail for better coloring
                    best_v = -1
                    best_deg = -1
                    avail_temp = avail
                    while avail_temp:
                        lsb = avail_temp & -avail_temp
                        v = lsb.bit_length() - 1
                        deg = (good_adj[v + 1] & avail).bit_count()
                        if deg > best_deg:
                            best_deg = deg
                            best_v = v
                        avail_temp ^= lsb
                    
                    if best_v == -1:
                        break
                    
                    v = best_v
                    color_class.append(v)
                    # Remove v and its neighbors from avail
                    avail &= ~(good_adj[v + 1] | (1 << v))
                
                colors.append(color_class)
                color_idx += 1
                mask_to_remove = 0
                for v in color_class:
                    mask_to_remove |= (1 << v)
                current_P &= ~mask_to_remove
            
            num_colors = len(colors)
            # Branch on vertices in reverse color order
            ordered_vertices = []
            for i in range(num_colors - 1, -1, -1):
                ordered_vertices.extend(colors[i])
            return num_colors, ordered_vertices

        k, ordered_vertices = color_bound(P)
        
        if R.bit_count() + k <= best_size:
            return

        for v in ordered_vertices:
            if time.time() > deadline:
                return
            
            if not (P & (1 << v)):
                continue
                
            new_R = R | (1 << v)
            new_P = P & good_adj[v + 1]
            
            expand(new_P, new_R)
            
            P &= ~(1 << v)
            k -= 1
            
            if R.bit_count() + k <= best_size:
                break

    # Start search
    P = (1 << N) - 1
    R = 0
    expand(P, R)

    result = []
    for i in range(N):
        if best_clique_mask & (1 << i):
            result.append(i + 1)
    
    if not result:
        result = [1]
        
    return result

# EVOLVE-BLOCK-END
