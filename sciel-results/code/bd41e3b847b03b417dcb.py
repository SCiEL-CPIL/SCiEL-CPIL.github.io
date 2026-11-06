"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 88.0

    N = 256
    squares = set()
    for k in range(2, 23):
        squares.add(k * k)

    individually_invalid = set()
    for a in range(1, N + 1):
        if 2 * a in squares:
            individually_invalid.add(a)

    valid_nodes = [i for i in range(1, N + 1) if i not in individually_invalid]
    valid_set = set(valid_nodes)
    n_valid = len(valid_nodes)
    
    node_to_idx = {node: i for i, node in enumerate(valid_nodes)}
    
    # Conflict graph adjacency as bitmasks
    adj = [0] * n_valid
    for i in range(n_valid):
        a = valid_nodes[i]
        mask = 0
        for s in squares:
            j = s - a
            if 1 <= j <= N and j != a and j in valid_set:
                j_idx = node_to_idx[j]
                mask |= (1 << j_idx)
        adj[i] = mask

    all_mask = (1 << n_valid) - 1
    # Complement adjacency for max clique
    comp_adj = [0] * n_valid
    for i in range(n_valid):
        comp_adj[i] = all_mask & ~(adj[i] | (1 << i))

    # Greedy initial solution
    def greedy_is():
        remaining = all_mask
        selected = []
        while remaining:
            best_node = -1
            best_deg = float('inf')
            r = remaining
            while r:
                lsb = r & (-r)
                idx = lsb.bit_length() - 1
                deg = (adj[idx] & remaining).bit_count()
                if deg < best_deg:
                    best_deg = deg
                    best_node = idx
                r ^= lsb
            if best_node == -1:
                break
            selected.append(best_node)
            remaining &= ~adj[best_node]
            remaining &= ~(1 << best_node)
        return selected

    best_clique = set(greedy_is())
    best_size = len(best_clique)

    # Tomita max clique with bitset coloring
    def color_bound(P):
        """Greedy coloring, returns (ordered_list, color_array) where ordered_list
        has vertices sorted by color descending, and color[v] is the color of v."""
        uncolored = P
        color = [0] * n_valid
        c = 0
        while uncolored:
            c += 1
            Q = uncolored
            while Q:
                # Pick vertex with max degree in Q for this color class
                best_v = -1
                best_deg = -1
                q = Q
                while q:
                    lsb = q & (-q)
                    v = lsb.bit_length() - 1
                    deg = (comp_adj[v] & Q).bit_count()
                    if deg > best_deg:
                        best_deg = deg
                        best_v = v
                    q ^= lsb
                if best_v == -1:
                    break
                color[best_v] = c
                Q &= ~(1 << best_v)
                uncolored &= ~(1 << best_v)
        
        vertices = [v for v in range(n_valid) if (P >> v) & 1]
        vertices.sort(key=lambda v: (-color[v], v))
        return vertices, color

    def max_clique(P, clique, size):
        nonlocal best_size, best_clique
        if time.time() > deadline:
            return
        if P == 0:
            if size > best_size:
                best_size = size
                best_clique = set(clique)
            return
        if size + P.bit_count() <= best_size:
            return
        
        vertices, color = color_bound(P)
        
        # Process vertices in reverse color order (highest color first)
        # For each vertex v, new candidate set is neighbors of v in P that appear after v in ordering
        for i in range(len(vertices) - 1, -1, -1):
            v = vertices[i]
            if time.time() > deadline:
                return
            if size + color[v] <= best_size:
                break
            
            # New P: comp_adj[v] & P & (vertices that come after v in the ordering)
            # Vertices after v are vertices[0:i] (since we're going from end)
            # Actually vertices is sorted by -color, so vertices[0] has highest color
            # We process from i=len-1 down to 0, so "after" means indices < i
            after_mask = 0
            for j in range(i):
                after_mask |= (1 << vertices[j])
            
            new_P = P & comp_adj[v] & after_mask
            max_clique(new_P, clique + [v], size + 1)
    
    max_clique(all_mask, [], 0)
    
    result = sorted(valid_nodes[v] for v in best_clique)
    
    # Verify solution
    for i in range(len(result)):
        for j in range(i, len(result)):
            s = result[i] + result[j]
            r = math.isqrt(s)
            if r * r == s:
                return list(range(1, 257, 3))
    
    if not result:
        return [1]
    return result
# EVOLVE-BLOCK-END
