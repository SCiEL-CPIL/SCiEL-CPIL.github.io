"""5-colourings of K_n without monochromatic C4.
Uses Simulated Annealing with efficient delta-energy updates to find
valid 5-colourings of K_n for large n.
"""
import time
import random
import numpy as np

_START = time.time()
_LIMIT = 150.0
_MARGIN = 8.0

def _time_left():
    return _LIMIT - _MARGIN - (time.time() - _START)

def _validate(n, colors):
    """Check that no colour class contains a 4-cycle (C4)."""
    if len(colors) != n * (n - 1) // 2:
        return False
    if any(c < 0 or c > 4 for c in colors):
        return False
    nbrs = [[set() for _ in range(n)] for _ in range(5)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            nbrs[c][i].add(j)
            nbrs[c][j].add(i)
            idx += 1
    for c in range(5):
        for i in range(n):
            for j in range(i + 1, n):
                if len(nbrs[c][i] & nbrs[c][j]) >= 2:
                    return False
    return True

def _edge_index(i, j, n):
    """Compute flat index for edge (i,j) where i < j."""
    if i > j:
        i, j = j, i
    return i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)

def _seed_construction():
    """Original K11 seed: split into five Hamiltonian cycles by circular distance 1..5."""
    n = 11
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}

def _cyclic_construction(n):
    """Cyclic 5-colouring using circular distance mapped to colour (d-1)%5."""
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            d = j - i
            d = min(d, n - d)
            c = (d - 1) % 5
            colors.append(c)
    return {'n': n, 'colors': colors}

def _simulated_annealing(n, max_time=120.0, seed=42):
    """
    Simulated Annealing with efficient delta-energy updates.
    
    Energy = number of monochromatic C4s (pairs of vertices sharing >=2 common neighbors in same color).
    We minimize this to 0.
    
    State: color assignment for all edges.
    Move: recolor one edge.
    Delta: compute change in C4 count when changing edge (u,v) from color c_old to c_new.
    """
    rng = random.Random(seed)
    m = n * (n - 1) // 2
    
    # Build adjacency list for edges: for each vertex, list of (neighbor, edge_idx)
    # And for each edge, the two endpoints
    edge_endpoints = []
    vertex_edges = [[] for _ in range(n)]  # vertex_edges[v] = list of (other_vertex, edge_idx)
    
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_endpoints.append((i, j))
            vertex_edges[i].append((j, idx))
            vertex_edges[j].append((i, idx))
            idx += 1
    
    # Initial coloring: cyclic
    colors = np.zeros(m, dtype=np.int8)
    for idx, (i, j) in enumerate(edge_endpoints):
        d = j - i
        d = min(d, n - d)
        colors[idx] = (d - 1) % 5
    
    # Build neighbor sets per color (as bitmasks for speed)
    # nbr_mask[c][v] = bitmask of neighbors of v in color c
    nbr_mask = [np.zeros(n, dtype=np.uint64) for _ in range(5)]
    for idx, (i, j) in enumerate(edge_endpoints):
        c = colors[idx]
        nbr_mask[c][i] |= (1 << j)
        nbr_mask[c][j] |= (1 << i)
    
    # Compute current energy: number of C4s
    # For each color c, for each pair (i,j), count common neighbors in color c
    # A C4 exists if common neighbors >= 2
    # Energy = sum over all (c, i, j) of max(0, common_neighbors - 1) ... 
    # Actually let's just count the number of violating pairs
    def compute_energy():
        energy = 0
        for c in range(5):
            for i in range(n):
                mask_i = nbr_mask[c][i]
                for j in range(i + 1, n):
                    common = mask_i & nbr_mask[c][j]
                    # Count bits
                    cnt = bin(common).count('1')
                    if cnt >= 2:
                        energy += cnt - 1
        return energy
    
    def compute_violation_count():
        """Count number of (c, i, j) triples where i,j share >=2 common neighbors in color c."""
        count = 0
        for c in range(5):
            for i in range(n):
                mask_i = nbr_mask[c][i]
                for j in range(i + 1, n):
                    common = mask_i & nbr_mask[c][j]
                    if bin(common).count('1') >= 2:
                        count += 1
        return count
    
    current_energy = compute_energy()
    
    if current_energy == 0:
        return {'n': n, 'colors': colors.tolist()}
    
    # SA parameters
    T_start = 10.0
    T_end = 0.01
    # We'll do iterations until time runs out
    deadline = time.time() + max_time
    best_energy = current_energy
    best_colors = colors.copy()
    
    # Precompute for each edge (u,v), the list of other vertices that form a "potential C4"
    # When we change edge (u,v) from c_old to c_new:
    # - In color c_old: removing edge (u,v) may reduce C4s involving u and v
    # - In color c_new: adding edge (u,v) may create new C4s
    
    # For efficiency, let's compute delta energy for changing edge e from c_old to c_new:
    # 
    # The C4s affected are those where (u,v) is one of the two "diagonal" edges or one of the 4 sides.
    # Actually, a C4 is determined by 4 vertices {a,b,c,d} with edges forming a cycle.
    # Edge (u,v) participates in C4s where u and v are adjacent in the C4.
    # 
    # A C4 containing edge (u,v) in color c: there exist vertices x,y such that
    # (u,x), (x,y), (y,v), (v,u) all in color c, and x != y, x != v, y != u.
    # Equivalently, x and y are common neighbors of u and v in color c, and (x,y) is also in color c.
    # Wait, that's not quite right. Let me think again.
    #
    # A 4-cycle in the subgraph of color c: vertices a-b-c-d-a.
    # Edge (u,v) is in the C4 if u,v are consecutive in the cycle.
    # So if edge (u,v) is in color c, then a C4 using this edge has form:
    # u - v - x - y - u, where (v,x), (x,y), (y,u) are all in color c.
    # Or equivalently: x is a neighbor of v in color c, y is a neighbor of u in color c,
    # and (x,y) is in color c, and x != u, y != v.
    #
    # So the number of C4s using edge (u,v) in color c = number of pairs (x,y) where
    # x in N_c(v), y in N_c(u), x != u, y != v, and (x,y) in color c.
    #
    # When we remove edge (u,v) from color c_old:
    #   We lose all C4s that used this edge in color c_old.
    # When we add edge (u,v) to color c_new:
    #   We gain all C4s that use this edge in color c_new.
    #
    # But also, changing the color of (u,v) affects C4s where (u,v) is NOT a side but...
    # Actually no, (u,v) can only be a side of a C4 if it's in that color.
    # 
    # Wait, I need to be more careful. The C4 is a cycle of 4 edges. If (u,v) is one edge,
    # then the other 3 edges must also be in the same color. So yes, the count above is correct.
    
    # Let me define:
    # c4_count(c, u, v) = number of C4s in color c that include edge (u,v)
    # = |{ (x,y) : x in N_c(v) \ {u}, y in N_c(u) \ {v}, edge(x,y) in color c }|
    
    # Delta energy when changing edge e=(u,v) from c_old to c_new:
    # delta = c4_count(c_new, u, v) - c4_count(c_old, u, v)
    # BUT we also need to consider that changing the color of (u,v) changes N_c(u) and N_c(v),
    # which affects c4_count for OTHER edges. However, for the purpose of the delta,
    # we only need the direct effect on C4s that include edge (u,v).
    # 
    # Hmm, actually that's not quite right either. Let me think about it differently.
    #
    # Total energy = sum over all C4s in the graph (counting each C4 once per color).
    # A C4 is a set of 4 edges forming a cycle. It's "monochromatic" if all 4 edges have the same color.
    #
    # When we change edge (u,v) from c_old to c_new:
    # - C4s in color c_old that included (u,v) are broken (energy decreases)
    # - C4s in color c_new that would include (u,v) are created (energy increases)
    # - C4s in other colors are unaffected
    # - C4s in color c_old that did NOT include (u,v) are unaffected
    # - C4s in color c_new that did NOT include (u,v) are unaffected
    #
    # So delta = (new C4s in c_new using (u,v)) - (old C4s in c_old using (u,v))
    #
    # To compute "new C4s in c_new using (u,v)":
    #   Count pairs (x,y) where x in N_{c_new}(v), y in N_{c_new}(u), x!=u, y!=v, and edge(x,y) in c_new.
    #   But wait, after adding (u,v) to c_new, N_{c_new}(u) includes v and N_{c_new}(v) includes u.
    #   So we should count x in N_{c_new}(v) \ {u}, y in N_{c_new}(u) \ {v}, edge(x,y) in c_new.
    #   This is the same as before the change (since (u,v) wasn't in c_new before).
    #
    # To compute "old C4s in c_old using (u,v)":
    #   Count pairs (x,y) where x in N_{c_old}(v) \ {u}, y in N_{c_old}(u) \ {v}, edge(x,y) in c_old.
    #   This is the count before the change.
    
    # So the delta computation is:
    # delta = count_pairs(c_new, u, v) - count_pairs(c_old, u, v)
    # where count_pairs(c, u, v) = |{ (x,y) : x in N_c(v)\{u}, y in N_c(u)\{v}, edge(x,y) in color c }|
    
    # For efficiency, let's precompute for each color c and each vertex v, the set of neighbors.
    # Then count_pairs can be computed by iterating over the smaller of N_c(u)\{v} and N_c(v)\{u}.
    
    # Let me implement this with bitmasks for speed.
    
    def count_c4s_using_edge(c, u, v):
        """Count C4s in color c that use edge (u,v)."""
        # x in N_c(v) \ {u}, y in N_c(u) \ {v}, and edge(x,y) in color c
        mask_v = nbr_mask[c][v] & ~(1 << u)  # neighbors of v in color c, excluding u
        mask_u = nbr_mask[c][u] & ~(1 << v)  # neighbors of u in color c, excluding v
        
        # For each x in mask_v, count y in mask_u such that edge(x,y) in color c
        # i.e., y in N_c(x) & mask_u
        count = 0
        # Iterate over set bits in mask_v
        mv = mask_v
        while mv:
            x_bit = mv & (-mv)
            x = x_bit.bit_length() - 1
            mv ^= x_bit
            # Count common neighbors of x and u in color c, excluding v
            common = nbr_mask[c][x] & mask_u
            count += bin(common).count('1')
        return count
    
    # SA loop
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        
        # Temperature schedule: exponential cooling
        # Use a fixed number of iterations per "phase"
        if iteration % 1000 == 0:
            T = max(T_end, T_start * (T_end / T_start) ** (iteration / 50000))
        else:
            T = max(T_end, T_start * (T_end / T_start) ** (iteration / 50000))
        
        # Pick a random edge
        e = rng.randrange(m)
        u, v = edge_endpoints[e]
        c_old = colors[e]
        
        # Pick a random new color different from c_old
        c_new = rng.randrange(5)
        while c_new == c_old:
            c_new = rng.randrange(5)
        
        # Compute delta
        # Before change:
        lost = count_c4s_using_edge(c_old, u, v)
        # After change (in c_new, the neighbors are the same as before since (u,v) wasn't in c_new):
        gained = count_c4s_using_edge(c_new, u, v)
        delta = gained - lost
        
        # Metropolis criterion
        if delta < 0 or rng.random() < np.exp(-delta / T):
            # Accept the move
            colors[e] = c_new
            # Update neighbor masks
            nbr_mask[c_old][u] &= ~(1 << v)
            nbr_mask[c_old][v] &= ~(1 << u)
            nbr_mask[c_new][u] |= (1 << v)
            nbr_mask[c_new][v] |= (1 << u)
            current_energy += delta
            if current_energy < best_energy:
                best_energy = current_energy
                best_colors = colors.copy()
                if best_energy == 0:
                    return {'n': n, 'colors': best_colors.tolist()}
    
    # Return best found
    if best_energy == 0:
        return {'n': n, 'colors': best_colors.tolist()}
    
    # Verify and return if valid
    if _validate(n, best_colors.tolist()):
        return {'n': n, 'colors': best_colors.tolist()}
    
    return None

def construct():
    """Entry point: return the best 5-colouring found."""
    if _time_left() <= 0:
        return _seed_construction()
    
    best = _seed_construction()
    
    # Try cyclic constructions for various n
    for n in [25, 20, 15, 10, 5]:
        if _time_left() < 5:
            break
        res = _cyclic_construction(n)
        if _validate(n, res['colors']):
            if n > best['n']:
                best = res
    
    # Try Simulated Annealing for n=26
    if best['n'] < 26 and _time_left() > 30:
        sa_time = min(_time_left() - 10, 100)
        res = _simulated_annealing(26, max_time=sa_time, seed=42)
        if res and _validate(res['n'], res['colors']):
            if res['n'] > best['n']:
                best = res
    
    # Try n=25 with SA if 26 failed
    if best['n'] < 25 and _time_left() > 20:
        sa_time = min(_time_left() - 10, 60)
        res = _simulated_annealing(25, max_time=sa_time, seed=123)
        if res and _validate(res['n'], res['colors']):
            if res['n'] > best['n']:
                best = res
    
    # Try n=24
    if best['n'] < 24 and _time_left() > 15:
        sa_time = min(_time_left() - 8, 40)
        res = _simulated_annealing(24, max_time=sa_time, seed=777)
        if res and _validate(res['n'], res['colors']):
            if res['n'] > best['n']:
                best = res
    
    return best

if __name__ == "__main__":
    res = construct()
    print(f"n={res['n']}, valid={_validate(res['n'], res['colors'])}")
    print(res['colors'][:20])
