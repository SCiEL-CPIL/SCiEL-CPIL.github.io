import time
import random
import numpy as np
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    def edge_idx(n, u, v):
        if u > v:
            u, v = v, u
        return u * (n - 1) - u * (u - 1) // 2 + (v - u - 1)

    def generate_cycles(n):
        """Generate all C6 cycles in K_n as arrays of edge indices."""
        cycles = []
        # Precompute edge indices for all pairs to speed up cycle generation
        edge_map = {}
        for u in range(n):
            for v in range(u + 1, n):
                edge_map[(u, v)] = edge_idx(n, u, v)
        
        for verts in itertools.combinations(range(n), 6):
            # Generate all distinct cycles on these 6 vertices
            seen = set()
            # Fix first vertex to avoid rotational symmetry
            # We iterate through permutations of the other 5
            # To avoid reflection symmetry, we can enforce a canonical form
            # e.g., the second vertex is smaller than the last vertex
            for perm in itertools.permutations(verts[1:]):
                seq = (verts[0],) + perm
                # Check reflection symmetry: seq[1] < seq[-1]
                if seq[1] > seq[-1]:
                    continue
                edges = []
                for i in range(6):
                    u, v = seq[i], seq[(i+1)%6]
                    if u > v:
                        u, v = v, u
                    e_idx = edge_map[(u, v)]
                    edges.append(e_idx)
                edge_set = frozenset(edges)
                if edge_set not in seen:
                    seen.add(edge_set)
                    cycles.append(list(edge_set))
        return np.array(cycles, dtype=np.int32)

    def calc_cost(colors_np, cycles_np):
        """Calculate the number of monochromatic C6s."""
        if len(cycles_np) == 0:
            return 0
        cyc_colors = colors_np[cycles_np]
        # Check if all 6 edges in the cycle have the same color
        is_mono = np.all(cyc_colors == cyc_colors[:, 0:1], axis=1)
        return int(np.sum(is_mono))

    def solve_local_search(n, time_limit):
        """
        Attempt to find a valid 4-coloring of K_n using local search.
        """
        num_edges = n * (n - 1) // 2
        rng = np.random.default_rng(42)
        
        # Generate cycles
        if time.time() > time_limit:
            return None
            
        cycles = generate_cycles(n)
        
        # Initialize colors randomly
        colors = rng.integers(0, 4, size=num_edges).astype(np.int32)
        
        current_cost = calc_cost(colors, cycles)
        if current_cost == 0:
            return colors.tolist()
            
        best_colors = colors.copy()
        best_cost = current_cost
        
        # Local Search
        while time.time() < time_limit:
            # Pick a random edge to flip
            e_idx = rng.integers(0, num_edges)
            old_c = colors[e_idx]
            
            best_new_cost = current_cost
            best_new_c = old_c
            
            for c in range(4):
                if c == old_c:
                    continue
                colors[e_idx] = c
                new_cost = calc_cost(colors, cycles)
                if new_cost < best_new_cost:
                    best_new_cost = new_cost
                    best_new_c = c
            
            # Accept the best color for this edge
            colors[e_idx] = best_new_c
            current_cost = best_new_cost
            
            if current_cost == 0:
                return colors.tolist()
                
            if current_cost < best_cost:
                best_cost = current_cost
                best_colors = colors.copy()
                
            # Random restart if stuck for too long? 
            # Or just continue.
            
        return best_colors.tolist() if best_cost == 0 else None

    # Strategy:
    # 1. Try n=18 with 60s limit.
    # 2. Try n=17 with 50s limit.
    # 3. Fallback to hardcoded n=16 or smaller.

    # Try n=18
    if time.time() < deadline - 60:
        res = solve_local_search(18, time.time() + 55)
        if res is not None:
            return {'n': 18, 'colors': res}

    # Try n=17
    if time.time() < deadline - 10:
        res = solve_local_search(17, time.time() + 5)
        if res is not None:
            return {'n': 17, 'colors': res}
            
    # Fallback: Return a known valid small solution
    # n=9 solution
    n_fb = 9
    colors_fb = [min(j - i, n_fb - j + i) - 1 for i in range(n_fb) for j in range(i + 1, n_fb)]
    return {'n': n_fb, 'colors': colors_fb}
