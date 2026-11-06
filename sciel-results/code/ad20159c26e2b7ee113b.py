"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill

    def count_c6s_in_graph(adj, n):
        """Count the number of distinct 6-cycles in a graph given as adjacency list.
        Uses enumeration of 6-vertex subsets and checks for Hamiltonian cycles."""
        count = 0
        # For each set of 6 vertices, check if there's a Hamiltonian cycle
        # To avoid counting each cycle multiple times, fix the smallest vertex as start
        # and require the second vertex < the last vertex (to fix reversal)
        for verts in combinations(range(n), 6):
            v0 = verts[0]
            # Try all permutations of the remaining 5 vertices
            # Fix reversal: perm[0] < perm[4]
            from itertools import permutations
            for perm in permutations(verts[1:]):
                if perm[0] >= perm[4]:
                    continue
                cycle = (v0,) + perm
                # Check if all edges exist
                ok = True
                for k in range(6):
                    a = cycle[k]
                    b = cycle[(k + 1) % 6]
                    if b not in adj[a]:
                        ok = False
                        break
                if ok:
                    count += 1
        return count

    def has_c6_adj(adj, n):
        """Check if graph with adjacency list adj contains any C6."""
        for a0 in range(n):
            for a1 in adj[a0]:
                if a1 <= a0:
                    continue
                for a2 in adj[a1]:
                    if a2 == a0:
                        continue
                    for a3 in adj[a2]:
                        if a3 in (a0, a1):
                            continue
                        for a4 in adj[a3]:
                            if a4 in (a0, a1, a2):
                                continue
                            for a5 in adj[a4]:
                                if a5 in (a0, a1, a2, a3):
                                    continue
                                if a0 in adj[a5]:
                                    return True
        return False

    def check_no_c6(colors, n):
        """Verify no color class contains a C6."""
        adj = [set() for _ in range(4)]
        # Build adjacency per color
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c].add((i, j))
                adj[c].add((j, i))
                idx += 1
        # Convert to list of sets for faster lookup
        adj_list = [[set() for _ in range(n)] for _ in range(4)]
        for c in range(4):
            for (u, v) in adj[c]:
                adj_list[c][u].add(v)
                adj_list[c][v].add(u)
        for c in range(4):
            if has_c6_adj(adj_list[c], n):
                return False
        return True

    def total_c6_count(colors, n):
        """Count total number of C6s across all 4 color classes."""
        adj_list = [[set() for _ in range(n)] for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj_list[c][i].add(j)
                adj_list[c][j].add(i)
                idx += 1
        total = 0
        for c in range(4):
            if has_c6_adj(adj_list[c], n):
                # Count them
                total += count_c6s_in_graph(adj_list[c], n)
        return total

    def simulate_annealing(n, time_limit):
        """Simulated annealing to minimize total C6 count."""
        num_edges = n * (n - 1) // 2
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        # Initial random coloring
        colors = [random.randint(0, 3) for _ in range(num_edges)]
        
        def get_adj_list(colors_arr):
            adj = [[set() for _ in range(n)] for _ in range(4)]
            for idx, (u, v) in enumerate(edges):
                c = colors_arr[idx]
                adj[c][u].add(v)
                adj[c][v].add(u)
            return adj
        
        current_adj = get_adj_list(colors)
        current_cost = total_c6_count(colors, n)
        best_colors = colors[:]
        best_cost = current_cost
        
        T = 10.0
        cooling = 0.999
        min_T = 0.01
        
        while time.time() < time_limit and T > min_T:
            # Pick a random edge and change its color
            eidx = random.randint(0, num_edges - 1)
            old_c = colors[eidx]
            new_c = random.choice([c for c in range(4) if c != old_c])
            u, v = edges[eidx]
            
            # Update adjacency
            current_adj[old_c][u].discard(v)
            current_adj[old_c][v].discard(u)
            current_adj[new_c][u].add(v)
            current_adj[new_c][v].add(u)
            
            colors[eidx] = new_c
            new_cost = total_c6_count(colors, n)
            
            delta = new_cost - current_cost
            if delta <= 0 or random.random() < np.exp(-delta / T):
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_colors = colors[:]
            else:
                # Revert
                current_adj[new_c][u].discard(v)
                current_adj[new_c][v].discard(u)
                current_adj[old_c][u].add(v)
                current_adj[old_c][v].add(u)
                colors[eidx] = old_c
            
            T *= cooling
        
        return best_colors, best_cost

    # Try n=18 first
    if time.time() < deadline - 40:
        print("Trying n=18 with simulated annealing...", flush=True)
        colors, cost = simulate_annealing(18, deadline - 40)
        if cost == 0 and check_no_c6(colors, 18):
            return {'n': 18, 'colors': colors}
    
    # Try n=17
    if time.time() < deadline - 20:
        print("Trying n=17 with simulated annealing...", flush=True)
        colors, cost = simulate_annealing(17, deadline - 20)
        if cost == 0 and check_no_c6(colors, 17):
            return {'n': 17, 'colors': colors}
    
    # Try n=16
    if time.time() < deadline - 10:
        print("Trying n=16 with simulated annealing...", flush=True)
        colors, cost = simulate_annealing(16, deadline - 10)
        if cost == 0 and check_no_c6(colors, 16):
            return {'n': 16, 'colors': colors}
    
    # Fallback: small n with simple construction
    # For n=9, use cyclic construction
    n = 9
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            d = (j - i) % n
            # Assign color based on distance
            c = min(d, n - d) - 1
            if c < 0:
                c = 0
            colors.append(c)
    # Verify
    if check_no_c6(colors, n):
        return {'n': n, 'colors': colors}
    
    # Ultimate fallback
    n = 5
    colors = [0] * (n * (n - 1) // 2)
    return {'n': n, 'colors': colors}
# EVOLVE-BLOCK-END
