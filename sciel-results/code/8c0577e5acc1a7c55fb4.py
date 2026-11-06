import time
import random
import sys
import itertools
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    def get_fallback(n):
        # For n=5, no C6 is possible, so any coloring works.
        # For n<6, any coloring works.
        m = n * (n - 1) // 2
        colors = [0] * m
        return {'n': n, 'colors': colors}

    best_result = get_fallback(5)

    def edge_index(i, j, n):
        if i > j:
            i, j = j, i
        # Formula for index of edge (i, j) in lexicographic order of (u, v) with u < v
        # Number of edges with first vertex < i is i * (n - 1) - i * (i + 1) / 2 ? 
        # Let's derive:
        # Edges with first vertex 0: n-1
        # Edges with first vertex 1: n-2
        # ...
        # Edges with first vertex i-1: n-1-(i-1) = n-i
        # Total edges before i: sum_{k=0}^{i-1} (n-1-k) = i*(n-1) - i*(i-1)/2
        # Within vertex i, edges are (i, i+1), (i, i+2), ..., (i, n-1)
        # Index of j among these: j - (i+1)
        # Total index = i*(n-1) - i*(i-1)/2 + j - i - 1
        # Let's simplify: i*n - i - i^2/2 + i/2 + j - i - 1 = i*n - i^2/2 - i/2 + j - 1
        # Alternatively: i * (2*n - i - 1) // 2 + j - i - 1
        # Let's check for n=4, i=0, j=1 -> 0 + 0 = 0. Correct.
        # i=0, j=2 -> 0 + 1 = 1. Correct.
        # i=1, j=2 -> 1*(8-1-1)//2 + 2-1-1 = 3 + 0 = 3. 
        # Edges: (0,1)=0, (0,2)=1, (0,3)=2, (1,2)=3. Correct.
        return i * (2 * n - i - 1) // 2 + j - i - 1

    def build_cycles_and_maps(n):
        """
        Returns:
        cycles: np.array of shape (N, 6) containing edge indices
        edge_to_other_edges: list of np.arrays, where edge_to_other_edges[e] 
                             is an array of shape (K, 5) containing the other 5 edges 
                             for each of the K cycles containing edge e.
        """
        m = n * (n - 1) // 2
        cycles_list = []
        verts = list(range(n))
        
        # Generate all C6 cycles
        # For each combination of 6 vertices, there are 60 distinct undirected cycles.
        # We fix the smallest vertex v0.
        # The remaining 5 vertices are permuted.
        # To avoid duplicates (reversals), we enforce v1 < v5.
        for combo in itertools.combinations(verts, 6):
            v0 = combo[0]
            others = combo[1:]
            for perm in itertools.permutations(others):
                if perm[0] > perm[-1]:
                    continue
                path = [v0] + list(perm)
                cycle_edges = []
                for k in range(6):
                    u = path[k]
                    v = path[(k + 1) % 6]
                    cycle_edges.append(edge_index(u, v, n))
                cycles_list.append(cycle_edges)
        
        if not cycles_list:
            return np.array([]).reshape(0, 6), [np.array([]).reshape(0, 5) for _ in range(m)]

        cycles = np.array(cycles_list, dtype=np.int32)
        N = len(cycles)
        
        # Map edge to list of cycle indices
        edge_to_cycles = [[] for _ in range(m)]
        for i in range(N):
            for e in cycles[i]:
                edge_to_cycles[e].append(i)
        
        # Precompute 'other edges' for each edge
        # edge_to_other_edges[e] will be a numpy array of shape (len(edge_to_cycles[e]), 5)
        edge_to_other_edges = [None] * m
        for e in range(m):
            cyc_ids = edge_to_cycles[e]
            if not cyc_ids:
                edge_to_other_edges[e] = np.array([]).reshape(0, 5)
                continue
            
            other_edges_list = []
            for cid in cyc_ids:
                ce = cycles[cid]
                # ce is a numpy array of 6 ints
                # Create a mask of where ce != e
                mask = ce != e
                # There should be exactly 5 True values
                other_edges_list.append(ce[mask])
            
            edge_to_other_edges[e] = np.array(other_edges_list, dtype=np.int32)
            
        return cycles, edge_to_other_edges

    def calculate_initial_bad(colors, cycles):
        """
        colors: np.array of shape (M,)
        cycles: np.array of shape (N, 6)
        Returns: number of monochromatic cycles
        """
        if len(cycles) == 0:
            return 0
        # Get colors for all edges in all cycles
        cycle_colors = colors[cycles]
        # Check if all 6 are the same
        first = cycle_colors[:, 0]
        is_mono = np.all(cycle_colors == first[:, np.newaxis], axis=1)
        return np.sum(is_mono)

    def search_n(n, time_budget, seed_offset=0):
        if time.time() > deadline - 5:
            return None

        m = n * (n - 1) // 2
        random.seed(42 + seed_offset)
        np.random.seed(42 + seed_offset)

        cycles, edge_to_other_edges = build_cycles_and_maps(n)
        N = len(cycles)

        if N == 0:
            colors = np.zeros(m, dtype=np.uint8)
            return {'n': n, 'colors': colors.tolist()}

        # Initial coloring
        colors = np.random.randint(0, 4, size=m, dtype=np.uint8)
        bad_count = calculate_initial_bad(colors, cycles)
        
        if bad_count == 0:
            return {'n': n, 'colors': colors.tolist()}

        best_colors = colors.copy()
        best_bad = bad_count

        T_start = 1.0
        T_end = 0.01
        
        start_iter_time = time.time()
        
        # Simulated Annealing
        while time.time() < start_iter_time + time_budget:
            if time.time() > deadline - 5:
                break

            # Pick random edge
            e = np.random.randint(0, m)
            c_old = colors[e]
            
            # Pick new color
            c_new = np.random.randint(0, 4)
            while c_new == c_old:
                c_new = np.random.randint(0, 4)

            # Calculate Delta
            other_edges = edge_to_other_edges[e]
            
            if len(other_edges) == 0:
                delta = 0
            else:
                # Get colors of other edges
                other_colors = colors[other_edges]
                
                # Check if all 5 are c_old
                is_all_old = np.all(other_colors == c_old, axis=1)
                # Check if all 5 are c_new
                is_all_new = np.all(other_colors == c_new, axis=1)
                
                delta = np.sum(is_all_new) - np.sum(is_all_old)

            # Acceptance
            accept = False
            if delta <= 0:
                accept = True
            else:
                # Temperature
                elapsed = time.time() - start_iter_time
                progress = min(1.0, elapsed / time_budget)
                T = T_start * (T_end / T_start) ** progress
                
                if delta / T > 50:
                    accept = False
                else:
                    prob = math.exp(-delta / T)
                    accept = random.random() < prob

            if accept:
                colors[e] = c_new
                bad_count += delta
                if bad_count < best_bad:
                    best_bad = bad_count
                    best_colors = colors.copy()
                    if best_bad == 0:
                        return {'n': n, 'colors': colors.tolist()}

        if best_bad == 0:
            return {'n': n, 'colors': best_colors.tolist()}
        return None

    # Try targets from high to low
    # We want to maximize n.
    # n=17 is the target.
    targets = [17, 16, 15, 14, 13, 12]
    
    # Allocate time based on difficulty.
    # n=17 is hardest.
    time_alloc = {17: 40, 16: 30, 15: 20, 14: 15, 13: 10, 12: 5}

    for n in targets:
        if time.time() > deadline - 10:
            break

        tb = time_alloc.get(n, 5)
        # Multiple restarts if time permits
        # For n=17, we might need multiple restarts.
        num_restarts = 1
        if n == 17:
            num_restarts = 3
        elif n == 16:
            num_restarts = 2
            
        found = None
        for r in range(num_restarts):
            if time.time() > deadline - 10:
                break
            result = search_n(n, tb / num_restarts, seed_offset=r*100)
            if result:
                found = result
                break
        
        if found:
            return found

    return best_result

if __name__ == "__main__":
    result = construct()
    print(result['n'])
