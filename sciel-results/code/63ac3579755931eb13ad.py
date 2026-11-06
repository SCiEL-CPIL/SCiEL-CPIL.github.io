import time
import itertools
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    def get_fallback(n):
        # A trivial valid coloring for small n (e.g., n=5 has no C6)
        # For n=5, M=10.
        colors = [0] * (n * (n - 1) // 2)
        return {'n': n, 'colors': colors}

    best_result = get_fallback(5)

    def edge_index(i, j, n):
        if i > j:
            i, j = j, i
        # Number of edges in rows 0 to i-1
        # Row 0 has n-1 edges
        # Row 1 has n-2 edges
        # ...
        # Row i-1 has n-i edges
        # Sum = (n-1) + (n-2) + ... + (n-i) = i*n - i(i+1)/2
        # Edge (i, j) is the (j-i-1)-th edge in row i
        return i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)

    def build_cycles_and_maps(n):
        """
        Precomputes all 6-cycles and maps edges to other edges in those cycles.
        """
        m = n * (n - 1) // 2
        cycles_list = []
        verts = list(range(n))
        
        # Generate all distinct 6-cycles
        # A 6-cycle is defined by 6 distinct vertices.
        # Number of cycles on 6 labeled vertices is (6-1)! / 2 = 60.
        # We iterate over combinations of 6 vertices, then permutations.
        # To avoid duplicates, we fix the first vertex to be the smallest in the combo?
        # No, the cycle is a set of edges.
        # Let's just generate all permutations of 6 distinct vertices from n,
        # and normalize the cycle representation.
        
        # Optimization: Iterate over combinations of 6 vertices.
        # For each combo, there are 60 distinct cycles.
        for combo in itertools.combinations(verts, 6):
            v0 = combo[0]
            others = combo[1:]
            # Generate all 5! / 2 = 60 unique cycles starting/ending at v0?
            # Actually, a cycle is v0-v1-v2-v3-v4-v5-v0.
            # v1..v5 are a permutation of others.
            # To avoid double counting (v0-v1... vs v0-v5...), we can require v1 < v5.
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
        Returns: number of monochromatic cycles
        """
        if len(cycles) == 0:
            return 0
        # Get colors for all edges in all cycles
        # cycle_colors shape (N, 6)
        cycle_colors = colors[cycles]
        # Check if all 6 are the same
        first = cycle_colors[:, 0]
        is_mono = np.all(cycle_colors == first[:, np.newaxis], axis=1)
        return int(np.sum(is_mono))

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
        T_end = 0.05
        start_iter_time = time.time()
        
        # Walk-SAT / SA Hybrid
        # We track a "walk" probability that increases if we don't improve
        no_improve_steps = 0
        max_no_improve = 1000
        
        while time.time() < start_iter_time + time_budget:
            if time.time() > deadline - 5:
                break

            # Pick random edge
            # Bias towards edges in bad cycles if we are stuck?
            # For simplicity and speed, we pick random edge.
            # If we want Walk-SAT behavior:
            # 1. Find a random bad cycle.
            # 2. Pick a random edge from that cycle.
            # 3. Change its color.
            
            # Let's implement a simple bias:
            # With probability p_walk, pick an edge from a bad cycle.
            # Else, pick random edge.
            
            # To pick an edge from a bad cycle efficiently, we'd need a list of bad cycles.
            # Maintaining that list is expensive.
            # Alternative: Just pick random edge. SA usually works well enough.
            
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
                other_colors = colors[other_edges]
                
                # Check if all 5 are c_old
                is_all_old = np.all(other_colors == c_old, axis=1)
                # Check if all 5 are c_new
                is_all_new = np.all(other_colors == c_new, axis=1)
                
                delta = int(np.sum(is_all_new)) - int(np.sum(is_all_old))

            # Acceptance
            accept = False
            if delta <= 0:
                accept = True
            else:
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
                    no_improve_steps = 0
                    if best_bad == 0:
                        return {'n': n, 'colors': colors.tolist()}
                else:
                    no_improve_steps += 1
                    
                    # Simple restart logic within the search
                    if no_improve_steps > max_no_improve:
                        # Randomize a few edges to escape local minima
                        for _ in range(10):
                            re = np.random.randint(0, m)
                            colors[re] = np.random.randint(0, 4)
                        bad_count = calculate_initial_bad(colors, cycles)
                        no_improve_steps = 0

        if best_bad == 0:
            return {'n': n, 'colors': best_colors.tolist()}
        return None

    # Try targets from high to low
    targets = [17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6]
    
    # Allocate time
    # n=17 is the most important. Give it most time.
    time_alloc = {
        17: 60, 16: 30, 15: 20, 14: 10, 13: 5, 12: 3, 11: 2, 10: 1, 9: 1, 8: 1, 7: 1, 6: 1
    }

    for n in targets:
        if time.time() > deadline - 10:
            break

        tb = time_alloc.get(n, 5)
        # Multiple restarts
        num_restarts = 3 if n <= 16 else 1
        
        for r in range(num_restarts):
            if time.time() > deadline - 10:
                break
            
            # Reduce time budget for subsequent restarts if time is tight
            current_budget = tb / (num_restarts - r) if r < num_restarts - 1 else tb
            # Ensure we don't exceed remaining time
            remaining = deadline - 10 - time.time()
            current_budget = min(current_budget, remaining)
            
            if current_budget <= 1:
                break
                
            result = search_n(n, current_budget, seed_offset=r*100)
            if result:
                # Found a valid coloring for n
                # Update best result
                best_result = result
                break # Found valid for this n, go to next lower n? 
                # Actually, we want to maximize n. If we found n=17, we are done.
                # If we found n=16, we should try to see if we can find n=17 with more time?
                # The loop goes 17, 16, 15...
                # If 17 fails, we try 16. If 16 succeeds, we return 16.
                # We should not stop at 16 if we haven't tried 17 enough.
                # But the loop structure tries 17 first.
                # If 17 returns None, we move to 16.
                # So this logic is correct.
        
        # If we found a result for n, we return it immediately as it's the highest possible for this iteration
        # Wait, the loop continues to lower n only if the current n failed.
        # So if search_n(17) returns None, we try 16.
        # If search_n(16) returns a result, we set best_result and break the inner loop.
        # Then the outer loop continues? No, we want to return the best n.
        # So if we find a valid coloring for n, we should return it immediately?
        # Yes, because we are iterating from high to low.
        
        if best_result['n'] == n:
            return best_result

    return best_result

# Fallback if construct fails or times out badly (shouldn't happen)
if __name__ == "__main__":
    res = construct()
    print(res['n'])
