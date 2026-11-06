import time
import random
import sys
import itertools
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    def get_fallback(n):
        colors = [0] * (n * (n - 1) // 2)
        return {'n': n, 'colors': colors}

    best_result = get_fallback(5)

    def edge_index(i, j, n):
        if i > j:
            i, j = j, i
        return i * (n - 2) - i * (i - 1) // 2 + j - 1

    def build_cycles_and_maps(n):
        m = n * (n - 1) // 2
        # Precompute edge index lookup for O(1) access
        # edge_lookup[i][j] where i < j
        edge_lookup = [[-1] * n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_lookup[i][j] = idx
                edge_lookup[j][i] = idx
                idx += 1

        cycles_list = []
        verts = list(range(n))
        
        # Generate all C6 cycles
        # Optimization: Fix v0, permute others.
        # To avoid duplicates (reversals), enforce perm[0] < perm[-1]
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
                    cycle_edges.append(edge_lookup[u][v])
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
        edge_to_other_edges = [None] * m
        for e in range(m):
            cyc_ids = edge_to_cycles[e]
            if not cyc_ids:
                edge_to_other_edges[e] = np.array([]).reshape(0, 5)
                continue
            other_edges_list = []
            for cid in cyc_ids:
                ce = cycles[cid]
                mask = ce != e
                other_edges_list.append(ce[mask])
            edge_to_other_edges[e] = np.array(other_edges_list, dtype=np.int32)
            
        return cycles, edge_to_other_edges

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
        
        # Helper to calculate bad count and mask
        def calc_bad():
            if N == 0: return 0, np.array([], dtype=bool)
            cycle_colors = colors[cycles]
            first = cycle_colors[:, 0]
            is_mono = np.all(cycle_colors == first[:, np.newaxis], axis=1)
            return np.sum(is_mono), is_mono

        bad_count, bad_mask = calc_bad()
        if bad_count == 0:
            return {'n': n, 'colors': colors.tolist()}

        best_colors = colors.copy()
        best_bad = bad_count

        # Phase 1: Greedy Repair
        # Iterate until no bad cycles or time runs out
        # We can't check every step, so we check every K steps or use a timer
        # Actually, let's just run a loop for a fixed number of greedy steps or time
        greedy_time_limit = time.time() + min(2.0, time_budget * 0.2)
        
        while time.time() < greedy_time_limit:
            if bad_count == 0:
                return {'n': n, 'colors': colors.tolist()}
            
            # Find a bad cycle
            bad_indices = np.where(bad_mask)[0]
            if len(bad_indices) == 0:
                bad_count, bad_mask = calc_bad()
                continue
                
            ci = bad_indices[random.randint(0, len(bad_indices) - 1)]
            cycle_edges = cycles[ci]
            # Pick a random edge from this cycle
            e = cycle_edges[random.randint(0, 5)]
            c_old = colors[e]
            
            # Try all other colors
            best_delta = float('inf')
            best_c_new = c_old
            
            other_edges = edge_to_other_edges[e]
            if len(other_edges) == 0:
                # Edge not in any cycle? Impossible for n>=6
                best_c_new = (c_old + 1) % 4
                best_delta = 0
            else:
                other_colors = colors[other_edges]
                # Precompute "all equal to c" for all 4 colors?
                # We only need to check the 3 new colors
                for c_new in range(4):
                    if c_new == c_old:
                        continue
                    is_all_old = np.all(other_colors == c_old, axis=1)
                    is_all_new = np.all(other_colors == c_new, axis=1)
                    delta = np.sum(is_all_new) - np.sum(is_all_old)
                    if delta < best_delta:
                        best_delta = delta
                        best_c_new = c_new
            
            # Apply best move
            if best_c_new != c_old:
                colors[e] = best_c_new
                bad_count += best_delta
                # Update mask for affected cycles?
                # It's cheaper to just recalculate mask if bad_count is small, 
                # or recalculate periodically.
                # For greedy phase, let's recalculate mask every 10 steps?
                # Or just rely on bad_count and recalc mask when needed.
                # To pick next bad cycle, we need mask.
                # Let's recalc mask if bad_count changed significantly or periodically.
                # For simplicity and speed, let's recalc mask every 5 greedy steps.
                # Actually, calc_bad is vectorized and fast (~1ms).
                # Let's just recalc every step if greedy steps are few.
                # If greedy steps are many, it might be slow.
                # Let's limit greedy steps to 1000 or so.
                pass

            # Recalculate mask periodically or if we suspect it's stale
            # To be safe and simple, let's just recalc every 10 iterations of this loop
            # But we don't have an iteration counter here easily.
            # Let's just recalc every time we pick a bad cycle?
            # No, that's 1ms * (number of greedy steps).
            # If we do 1000 steps, that's 1s. Acceptable.
            bad_count, bad_mask = calc_bad()
            
            if bad_count < best_bad:
                best_bad = bad_count
                best_colors = colors.copy()

        if bad_count == 0:
            return {'n': n, 'colors': colors.tolist()}

        # Phase 2: Simulated Annealing
        T_start = 1.0
        T_end = 0.01
        start_iter_time = time.time()
        
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
                other_colors = colors[other_edges]
                is_all_old = np.all(other_colors == c_old, axis=1)
                is_all_new = np.all(other_colors == c_new, axis=1)
                delta = np.sum(is_all_new) - np.sum(is_all_old)

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
                    if best_bad == 0:
                        return {'n': n, 'colors': colors.tolist()}

        if best_bad == 0:
            return {'n': n, 'colors': best_colors.tolist()}
        return None

    # Try targets from high to low
    targets = [17, 16, 15, 14]
    
    # Allocate time
    time_alloc = {17: 40, 16: 20, 15: 10, 14: 5}

    for n in targets:
        if time.time() > deadline - 10:
            break

        tb = time_alloc.get(n, 5)
        # Multiple restarts
        num_restarts = 5 if n >= 15 else 2
        restart_budget = tb / num_restarts

        for r in range(num_restarts):
            if time.time() > deadline - 10:
                break
            result = search_n(n, restart_budget, seed_offset=r * 1000 + n)
            if result:
                best_result = result
                return best_result

    return best_result

if __name__ == "__main__":
    res = construct()
    print(f"Found n={res['n']}")
