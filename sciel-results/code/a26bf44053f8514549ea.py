<<<<<<< SEARCH
"""Strong edge colourings of three fixed bounded-degree graphs."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def instance(n, d, seed):
        rng = random.Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_conflicts(edges, adj):
        """Build conflict neighbor lists and bitmasks."""
        m = len(edges)
        if m == 0:
            return [], []
        
        conflict_bits = [0] * m
        conflict_list = [[] for _ in range(m)]
        
        for i in range(m):
            u1, v1 = edges[i]
            for j in range(i + 1, m):
                u2, v2 = edges[j]
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    conflict_bits[i] |= (1 << j)
                    conflict_bits[j] |= (1 << i)
                    conflict_list[i].append(j)
                    conflict_list[j].append(i)
                    continue
                if (u2 in adj.get(u1, set()) or v2 in adj.get(u1, set()) or
                    u2 in adj.get(v1, set()) or v2 in adj.get(v1, set())):
                    conflict_bits[i] |= (1 << j)
                    conflict_bits[j] |= (1 << i)
                    conflict_list[i].append(j)
                    conflict_list[j].append(i)
        
        return conflict_bits, conflict_list

    def dsatur_greedy_fast(conflict_list, m, rng):
        """
        Fast DSATUR using incremental saturation tracking.
        Returns (color_array, num_colors).
        """
        color = [-1] * m
        sat_mask = [0] * m
        colored_count = [0] * m
        num_colors = 0
        
        uncolored = set(range(m))
        deg_cache = {i: len(conflict_list[i]) for i in range(m)}
        
        while uncolored:
            if time.time() > deadline - 10:
                break
            best_edge = -1
            best_sat = -1
            best_deg = -1
            
            for i in list(uncolored):
                sat = bin(sat_mask[i]).count('1')
                deg = deg_cache[i]
                if sat > best_sat or (sat == best_sat and deg > best_deg):
                    best_sat = sat
                    best_deg = deg
                    best_edge = i
            
            if best_edge == -1:
                break
            
            used = sat_mask[best_edge]
            c = 0
            while (used >> c) & 1:
                c += 1
            
            color[best_edge] = c
            if c >= num_colors:
                num_colors = c + 1
            
            uncolored.discard(best_edge)
            for j in conflict_list[best_edge]:
                if j in uncolored:
                    sat_mask[j] |= (1 << c)
                    colored_count[j] += 1
        
        return color, num_colors

    def try_remove_color_fast(color, conflict_list, target_color, m):
        """
        Try to eliminate target_color by reassigning its edges to other colors.
        Returns True if successful (mutates color in place).
        """
        target_edges = [i for i in range(m) if color[i] == target_color]
        if not target_edges:
            return True
        
        target_edges.sort(key=lambda i: len(conflict_list[i]))
        
        for idx in target_edges:
            neighbor_colors = set()
            for j in conflict_list[idx]:
                if color[j] != -1:
                    neighbor_colors.add(color[j])
            
            found = False
            max_c = max(color) if any(c != -1 for c in color) else 0
            for c in range(max_c + 1):
                if c == target_color:
                    continue
                if c not in neighbor_colors:
                    color[idx] = c
                    found = True
                    break
            
            if not found:
                return False
        
        return True

    def kempe_swap(color, conflict_list, target_color, m, rng):
        """
        Try a Kempe-chain style swap.
        """
        target_edges = [i for i in range(m) if color[i] == target_color]
        if not target_edges:
            return False
        
        max_c = max(color) if any(c != -1 for c in color) else 0
        
        for c in range(max_c + 1):
            if c == target_color:
                continue
            
            for idx in target_edges:
                neighbor_colors = set()
                for j in conflict_list[idx]:
                    if color[j] != -1:
                        neighbor_colors.add(color[j])
                if c not in neighbor_colors:
                    trial = color[:]
                    trial[idx] = c
                    if try_remove_color_fast(trial, conflict_list, target_color, m):
                        color[:] = trial
                        return True
        
        return False

    def local_search(color, conflict_list, m, max_iters=3000):
        """
        Local search with 2-opt style edge swaps.
        """
        best = color[:]
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            cur_nc = max(best) + 1 if best else 0
            if cur_nc <= 1:
                break
            
            target = cur_nc - 1
            target_edges = [i for i in range(m) if best[i] == target]
            
            if not target_edges:
                continue
            
            trial = best[:]
            if try_remove_color_fast(trial, conflict_list, target, m):
                best = trial
                continue
            
            if kempe_swap(best, conflict_list, target, m, random):
                continue
            
            # 2-opt: swap colors of two non-target edges
            if len(target_edges) >= 2:
                i1, i2 = random.sample(target_edges, 2)
                # Find non-target edges that conflict with i1 or i2
                candidates = []
                for j in range(m):
                    if best[j] != target and best[j] != -1:
                        if j in conflict_list[i1] or j in conflict_list[i2]:
                            candidates.append(j)
                
                if len(candidates) >= 2:
                    j1, j2 = random.sample(candidates, 2)
                    # Check if swap is valid
                    if best[i1] == best[j1] and best[i2] == best[j2]:
                        # Swap i1<->j2, i2<->j1
                        trial = best[:]
                        tmp = trial[i1]
                        trial[i1] = trial[j2]
                        trial[j2] = tmp
                        tmp = trial[i2]
                        trial[i2] = trial[j1]
                        trial[j1] = tmp
                        
                        # Verify
                        if try_remove_color_fast(trial, conflict_list, target, m):
                            best = trial
                            continue
            
            idx = random.choice(target_edges)
            neighbor_colors = set()
            for j in conflict_list[idx]:
                if best[j] != -1:
                    neighbor_colors.add(best[j])
            available = [c for c in range(cur_nc) if c != target and c not in neighbor_colors]
            if available:
                best[idx] = random.choice(available)
                trial = best[:]
                if try_remove_color_fast(trial, conflict_list, target, m):
                    best = trial
        
        return best

    def polish(color, conflict_list, m):
        """
        Final polish: multiple passes trying to remove colors from high to low.
        """
        best = color[:]
        
        for _ in range(10):
            if time.time() > deadline - 3:
                break
            
            cur_nc = max(best) + 1 if best else 0
            
            for target in range(cur_nc - 1, -1, -1):
                if time.time() > deadline - 3:
                    break
                
                target_edges = [i for i in range(m) if best[i] == target]
                if not target_edges:
                    continue
                
                trial = best[:]
                if try_remove_color_fast(trial, conflict_list, target, m):
                    best = trial
                    break
        
        return best

    def solve_instance(n, d, seed, time_budget):
        edges = instance(n, d, seed)
        m = len(edges)
        
        if m == 0:
            return []
        
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        conflict_bits, conflict_list = build_conflicts(edges, adj)
        
        best_color = None
        best_nc = float('inf')
        
        num_restarts = max(5, min(15, int(time_budget / 10)))
        
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break
            
            rng = random.Random(seed + restart * 1000)
            color, nc = dsatur_greedy_fast(conflict_list, m, rng)
            
            if nc < best_nc:
                best_nc = nc
                best_color = color[:]
            
            color = local_search(color, conflict_list, m, max_iters=3000)
            nc = max(color) + 1 if color else 0
            
            if nc < best_nc:
                best_nc = nc
                best_color = color[:]
        
        if best_color is not None:
            best_color = polish(best_color, conflict_list, m)
        
        return best_color if best_color is not None else []

    total_budget = deadline - start_time - 5
    weights = [40*4, 60*5, 80*6]
    total_w = sum(weights)
    budgets = [total_budget * w / total_w for w in weights]
    
    out = []
    for i, spec in enumerate(((40, 4, 101), (60, 5, 102), (80, 6, 103))):
        n, d, seed = spec
        remaining = deadline - time.time() - 5
        budget = min(budgets[i], remaining)
        if budget < 5:
            out.append([])
            continue
        out.append(solve_instance(n, d, seed, budget))
    
    while len(out) < 3:
        out.append([])
    
    return out
# EVOLVE-BLOCK-END
=======
"""Strong edge colourings of three fixed bounded-degree graphs."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def instance(n, d, seed):
        rng = random.Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_conflicts(edges, adj):
        """Build conflict neighbor lists and bitmasks."""
        m = len(edges)
        if m == 0:
            return [], []
        
        conflict_bits = [0] * m
        conflict_list = [[] for _ in range(m)]
        
        for i in range(m):
            u1, v1 = edges[i]
            for j in range(i + 1, m):
                u2, v2 = edges[j]
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    conflict_bits[i] |= (1 << j)
                    conflict_bits[j] |= (1 << i)
                    conflict_list[i].append(j)
                    conflict_list[j].append(i)
                    continue
                if (u2 in adj.get(u1, set()) or v2 in adj.get(u1, set()) or
                    u2 in adj.get(v1, set()) or v2 in adj.get(v1, set())):
                    conflict_bits[i] |= (1 << j)
                    conflict_bits[j] |= (1 << i)
                    conflict_list[i].append(j)
                    conflict_list[j].append(i)
        
        return conflict_bits, conflict_list

    def dsatur_greedy_fast(conflict_list, m, rng):
        """
        Fast DSATUR using incremental saturation tracking.
        Returns (color_array, num_colors).
        """
        color = [-1] * m
        sat_mask = [0] * m
        colored_count = [0] * m
        num_colors = 0
        
        uncolored = set(range(m))
        deg_cache = {i: len(conflict_list[i]) for i in range(m)}
        
        while uncolored:
            if time.time() > deadline - 10:
                break
            best_edge = -1
            best_sat = -1
            best_deg = -1
            
            for i in list(uncolored):
                sat = bin(sat_mask[i]).count('1')
                deg = deg_cache[i]
                if sat > best_sat or (sat == best_sat and deg > best_deg):
                    best_sat = sat
                    best_deg = deg
                    best_edge = i
            
            if best_edge == -1:
                break
            
            used = sat_mask[best_edge]
            c = 0
            while (used >> c) & 1:
                c += 1
            
            color[best_edge] = c
            if c >= num_colors:
                num_colors = c + 1
            
            uncolored.discard(best_edge)
            for j in conflict_list[best_edge]:
                if j in uncolored:
                    sat_mask[j] |= (1 << c)
                    colored_count[j] += 1
        
        return color, num_colors

    def try_remove_color_fast(color, conflict_list, target_color, m):
        """
        Try to eliminate target_color by reassigning its edges to other colors.
        Returns True if successful (mutates color in place).
        """
        target_edges = [i for i in range(m) if color[i] == target_color]
        if not target_edges:
            return True
        
        target_edges.sort(key=lambda i: len(conflict_list[i]))
        
        for idx in target_edges:
            neighbor_colors = set()
            for j in conflict_list[idx]:
                if color[j] != -1:
                    neighbor_colors.add(color[j])
            
            found = False
            max_c = max(color) if any(c != -1 for c in color) else 0
            for c in range(max_c + 1):
                if c == target_color:
                    continue
                if c not in neighbor_colors:
                    color[idx] = c
                    found = True
                    break
            
            if not found:
                return False
        
        return True

    def kempe_swap(color, conflict_list, target_color, m, rng):
        """
        Try a Kempe-chain style swap.
        """
        target_edges = [i for i in range(m) if color[i] == target_color]
        if not target_edges:
            return False
        
        max_c = max(color) if any(c != -1 for c in color) else 0
        
        for c in range(max_c + 1):
            if c == target_color:
                continue
            
            for idx in target_edges:
                neighbor_colors = set()
                for j in conflict_list[idx]:
                    if color[j] != -1:
                        neighbor_colors.add(color[j])
                if c not in neighbor_colors:
                    trial = color[:]
                    trial[idx] = c
                    if try_remove_color_fast(trial, conflict_list, target_color, m):
                        color[:] = trial
                        return True
        
        return False

    def local_search(color, conflict_list, m, max_iters=3000):
        """
        Local search with 2-opt style edge swaps.
        """
        best = color[:]
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            cur_nc = max(best) + 1 if best else 0
            if cur_nc <= 1:
                break
            
            target = cur_nc - 1
            target_edges = [i for i in range(m) if best[i] == target]
            
            if not target_edges:
                continue
            
            trial = best[:]
            if try_remove_color_fast(trial, conflict_list, target, m):
                best = trial
                continue
            
            if kempe_swap(best, conflict_list, target, m, random):
                continue
            
            # 2-opt: swap colors of two non-target edges
            if len(target_edges) >= 2:
                i1, i2 = random.sample(target_edges, 2)
                # Find non-target edges that conflict with i1 or i2
                candidates = []
                for j in range(m):
                    if best[j] != target and best[j] != -1:
                        if j in conflict_list[i1] or j in conflict_list[i2]:
                            candidates.append(j)
                
                if len(candidates) >= 2:
                    j1, j2 = random.sample(candidates, 2)
                    # Check if swap is valid
                    if best[i1] == best[j1] and best[i2] == best[j2]:
                        # Swap i1<->j2, i2<->j1
                        trial = best[:]
                        tmp = trial[i1]
                        trial[i1] = trial[j2]
                        trial[j2] = tmp
                        tmp = trial[i2]
                        trial[i2] = trial[j1]
                        trial[j1] = tmp
                        
                        # Verify
                        if try_remove_color_fast(trial, conflict_list, target, m):
                            best = trial
                            continue
            
            idx = random.choice(target_edges)
            neighbor_colors = set()
            for j in conflict_list[idx]:
                if best[j] != -1:
                    neighbor_colors.add(best[j])
            available = [c for c in range(cur_nc) if c != target and c not in neighbor_colors]
            if available:
                best[idx] = random.choice(available)
                trial = best[:]
                if try_remove_color_fast(trial, conflict_list, target, m):
                    best = trial
        
        return best

    def polish(color, conflict_list, m):
        """
        Final polish: multiple passes trying to remove colors from high to low.
        """
        best = color[:]
        
        for _ in range(10):
            if time.time() > deadline - 3:
                break
            
            cur_nc = max(best) + 1 if best else 0
            
            for target in range(cur_nc - 1, -1, -1):
                if time.time() > deadline - 3:
                    break
                
                target_edges = [i for i in range(m) if best[i] == target]
                if not target_edges:
                    continue
                
                trial = best[:]
                if try_remove_color_fast(trial, conflict_list, target, m):
                    best = trial
                    break
        
        return best

    def backtracking_min_colors(conflict_list, m, max_colors, time_limit):
        """
        Backtracking solver with forward checking to find a coloring
        using at most max_colors colors. Returns color array or None.
        """
        color = [-1] * m
        # Order vertices by descending degree for better pruning
        order = sorted(range(m), key=lambda i: len(conflict_list[i]), reverse=True)
        
        # Precompute conflict sets for fast lookup
        conflict_sets = [set(conflict_list[i]) for i in range(m)]
        
        def backtrack(pos, used_colors):
            if time.time() > time_limit:
                return False
            if pos == m:
                return True
            
            idx = order[pos]
            # Find colors used by already-colored neighbors
            neighbor_colors = set()
            for j in conflict_list[idx]:
                if color[j] != -1:
                    neighbor_colors.add(color[j])
            
            # Try colors in order: prefer colors already used (to minimize total)
            for c in range(max_colors):
                if c not in neighbor_colors:
                    color[idx] = c
                    if backtrack(pos + 1, used_colors | {c}):
                        return True
                    color[idx] = -1
                    if time.time() > time_limit:
                        return False
            
            return False
        
        if backtrack(0, set()):
            return color
        return None

    def solve_instance(n, d, seed, time_budget):
        edges = instance(n, d, seed)
        m = len(edges)
        
        if m == 0:
            return []
        
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        conflict_bits, conflict_list = build_conflicts(edges, adj)
        
        best_color = None
        best_nc = float('inf')
        
        # Phase 1: Get initial solution via DSATUR + local search
        num_restarts = max(3, min(8, int(time_budget / 15)))
        
        for restart in range(num_restarts):
            if time.time() > deadline - 20:
                break
            
            rng = random.Random(seed + restart * 1000)
            color, nc = dsatur_greedy_fast(conflict_list, m, rng)
            
            if nc < best_nc:
                best_nc = nc
                best_color = color[:]
            
            color = local_search(color, conflict_list, m, max_iters=2000)
            nc = max(color) + 1 if color else 0
            
            if nc < best_nc:
                best_nc = nc
                best_color = color[:]
        
        if best_color is not None:
            best_color = polish(best_color, conflict_list, m)
            best_nc = max(best_color) + 1 if best_color else 0
        
        # Phase 2: Try to improve via backtracking with decreasing max_colors
        remaining_time = deadline - time.time() - 5
        if remaining_time > 10 and best_color is not None and best_nc > 1:
            # Try to find coloring with fewer colors
            for target_nc in range(best_nc - 1, 0, -1):
                if time.time() > deadline - 8:
                    break
                # Give backtracking a time budget
                bt_time_limit = min(deadline - 8, time.time() + max(5, remaining_time / (best_nc - target_nc + 1)))
                result = backtracking_min_colors(conflict_list, m, target_nc, bt_time_limit)
                if result is not None:
                    best_color = result
                    best_nc = target_nc
                    break
        
        return best_color if best_color is not None else []

    total_budget = deadline - start_time - 5
    weights = [40*4, 60*5, 80*6]
    total_w = sum(weights)
    budgets = [total_budget * w / total_w for w in weights]
    
    out = []
    for i, spec in enumerate(((40, 4, 101), (60, 5, 102), (80, 6, 103))):
        n, d, seed = spec
        remaining = deadline - time.time() - 5
        budget = min(budgets[i], remaining)
        if budget < 5:
            out.append([])
            continue
        out.append(solve_instance(n, d, seed, budget))
    
    while len(out) < 3:
        out.append([])
    
    return out
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
