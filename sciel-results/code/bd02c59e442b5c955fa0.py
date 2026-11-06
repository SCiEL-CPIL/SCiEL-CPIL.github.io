import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    def get_seed_solution():
        # Fallback seed, n=11
        n = 11
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                d = min(j - i, n - (j - i))
                colors.append(d - 1)
        return n, colors

    best_n, best_colors = get_seed_solution()

    def edge_idx(i, j, n):
        if i > j:
            i, j = j, i
        return i * (2 * n - i - 1) // 2 + j - i - 1

    def compute_violation_counts(neighbors, n):
        """Compute violation count per vertex pair per color using bitsets."""
        # viol_count[c][u][v] = number of common neighbors of u,v in color c
        # We only need to know if >= 2
        # Use a 2D array per color
        viol = [[0] * n for _ in range(n)]
        total = 0
        for c in range(5):
            for u in range(n):
                nu = neighbors[c][u]
                if nu == 0:
                    continue
                for v in range(u + 1, n):
                    nv = neighbors[c][v]
                    common = nu & nv
                    if common & (common - 1):
                        viol[u][v] += 1
                        total += 1
        return total, viol

    def incremental_delta(neighbors, n, i, j, c_old, c_new):
        """
        Compute delta cost when changing edge (i,j) from c_old to c_new.
        Returns the change in total violation count.
        
        The cost is sum over all pairs (u,v) of [number of colors where u,v have >=2 common neighbors].
        Changing (i,j) from c_old to c_new affects:
        - In c_old: removing edge (i,j) changes common neighbors for pairs (i,w) and (j,w)
        - In c_new: adding edge (i,j) changes common neighbors for pairs (i,w) and (j,w)
        
        For a pair (u,v) in color c, the common neighbors are N_c(u) ∩ N_c(v).
        When we add/remove edge (i,j) in color c:
        - For pair (i,w): common neighbors change by ±1 if j∈N_c(w) (i.e., edge (j,w) exists in c)
        - For pair (j,w): common neighbors change by ±1 if i∈N_c(w) (i.e., edge (i,w) exists in c)
        - Other pairs are unaffected.
        """
        delta = 0
        
        # Process c_old: removing edge (i,j)
        # For each w != i,j, check if removing j from N(i) or i from N(j) changes violation status
        for w in range(n):
            if w == i or w == j:
                continue
            # Pair (i, w) in c_old: N(i) loses j
            # j was a common neighbor of i and w if (j,w) exists in c_old
            if neighbors[c_old][j] & (1 << w):
                # j was in N(i) ∩ N(w)
                # Count common neighbors of i and w in c_old BEFORE removal
                common_before = neighbors[c_old][i] & neighbors[c_old][w]
                # j is in common_before since (i,j) and (j,w) both in c_old
                cnt_before = bin(common_before).count('1')
                # After removal, j is no longer in N(i), so common loses j
                cnt_after = cnt_before - 1
                # Check if violation status changed
                was_violation = cnt_before >= 2
                is_violation = cnt_after >= 2
                if was_violation and not is_violation:
                    delta -= 1  # Violation removed
                elif not was_violation and is_violation:
                    delta += 1  # Violation added (shouldn't happen on removal)
            
            # Pair (j, w) in c_old: N(j) loses i
            if neighbors[c_old][i] & (1 << w):
                common_before = neighbors[c_old][j] & neighbors[c_old][w]
                cnt_before = bin(common_before).count('1')
                cnt_after = cnt_before - 1
                was_violation = cnt_before >= 2
                is_violation = cnt_after >= 2
                if was_violation and not is_violation:
                    delta -= 1
                elif not was_violation and is_violation:
                    delta += 1
        
        # Process c_new: adding edge (i,j)
        for w in range(n):
            if w == i or w == j:
                continue
            # Pair (i, w) in c_new: N(i) gains j
            if neighbors[c_new][j] & (1 << w):
                # j will be in N(i) ∩ N(w) after addition
                common_before = neighbors[c_new][i] & neighbors[c_new][w]
                cnt_before = bin(common_before).count('1')
                cnt_after = cnt_before + 1
                was_violation = cnt_before >= 2
                is_violation = cnt_after >= 2
                if not was_violation and is_violation:
                    delta += 1
                elif was_violation and not is_violation:
                    delta -= 1  # Shouldn't happen on addition
            
            # Pair (j, w) in c_new: N(j) gains i
            if neighbors[c_new][i] & (1 << w):
                common_before = neighbors[c_new][j] & neighbors[c_new][w]
                cnt_before = bin(common_before).count('1')
                cnt_after = cnt_before + 1
                was_violation = cnt_before >= 2
                is_violation = cnt_after >= 2
                if not was_violation and is_violation:
                    delta += 1
                elif was_violation and not is_violation:
                    delta -= 1
        
        return delta

    def solve_local_search(n, time_limit, random_seed=None):
        """
        Local search with incremental cost updates.
        """
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random
        
        total_edges = n * (n - 1) // 2
        
        # Initialize with random colors
        colors = [rng.randint(0, 4) for _ in range(total_edges)]
        
        # Build neighbor bitsets
        neighbors = [[0] * n for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                neighbors[c][i] |= (1 << j)
                neighbors[c][j] |= (1 << i)
                idx += 1
        
        # Compute initial cost
        total_cost, _ = compute_violation_counts(neighbors, n)
        
        if total_cost == 0:
            return colors
        
        search_start = time.time()
        stagnant = 0
        max_stagnant = 2000
        
        while time.time() - search_start < time_limit:
            # Pick random edge
            i = rng.randint(0, n - 1)
            j = rng.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            eidx = edge_idx(i, j, n)
            c_old = colors[eidx]
            
            # Pick a new color (different from old)
            c_new = rng.randint(0, 4)
            if c_new == c_old:
                c_new = (c_new + 1) % 5
            
            # Calculate delta cost BEFORE modifying neighbors
            delta = incremental_delta(neighbors, n, i, j, c_old, c_new)
            
            # Apply move
            neighbors[c_old][i] &= ~(1 << j)
            neighbors[c_old][j] &= ~(1 << i)
            neighbors[c_new][i] |= (1 << j)
            neighbors[c_new][j] |= (1 << i)
            colors[eidx] = c_new
            
            new_cost = total_cost + delta
            
            if new_cost < total_cost:
                total_cost = new_cost
                stagnant = 0
                if total_cost == 0:
                    return colors
            else:
                # Revert
                neighbors[c_new][i] &= ~(1 << j)
                neighbors[c_new][j] &= ~(1 << i)
                neighbors[c_old][i] |= (1 << j)
                neighbors[c_old][j] |= (1 << i)
                colors[eidx] = c_old
                stagnant += 1
            
            if stagnant > max_stagnant:
                break
        
        return None

    # Strategy: Try n=26 first with multiple restarts, then smaller n
    targets = [26, 25, 24, 23, 22, 21]
    
    for n in targets:
        if time.time() > deadline - 5:
            break
        
        # Allocate time based on n
        if n == 26:
            t_limit = 45.0
        elif n == 25:
            t_limit = 25.0
        elif n == 24:
            t_limit = 15.0
        elif n == 23:
            t_limit = 10.0
        else:
            t_limit = 5.0
            
        t_limit = min(t_limit, deadline - time.time() - 2)
        if t_limit < 1.0:
            break
        
        # Run multiple restarts with different seeds
        found = False
        search_start = time.time()
        restart_num = 0
        while time.time() - search_start < t_limit:
            # Allocate time per restart
            time_per_restart = min(5.0, t_limit - (time.time() - search_start))
            if time_per_restart < 0.5:
                break
            
            result = solve_local_search(n, time_per_restart, random_seed=restart_num * 12345 + 67890)
            restart_num += 1
            
            if result is not None:
                found = True
                best_n = n
                best_colors = result
                break
        
        if found:
            break
            
    return {'n': best_n, 'colors': best_colors}

if __name__ == '__main__':
    res = construct()
    print(f"n={res['n']}")
