import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    def get_idx(i, j, n):
        if i > j:
            i, j = j, i
        return i * (i - 1) // 2 + (j - i - 1)

    def count_c4s(n, colors):
        """
        Counts total number of monochromatic C4s.
        A C4 is defined by 4 vertices a,b,c,d such that edges ab, bc, cd, da are same color.
        Equivalently, for each color c, and each pair of vertices (u,v), 
        let k = |N_c(u) ∩ N_c(v)|.
        The number of C4s in color c is sum_{u<v} C(k, 2).
        """
        neighbors = [set() for _ in range(5)]
        for c in range(5):
            neighbors[c] = [set() for _ in range(n)]
        
        for i in range(n):
            for j in range(i+1, n):
                idx = get_idx(i, j, n)
                c = colors[idx]
                neighbors[c][i].add(j)
                neighbors[c][j].add(i)
                
        bad_total = 0
        for c in range(5):
            for u in range(n):
                for v in range(u+1, n):
                    cnt = len(neighbors[c][u] & neighbors[c][v])
                    if cnt >= 2:
                        bad_total += cnt * (cnt - 1) // 2
        return bad_total

    def local_search(n, initial_colors, max_steps, deadline):
        colors = list(initial_colors)
        
        # Precompute neighbors and common counts
        neighbors = [set() for _ in range(5)]
        for c in range(5):
            neighbors[c] = [set() for _ in range(n)]
        
        for i in range(n):
            for j in range(i+1, n):
                idx = get_idx(i, j, n)
                c = colors[idx]
                neighbors[c][i].add(j)
                neighbors[c][j].add(i)
        
        common = [[[0]*n for _ in range(n)] for _ in range(5)]
        bad_total = 0
        for c in range(5):
            for u in range(n):
                for v in range(u+1, n):
                    cnt = len(neighbors[c][u] & neighbors[c][v])
                    common[c][u][v] = cnt
                    common[c][v][u] = cnt
                    if cnt >= 2:
                        bad_total += cnt * (cnt - 1) // 2
        
        if bad_total == 0:
            return colors, True
        
        best_bad = bad_total
        best_colors = list(colors)
        
        # Simulated Annealing parameters
        T = 1.0
        cool_rate = 0.999
        min_T = 0.001
        
        step = 0
        while step < max_steps:
            if time.time() > deadline:
                break
            
            # Pick random edge
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            idx = get_idx(i, j, n)
            
            old_c = colors[idx]
            new_c = random.randint(0, 4)
            if new_c == old_c:
                continue
            
            delta = 0
            
            # Update common counts for old_c (removing edge i-j)
            # For each k, if k is neighbor of i in old_c, common[old_c][k][j] decreases?
            # No, common[c][u][v] = |N_c(u) ∩ N_c(v)|.
            # Removing (i,j) from color c:
            # 1. j is removed from N_c(i).
            # 2. i is removed from N_c(j).
            # 
            # Effect on common[c][i][k]:
            #   old: |N_c(i) ∩ N_c(k)|
            #   new: |(N_c(i) \ {j}) ∩ N_c(k)|
            #   Change: -1 if j in N_c(k), else 0.
            #
            # Effect on common[c][j][k]:
            #   old: |N_c(j) ∩ N_c(k)|
            #   new: |(N_c(j) \ {i}) ∩ N_c(k)|
            #   Change: -1 if i in N_c(k), else 0.
            
            for k in range(n):
                if k == i or k == j:
                    continue
                
                # Remove from old_c
                if j in neighbors[old_c][k]:
                    old_cnt = common[old_c][i][k]
                    new_cnt = old_cnt - 1
                    delta -= (old_cnt*(old_cnt-1)//2) - (new_cnt*(new_cnt-1)//2)
                    common[old_c][i][k] = new_cnt
                    common[old_c][k][i] = new_cnt
                
                if i in neighbors[old_c][k]:
                    old_cnt = common[old_c][j][k]
                    new_cnt = old_cnt - 1
                    delta -= (old_cnt*(old_cnt-1)//2) - (new_cnt*(new_cnt-1)//2)
                    common[old_c][j][k] = new_cnt
                    common[old_c][k][j] = new_cnt
                
                # Add to new_c
                if j in neighbors[new_c][k]:
                    old_cnt = common[new_c][i][k]
                    new_cnt = old_cnt + 1
                    delta += (new_cnt*(new_cnt-1)//2) - (old_cnt*(old_cnt-1)//2)
                    common[new_c][i][k] = new_cnt
                    common[new_c][k][i] = new_cnt
                
                if i in neighbors[new_c][k]:
                    old_cnt = common[new_c][j][k]
                    new_cnt = old_cnt + 1
                    delta += (new_cnt*(new_cnt-1)//2) - (old_cnt*(old_cnt-1)//2)
                    common[new_c][j][k] = new_cnt
                    common[new_c][k][j] = new_cnt
            
            new_bad = bad_total + delta
            
            # Acceptance criterion
            if delta < 0 or (T > min_T and random.random() < pow(2.71828, -delta / T)):
                # Accept
                bad_total = new_bad
                neighbors[old_c][i].discard(j)
                neighbors[old_c][j].discard(i)
                neighbors[new_c][i].add(j)
                neighbors[new_c][j].add(i)
                colors[idx] = new_c
                
                if bad_total == 0:
                    return colors, True
                
                if bad_total < best_bad:
                    best_bad = bad_total
                    best_colors = list(colors)
            else:
                # Revert
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if j in neighbors[old_c][k]:
                        common[old_c][i][k] += 1
                        common[old_c][k][i] += 1
                    if i in neighbors[old_c][k]:
                        common[old_c][j][k] += 1
                        common[old_c][k][j] += 1
                    if j in neighbors[new_c][k]:
                        common[new_c][i][k] -= 1
                        common[new_c][k][i] -= 1
                    if i in neighbors[new_c][k]:
                        common[new_c][j][k] -= 1
                        common[new_c][k][j] -= 1
            
            T *= cool_rate
            step += 1
        
        if best_bad < bad_total:
            return best_colors, (best_bad == 0)
        return colors, (bad_total == 0)

    # Try n from 26 down to 11
    best_n = 0
    best_colors = []
    
    for n in range(26, 10, -1):
        if time.time() > deadline - 10:
            break
        
        found = False
        # Number of restarts depends on n
        max_restarts = 20 if n <= 15 else 5
        steps_per_restart = 50000 if n <= 15 else 20000
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            random.seed(n * 1000 + restart)
            initial = [random.randint(0, 4) for _ in range(n*(n-1)//2)]
            
            res_colors, is_valid = local_search(n, initial, steps_per_restart, deadline - 2)
            
            if is_valid:
                best_n = n
                best_colors = res_colors
                found = True
                break
        
        if found:
            # Try to extend to n+1?
            # For now, just return the largest valid n found
            # We can try one extension step if time permits
            if n < 26 and time.time() < deadline - 20:
                # Try extending best_colors to n+1
                new_n = n + 1
                new_edges = new_n * (new_n - 1) // 2
                # Copy old colors
                new_colors = list(best_colors)
                # Add random colors for new edges
                for _ in range(new_edges - len(best_colors)):
                    new_colors.append(random.randint(0, 4))
                
                # Run local search on new_n
                res_ext, is_valid_ext = local_search(new_n, new_colors, 20000, deadline - 5)
                if is_valid_ext:
                    best_n = new_n
                    best_colors = res_ext
            break # Break the n loop, we found a valid one
    
    if best_n == 0:
        # Fallback: return a small valid coloring
        # n=11 is likely easy to find or hardcode
        # Let's just run local search for n=11 with high restarts if we didn't find anything
        # Or just return n=11 with a known valid pattern?
        # The seed had a valid n=11? "initial best_colors for n=11 contained a monochromatic C4"
        # So we must find a valid one.
        for n in range(11, 5, -1):
            for restart in range(50):
                if time.time() > deadline - 1:
                    break
                random.seed(999 + n + restart)
                initial = [random.randint(0, 4) for _ in range(n*(n-1)//2)]
                res_colors, is_valid = local_search(n, initial, 10000, deadline - 1)
                if is_valid:
                    best_n = n
                    best_colors = res_colors
                    break
            if best_n > 0:
                break

    if best_n == 0:
        # Absolute fallback: n=2
        best_n = 2
        best_colors = [0]

    return {'n': best_n, 'colors': best_colors}
