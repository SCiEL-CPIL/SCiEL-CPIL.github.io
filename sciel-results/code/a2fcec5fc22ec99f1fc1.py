import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Fallback: known small valid coloring (n=11 from seed, or we can just start from n=12 random)
    # We will try to maximize n.
    
    def get_idx(i, j, n):
        if i > j:
            i, j = j, i
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)
        # Alternative faster index:
        # if i > j: i, j = j, i
        # return (n * (n - 1) // 2) - (n - 1 - i) * (n - i) // 2 + (j - i - 1) ? No.
        # Standard lexicographic:
        # (0,1) -> 0
        # (0,2) -> 1
        # ...
        # (0,n-1) -> n-2
        # (1,2) -> n-1
        # Index for (i, j) with i < j:
        # Sum_{k=0}^{i-1} (n - 1 - k) + (j - i - 1)
        # = i * (n - 1) - i * (i - 1) // 2 + j - i - 1
        
    def count_violations(colors, n):
        # Returns total number of C4s (counting each C4 once? or number of pairs of common neighbors?)
        # The objective in local search is usually sum of C(common, 2) over all pairs.
        # This is proportional to the number of C4s.
        common = [[0]*n for _ in range(n)]
        neighbors = [[] for _ in range(n)] # neighbors[v] = list of (neighbor, color) ? No, we need color-specific.
        
        # This function is for initial check or verification. 
        # For SA, we maintain state incrementally.
        pass

    def solve_sa(n, max_time=10):
        # Initialize
        m = n * (n - 1) // 2
        colors = [random.randint(0, 4) for _ in range(m)]
        
        # Build neighbor lists for each color
        # neighbors[c][v] = list of neighbors of v in color c
        neighbors = [[[] for _ in range(n)] for _ in range(5)]
        for i in range(n):
            for j in range(i + 1, n):
                idx = get_idx(i, j, n)
                c = colors[idx]
                neighbors[c][i].append(j)
                neighbors[c][j].append(i)
        
        # common[c][u][v] = number of common neighbors of u and v in color c
        # We only need u < v.
        # common is a 3D array: common[c][u][v]
        common = [[[0]*n for _ in range(n)] for _ in range(5)]
        
        bad_total = 0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    # Count intersection of neighbors[c][u] and neighbors[c][v]
                    # Since degrees are small, we can use set or sorted list.
                    # For speed, let's just iterate.
                    # Optimization: use a boolean array for neighbors of u?
                    # Or just count.
                    cnt = 0
                    nu = neighbors[c][u]
                    nv = neighbors[c][v]
                    # Small lists
                    for x in nu:
                        if x in nv: # O(deg)
                            cnt += 1
                    common[c][u][v] = cnt
                    common[c][v][u] = cnt
                    if cnt >= 2:
                        bad_total += cnt * (cnt - 1) // 2
                        
        if bad_total == 0:
            return colors, True

        T = 1.0
        cooling = 0.999
        best_colors = list(colors)
        best_bad = bad_total
        current_bad = bad_total
        
        time_limit = start_time + max_time
        while time.time() < time_limit:
            # Pick random edge
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
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
            
            # Update common counts for old_c (removing edge)
            # For each k in neighbors[old_c][i] (excluding j), common[old_c][k][j] decreases?
            # Wait. 
            # Removing (i, j) from old_c.
            # For any k:
            #   common[old_c][i][k]: j was a common neighbor if j in N(old_c, k).
            #   So if j in neighbors[old_c][k], common[old_c][i][k] decreases by 1.
            #   common[old_c][j][k]: i was a common neighbor if i in N(old_c, k).
            #   So if i in neighbors[old_c][k], common[old_c][j][k] decreases by 1.
            
            # We need to iterate over k.
            # To be fast, we can iterate over neighbors of i and j.
            
            # Let's precompute sets for fast membership?
            # Or just iterate k from 0 to n-1. n=26, so 26 iterations is fine.
            
            for k in range(n):
                if k == i or k == j:
                    continue
                
                # old_c removal
                if j in neighbors[old_c][k]:
                    old_cnt = common[old_c][i][k]
                    new_cnt = old_cnt - 1
                    delta -= (old_cnt * (old_cnt - 1) // 2) - (new_cnt * (new_cnt - 1) // 2)
                    common[old_c][i][k] = new_cnt
                    common[old_c][k][i] = new_cnt
                
                if i in neighbors[old_c][k]:
                    old_cnt = common[old_c][j][k]
                    new_cnt = old_cnt - 1
                    delta -= (old_cnt * (old_cnt - 1) // 2) - (new_cnt * (new_cnt - 1) // 2)
                    common[old_c][j][k] = new_cnt
                    common[old_c][k][j] = new_cnt
                
                # new_c addition
                if j in neighbors[new_c][k]:
                    old_cnt = common[new_c][i][k]
                    new_cnt = old_cnt + 1
                    delta += (new_cnt * (new_cnt - 1) // 2) - (old_cnt * (old_cnt - 1) // 2)
                    common[new_c][i][k] = new_cnt
                    common[new_c][k][i] = new_cnt
                
                if i in neighbors[new_c][k]:
                    old_cnt = common[new_c][j][k]
                    new_cnt = old_cnt + 1
                    delta += (new_cnt * (new_cnt - 1) // 2) - (old_cnt * (old_cnt - 1) // 2)
                    common[new_c][j][k] = new_cnt
                    common[new_c][k][j] = new_cnt
            
            # Note: common[old_c][i][j] and common[new_c][i][j] do not change 
            # because i is not in N(i) and j is not in N(j).
            
            new_bad = current_bad + delta
            
            if new_bad < current_bad or random.random() < math.exp(-delta / T):
                # Accept
                current_bad = new_bad
                colors[idx] = new_c
                
                # Update neighbor lists
                # Remove j from neighbors[old_c][i] and i from neighbors[old_c][j]
                # List removal is O(deg). Deg is small.
                try:
                    neighbors[old_c][i].remove(j)
                except ValueError:
                    pass
                try:
                    neighbors[old_c][j].remove(i)
                except ValueError:
                    pass
                
                neighbors[new_c][i].append(j)
                neighbors[new_c][j].append(i)
                
                if current_bad < best_bad:
                    best_bad = current_bad
                    best_colors = list(colors)
                    if best_bad == 0:
                        return best_colors, True
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
            
            T *= cooling
        
        return best_colors, (best_bad == 0)

    best_n = 11
    best_colors = [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]
    
    # Try to find valid colorings for increasing n
    # We know 26 is possible. Let's try 26, 25, ..., 12.
    # If we find one for n, we can try n+1? 
    # Or just find the max n that yields a valid coloring in the time budget.
    
    # Strategy: 
    # 1. Try n=26 with several restarts.
    # 2. If fail, try n=25.
    # ...
    # 3. Return best found.
    
    for n in range(26, 11, -1):
        if time.time() > deadline - 10:
            break
        
        found = False
        # Number of restarts depends on time remaining
        restarts = 5
        for r in range(restarts):
            if time.time() > deadline - 5:
                break
            random.seed(n * 1000 + r)
            colors, is_valid = solve_sa(n, max_time=20)
            if is_valid:
                best_n = n
                best_colors = colors
                found = True
                break
        
        if found:
            # If we found n, we can try n+1? 
            # The problem asks to maximize n.
            # If we found 26, we are done (known bound).
            # If we found 25, we should try 26 again or 27?
            # The contract says 27 or 28 would be new.
            # So if we find 26, we stop.
            if n == 26:
                break
            # If we found lower, we continue to lower n? 
            # No, we iterate downwards. So if we find 25, we return 25?
            # But maybe 26 is possible with more time?
            # Let's just return the highest n found.
            # Since we iterate 26->12, if we find 25, we break and return 25.
            # But we might have skipped 26 because it failed in 5 restarts.
            # This is a heuristic.
            break

    return {'n': best_n, 'colors': best_colors}
