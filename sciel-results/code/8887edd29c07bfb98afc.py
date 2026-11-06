import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin for output serialization etc.

    def get_seed_solution():
        def mul(a, b):
            r = 0
            for i in range(4):
                if b >> i & 1:
                    r ^= a << i
            for i in range(7, 3, -1):
                if r >> i & 1:
                    r ^= 0b10011 << (i - 4)
            return r
        log, x = {}, 1
        for e in range(15):
            log[x] = e
            x = mul(x, 2)
        colors = []
        for i in range(32):
            for j in range(i + 1, 32):
                colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
        return {'n': 32, 'colors': colors}

    # Helper to flatten a 2D matrix (upper triangle) to a list
    def mat_to_list(mat, n):
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                colors.append(mat[i][j])
        return colors

    # Helper to verify a coloring (flat list) is valid
    def is_valid(n, colors):
        # Build matrix for easy access
        mat = [[0] * n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                mat[i][j] = mat[j][i] = colors[idx]
                idx += 1
        for i in range(n):
            for j in range(i + 1, n):
                c = mat[i][j]
                for k in range(j + 1, n):
                    if mat[i][k] == c and mat[j][k] == c:
                        return False
        return True

    def try_find_coloring(n, time_limit):
        """
        Attempts to find a 4-coloring of K_n with no monochromatic triangles
        using Min-Conflicts Local Search with incremental updates.
        Returns list of colors if found, None otherwise.
        """
        if n < 2:
            return [0] * (n * (n - 1) // 2)
            
        def idx(i, j):
            if i > j: i, j = j, i
            return i * n - i * (i + 1) // 2 + (j - i - 1)
        
        best_n_found = 0
        best_colors = None
        
        # We will do multiple restarts
        # Estimate time per restart
        # O(n^3) per iteration. n=50 -> 125k ops. Python ~10M ops/s -> 12ms/iter.
        # 1000 iters -> 12s. 
        # We have time_limit.
        
        for attempt in range(100): # Max restarts, will break on time
            if time.time() > time_limit:
                break
            
            # Initialize random coloring
            colors = [random.randint(0, 3) for _ in range(n * (n - 1) // 2)]
            
            # Precompute edge indices for faster access? 
            # Let's just use the idx function.
            
            # Maintain counts of "bad" triangles for each edge.
            # bad[i][j] = number of k such that (i, j, k) is a monochromatic triangle.
            # This allows O(n) update for changing edge (i, j).
            
            bad = [[0] * n for _ in range(n)]
            total_bad = 0
            
            # Initialize bad counts
            # O(n^3)
            for i in range(n):
                for j in range(i + 1, n):
                    c_ij = colors[idx(i, j)]
                    for k in range(n):
                        if k == i or k == j:
                            continue
                        c_ik = colors[idx(i, k)]
                        c_jk = colors[idx(j, k)]
                        if c_ik == c_jk == c_ij:
                            bad[i][j] += 1
                            bad[i][k] += 1
                            bad[j][k] += 1
                            total_bad += 1
                            
            # Note: total_bad is 3 * (number of mono triangles).
            # We want to minimize total_bad.
            
            if total_bad == 0:
                return colors.copy()
            
            # Local Search
            # We will pick the edge with the largest decrease in total_bad.
            # This is a greedy step.
            
            # To avoid O(n^3) per step, we can maintain a priority queue or just scan.
            # Scanning O(n^2) edges, and for each, calculating delta is O(1) if we had precomputed counts.
            # But we don't have precomputed counts for "what if I change to color c".
            
            # Let's use a simpler heuristic: 
            # Pick a random edge with high bad count, and try to change its color to the one that minimizes new bad count.
            # Or, scan all edges, find the one with max bad[i][j], and try to fix it.
            
            # Better: 
            # For each edge (i, j), we can calculate the "cost" of changing it to each color c.
            # cost(c) = number of k such that c_ik == c_jk == c.
            # We want to minimize sum of costs.
            # Currently, the cost for edge (i, j) with color c_old is bad[i][j] (roughly, but bad[i][j] counts triangles where ALL 3 edges match).
            # Wait, bad[i][j] is the number of triangles (i, j, k) that are monochromatic.
            # If we change (i, j) to c_new, the new bad count for (i, j) will be the number of k such that c_ik == c_jk == c_new.
            # Let's call this new_bad_ij.
            # The change in total_bad is:
            # Delta = (new_bad_ij - old_bad_ij) + (changes in bad[i][k] and bad[j][k] for all k)
            # This is complex to update incrementally without knowing the new state.
            
            # Let's stick to the "recompute delta for candidate moves" approach but optimize it.
            # For a fixed edge (i, j), we can compute the counts for all 4 colors in O(n).
            # counts[c] = number of k such that c_ik == c_jk == c.
            # Then delta for changing to c_new is counts[c_new] - counts[old_c].
            # This is O(n) per edge.
            # Total O(n^3) per full scan.
            
            # We can do a "steepest descent" step:
            # Find the edge (i, j) and color c_new that minimizes total_bad the most.
            
            for step in range(2000):
                if time.time() > time_limit:
                    break
                
                best_delta = 0
                best_move = None
                best_new_color = None
                best_i, best_j = -1, -1
                
                # Sample a subset of edges if n is large to save time?
                # For n=50, n^2/2 = 1250 edges.
                # O(n^3) = 125,000 ops.
                # This is fast enough in C, but in Python ~10-20ms.
                # 2000 steps * 20ms = 40s.
                # Too slow for 5s limit per restart.
                
                # We need to be smarter.
                # Only consider edges with high bad count?
                # Or just pick a random edge and try to improve it?
                
                # Let's try: Pick a random edge. Calculate best color for it.
                # If it improves, accept.
                
                # This is "Randomized Local Search" or "Min-Conflicts" style.
                
                i, j = random.sample(range(n), 2)
                if i > j: i, j = j, i
                
                old_c = colors[idx(i, j)]
                
                # Calculate counts for this edge
                cnt = [0, 0, 0, 0]
                for k in range(n):
                    if k == i or k == j: continue
                    c_ik = colors[idx(i, k)]
                    c_jk = colors[idx(j, k)]
                    if c_ik == c_jk:
                        cnt[c_ik] += 1
                
                # Current bad contribution of this edge is cnt[old_c]
                # Note: This is NOT exactly bad[i][j] because bad[i][j] counts triangles where ALL 3 edges match.
                # If c_ik == c_jk == c, then triangle (i, j, k) is mono ONLY IF c_ij == c.
                # So cnt[c] is the number of "potential" mono triangles for edge (i, j) if it were color c.
                # The actual bad[i][j] is cnt[old_c].
                
                # If we change to new_c, the new bad contribution is cnt[new_c].
                # Delta for this edge is cnt[new_c] - cnt[old_c].
                
                # However, changing (i, j) also affects bad[i][k] and bad[j][k].
                # Specifically, for each k:
                # If c_ik == c_jk == old_c, then triangle (i, j, k) was mono.
                # It contributed 1 to bad[i][j], bad[i][k], bad[j][k].
                # If we change (i, j) to new_c:
                # - If new_c == old_c, no change.
                # - If new_c != old_c:
                #   - Triangle (i, j, k) is no longer mono (unless c_ik == c_jk == new_c, which is impossible if old_c != new_c and c_ik==c_jk).
                #   - So we remove 1 from bad[i][j], bad[i][k], bad[j][k].
                #   - BUT, if c_ik == c_jk == new_c, then it BECOMES a mono triangle.
                #   - So we add 1 to bad[i][j], bad[i][k], bad[j][k].
                
                # This means the total change in "sum of bad" is:
                # For each k:
                #   Let old_mono = 1 if (c_ik == c_jk == old_c) else 0
                #   Let new_mono = 1 if (c_ik == c_jk == new_c) else 0
                #   Change in bad[i][j] += (new_mono - old_mono)
                #   Change in bad[i][k] += (new_mono - old_mono)  <-- ONLY IF the triangle was mono before or is mono now?
                #   Wait. bad[i][k] counts triangles (i, k, x) that are mono.
                #   The triangle (i, j, k) is one such triangle for edge (i, k).
                #   So yes, if the triangle status changes, bad[i][k] changes by (new_mono - old_mono).
                
                # So the total change in the SUM of all bad[i][j] is:
                # Sum_{k} 3 * (new_mono(k) - old_mono(k))
                # = 3 * (cnt[new_c] - cnt[old_c])
                
                # So minimizing the sum of bads is equivalent to minimizing cnt[new_c] - cnt[old_c].
                # This is great! We don't need to update the bad matrix for other edges to know the GLOBAL impact?
                # Wait, this logic holds if we are minimizing the SUM of bads.
                # But we want to reach 0 triangles.
                # Min-conflicts usually minimizes the number of conflicts.
                # Here, "conflict" is a mono triangle.
                # The total number of mono triangles is T = Sum_{i<j<k} [mono(i,j,k)].
                # Also T = (1/3) * Sum_{i<j} bad[i][j].
                # So minimizing Sum bad is equivalent to minimizing T.
                
                # So, we can just pick the move that minimizes cnt[new_c] - cnt[old_c].
                # And we don't even need to update the bad matrix for other edges if we just recompute T periodically?
                # No, we need to update the bad matrix to know the current state for future moves?
                # Actually, if we just use the "cnt" calculation on the fly, we don't need the bad matrix at all!
                # We just need to check if T=0.
                # But checking T=0 is O(n^3).
                
                # So the algorithm:
                # 1. Pick random edge (i, j).
                # 2. Compute cnt for all 4 colors. O(n).
                # 3. Find best new_c that minimizes cnt[new_c].
                # 4. If cnt[best_c] < cnt[old_c], accept.
                # 5. Periodically check if T=0. O(n^3).
                
                best_new_c = old_c
                min_cnt = cnt[old_c]
                for c in range(4):
                    if cnt[c] < min_cnt:
                        min_cnt = cnt[c]
                        best_new_c = c
                
                if best_new_c != old_c:
                    colors[idx(i, j)] = best_new_c
                    
                    # Check for 0 triangles every 50 steps
                    if step % 50 == 0:
                        T = 0
                        for i in range(n):
                            for j in range(i + 1, n):
                                c_ij = colors[idx(i, j)]
                                for k in range(j + 1, n):
                                    if colors[idx(i, k)] == c_ij and colors[idx(j, k)] == c_ij:
                                        T += 1
                        if T == 0:
                            return colors.copy()
                        # If T is very small, we might be close
                        if T < 5:
                            # Try more steps
                            pass
                            
        return None

    # Try to find coloring for n=50 down to 33
    for n in range(50, 32, -1):
        time_left = deadline - time.time()
        if time_left < 2.0:
            break
        # Allocate time
        t_limit = time.time() + min(10.0, time_left * 0.8)
        res = try_find_coloring(n, t_limit)
        if res:
            # Verify
            if is_valid(n, res):
                return {'n': n, 'colors': res}
            else:
                print(f"Found invalid coloring for n={n}, ignoring.")
            
    # Fallback to seed
    return get_seed_solution()

if __name__ == '__main__':
    res = construct()
    print(f"Result: n={res['n']}")
