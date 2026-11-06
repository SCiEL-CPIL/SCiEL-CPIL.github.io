import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin for output serialization etc.
    
    # Fallback: The seed solution for n=32
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
    
    def try_find_coloring(n, time_limit):
        """
        Attempts to find a 4-coloring of K_n with no monochromatic triangles
        using Min-Conflicts Local Search.
        Returns list of colors if found, None otherwise.
        """
        if n < 2:
            return [0] * (n * (n - 1) // 2)
            
        # Map (i, j) to index in the flat list
        # Index = i * n - i * (i + 1) // 2 + (j - i - 1)
        # Or simply use a 2D matrix for speed and convert at the end.
        # Matrix is n x n, upper triangle used.
        
        def idx(i, j):
            if i > j: i, j = j, i
            return i * n - i * (i + 1) // 2 + (j - i - 1)
        
        # Initialize random coloring
        # Using 4 colors: 0, 1, 2, 3
        colors = [random.randint(0, 3) for _ in range(n * (n - 1) // 2)]
        
        # Precompute neighbors? No, we can compute on the fly.
        # To speed up, we maintain counts of "bad" triangles.
        # Total triangles = sum over all (u, v, w) of 1 if c_uv == c_vw == c_wu.
        
        def count_triangles():
            total = 0
            for i in range(n):
                for j in range(i + 1, n):
                    c_ij = colors[idx(i, j)]
                    # For each k, check if (i, k) and (j, k) are also c_ij
                    # This is O(n^3) which is too slow for n=50 if done naively every step.
                    # We need incremental updates.
            return total

        # Efficient Local Search
        # We maintain the total number of monochromatic triangles.
        # And for each edge (i, j), we maintain the number of triangles involving it that are monochromatic.
        # Let bad[i][j] be the number of k such that (i, j, k) is a mono triangle.
        # When we change color of (i, j) from old_c to new_c:
        # 1. For each k, check if (i, k) and (j, k) are old_c. If so, we remove 1 from bad[i][j], bad[i][k], bad[j][k].
        #    Wait, bad[i][k] counts triangles involving edge (i, k). The triangle is (i, k, j).
        #    If c_ik == c_kj == old_c, then (i, k, j) was a bad triangle.
        #    Changing c_ij to new_c breaks it.
        # 2. For each k, check if (i, k) and (j, k) are new_c. If so, we add 1 to bad[i][j], bad[i][k], bad[j][k].
        
        # This update is O(n) per move.
        # We pick the move with the largest decrease in total triangles.
        
        bad = [[0] * n for _ in range(n)]
        total_bad = 0
        
        # Initialize bad counts
        for i in range(n):
            for j in range(i + 1, n):
                c_ij = colors[idx(i, j)]
                for k in range(n):
                    if k == i or k == j:
                        continue
                    c_ik = colors[idx(i, k)]
                    c_jk = colors[idx(j, k)]
                    if c_ik == c_jk == c_ij:
                        # Triangle (i, j, k) is mono
                        bad[i][j] += 1
                        bad[i][k] += 1
                        bad[j][k] += 1
                        total_bad += 1
        # Note: Each triangle is counted 3 times in total_bad if we sum bad[i][j].
        # Actually, total_bad is sum of bad[i][j] / 3?
        # Let's just track the actual number of triangles.
        # My update logic:
        # If I change (i, j), I affect triangles (i, j, k).
        # There are n-2 such triangles.
        # I can just recompute the contribution of (i, j) to the total.
        
        # Let's refine the data structure.
        # We want to minimize T = sum_{i<j<k} [c_ij == c_jk == c_ki]
        # When changing c_ij:
        # Delta T = (New count of mono triangles involving ij) - (Old count of mono triangles involving ij)
        # Old count: count of k such that c_ik == c_kj == c_ij_old
        # New count: count of k such that c_ik == c_kj == c_ij_new
        
        # We can precompute for each pair (i, j) and color c, the count of k such that c_ik == c_kj == c.
        # Let cnt[i][j][c] be this count.
        # Then old_count = cnt[i][j][old_c]
        # new_count = cnt[i][j][new_c]
        # Delta = new_count - old_count
        
        # When we change c_ij from old_c to new_c, we must update cnt for all edges involving i or j.
        # This is complex.
        
        # Simpler: Just calculate the delta on the fly for the best move selection.
        # For each edge (i, j) and each new color c_new:
        #   delta = count(k | c_ik == c_kj == c_new) - count(k | c_ik == c_kj == c_old)
        # We pick the move with min delta.
        # This is O(n^2 * 4 * n) = O(n^3) per iteration.
        # n=50, n^3 = 125,000. 1000 iterations = 125M ops. Python might be slow (10-20s).
        # We need to be careful.
        
        # Let's optimize the inner loop.
        # For a fixed (i, j), we can compute the counts for all 4 colors in one pass over k.
        # counts = [0, 0, 0, 0]
        # for k in range(n):
        #    if k != i and k != j:
        #       c1 = c_ik
        #       c2 = c_jk
        #       if c1 == c2:
        #           counts[c1] += 1
        # Then delta for new_c is counts[new_c] - counts[old_c].
        
        # This is O(n) per edge. Total O(n^3) per full pass.
        
        best_n_found = 0
        best_colors = None
        
        for attempt in range(20): # Max 20 restarts
            if time.time() > time_limit:
                break
            # Re-initialize
            colors = [random.randint(0, 3) for _ in range(n * (n - 1) // 2)]
            
            # Calculate initial total triangles
            total_tri = 0
            for i in range(n):
                for j in range(i + 1, n):
                    c_ij = colors[idx(i, j)]
                    cnt = [0, 0, 0, 0]
                    for k in range(n):
                        if k == i or k == j: continue
                        c_ik = colors[idx(i, k)]
                        c_jk = colors[idx(j, k)]
                        if c_ik == c_jk:
                            cnt[c_ik] += 1
                    total_tri += cnt[c_ij]
            # Note: This total_tri is the number of mono triangles.
            # Each triangle is counted once for each of its 3 edges?
            # No. For a fixed triangle (a,b,c), it is counted in:
            # - Edge (a,b) when k=c
            # - Edge (a,c) when k=b
            # - Edge (b,c) when k=a
            # So total_tri is 3 * (number of mono triangles).
            # We just need to minimize total_tri.
            
            if total_tri == 0:
                return colors.copy()
            
            for step in range(1000):
                if time.time() > time_limit:
                    break
                
                best_delta = 0
                best_move = None
                best_new_color = None
                best_i, best_j = -1, -1
                
                # Find the best move
                for i in range(n):
                    for j in range(i + 1, n):
                        old_c = colors[idx(i, j)]
                        cnt = [0, 0, 0, 0]
                        for k in range(n):
                            if k == i or k == j: continue
                            c_ik = colors[idx(i, k)]
                            c_jk = colors[idx(j, k)]
                            if c_ik == c_jk:
                                cnt[c_ik] += 1
                        
                        # Current contribution of this edge to total_tri is cnt[old_c]
                        # If we change to new_c, contribution becomes cnt[new_c]
                        # Delta = cnt[new_c] - cnt[old_c]
                        for new_c in range(4):
                            if new_c == old_c: continue
                            delta = cnt[new_c] - cnt[old_c]
                            if delta < best_delta:
                                best_delta = delta
                                best_move = (i, j)
                                best_new_color = new_c
                
                if best_move is None:
                    # Stuck in local minimum? Or no improving move.
                    # Try a random move to escape
                    i, j = random.sample(range(n), 2)
                    if i > j: i, j = j, i
                    new_c = random.randint(0, 3)
                    old_c = colors[idx(i, j)]
                    if new_c == old_c:
                        new_c = (new_c + 1) % 4
                    # Apply move
                    # We don't strictly need to update total_tri if we recompute, 
                    # but for efficiency we should. 
                    # However, recomputing total_tri is O(n^3).
                    # Let's just recompute total_tri every 10 steps or if we want to check for 0.
                    # Actually, we can just check if total_tri becomes 0 by counting.
                    
                    # For simplicity and correctness, let's just apply the move and recompute total_tri
                    # if we suspect we are close, or every 50 steps.
                    # But we need to know if we reached 0.
                    
                    # Let's update total_tri incrementally.
                    # Delta for the edge (i, j) itself is cnt[new_c] - cnt[old_c].
                    # But this changes the counts for other edges!
                    # This is the hard part.
                    
                    # Given the complexity, let's just recompute total_tri every step.
                    # O(n^3) = 125,000 ops. 1000 steps = 125M ops.
                    # Python can do ~10-50M simple ops/sec.
                    # So 125M ops might take 2-10 seconds.
                    # 20 restarts * 10s = 200s. Too slow.
                    
                    # We must optimize.
                    # Let's limit steps or n.
                    pass
                
                # Apply best move
                if best_move:
                    i, j = best_move
                    old_c = colors[idx(i, j)]
                    new_c = best_new_color
                    colors[idx(i, j)] = new_c
                    # We need to update total_tri.
                    # Since we don't maintain incremental state easily, we recompute.
                    # But recomputing is expensive.
                    
                    # Alternative: Just check for 0 triangles periodically.
                    # If we find 0, we return.
                    # If we don't, we continue.
                    
                    # Let's recompute total_tri every 20 steps to check for 0.
                    if step % 20 == 0:
                         total_tri = 0
                         for i in range(n):
                             for j in range(i + 1, n):
                                 c_ij = colors[idx(i, j)]
                                 cnt = [0, 0, 0, 0]
                                 for k in range(n):
                                     if k == i or k == j: continue
                                     c_ik = colors[idx(i, k)]
                                     c_jk = colors[idx(j, k)]
                                     if c_ik == c_jk:
                                         cnt[c_ik] += 1
                                 total_tri += cnt[c_ij]
                         if total_tri == 0:
                             return colors.copy()
        
        return None

    # Try to find coloring for n=50 down to 33
    for n in range(50, 32, -1):
        time_left = deadline - time.time()
        if time_left < 2.0:
            break
        # Allocate time proportional to difficulty? 
        # Just give it a chunk.
        t_limit = time.time() + min(5.0, time_left * 0.8)
        res = try_find_coloring(n, t_limit)
        if res:
            return {'n': n, 'colors': res}
            
    # Fallback to seed
    return get_seed_solution()
