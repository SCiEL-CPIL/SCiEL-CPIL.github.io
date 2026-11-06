import time
import random
import math

def construct():
    deadline = time.time() + 115  # Safety margin
    
    # Edge index mapping for K_n
    # Index for (i, j) with i < j
    # idx = i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)
    # This formula is O(1) but involves division. 
    # For n=50, n(n-1)/2 = 1225 edges.
    
    def get_idx(i, j, n):
        if i > j:
            i, j = j, i
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)

    def count_mono_triangles(colors, n):
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                c_ij = colors[get_idx(i, j, n)]
                for k in range(j + 1, n):
                    c_ik = colors[get_idx(i, k, n)]
                    c_jk = colors[get_idx(j, k, n)]
                    if c_ij == c_ik == c_jk:
                        count += 1
        return count

    def try_find_coloring(n, time_limit, rng):
        """
        Attempts to find a valid 4-coloring of K_n using Simulated Annealing.
        Returns the colors list if found, else None.
        """
        num_edges = n * (n - 1) // 2
        colors = [rng.randint(0, 3) for _ in range(num_edges)]
        
        # Precompute neighbors for faster triangle updates?
        # For each edge (i,j), the triangles are (i,j,k) for k != i,j.
        # We can just iterate k.
        
        current_triangles = count_mono_triangles(colors, n)
        if current_triangles == 0:
            return colors

        start_time = time.time()
        T = 1.0
        T_min = 0.01
        cooling_rate = 0.999
        
        best_colors = colors[:]
        best_triangles = current_triangles
        
        while time.time() < start_time + time_limit and T > T_min:
            # Pick a random edge
            e_idx = rng.randint(0, num_edges - 1)
            old_color = colors[e_idx]
            new_color = rng.randint(0, 3)
            if new_color == old_color:
                continue
            
            # Identify i, j from e_idx
            # Reverse of get_idx
            # This is O(sqrt(n)) or we can precompute edge_to_pair
            # For n=50, sqrt(1225) ~ 35. Acceptable.
            # But let's precompute edge_to_pair for speed.
            
            # We need to map e_idx to (i,j)
            # i * (n-1) - i(i-1)/2 <= e_idx < (i+1)(n-1) - (i+1)i/2
            # Let's just loop i to find the pair, or precompute.
            # Precomputation is better.
            
            # We will create edge_to_pair once per n
            # But this function is called for different n.
            # Let's pass edge_to_pair or compute it inside.
            
            # Actually, computing i,j from idx:
            # i = floor((sqrt(8 * e_idx + 1) - 1) / 2) ... wait, that's for 0..n-1 simple indexing.
            # The formula used is specific.
            # Let's just precompute edge_to_pair for the given n.
            
            # To avoid overhead, we compute i,j here.
            # i is such that i*(n-1) - i(i-1)/2 <= e_idx
            i = 0
            # Binary search or linear scan. n is small.
            while i < n:
                start = i * (n - 1) - i * (i - 1) // 2
                end = start + (n - 1 - i)
                if e_idx < end:
                    j = i + 1 + (e_idx - start)
                    break
                i += 1
            
            # Calculate delta in triangle count
            delta = 0
            # Triangles involving edge (i,j) are (i,j,k) for k in 0..n-1, k!=i, k!=j
            for k in range(n):
                if k == i or k == j:
                    continue
                
                # Get colors of (i,k) and (j,k)
                # Ensure correct index order
                idx_ik = get_idx(i, k, n)
                idx_jk = get_idx(j, k, n)
                
                c_ik = colors[idx_ik]
                c_jk = colors[idx_jk]
                
                # Old state
                if old_color == c_ik and old_color == c_jk:
                    delta -= 1  # Was mono, now not (unless new_color is also that, but we check new)
                
                # New state
                if new_color == c_ik and new_color == c_jk:
                    delta += 1  # Was not mono (or was, handled above), now is
            
            # Apply move
            colors[e_idx] = new_color
            current_triangles += delta
            
            if current_triangles < best_triangles:
                best_triangles = current_triangles
                best_colors = colors[:]
                if best_triangles == 0:
                    return best_colors
            
            # Acceptance
            if delta > 0:
                # Calculate probability
                # Avoid overflow in exp
                if delta / T < 700: 
                    prob = math.exp(-delta / T)
                else:
                    prob = 0.0
                
                if rng.random() > prob:
                    # Reject
                    colors[e_idx] = old_color
                    current_triangles -= delta
            
            T *= cooling_rate
            
        return best_colors if best_triangles == 0 else None

    # Precompute edge_to_pair is not strictly necessary if we do the loop, 
    # but for n=50, the loop to find i is O(n) which is 50 steps. 
    # 1225 edges * 50 steps = 60k ops per step. 
    # SA steps can be millions. 60k * 1M = 60B ops. Too slow.
    # We MUST precompute edge_to_pair.
    
    def try_find_coloring_optimized(n, time_limit, rng):
        num_edges = n * (n - 1) // 2
        edge_to_pair = [None] * num_edges
        pair_to_idx = {}
        
        for i in range(n):
            for j in range(i + 1, n):
                idx = i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)
                edge_to_pair[idx] = (i, j)
                pair_to_idx[(i, j)] = idx
                pair_to_idx[(j, i)] = idx # Symmetric, though we always use i<j in get_idx

        colors = [rng.randint(0, 3) for _ in range(num_edges)]
        
        # Initial count
        current_triangles = 0
        for i in range(n):
            for j in range(i + 1, n):
                c_ij = colors[pair_to_idx[(i, j)]]
                for k in range(j + 1, n):
                    c_ik = colors[pair_to_idx[(i, k)]]
                    c_jk = colors[pair_to_idx[(j, k)]]
                    if c_ij == c_ik == c_jk:
                        current_triangles += 1
        
        if current_triangles == 0:
            return colors

        start_time = time.time()
        T = 1.0
        T_min = 0.01
        cooling_rate = 0.9995 # Slower cooling for more search
        
        best_colors = colors[:]
        best_triangles = current_triangles
        
        while time.time() < start_time + time_limit and T > T_min:
            e_idx = rng.randint(0, num_edges - 1)
            i, j = edge_to_pair[e_idx]
            
            old_color = colors[e_idx]
            new_color = rng.randint(0, 3)
            if new_color == old_color:
                continue
            
            delta = 0
            # Iterate over k
            # We can optimize by precomputing neighbors, but n is small.
            for k in range(n):
                if k == i or k == j:
                    continue
                
                c_ik = colors[pair_to_idx[(i, k)]]
                c_jk = colors[pair_to_idx[(j, k)]]
                
                if old_color == c_ik and old_color == c_jk:
                    delta -= 1
                if new_color == c_ik and new_color == c_jk:
                    delta += 1
            
            colors[e_idx] = new_color
            current_triangles += delta
            
            if current_triangles < best_triangles:
                best_triangles = current_triangles
                best_colors = colors[:]
                if best_triangles == 0:
                    return best_colors
            
            if delta > 0:
                if delta / T < 700:
                    prob = math.exp(-delta / T)
                else:
                    prob = 0.0
                
                if rng.random() > prob:
                    colors[e_idx] = old_color
                    current_triangles -= delta
            
            T *= cooling_rate
        
        return best_colors if best_triangles == 0 else None

    # Try to find coloring for n=50, 49, ...
    # We allocate time based on n. Larger n is harder, but we want to hit 50.
    # If 50 fails, we try 49.
    
    rng = random.Random(42)
    
    # Strategy: 
    # 1. Try n=50 for 80s.
    # 2. If fail, try n=49 for 20s.
    # 3. If fail, try n=48 for 5s.
    # 4. If fail, return best known (or base 32).
    
    # Actually, if we find 50, we can try to extend to 51?
    # The task says "Maximize n". 51 is the theoretical max if R(3,3,3,3)=52.
    # But 51 is very hard. Let's stick to finding a valid 50 first.
    
    # We will use the base 32 as a fallback if SA fails completely (unlikely).
    # But SA is very strong.
    
    # Let's do a few restarts for n=50.
    # Total time ~ 110s.
    
    best_n = 32
    best_colors = None
    
    # Fallback: The seed's 32-vertex coloring
    def _build_k32_colors():
        def _mul(a, b):
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
            x = _mul(x, 2)
        colors = []
        for i in range(32):
            for j in range(i + 1, 32):
                colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
        return colors
    
    fallback_colors = _build_k32_colors()
    
    # Try n=50
    # We can run multiple independent SA runs in parallel? No, Python GIL.
    # We run sequential runs.
    
    # Time allocation:
    # 3 runs of 30s each for n=50.
    # If any succeeds, return.
    
    for attempt in range(3):
        if time.time() > deadline - 5:
            break
        t_start = time.time()
        time_left = min(30, deadline - t_start - 2)
        if time_left < 1:
            break
            
        colors_50 = try_find_coloring_optimized(50, time_left, rng)
        if colors_50 is not None:
            return {'n': 50, 'colors': colors_50}
        
        # If we have time, try n=49?
        # Actually, if 50 fails, 49 is likely to succeed quickly.
        # But we want to maximize n.
        # Let's check if we have time for 49.
        if time.time() < deadline - 10:
            t_start_49 = time.time()
            time_left_49 = min(10, deadline - t_start_49 - 2)
            if time_left_49 > 1:
                colors_49 = try_find_coloring_optimized(49, time_left_49, rng)
                if colors_49 is not None:
                    return {'n': 49, 'colors': colors_49}
    
    # If all failed, return the fallback 32
    return {'n': 32, 'colors': fallback_colors}
