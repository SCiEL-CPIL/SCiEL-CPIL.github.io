import time
import random
import math

def construct():
    t0 = time.time()
    DEADLINE = t0 + 115.0  # Safety margin
    
    # Target n to try to achieve
    # The record is 50. We will try to find a valid coloring for n=50.
    # If we fail, we might try 49, 48, etc.
    target_n = 50
    
    def edge_idx(i, j, n):
        # Lexicographic order: (0,1), (0,2), ..., (0,n-1), (1,2), ...
        # Index of (i, j) where i < j
        # Number of edges starting with 0: n-1
        # Number of edges starting with 1: n-2
        # ...
        # Number of edges starting with i: n-1-i
        # Edges before i: sum_{k=0}^{i-1} (n-1-k) = i*(n-1) - i*(i-1)/2
        # Offset within i: j - i - 1
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)

    def count_mono_triangles(colors, n):
        """
        Count the number of monochromatic triangles in the graph defined by colors.
        colors is a list of length n*(n-1)/2.
        """
        count = 0
        # To make it faster, we can precompute adjacency lists by color
        # adj[c][u] = list of v such that edge(u,v) is color c
        adj = [[[] for _ in range(n)] for _ in range(4)]
        
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[edge_idx(i, j, n)]
                adj[c][i].append(j)
                adj[c][j].append(i)
        
        # For each color c, count triangles.
        # A triangle (i, j, k) is monochromatic in color c if 
        # adj[c][i] contains j and k, etc.
        # Efficient way: For each vertex i, and each pair of neighbors j, k in adj[c][i],
        # check if j and k are connected by color c.
        # This is O(sum over i of (deg_c(i))^2).
        
        # Simpler for small n: Iterate all triangles.
        # Number of triangles is n*(n-1)*(n-2)/6. For n=50, ~19600.
        # This is very fast.
        
        for i in range(n):
            for j in range(i + 1, n):
                c_ij = colors[edge_idx(i, j, n)]
                for k in range(j + 1, n):
                    c_ik = colors[edge_idx(i, k, n)]
                    c_jk = colors[edge_idx(j, k, n)]
                    if c_ij == c_ik == c_jk:
                        count += 1
        return count

    def try_solve(n):
        """
        Attempt to find a 4-coloring of K_n with 0 monochromatic triangles
        using Simulated Annealing.
        Returns the list of colors if successful, None otherwise.
        """
        # Random initialization
        num_edges = n * (n - 1) // 2
        colors = [random.randint(0, 3) for _ in range(num_edges)]
        
        current_energy = count_mono_triangles(colors, n)
        
        # Simulated Annealing parameters
        T = 1.0
        cooling = 0.999
        min_T = 1e-5
        max_steps = 500000  # Max moves per restart
        
        # To speed up evaluation of moves, we can precompute neighbor lists
        # But for n=50, direct calculation is fine.
        # Let's optimize the move evaluation.
        
        # Precompute edge indices for all pairs to avoid recalculating
        # Actually, we can just use the formula.
        
        # We will iterate through edges randomly.
        # For a move on edge (u, v) from color c to c':
        # Delta E = (new triangles involving (u,v) with c') - (old triangles involving (u,v) with c)
        # Triangles involving (u,v) with color c:
        # Count of w such that (u,w) is c and (v,w) is c.
        
        # Let's maintain a 4D array or list of lists to make this O(1) or O(n) fast.
        # count_c[u][v] is not needed.
        # We need to count w where color(u,w)==c and color(v,w)==c.
        
        # We can precompute for each color c, and each vertex u, the set of neighbors v with color c.
        # Let's use bitmasks for speed since n <= 62 fits in a 64-bit int.
        
        # adj_mask[c][u] is a bitmask of vertices v such that edge(u,v) has color c.
        
        def update_masks(colors, n):
            masks = [[0] * n for _ in range(4)]
            for i in range(n):
                for j in range(i + 1, n):
                    c = colors[edge_idx(i, j, n)]
                    masks[c][i] |= (1 << j)
                    masks[c][j] |= (1 << i)
            return masks

        masks = update_masks(colors, n)
        
        def get_delta(u, v, old_c, new_c):
            # Triangles with old_c: count w where (u,w) is old_c and (v,w) is old_c
            # This is popcount(masks[old_c][u] & masks[old_c][v])
            # Note: u is in masks[old_c][v] and v is in masks[old_c][u], so the intersection includes u and v?
            # No, edge (u,v) is old_c. So u is in neighbor list of v for color old_c.
            # The intersection includes u and v.
            # But u and v do not form a triangle with themselves.
            # A triangle is (u, v, w) with w distinct from u, v.
            # So we need to subtract the bits for u and v if they are set.
            # masks[old_c][u] has bit v set.
            # masks[old_c][v] has bit u set.
            # Intersection has bits u and v set.
            # So count = popcount(intersection) - 2.
            
            inter_old = masks[old_c][u] & masks[old_c][v]
            # Clear bits u and v
            inter_old &= ~(1 << u)
            inter_old &= ~(1 << v)
            old_count = bin(inter_old).count('1')
            
            inter_new = masks[new_c][u] & masks[new_c][v]
            inter_new &= ~(1 << u)
            inter_new &= ~(1 << v)
            new_count = bin(inter_new).count('1')
            
            return new_count - old_count

        def apply_move(u, v, old_c, new_c):
            idx = edge_idx(u, v, n)
            colors[idx] = new_c
            # Update masks
            masks[old_c][u] &= ~(1 << v)
            masks[old_c][v] &= ~(1 << u)
            masks[new_c][u] |= (1 << v)
            masks[new_c][v] |= (1 << u)

        def revert_move(u, v, old_c, new_c):
            idx = edge_idx(u, v, n)
            colors[idx] = old_c
            masks[new_c][u] &= ~(1 << v)
            masks[new_c][v] &= ~(1 << u)
            masks[old_c][u] |= (1 << v)
            masks[old_c][v] |= (1 << u)

        best_energy = current_energy
        best_colors = colors[:]
        
        step = 0
        while T > min_T:
            if time.time() > DEADLINE - 5:
                break
            
            # Random edge
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            old_c = colors[edge_idx(i, j, n)]
            
            # Choose new color
            # Try all other colors or just one random?
            # Trying all 3 and picking best is a "steepest descent" variant.
            # Or pick one random.
            # Let's pick a random different color.
            new_c = random.randint(0, 3)
            if new_c == old_c:
                new_c = (new_c + 1) % 4
            
            delta = get_delta(i, j, old_c, new_c)
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                apply_move(i, j, old_c, new_c)
                current_energy += delta
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_colors = colors[:]
                    if best_energy == 0:
                        return best_colors
            # else: reject
            
            step += 1
            if step % 1000 == 0:
                T *= cooling
                
            # If stuck, restart or perturb
            if step > max_steps:
                break
                
        return best_colors if best_energy == 0 else None

    # Try to solve for n=50
    # We might need multiple restarts
    # Since SA can be slow, we run it for a few seconds.
    
    # To be safe, we can try smaller n if 50 fails, but 50 is the target.
    # The seed found 32. We want to beat that.
    
    # Run SA for n=50
    result = try_solve(50)
    
    if result is None:
        # Try n=49
        if time.time() < DEADLINE - 10:
            result = try_solve(49)
    
    if result is None:
        # Try n=48
        if time.time() < DEADLINE - 10:
            result = try_solve(48)
        
    if result is None:
        # Fallback to seed logic if SA fails completely (unlikely)
        # But we should return something valid.
        # Let's return a known valid small one, e.g., n=10 or whatever we can find quickly.
        # Actually, if SA fails, it means we didn't find 0 triangles.
        # We should return the best we found, but it must be valid (0 triangles).
        # If we don't have a valid one, we can't return it.
        # We must return a valid coloring.
        # Let's try a very simple greedy for small n to ensure validity.
        
        # Greedy for n=10
        n = 10
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                colors.append(random.randint(0, 3))
        # This might have triangles. We need to fix it or use a known valid one.
        # For n=10, a random one is likely valid? No, probability of no mono triangle is low.
        # Let's just run SA for n=10 which will be very fast.
        result = try_solve(10)
        
    if result is not None:
        n = 50 if len(result) == 50*49//2 else (49 if len(result) == 49*48//2 else 10)
        # Determine n from length
        # n(n-1)/2 = L => n = (1 + sqrt(1+8L))/2
        L = len(result)
        n = int((1 + math.sqrt(1 + 8 * L)) / 2)
        return {'n': n, 'colors': result}
    else:
        # Last resort: return n=1
        return {'n': 1, 'colors': []}

# To increase success probability, we can run multiple restarts in parallel or sequentially.
# The above try_solve does one SA run.
# Let's modify construct to try multiple restarts if the first one fails.

def construct():
    t0 = time.time()
    DEADLINE = t0 + 115.0
    
    def edge_idx(i, j, n):
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)

    def count_mono_triangles(colors, n):
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                c_ij = colors[edge_idx(i, j, n)]
                for k in range(j + 1, n):
                    c_ik = colors[edge_idx(i, k, n)]
                    c_jk = colors[edge_idx(j, k, n)]
                    if c_ij == c_ik == c_jk:
                        count += 1
        return count

    def try_solve(n, time_budget):
        num_edges = n * (n - 1) // 2
        colors = [random.randint(0, 3) for _ in range(num_edges)]
        
        def update_masks(colors, n):
            masks = [[0] * n for _ in range(4)]
            for i in range(n):
                for j in range(i + 1, n):
                    c = colors[edge_idx(i, j, n)]
                    masks[c][i] |= (1 << j)
                    masks[c][j] |= (1 << i)
            return masks

        masks = update_masks(colors, n)
        
        def get_delta(u, v, old_c, new_c):
            inter_old = masks[old_c][u] & masks[old_c][v]
            inter_old &= ~(1 << u)
            inter_old &= ~(1 << v)
            old_count = bin(inter_old).count('1')
            
            inter_new = masks[new_c][u] & masks[new_c][v]
            inter_new &= ~(1 << u)
            inter_new &= ~(1 << v)
            new_count = bin(inter_new).count('1')
            
            return new_count - old_count

        def apply_move(u, v, old_c, new_c):
            idx = edge_idx(u, v, n)
            colors[idx] = new_c
            masks[old_c][u] &= ~(1 << v)
            masks[old_c][v] &= ~(1 << u)
            masks[new_c][u] |= (1 << v)
            masks[new_c][v] |= (1 << u)

        def revert_move(u, v, old_c, new_c):
            idx = edge_idx(u, v, n)
            colors[idx] = old_c
            masks[new_c][u] &= ~(1 << v)
            masks[new_c][v] &= ~(1 << u)
            masks[old_c][u] |= (1 << v)
            masks[old_c][v] |= (1 << u)

        current_energy = count_mono_triangles(colors, n)
        best_energy = current_energy
        best_colors = colors[:]
        
        T = 1.0
        cooling = 0.999
        min_T = 1e-5
        
        while time.time() < t0 + time_budget:
            if T < min_T:
                break
                
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            old_c = colors[edge_idx(i, j, n)]
            new_c = random.randint(0, 3)
            if new_c == old_c:
                new_c = (new_c + 1) % 4
            
            delta = get_delta(i, j, old_c, new_c)
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                apply_move(i, j, old_c, new_c)
                current_energy += delta
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_colors = colors[:]
                    if best_energy == 0:
                        return best_colors
            T *= cooling
            
        return best_colors if best_energy == 0 else None

    # Try n=50 with a generous time budget
    # Split time among a few restarts
    time_per_run = 30.0
    max_restarts = 3
    
    for _ in range(max_restarts):
        if time.time() > DEADLINE - 5:
            break
        result = try_solve(50, time_per_run)
        if result is not None:
            return {'n': 50, 'colors': result}
            
    # If 50 fails, try 49
    if time.time() < DEADLINE - 10:
        result = try_solve(49, 20.0)
        if result is not None:
            return {'n': 49, 'colors': result}
            
    # If 49 fails, try 48
    if time.time() < DEADLINE - 10:
        result = try_solve(48, 20.0)
        if result is not None:
            return {'n': 48, 'colors': result}

    # Fallback
    return {'n': 1, 'colors': []}

# We should make sure random seed is not fixed to allow different runs?
# The task says "random.seed(42)" in the seed.
# If we use a fixed seed, we might always fail or always succeed.
# It's better to use system time or allow different seeds.
# However, if we use a fixed seed, the result is deterministic.
# If the deterministic result is 50, we get 1.0 (or better if seed was 32).
# If the deterministic result is <50, we might fail.
# Let's use a different seed for each restart to explore more.
# But the first call should be reproducible? No, the evaluator runs it once.
# We can just let random use its default seed (time-based) or set a seed.
# Setting a seed makes it reproducible. If we pick a bad seed, we fail.
# Not setting a seed might give a different result each time.
# The evaluator runs it once.
# Let's use a seed based on time to maximize chance of finding a solution?
# Or just use the default random state.

# Let's stick to the implementation above but ensure we don't hardcode a bad seed.
# The `try_solve` uses `random.randint`.
# I will remove `random.seed(42)` to allow variety, or use a different seed.
# Actually, if I use `random.seed(42)`, and it fails, I'm stuck.
# If I don't seed, I might get lucky.
# Given the complexity, I will not seed, or seed with time.
# Python's random module seeds from system time by default.

# One more check: The `count_mono_triangles` in `try_solve` is called once at the start.
# It is O(n^3). For n=50, it's fast.
# The main loop is O(1) per move (using bitmasks).
# This should be very fast.

# Final check on the logic:
# `get_delta` calculates change in number of mono triangles.
# If we change (u,v) from c to c':
# We lose triangles where (u,v) was part of a mono-c triangle.
# We gain triangles where (u,v) is part of a mono-c' triangle.
# A triangle (u,v,w) is mono-c if (u,v)=c, (u,w)=c, (v,w)=c.
# So for a fixed (u,v)=c, the number of such triangles is the number of w such that (u,w)=c and (v,w)=c.
# This is exactly `popcount(masks[c][u] & masks[c][v]) - 2` (excluding u and v themselves).
# Correct.

# The code looks solid.
