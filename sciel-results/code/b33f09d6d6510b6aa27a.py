import time
import random
import math

def construct():
    # Target n=50, as Chung's construction proves existence.
    # We use Simulated Annealing to find a valid 4-colouring.
    
    def count_mono_triangles(colors, n):
        # colors is a list of length n*(n-1)//2
        # Index mapping: pair (i, j) with i < j
        # idx = i * (n-1) - i * (i-1) // 2 + (j - i - 1) ? 
        # Simpler: use a dictionary or 2D array if n is small.
        # For n=50, n^2 = 2500, which is small.
        # Let's use a 2D array for speed in the search loop.
        pass # This function is for initial check, we will optimize inside the loop.

    def idx(i, j, n):
        if i > j: i, j = j, i
        return i * (2 * n - i - 3) // 2 + (j - i - 1)
        
    # Better index: 
    # 0: (0,1), (0,2)...(0,n-1)
    # 1: (1,2)...(1,n-1)
    # ...
    # idx(i, j) = sum_{k=0}^{i-1} (n-1-k) + (j-i-1)
    # sum_{k=0}^{i-1} (n-1-k) = i(n-1) - i(i-1)/2
    # idx = i(n-1) - i(i-1)//2 + j - i - 1
    
    def get_idx(i, j, n):
        if i > j:
            i, j = j, i
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)

    def set_edge(colors, i, j, c, n):
        colors[get_idx(i, j, n)] = c

    def get_edge(colors, i, j, n):
        return colors[get_idx(i, j, n)]

    # Precompute neighbors for fast triangle updates?
    # For an edge (u, v), the triangles are (u, v, w) for all w != u, v.
    
    def calc_delta(colors, u, v, new_c, n):
        # Returns change in number of mono-triangles if we set (u,v) to new_c
        old_c = colors[get_idx(u, v, n)]
        if old_c == new_c:
            return 0
        
        delta = 0
        for w in range(n):
            if w == u or w == v:
                continue
            # Triangle (u, v, w)
            c_uw = colors[get_idx(u, w, n)]
            c_vw = colors[get_idx(v, w, n)]
            
            # Check if (u, v, w) was mono before
            if c_uw == c_vw == old_c:
                delta -= 1 # We broke a mono-triangle
            
            # Check if (u, v, w) becomes mono after
            if c_uw == c_vw == new_c:
                delta += 1 # We created a mono-triangle
                
        return delta

    def try_n(n, time_limit):
        # Initialize
        num_edges = n * (n - 1) // 2
        colors = [random.randint(0, 3) for _ in range(num_edges)]
        
        # Initial cost
        cost = 0
        # We can compute initial cost by checking all triangles, O(n^3)
        # Or just start SA and let it converge.
        # For n=50, O(n^3) = 125k, very fast.
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    c_ij = colors[get_idx(i, j, n)]
                    c_ik = colors[get_idx(i, k, n)]
                    c_jk = colors[get_idx(j, k, n)]
                    if c_ij == c_ik == c_jk:
                        cost += 1
                        
        best_colors = colors[:]
        best_cost = cost
        
        start_time = time.time()
        T = 1.0
        T_min = 0.001
        cooling = 0.999
        
        while time.time() - start_time < time_limit:
            # Pick random edge
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            old_c = colors[get_idx(i, j, n)]
            new_c = random.randint(0, 3)
            if old_c == new_c:
                continue
            
            delta = calc_delta(colors, i, j, new_c, n)
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                colors[get_idx(i, j, n)] = new_c
                cost += delta
                
                if cost < best_cost:
                    best_cost = cost
                    best_colors = colors[:]
                    if cost == 0:
                        break
                
            T *= cooling
            if T < T_min:
                T = T_min
                
        if best_cost == 0:
            return best_colors
        return None

    # Try n=50
    target_n = 50
    # Use most of the time for n=50
    # If 50 fails, try 40? No, 50 is the target.
    # If 50 succeeds, can we try 51?
    # Let's reserve 10s for 51 if 50 is found quickly.
    
    time_spent = 0
    total_time = 110 # Safety margin
    
    # Phase 1: Try 50
    t0 = time.time()
    res_50 = try_n(50, 60)
    t1 = time.time()
    
    if res_50 is not None:
        # Found valid 50
        # Check if we have time for 51
        remaining = total_time - (t1 - t0)
        if remaining > 5:
            # Try 51 with remaining time
            res_51 = try_n(51, remaining - 1)
            if res_51 is not None:
                return {'n': 51, 'colors': res_51}
        
        return {'n': 50, 'colors': res_50}
    else:
        # Failed 50, fallback to 32 (seed)
        # We can try to generate 32 quickly to ensure score >= 1.0
        # The seed program does this.
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
