import time
import random
import sys
import math

def construct():
    """
    Uses Simulated Annealing to find a 4-coloring of K_n with no monochromatic triangles.
    Targets n=50 (the known record). Falls back to lower n if time runs out.
    """
    start_time = time.time()
    time_limit = 110.0  # Leave 10s buffer

    def get_idx(n, i, j):
        if i > j:
            i, j = j, i
        # Index for (i, j) in lexicographic order of pairs (a, b) with a < b
        # Number of pairs before row i is i*n - i*(i+1)//2
        # Position in row i is j - i - 1
        return i * n - i * (i + 1) // 2 + (j - i - 1)

    def get_color(colors, n, i, j):
        if i > j:
            i, j = j, i
        return colors[get_idx(n, i, j)]

    def set_color(colors, n, i, j, c):
        if i > j:
            i, j = j, i
        colors[get_idx(n, i, j)] = c

    def count_energy(colors, n):
        """Count total number of monochromatic triangles."""
        energy = 0
        for i in range(n):
            for j in range(i + 1, n):
                c_ij = colors[get_idx(n, i, j)]
                for k in range(j + 1, n):
                    c_ik = colors[get_idx(n, i, k)]
                    c_jk = colors[get_idx(n, j, k)]
                    if c_ij == c_ik == c_jk:
                        energy += 1
        return energy

    def delta_energy(colors, n, u, v, old_c, new_c):
        """
        Calculate change in energy when changing color of edge (u, v) from old_c to new_c.
        Only triangles containing edge (u, v) are affected.
        """
        if old_c == new_c:
            return 0
        
        delta = 0
        # Triangles are (u, v, w) for all w != u, v
        for w in range(n):
            if w == u or w == v:
                continue
            
            c_uw = get_color(colors, n, u, w)
            c_vw = get_color(colors, n, v, w)
            
            # Check if triangle (u, v, w) was monochromatic with old_c
            was_mono = (old_c == c_uw == c_vw)
            
            # Check if triangle (u, v, w) will be monochromatic with new_c
            is_mono = (new_c == c_uw == c_vw)
            
            if was_mono and not is_mono:
                delta -= 1
            elif is_mono and not was_mono:
                delta += 1
            # If both or neither, delta is 0 for this triangle
            
        return delta

    def simulate_annealing(n, time_budget):
        """
        Run simulated annealing for a given n.
        Returns (colors, success)
        """
        m = n * (n - 1) // 2
        colors = [random.randint(0, 3) for _ in range(m)]
        
        # Initial energy
        current_energy = count_energy(colors, n)
        
        # If we start with 0 energy (unlikely for n>3), we are done
        if current_energy == 0:
            return colors, True
            
        # Simulated Annealing Parameters
        # T starts high, decreases exponentially
        T_start = 1.0
        T_end = 0.001
        # Number of steps depends on time budget
        # Each step is O(n). For n=50, ~50 ops. 
        # We can do ~100k steps in 1 second easily in Python.
        # Let's estimate max steps based on time budget.
        
        steps = 0
        start_sa_time = time.time()
        
        # We will run until time budget is exceeded or energy is 0
        # To avoid getting stuck, we might restart if no improvement for a long time
        # But for simplicity, we just run SA.
        
        # Adaptive step size or just fixed? Fixed is fine.
        # We need to pick a random edge (u, v) and a random new color.
        
        # Precompute list of edges for faster random selection?
        # Or just pick random i, j.
        
        while time.time() - start_sa_time < time_budget:
            # Pick random edge
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            
            u, v = (i, j) if i < j else (j, i)
            
            old_c = colors[get_idx(n, u, v)]
            # Pick new color different from old
            new_c = random.randint(0, 3)
            if new_c == old_c:
                new_c = (new_c + 1) % 4
            
            dE = delta_energy(colors, n, u, v, old_c, new_c)
            
            # Acceptance criterion
            if dE < 0:
                # Improvement, always accept
                colors[get_idx(n, u, v)] = new_c
                current_energy += dE
                if current_energy == 0:
                    return colors, True
            else:
                # Worse, accept with probability exp(-dE / T)
                # T should decrease over time
                # Simple linear or exponential cooling
                # Let's use exponential cooling based on steps or time
                # Since we don't know total steps, let's use time-based cooling
                elapsed = time.time() - start_sa_time
                # Map elapsed time to T
                # T = T_start * (T_end / T_start) ** (elapsed / time_budget)
                # This ensures T reaches T_end at the end of the budget
                if time_budget > 0:
                    progress = elapsed / time_budget
                    if progress >= 1.0:
                        T = T_end
                    else:
                        T = T_start * math.exp(math.log(T_end / T_start) * progress)
                else:
                    T = T_end
                    
                if dE == 0:
                    # Neutral move, accept
                    colors[get_idx(n, u, v)] = new_c
                else:
                    # dE > 0
                    if random.random() < math.exp(-dE / T):
                        colors[get_idx(n, u, v)] = new_c
                        current_energy += dE
                        if current_energy == 0:
                            return colors, True
        
        # If we exit loop, check if we ended up with 0 energy
        if current_energy == 0:
            return colors, True
            
        return colors, False

    # Try to find a solution for n=50, then 49, etc.
    # We allocate time to each attempt.
    # n=50 is the target.
    # Let's give 60s to n=50, 30s to n=49, 10s to n=48, etc.
    
    targets = [50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40]
    time_allocations = [60, 30, 10, 5, 2, 1, 1, 1, 1, 1, 1]
    
    for n, t_alloc in zip(targets, time_allocations):
        if time.time() - start_time > time_limit:
            break
            
        # If we have already found a solution with higher n, we wouldn't be here.
        # But this loop is sequential.
        
        # We can try multiple restarts within the time allocation if needed, 
        # but SA is usually good enough with one long run.
        # Let's just run SA for the allocated time.
        
        colors, success = simulate_annealing(n, t_alloc)
        
        if success:
            return {'n': n, 'colors': colors}
            
    # If we failed to find any solution for n >= 40 (unlikely), 
    # we must return a valid solution to score > 0.
    # The seed program had a valid n=32 solution. 
    # We should implement a fallback to ensure we don't return invalid or None.
    # However, SA is very likely to find n=50.
    # For safety, let's construct a known small valid coloring if all else fails.
    # A simple valid coloring for n=3 is [0, 0, 0] (no triangles? No, K3 has 1 triangle. 
    # Wait, "No colour class may contain a triangle". 
    # In K3, the only triangle is the whole graph. If all edges are color 0, it's a monochromatic triangle. Invalid.
    # So for n=3, we need at least 2 colors. e.g. 0, 0, 1.
    # Let's just return a random valid one if we fail, or better, a hardcoded small one.
    
    # Hardcoded valid coloring for n=5 (no mono triangles)
    # Actually, let's just try n=10 with SA quickly if we get here.
    if time.time() - start_time < time_limit - 5:
         colors, success = simulate_annealing(10, 5.0)
         if success:
             return {'n': 10, 'colors': colors}
             
    # Absolute fallback: n=1 is always valid.
    # But score will be low.
    # Let's try to construct n=32 using the seed logic as a last resort fallback.
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
