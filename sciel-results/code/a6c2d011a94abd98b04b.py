<<<<<<< SEARCH
def _simulate_annealing(seed_points, max_time=10.0):
    """Run Simulated Annealing to optimize a configuration."""
    rng = random.Random(int(time.time() * 1000) % 2**32)
    cur = [p[:] for p in seed_points]
    if not _valid(cur):
        # Try to fix validity
        for i in range(12):
            if not _valid(cur):
                cur[i] = _random_point()
        if not _valid(cur):
            return None, float('inf')
            
    cur_obj = _objective(cur)
    best = [p[:] for p in cur]
    best_obj = cur_obj
    
    T = 1000.0
    T_min = 0.1
    cooling_rate = 0.999
    
    start_time = time.time()
    
    while T > T_min and time.time() - start_time < max_time:
        # Generate neighbor
        new_pts = [p[:] for p in cur]
        move_type = rng.random()
        
        if move_type < 0.5:
            # Small perturbation
            idx = rng.randrange(12)
            new_pts[idx][0] += rng.randint(-5, 5)
            new_pts[idx][1] += rng.randint(-5, 5)
        elif move_type < 0.8:
            # Snap to line
            i = rng.randrange(12)
            j = rng.randrange(12)
            k = rng.randrange(12)
            if i == j or i == k or j == k:
                continue
            x1, y1 = new_pts[j]
            x2, y2 = new_pts[k]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                continue
            # Find integer point on line y - y1 = m (x - x1)
            # y = m x + c
            # We want to move point i to a point on this line.
            # Simple approach: pick a random parameter t and see if it's integer.
            # Line: (y2-y1)x - (x2-x1)y + (x2*y1 - x1*y2) = 0
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            g = gcd(gcd(abs(A), abs(B)), abs(C)) if (A or B or C) else 1
            if g:
                A //= g
                B //= g
                C //= g
            # A x + B y = -C
            # We need integer solutions.
            # If gcd(A,B) divides -C, solutions exist.
            g_ab = gcd(abs(A), abs(B))
            if -C % g_ab != 0:
                continue
            # Find one solution
            # A/g x + B/g y = -C/g
            a = A // g_ab
            b = B // g_ab
            c = -C // g_ab
            # Extended Euclid for a x0 + b y0 = 1
            def egcd(a, b):
                if b == 0:
                    return (a, 1, 0)
                g, x1, y1 = egcd(b, a % b)
                return (g, y1, x1 - (a // b) * y1)
            g_gcd, x0, y0 = egcd(a, b)
            if g_gcd != 1:
                continue
            x0 *= c
            y0 *= c
            # General solution: x = x0 + b t, y = y0 - a t
            t = rng.randint(-10, 10)
            nx = x0 + b * t
            ny = y0 - a * t
            if abs(nx) > 10**9 or abs(ny) > 10**9:
                continue
            new_pts[i] = [nx, ny]
        else:
            # Large jump
            idx = rng.randrange(12)
            new_pts[idx] = _random_point()
            
        if not _valid(new_pts):
            T *= cooling_rate
            continue
            
        new_obj = _objective(new_pts)
        delta = new_obj - cur_obj
        
        if delta < 0 or rng.random() < pow(2.71828, -delta / T):
            cur = new_pts
            cur_obj = new_obj
            if cur_obj < best_obj:
                best_obj = cur_obj
                best = [p[:] for p in cur]
                
        T *= cooling_rate
        
    return best, best_obj
=======
def _simulate_annealing(seed_points, max_time=10.0):
    """Run Simulated Annealing to optimize a configuration."""
    rng = random.Random(int(time.time() * 1000) % 2**32)
    cur = [p[:] for p in seed_points]
    if not _valid(cur):
        # Try to fix validity
        for i in range(12):
            if not _valid(cur):
                cur[i] = _random_point()
        if not _valid(cur):
            return None, float('inf')
            
    cur_obj = _objective(cur)
    best = [p[:] for p in cur]
    best_obj = cur_obj
    
    # Early exit if we already found the theoretical minimum
    if _count_ordinary(cur) <= 6:
        return cur, cur_obj

    T = 1000.0
    T_min = 0.1
    cooling_rate = 0.999
    
    start_time = time.time()
    
    while T > T_min and time.time() - start_time < max_time:
        # Generate neighbor
        new_pts = [p[:] for p in cur]
        move_type = rng.random()
        
        if move_type < 0.5:
            # Small perturbation
            idx = rng.randrange(12)
            new_pts[idx][0] += rng.randint(-5, 5)
            new_pts[idx][1] += rng.randint(-5, 5)
        elif move_type < 0.8:
            # Snap to line
            i = rng.randrange(12)
            j = rng.randrange(12)
            k = rng.randrange(12)
            if i == j or i == k or j == k:
                continue
            x1, y1 = new_pts[j]
            x2, y2 = new_pts[k]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                continue
            # Find integer point on line y - y1 = m (x - x1)
            # y = m x + c
            # We want to move point i to a point on this line.
            # Simple approach: pick a random parameter t and see if it's integer.
            # Line: (y2-y1)x - (x2-x1)y + (x2*y1 - x1*y2) = 0
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            g = gcd(gcd(abs(A), abs(B)), abs(C)) if (A or B or C) else 1
            if g:
                A //= g
                B //= g
                C //= g
            # A x + B y = -C
            # We need integer solutions.
            # If gcd(A,B) divides -C, solutions exist.
            g_ab = gcd(abs(A), abs(B))
            if -C % g_ab != 0:
                continue
            # Find one solution
            # A/g x + B/g y = -C/g
            a = A // g_ab
            b = B // g_ab
            c = -C // g_ab
            # Extended Euclid for a x0 + b y0 = 1
            def egcd(a, b):
                if b == 0:
                    return (a, 1, 0)
                g, x1, y1 = egcd(b, a % b)
                return (g, y1, x1 - (a // b) * y1)
            g_gcd, x0, y0 = egcd(a, b)
            if g_gcd != 1:
                continue
            x0 *= c
            y0 *= c
            # General solution: x = x0 + b t, y = y0 - a t
            t = rng.randint(-10, 10)
            nx = x0 + b * t
            ny = y0 - a * t
            if abs(nx) > 10**9 or abs(ny) > 10**9:
                continue
            new_pts[i] = [nx, ny]
        else:
            # Large jump
            idx = rng.randrange(12)
            new_pts[idx] = _random_point()
            
        if not _valid(new_pts):
            T *= cooling_rate
            continue
            
        new_obj = _objective(new_pts)
        delta = new_obj - cur_obj
        
        if delta < 0 or rng.random() < pow(2.71828, -delta / T):
            cur = new_pts
            cur_obj = new_obj
            if cur_obj < best_obj:
                best_obj = cur_obj
                best = [p[:] for p in cur]
                
            # Early exit if we found the theoretical minimum
            if _count_ordinary(cur) <= 6:
                return cur, cur_obj

        T *= cooling_rate
        
    return best, best_obj
>>>>>>> REPLACE

This adds an early exit. This will save time if we find 6 lines, allowing us to run more seeds or longer if we don't.

To further improve, I will add a **Böröczky seed** generator.
I will use a known 12-point configuration.
One such configuration is derived from the **affine plane of order 4** (16 points) by removing 4 points?
Or simply, I will use a **randomized grid** that is less dense.

Actually, the most effective change is to **increase the time budget** for SA by reducing the overhead of `_count_ordinary`.
The current `_count_ordinary` is fine.

I will also add a **second phase** of SA with a different cooling schedule if the first phase doesn't find 6.

But the most important thing is to ensure we **don't miss the 6-line solution**.
The current SA might be too slow to find it from a grid.
By adding the early exit, we ensure that if we find it, we stop wasting time.

I will also modify `construct()` to run SA for **more seeds** if time permits.

