import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    def count_and_find_triangle(N, n):
        """
        Counts monochromatic triangles and returns one if found.
        N is a list of 4 lists of bitmasks. N[c][i] is bitmask of neighbors of i with color c.
        Returns (count, (c, i, j, k) or None)
        """
        total = 0
        found = None
        for c in range(4):
            Nc = N[c]
            for i in range(n):
                Ni = Nc[i]
                for j in range(i + 1, n):
                    inter = Ni & Nc[j]
                    if inter:
                        # Find a set bit k
                        # k is a neighbor of i and j in color c
                        # Triangle is (i, j, k)
                        lsb = inter & -inter
                        k = lsb.bit_length() - 1
                        total += inter.bit_count()
                        if found is None:
                            found = (c, i, j, k)
        return total // 3, found

    def solve(n, time_limit):
        # Initialize random coloring
        # N[c][i] bitmask
        N = [[0] * n for _ in range(4)]
        # Assign random colors
        for i in range(n):
            for j in range(i + 1, n):
                c = random.randint(0, 3)
                N[c][i] |= (1 << j)
                N[c][j] |= (1 << i)
        
        best_count, _ = count_and_find_triangle(N, n)
        if best_count == 0:
            return N

        # Local Search
        while time.time() < time_limit:
            # Find a triangle
            _, tri = count_and_find_triangle(N, n)
            if tri is None:
                return N
            
            c, i, j, k = tri
            # Change color of one of the edges (i,j), (j,k), (i,k)
            # Pick a random edge from the triangle
            edge = random.choice([(i, j), (j, k), (i, k)])
            u, v = edge
            
            # Find current color of (u,v)
            # It must be c (since it's in the mono-triangle of color c)
            # But wait, the triangle (i,j,k) is in color c.
            # So edges (i,j), (j,k), (i,k) are all color c.
            # We want to change one to a different color.
            new_c = random.choice([x for x in range(4) if x != c])
            
            # Update N
            # Remove from c
            N[c][u] &= ~(1 << v)
            N[c][v] &= ~(1 << u)
            # Add to new_c
            N[new_c][u] |= (1 << v)
            N[new_c][v] |= (1 << u)
            
            # Check if we improved
            # We don't need full count every time, but let's check if triangle is gone?
            # Actually, just let the main loop find the next triangle.
            # To avoid getting stuck, we can do a simple restart if no improvement for a while?
            # For now, just keep going.
            
            # Optional: Accept worse moves (Simulated Annealing)?
            # Simple hill climbing might get stuck.
            # Let's add a small probability of random restart or accepting worse states.
            # Given the time limit, simple hill climbing with random restarts is safer.
            # We'll just let it run. If it gets stuck, it will just keep flipping.
            # To improve, we can track best_count and if no improvement for X steps, restart.
            # But for now, let's see if simple works.
            
            # Actually, let's implement a simple restart strategy.
            # If we don't find 0 in a while, restart.
            # But we don't know "a while" without a counter.
            # Let's just run until deadline.

        return None

    def N_to_colors(N, n):
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                # Find color
                for c in range(4):
                    if (N[c][i] >> j) & 1:
                        colors.append(c)
                        break
        return colors

    # Try 50
    result_50 = solve(50, deadline - 5.0) # Leave 5s for 40/32
    if result_50 is not None:
        # Verify
        cnt, _ = count_and_find_triangle(result_50, 50)
        if cnt == 0:
            return {'n': 50, 'colors': N_to_colors(result_50, 50)}

    # Try 40
    if time.time() < deadline - 2.0:
        result_40 = solve(40, deadline - 2.0)
        if result_40 is not None:
            cnt, _ = count_and_find_triangle(result_40, 40)
            if cnt == 0:
                return {'n': 40, 'colors': N_to_colors(result_40, 40)}

    # Fallback to seed 32
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
