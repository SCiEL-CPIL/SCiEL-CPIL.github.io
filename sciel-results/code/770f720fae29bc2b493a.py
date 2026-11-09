import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Fallback: 32-vertex coloring from seed
    def get_seed_coloring():
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

    best_result = get_seed_coloring()
    
    for n in range(50, 32, -1):
        if time.time() > deadline - 5:
            break
        
        # Try to find a coloring for n
        result = solve_n(n, deadline)
        if result:
            best_result = result
            break # If we found 50, we are done. If we found 49, we are done (maximized).
            # Actually, if we find 49, we should return it. We iterate downwards, so first success is max.
            
    return best_result

def solve_n(n, deadline):
    # Local search
    # colors[i][j] is color of edge (i,j)
    # N[i][c] is bitset of neighbors k such that (i,k) is color c
    
    # Initialization
    while time.time() < deadline:
        colors = [[0]*n for _ in range(n)]
        N = [[0]*4 for _ in range(n)]
        
        # Random init
        for i in range(n):
            for j in range(i+1, n):
                c = random.randint(0, 3)
                colors[i][j] = c
                colors[j][i] = c
                N[i][c] |= (1 << j)
                N[j][c] |= (1 << i)
        
        # Count triangles
        tri_count = count_triangles(N, n)
        
        if tri_count == 0:
            return format_result(n, colors)
            
        # Local search
        last_improvement = time.time()
        stuck_count = 0
        
        while time.time() < deadline:
            if tri_count == 0:
                return format_result(n, colors)
                
            # Find a random mono-triangle
            tri = find_random_triangle(N, n)
            if not tri:
                # Should not happen if tri_count > 0, but safety
                break
                
            i, j, k, c = tri
            # Pick a random edge to recolor
            edge = random.choice([(i,j), (j,k), (k,i)])
            u, v = edge
            
            # Pick a new color
            old_c = colors[u][v]
            new_c = random.choice([c for c in range(4) if c != old_c])
            
            # Update
            colors[u][v] = new_c
            colors[v][u] = new_c
            
            # Update N
            N[u][old_c] &= ~(1 << v)
            N[v][old_c] &= ~(1 << u)
            N[u][new_c] |= (1 << v)
            N[v][new_c] |= (1 << u)
            
            # Recount
            new_tri_count = count_triangles(N, n)
            
            if new_tri_count < tri_count:
                tri_count = new_tri_count
                last_improvement = time.time()
                stuck_count = 0
            else:
                stuck_count += 1
                # Accept with some probability? Or just restart if stuck too long.
                if stuck_count > 1000:
                    break # Restart
            
        # If stuck, restart initialization
        if time.time() > deadline - 2:
            break

    return None

def count_triangles(N, n):
    count = 0
    for i in range(n):
        for c in range(4):
            bits = N[i][c]
            # For each neighbor j in bits, check common neighbors
            # This is O(n^2) per i? No, iterating bits is fast.
            # But we double count.
            # Each triangle is counted 6 times (once for each vertex, once for each direction? No, N is symmetric).
            # Triangle (i,j,k) with color c:
            # i sees j and k. j is in N[i][c]. k is in N[i][c].
            # When i is current, we check j. N[j][c] contains k. So (i,j,k) found.
            # When i is current, we check k. N[k][c] contains j. So (i,k,j) found.
            # So each triangle is counted 2 times for vertex i?
            # Total count will be 6 * (number of triangles).
            
            # Optimization: Only count triangles where i < j < k?
            # Hard with bitsets.
            # Let's just count all and divide by 6?
            # Or just use the count as a heuristic.
            
            # Efficient counting:
            # For each i, for each c:
            #   for each j in N[i][c]:
            #      if j > i: # to avoid double counting in i loop?
            #         common = N[i][c] & N[j][c]
            #         count += popcount(common)
            # This counts each triangle exactly once?
            # Triangle (i,j,k) with i<j<k.
            # i loop: j in N[i][c]. common has k. Count 1.
            # i loop: k in N[i][c]. common has j. But j < i? No, j>i.
            # Wait, if we don't restrict j>i, we count (i,j) and (i,k).
            # Let's restrict:
            # For i in range(n):
            #   for c in range(4):
            #     for j in bits(N[i][c]):
            #       if j > i:
            #         common = N[i][c] & N[j][c]
            #         count += popcount(common)
            # This counts each triangle once.
            
            pass # Implementation in function body

    # Correct implementation of count_triangles
    count = 0
    for i in range(n):
        for c in range(4):
            bits = N[i][c]
            # Iterate set bits
            while bits:
                lsb = bits & -bits
                j = lsb.bit_length() - 1
                if j > i:
                    common = N[i][c] & N[j][c]
                    count += common.bit_count()
                bits -= lsb
    return count

def find_random_triangle(N, n):
    # Pick random i, c, j in N[i][c]
    for _ in range(100): # Try 100 times
        i = random.randint(0, n-1)
        c = random.randint(0, 3)
        bits = N[i][c]
        if bits == 0:
            continue
        # Pick random bit
        # Convert to list? Slow.
        # Pick random index in range(n) and check if bit is set?
        # Or pick random int from 0 to sum(bits) - 1?
        
        # Simple: Pick random j
        j = random.randint(0, n-1)
        if not (bits & (1 << j)):
            continue
        
        common = N[i][c] & N[j][c]
        if common:
            # Pick random k from common
            # Convert to list of indices
            ks = []
            tmp = common
            while tmp:
                lsb = tmp & -tmp
                k = lsb.bit_length() - 1
                ks.append(k)
                tmp -= lsb
            k = random.choice(ks)
            return (i, j, k, c)
            
    return None

def format_result(n, colors):
    res_colors = []
    for i in range(n):
        for j in range(i+1, n):
            res_colors.append(colors[i][j])
    return {'n': n, 'colors': res_colors}
