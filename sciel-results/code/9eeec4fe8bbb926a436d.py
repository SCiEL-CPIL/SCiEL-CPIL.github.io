def construct():
    import time
    start_time = time.time()
    deadline = start_time + 110  # 10 seconds margin

    def is_valid_coloring(n, colors):
        # Build adjacency lists for color 0 and 1
        adj0 = [0] * n
        adj1 = [0] * n
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                idx += 1
                if c == 0:
                    adj0[i] |= (1 << j)
                    adj0[j] |= (1 << i)
                else:
                    adj1[i] |= (1 << j)
                    adj1[j] |= (1 << i)
        
        def has_k222(adj):
            # Check for K2,2,2 in graph defined by adj
            # Iterate over pairs (u, v)
            for u in range(n):
                for v in range(u + 1, n):
                    if not (adj[u] & (1 << v)):
                        continue
                    # M_uv are vertices connected to both u and v
                    m_uv = adj[u] & adj[v]
                    # Remove u and v from m_uv
                    m_uv &= ~(1 << u)
                    m_uv &= ~(1 << v)
                    
                    # Iterate over pairs (x, y) in m_uv
                    # To be efficient, we can iterate over set bits
                    # But n=29 is small, so we can just loop
                    xs = []
                    for x in range(n):
                        if x != u and x != v and (m_uv & (1 << x)):
                            xs.append(x)
                            
                    for xi in range(len(xs)):
                        x = xs[xi]
                        for yi in range(xi + 1, len(xs)):
                            y = xs[yi]
                            # Check if x and y are connected to u and v (they are, by definition of m_uv)
                            # Now check if there are at least 2 other vertices w, z connected to u, v, x, y
                            common = adj[u] & adj[v] & adj[x] & adj[y]
                            # Remove u, v, x, y
                            common &= ~(1 << u)
                            common &= ~(1 << v)
                            common &= ~(1 << x)
                            common &= ~(1 << y)
                            # Count bits
                            if common.bit_count() >= 2:
                                return True
            return False

        if has_k222(adj0):
            return False
        if has_k222(adj1):
            return False
        return True

    def paley_coloring(n):
        # n must be a prime p = 4k+1
        # Color 0 if (j-i)^2 is a quadratic residue mod n (and non-zero)
        # Wait, the seed uses: 0 if (j-i)%17 in squares else 1
        # Squares are non-zero squares.
        # So if diff is 0 (impossible for i<j) or non-square, it's 1.
        # If diff is a non-zero square, it's 0.
        
        if n < 2:
            return []
        
        # Check if n is prime and n % 4 == 1
        def is_prime(num):
            if num < 2:
                return False
            for i in range(2, int(num**0.5) + 1):
                if num % i == 0:
                    return False
            return True
            
        if not is_prime(n) or n % 4 != 1:
            return None

        squares = set()
        for x in range(1, n):
            squares.add((x * x) % n)
        
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                diff = (j - i) % n
                if diff in squares:
                    colors.append(0)
                else:
                    colors.append(1)
        return colors

    # Try n=29
    if time.time() < deadline:
        colors29 = paley_coloring(29)
        if colors29 is not None:
            if is_valid_coloring(29, colors29):
                return {'n': 29, 'colors': colors29}
                
    # Try n=17 (Fallback to seed)
    colors17 = paley_coloring(17)
    if colors17 is not None:
        # We know 17 is valid, but check just in case
        if is_valid_coloring(17, colors17):
            return {'n': 17, 'colors': colors17}
            
    # Fallback to 1
    return {'n': 1, 'colors': []}
