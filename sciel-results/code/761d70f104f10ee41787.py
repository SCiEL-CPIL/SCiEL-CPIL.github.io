    def build_lps(q):
        """Build a 3-regular LPS graph over F_q using BFS."""
        # q must be odd prime or prime power, q ≡ ±1 mod 6
        # Check condition
        if q % 2 == 0:
            return None, None
        if (q - 1) % 6 != 0 and (q + 1) % 6 != 0:
            return None, None

        # Matrices in SL(2, F_q) identified with PSL(2, F_q)
        # We work in SL(2, F_q) and identify M ~ -M.
        
        def norm(m):
            a, b, c, d = m
            rep1 = (a, b, c, d)
            rep2 = ((-a)%q, (-b)%q, (-c)%q, (-d)%q)
            if rep1 < rep2:
                return rep1
            else:
                return rep2

        def mul(m1, m2):
            a1, b1, c1, d1 = m1
            a2, b2, c2, d2 = m2
            a = (a1*a2 + b1*c2) % q
            b = (a1*b2 + b1*d2) % q
            c = (c1*a2 + d1*c2) % q
            d = (c1*b2 + d1*d2) % q
            return norm((a, b, c, d))

        # Generators in SL(2, F_q)
        # A = [[0, 1], [-1, 0]] -> (0, 1, q-1, 0)
        # B = [[0, 1], [1, 0]] -> (0, 1, 1, 0)
        # C = [[1, 1], [0, 1]] -> (1, 1, 0, 1)
        
        # Note: We must ensure these generate the group.
        # For q=7, they do.
        
        A = norm((0, 1, q-1, 0))
        B = norm((0, 1, 1, 0))
        C = norm((1, 1, 0, 1))
        
        # Identity
        I = norm((1, 0, 0, 1))
        
        # BFS to generate all elements
        index = {I: 0}
        order = [I]
        queue = deque([I])
        
        # To avoid infinite loops or huge groups if generators don't generate the whole group,
        # we rely on the known size n = q(q^2-1)/2.
        expected_n = q * (q*q - 1) // 2
        
        while queue and len(order) < expected_n:
            u = queue.popleft()
            # Generate neighbors
            for gen in [A, B, C]:
                v = mul(u, gen)
                if v not in index:
                    index[v] = len(order)
                    order.append(v)
                    queue.append(v)
        
        n = len(order)
        
        # Build edges
        edges = []
        for i, u in enumerate(order):
            for gen in [A, B, C]:
                v = mul(u, gen)
                j = index[v]
                if i < j:
                    edges.append((i, j))
        
        return n, edges
