    # Greedy initialization
    def greedy_decompose():
        rem = N
        last = 12
        factors = []
        while rem > 1:
            found = False
            for d in divs_map.get(rem, []):
                if d > last:
                    factors.append(d)
                    rem //= d
                    last = d
                    found = True
                    break
            if not found:
                return None
        return factors
        
    g_factors = greedy_decompose()
    if g_factors:
        best_max = max(g_factors)
    else:
        best_max = float('inf')
