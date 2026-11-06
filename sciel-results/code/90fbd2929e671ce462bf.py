    # Precompute 3-term progressions as well
    aps3 = []
    for a in range(N):
        for d in range(1, (N - 1 - a) // 2 + 1):
            b = a + d
            c = a + 2 * d
            aps3.append((a, b, c))
            
    def get_forbidden_mask(i, colors, max_colors):
        forbidden = 0
        # Check 4-APs
        for (a, b, c, e) in aps:
            if i == a:
                others = [colors[b], colors[c], colors[e]]
                if -1 in others: continue
                distinct = set(others)
                if len(distinct) == 1: return (1 << max_colors) - 1
                elif len(distinct) == 2:
                    for c_val in distinct: forbidden |= (1 << c_val)
            elif i == b:
                others = [colors[a], colors[c], colors[e]]
                if -1 in others: continue
                distinct = set(others)
                if len(distinct) == 1: return (1 << max_colors) - 1
                elif len(distinct) == 2:
                    for c_val in distinct: forbidden |= (1 << c_val)
            elif i == c:
                others = [colors[a], colors[b], colors[e]]
                if -1 in others: continue
                distinct = set(others)
                if len(distinct) == 1: return (1 << max_colors) - 1
                elif len(distinct) == 2:
                    for c_val in distinct: forbidden |= (1 << c_val)
            elif i == e:
                others = [colors[a], colors[b], colors[c]]
                if -1 in others: continue
                distinct = set(others)
                if len(distinct) == 1: return (1 << max_colors) - 1
                elif len(distinct) == 2:
                    for c_val in distinct: forbidden |= (1 << c_val)
        
        # Check 3-APs to prevent mono 3-APs
        for (a, b, c) in aps3:
            if i == a:
                if colors[b] != -1 and colors[c] != -1 and colors[b] == colors[c]:
                    forbidden |= (1 << colors[b])
            elif i == b:
                if colors[a] != -1 and colors[c] != -1 and colors[a] == colors[c]:
                    forbidden |= (1 << colors[a])
            elif i == c:
                if colors[a] != -1 and colors[b] != -1 and colors[a] == colors[b]:
                    forbidden |= (1 << colors[a])
                    
        return forbidden
