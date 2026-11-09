def construct():
    n = 128
    max_colors = 4  # try up to 4 colors
    
    for num_colors in range(3, max_colors + 1):
        colors = [-1] * (n + 1)  # 1-indexed
        success = True
        
        for x in range(1, n + 1):
            placed = False
            for c in range(num_colors):
                colors[x] = c
                # Check if this assignment is valid so far
                valid = True
                # Check all 4-APs ending at x or involving x
                for d in range(1, (x - 1) // 3 + 1):
                    a = x - 3 * d
                    if a < 1:
                        break
                    ap = [a, a + d, a + 2 * d, a + 3 * d]
                    if all(colors[pos] != -1 for pos in ap):
                        if len(set(colors[pos] for pos in ap)) < 3:
                            valid = False
                            break
                if not valid:
                    continue
                
                # Look-ahead: try to greedily place next k positions
                k = 4
                temp_colors = colors[:]
                la_valid = True
                for lx in range(x + 1, min(x + 1 + k, n + 1)):
                    la_placed = False
                    for lc in range(num_colors):
                        temp_colors[lx] = lc
                        la_check = True
                        for ld in range(1, (lx - 1) // 3 + 1):
                            la = lx - 3 * ld
                            if la < 1:
                                break
                            lap = [la, la + ld, la + 2 * ld, la + 3 * ld]
                            if all(temp_colors[pos] != -1 for pos in lap):
                                if len(set(temp_colors[pos] for pos in lap)) < 3:
                                    la_check = False
                                    break
                        if la_check:
                            la_placed = True
                            break
                        else:
                            temp_colors[lx] = -1
                    if not la_placed:
                        la_valid = False
                        break
                
                if la_valid:
                    placed = True
                    break
                else:
                    colors[x] = -1
            
            if not placed:
                success = False
                break
        
        if success:
            return colors[1:]
    
    # Fallback: should not reach here
    return list(range(n))
