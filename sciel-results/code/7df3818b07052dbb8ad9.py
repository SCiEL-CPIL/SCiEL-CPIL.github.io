def construct():
    from itertools import combinations
    import numpy as np

    def check_valid(edge_colors):
        """Check every K4 in K16 has at least 5 distinct colors."""
        for quad in combinations(range(16), 4):
            a, b, c, d = quad
            cols = {
                edge_colors[a][b], edge_colors[a][c], edge_colors[a][d],
                edge_colors[b][c], edge_colors[b][d], edge_colors[c][d]
            }
            if len(cols) < 5:
                return False
        return True

    # We want to find a coloring with the minimum number of colors.
    # The current best known is 16 (from p=17 multiplicative).
    # Can we do 15?
    # Let's try to construct a 15-color coloring.
    # One approach: Start with p=17 multiplicative (16 colors), and try to merge two colors if possible.
    # Or try other modular schemes.
    
    # Let's first verify the p=17 multiplicative scheme and count its colors.
    def get_colors_for_scheme(scheme_type, p, shift=0):
        edge_colors = [[0]*16 for _ in range(16)]
        for i in range(16):
            for j in range(i+1, 16):
                if scheme_type == 'mult':
                    c = ((i+1)*(j+1)) % p
                elif scheme_type == 'add':
                    c = (i + j) % p
                elif scheme_type == 'mixed':
                    c = ((i+1)*(j+1) + i + j) % p
                elif scheme_type == 'mult_shift':
                    c = ((i+shift)*(j+shift)) % p
                elif scheme_type == 'mult_diff':
                    c = (i*j) % p if i*j % p != 0 else p-1 # Avoid 0?
                else:
                    c = 0
                edge_colors[i][j] = c
                edge_colors[j][i] = c
        return edge_colors

    candidates = []

    # Try multiplicative schemes with various primes
    for p in [17, 19, 23, 29, 31, 37, 41]:
        edge_colors = get_colors_for_scheme('mult', p)
        if check_valid(edge_colors):
            num = len(set(edge_colors[i][j] for i in range(16) for j in range(i+1, 16)))
            candidates.append((num, p, 'mult', 0))

    # Try additive schemes
    for p in [17, 19, 23, 29, 31]:
        edge_colors = get_colors_for_scheme('add', p)
        if check_valid(edge_colors):
            num = len(set(edge_colors[i][j] for i in range(16) for j in range(i+1, 16)))
            candidates.append((num, p, 'add', 0))

    # Try mixed: (i*j + i + j) % p
    for p in [17, 19, 23]:
        edge_colors = get_colors_for_scheme('mixed', p)
        if check_valid(edge_colors):
            num = len(set(edge_colors[i][j] for i in range(16) for j in range(i+1, 16)))
            candidates.append((num, p, 'mixed', 0))

    # Try (i*j) % p with shift
    for p in [17, 19, 23]:
        for shift in range(p):
            edge_colors = get_colors_for_scheme('mult_shift', p, shift)
            if check_valid(edge_colors):
                num = len(set(edge_colors[i][j] for i in range(16) for j in range(i+1, 16)))
                candidates.append((num, p, 'mult_shift', shift))
                break  # only need first valid shift for this p

    # Try to find a 15-color solution by modifying the 16-color solution
    # We know p=17 mult is likely 16 colors.
    # Can we merge two colors in the p=17 mult solution?
    # Let's get the p=17 mult solution and try to merge pairs of colors.
    
    best_candidate = None
    if candidates:
        candidates.sort()
        best_num, best_p, best_type, best_shift = candidates[0]
        best_candidate = (best_num, best_p, best_type, best_shift)
    
    # If best is 16, try to improve to 15
    if best_candidate and best_candidate[0] <= 16:
        base_p = best_candidate[1]
        base_type = best_candidate[2]
        base_shift = best_candidate[3]
        base_edge_colors = get_colors_for_scheme(base_type, base_p, base_shift)
        
        # Get the set of colors used
        used_colors = set(base_edge_colors[i][j] for i in range(16) for j in range(i+1, 16))
        
        # Try to merge each pair of colors
        color_list = list(used_colors)
        improved = False
        for c1_idx in range(len(color_list)):
            for c2_idx in range(c1_idx+1, len(color_list)):
                c1 = color_list[c1_idx]
                c2 = color_list[c2_idx]
                
                # Create a new edge coloring where c2 is mapped to c1
                new_edge_colors = [[0]*16 for _ in range(16)]
                for i in range(16):
                    for j in range(i+1, 16):
                        c = base_edge_colors[i][j]
                        if c == c2:
                            c = c1
                        new_edge_colors[i][j] = c
                        new_edge_colors[j][i] = c
                
                if check_valid(new_edge_colors):
                    num = len(set(new_edge_colors[i][j] for i in range(16) for j in range(i+1, 16)))
                    if num < best_candidate[0]:
                        best_candidate = (num, base_p, base_type, base_shift, new_edge_colors)
                        improved = True
                        break
            if improved:
                break

    # Reconstruct the best candidate
    if best_candidate:
        if len(best_candidate) == 5:
            # It's a modified edge coloring
            return [best_candidate[4][i][j] for i in range(16) for j in range(i+1, 16)]
        else:
            num, p, typ, shift = best_candidate
            edge_colors = get_colors_for_scheme(typ, p, shift)
            return [edge_colors[i][j] for i in range(16) for j in range(i+1, 16)]
    
    # Fallback
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
