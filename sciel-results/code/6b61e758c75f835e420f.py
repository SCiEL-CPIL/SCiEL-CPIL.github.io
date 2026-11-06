import time
import random

def construct():
    start_time = time.time()
    deadline = 145  # Leave 5s for overhead
    
    def get_idx(i, j, n):
        return i * (2 * n - i - 1) // 2 + (j - i - 1)

    def get_color(i, j, colors, n):
        if i > j:
            i, j = j, i
        return colors[get_idx(i, j, n)]

    def set_color(i, j, c, colors, n):
        if i > j:
            i, j = j, i
        colors[get_idx(i, j, n)] = c

    def count_violations(n, colors):
        # N[c][v] is a bitmask of neighbors of v with color c
        N = [[0] * n for _ in range(5)]
        
        # Build adjacency bitmasks
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[get_idx(i, j, n)]
                N[c][i] |= (1 << j)
                N[c][j] |= (1 << i)
        
        violations = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[get_idx(i, j, n)]
                common = N[c][i] & N[c][j]
                if bin(common).count('1') >= 2:
                    violations += 1
        return violations

    def is_valid(n, colors):
        return count_violations(n, colors) == 0

    def local_search(n, initial_colors, max_iters=10000):
        colors = initial_colors[:]
        n_edges = n * (n - 1) // 2
        
        # Initial check
        if is_valid(n, colors):
            return colors
        
        # Precompute indices for faster access
        edge_list = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Find a random violation
            # Instead of checking all, we can pick a random edge and check if it's part of a violation
            # Or just pick a random edge and try to fix it if it has a violation.
            # Better: Maintain a list of violating edges? Too complex.
            # Let's pick a random edge, check if it's violating, if so, fix it.
            # If not, try another.
            
            fixed = False
            for _ in range(50): # Try 50 random edges
                idx = random.randint(0, n_edges - 1)
                # Map idx back to (i, j) is hard without precomputed list.
                # Let's use the precomputed edge_list.
                i, j = edge_list[idx]
                c = colors[get_idx(i, j, n)]
                
                # Check violation for this edge
                # Need N bitmasks. This is expensive to rebuild every time.
                # Alternative: Check common neighbors on the fly?
                # Common neighbors in color c:
                # k such that color(i,k)==c and color(j,k)==c
                
                common = 0
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if get_color(i, k, colors, n) == c and get_color(j, k, colors, n) == c:
                        common += 1
                        if common >= 2:
                            break
                
                if common >= 2:
                    # Try to recolor
                    for new_c in random.sample(range(5), 5):
                        if new_c == c:
                            continue
                        # Check if new_c causes violation
                        new_common = 0
                        for k in range(n):
                            if k == i or k == j:
                                continue
                            if get_color(i, k, colors, n) == new_c and get_color(j, k, colors, n) == new_c:
                                new_common += 1
                                if new_common >= 2:
                                    break
                        if new_common < 2:
                            set_color(i, j, new_c, colors, n)
                            fixed = True
                            break
                if fixed:
                    break
            
            if time.time() > deadline:
                break
        
        return colors

    best_n = 11
    best_colors = [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]

    # Try to improve
    for n in range(12, 21):
        if time.time() > deadline:
            break
        
        found_valid = False
        # Try multiple random starts
        for restart in range(10):
            if time.time() > deadline:
                break
            
            # Initialize
            init_colors = [random.randint(0, 4) for _ in range(n * (n - 1) // 2)]
            
            # Run local search
            # The local_search above is a bit slow due to on-the-fly checks.
            # Let's optimize the violation check using bitmasks in the loop.
            
            # Actually, the bitmask method is much faster.
            # Let's implement a faster local search using bitmasks.
            
            colors = init_colors
            N = [[0] * n for _ in range(5)]
            
            # Build initial N
            for i in range(n):
                for j in range(i + 1, n):
                    c = colors[get_idx(i, j, n)]
                    N[c][i] |= (1 << j)
                    N[c][j] |= (1 << i)
            
            # Check initial validity
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    c = colors[get_idx(i, j, n)]
                    if bin(N[c][i] & N[c][j]).count('1') >= 2:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                best_n = n
                best_colors = colors
                found_valid = True
                break
            
            # If not valid, perform local search steps
            # We need to update N when we change a color.
            
            for step in range(5000):
                if time.time() > deadline:
                    break
                
                # Pick a random edge
                idx = random.randint(0, n * (n - 1) // 2 - 1)
                # Find i, j from idx
                # i*(2n-i-1)/2 + (j-i-1) = idx
                # This mapping is 1-1.
                # We can precompute a reverse map for speed if n is fixed, but n changes.
                # Let's just iterate to find i,j? No, too slow.
                # Let's stick to the edge_list approach for random picking, but update N efficiently.
                
                # Re-reading: I used `edge_list` in the previous thought but not here.
                # Let's generate random i, j directly.
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i == j:
                    continue
                if i > j:
                    i, j = j, i
                
                c = colors[get_idx(i, j, n)]
                
                # Check violation
                common = N[c][i] & N[c][j]
                if bin(common).count('1') < 2:
                    continue # No violation on this edge
                
                # Try to recolor
                recolor_success = False
                for new_c in random.sample(range(5), 5):
                    if new_c == c:
                        continue
                    
                    # Check if new_c is safe
                    new_common = N[new_c][i] & N[new_c][j]
                    if bin(new_common).count('1') < 2:
                        # Apply change
                        # Remove from N[c]
                        N[c][i] &= ~(1 << j)
                        N[c][j] &= ~(1 << i)
                        
                        # Add to N[new_c]
                        N[new_c][i] |= (1 << j)
                        N[new_c][j] |= (1 << i)
                        
                        colors[get_idx(i, j, n)] = new_c
                        recolor_success = True
                        break
                
                if recolor_success:
                    # Check if fully valid (expensive, do occasionally)
                    # Or just continue
                    pass
            
            # Final check after search
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    c = colors[get_idx(i, j, n)]
                    if bin(N[c][i] & N[c][j]).count('1') >= 2:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                best_n = n
                best_colors = colors
                found_valid = True
                break
        
        if found_valid:
            # Try to go higher?
            continue

    return {'n': best_n, 'colors': best_colors}
