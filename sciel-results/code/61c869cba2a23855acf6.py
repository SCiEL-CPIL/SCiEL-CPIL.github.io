import time
import random

def construct():
    q = 29
    squares = {x * x % q for x in range(1, q)}
    
    n_base = q
    colors_base = []
    for i in range(n_base):
        for j in range(i + 1, n_base):
            diff = (j - i) % q
            if diff in squares:
                colors_base.append(0)
            else:
                colors_base.append(1)
                
    # Build adjacency bitmasks for base graph
    # adj[c][i] is a bitmask of vertices j such that (i,j) is color c
    adj = [[0] * n_base for _ in range(2)]
    idx = 0
    for i in range(n_base):
        for j in range(i + 1, n_base):
            c = colors_base[idx]
            idx += 1
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            
    # Add vertex n_base
    n_new = n_base + 1
    # colors_new[i] is the color of edge (i, n_base)
    colors_new = [random.randint(0, 1) for _ in range(n_base)]
    
    def count_k22_in_subset(subset_mask, color):
        """
        Count K_{2,2} subgraphs in the induced subgraph of subset_mask with edges of given color.
        A K_{2,2} is a set of 4 vertices {a,b,c,d} partitioned into {a,b} and {c,d}
        such that all 4 cross edges are of the given color.
        """
        verts = [i for i in range(n_base) if (subset_mask >> i) & 1]
        count = 0
        # Iterate over pairs of pairs
        # There are 3 ways to partition 4 vertices into 2 pairs.
        # We can just iterate over all 4-subsets and check.
        # For efficiency, we can precompute for the subset?
        # Subset size is small (<30).
        import itertools
        for combo in itertools.combinations(verts, 4):
            a, b, c, d = combo
            # Check partition {a,b} vs {c,d}
            if (adj[color][a] & (1 << c)) and (adj[color][a] & (1 << d)) and \
               (adj[color][b] & (1 << c)) and (adj[color][b] & (1 << d)):
                count += 1
            # Check partition {a,c} vs {b,d}
            if (adj[color][a] & (1 << b)) and (adj[color][a] & (1 << d)) and \
               (adj[color][c] & (1 << b)) and (adj[color][c] & (1 << d)):
                count += 1
            # Check partition {a,d} vs {b,c}
            if (adj[color][a] & (1 << b)) and (adj[color][a] & (1 << c)) and \
               (adj[color][d] & (1 << b)) and (adj[color][d] & (1 << c)):
                count += 1
        return count

    def total_violations(colors_new):
        # Build masks for new vertex neighbors
        mask0 = 0
        mask1 = 0
        for i in range(n_base):
            if colors_new[i] == 0:
                mask0 |= (1 << i)
            else:
                mask1 |= (1 << i)
        
        v0 = count_k22_in_subset(mask0, 0)
        v1 = count_k22_in_subset(mask1, 1)
        return v0 + v1

    start_time = time.time()
    deadline = 110.0 # Safety margin
    
    best_violations = total_violations(colors_new)
    best_colors = colors_new[:]
    
    if best_violations == 0:
        # Found solution immediately
        final_colors = colors_base + colors_new
        return {'n': n_new, 'colors': final_colors}

    while time.time() < deadline:
        # Pick a random index to flip
        idx = random.randint(0, n_base - 1)
        old_color = colors_new[idx]
        new_color = 1 - old_color
        colors_new[idx] = new_color
        
        new_violations = total_violations(colors_new)
        
        if new_violations < best_violations:
            best_violations = new_violations
            best_colors = colors_new[:]
            if best_violations == 0:
                final_colors = colors_base + best_colors
                return {'n': n_new, 'colors': final_colors}
        else:
            # Revert if not better (strict hill climbing)
            colors_new[idx] = old_color
            
    # If no solution found for n=30, return n=29
    return {'n': n_base, 'colors': colors_base}
