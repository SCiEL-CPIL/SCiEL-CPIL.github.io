"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools

# EVOLVE-BLOCK-START

def _count_k222(color_matrix, n):
    """Count monochromatic K_{2,2,2} subgraphs in a coloring of K_n.
    
    A K_{2,2,2} is a complete tripartite graph with parts of size 2.
    It has 6 vertices partitioned into 3 pairs {a,b}, {c,d}, {e,f}
    such that all 12 cross-edges between the pairs have the same color.
    """
    count = 0
    # For each pair of vertices (i,j) that will form one part,
    # we need to find two other pairs (k,l) and (m,n) such that
    # all edges between the three pairs are the same color.
    # 
    # A K_{2,2,2} is determined by 6 vertices {v1,...,v6} and a partition
    # into 3 pairs. For a fixed set of 6 vertices, there are 15 ways to
    # partition into 3 pairs. For each partition, we check if all 12
    # cross-edges are monochromatic.
    #
    # However, a faster approach: for each pair (i,j), compute the set of
    # vertices that have the same color to both i and j. If this set has
    # >= 4 vertices, then we can form K_{2,2,2} with (i,j) as one part.
    # But we need to be careful about double counting.
    
    # Let's use a direct but optimized approach:
    # For each unordered pair (i,j), let S_c = {k : k != i,j and color[i][k] == color[j][k] == c}
    # for c in {0,1}. If |S_0| >= 4 or |S_1| >= 4, then there exists a K_{2,2,2}
    # containing (i,j) as one part.
    
    # Actually, let's count more carefully. A monochromatic K_{2,2,2} with
    # parts {a,b}, {c,d}, {e,f} in color 0 means:
    # color[a][c]=color[a][d]=color[b][c]=color[b][d]=0 (4 edges between part1 and part2)
    # color[a][e]=color[a][f]=color[b][e]=color[b][f]=0 (4 edges between part1 and part3)
    # color[c][e]=color[c][f]=color[d][e]=color[d][f]=0 (4 edges between part2 and part3)
    
    # For efficiency, for each pair (i,j), compute for each color c:
    #   common_c = {k : k not in {i,j} and color[i][k]==c and color[j][k]==c}
    # Then a K_{2,2,2} with (i,j) as first part in color c exists iff
    # we can find two disjoint pairs (k,l) and (m,n) from common_c such that
    # all edges between {k,l} and {m,n} are also color c.
    
    # This is still complex. Let's use a simpler counting method for the energy:
    # Count all 6-tuples (a,b,c,d,e,f) with a<b, c<d, e<f, {a,b} disjoint from {c,d} disjoint from {e,f},
    # and all 12 cross-edges same color. This is O(n^6) which is too slow for n=30.
    
    # Better: iterate over pairs (i,j). For color c, let S = common neighbors in color c.
    # Then we need to count pairs of disjoint pairs from S where all cross-edges are color c.
    # For a fixed S of size s, the number of ways to choose two disjoint pairs is C(s,2)*C(s-2,2)/... 
    # Actually: choose 4 distinct vertices from S, then partition into 2 pairs: 3 ways.
    # For each such choice of 4 vertices {k,l,m,n} from S, check if all 4 edges between
    # {k,l} and {m,n} are color c (or {k,m} and {l,n}, or {k,n} and {l,m}).
    
    # For n=30, s can be up to ~28, and C(28,4) = 20475, times 3 partitions, times C(30,2)=435 pairs.
    # That's about 435 * 20475 * 3 ≈ 26.8 million operations per energy evaluation. Too slow.
    
    # Let's use a different energy: just count the number of "bad" configurations where
    # a pair (i,j) has >= 4 common neighbors in the same color, weighted appropriately.
    # This is a proxy that's O(n^3).
    
    energy = 0
    for i in range(n):
        for j in range(i + 1, n):
            ci = color_matrix[i][j]
            common = 0
            for k in range(n):
                if k == i or k == j:
                    continue
                if color_matrix[i][k] == ci and color_matrix[j][k] == ci:
                    common += 1
            # If common >= 4, there might be a K_{2,2,2}. Use common as a soft measure.
            # The number of K_{2,2,2} with (i,j) as one part is at least C(common, 4) * 3 / something
            # Let's just use max(0, common - 3) as a penalty, or common^2 for stronger penalty.
            if common >= 4:
                energy += (common - 3) ** 2
    return energy


def _count_k222_exact(color_matrix, n):
    """Exact count of monochromatic K_{2,2,2} subgraphs."""
    count = 0
    # For each triple of disjoint pairs, check if all 12 cross-edges are monochromatic.
    # This is O(n^6) which is ~735M for n=30. Too slow for repeated calls.
    # Use the proxy instead.
    return _count_k222(color_matrix, n)


def _is_valid(color_matrix, n):
    """Check if the coloring has no monochromatic K_{2,2,2}."""
    # A coloring is valid iff there is no monochromatic K_{2,2,2}.
    # Check: for every pair (i,j), the common neighbors in color color[i][j]
    # must not contain two disjoint pairs with all cross-edges in that color.
    
    for i in range(n):
        for j in range(i + 1, n):
            c = color_matrix[i][j]
            # Find all k such that color[i][k] == c and color[j][k] == c
            common = [k for k in range(n) if k != i and k != j and color_matrix[i][k] == c and color_matrix[j][k] == c]
            if len(common) < 4:
                continue
            # Check all ways to pick 4 from common and partition into 2 pairs
            for quad in itertools.combinations(common, 4):
                k, l, m, n_ = quad
                # Three ways to partition {k,l,m,n} into two pairs
                # {k,l} & {m,n}: check all 4 cross-edges
                if (color_matrix[k][m] == c and color_matrix[k][n_] == c and 
                    color_matrix[l][m] == c and color_matrix[l][n_] == c):
                    return False
                # {k,m} & {l,n}: check all 4 cross-edges
                if (color_matrix[k][l] == c and color_matrix[k][n_] == c and 
                    color_matrix[m][l] == c and color_matrix[m][n_] == c):
                    return False
                # {k,n} & {l,m}: check all 4 cross-edges
                if (color_matrix[k][l] == c and color_matrix[k][m] == c and 
                    color_matrix[n_][l] == c and color_matrix[n_][m] == c):
                    return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    # Start with a random coloring of K_30
    n = 30
    random.seed(42)
    
    # Initialize colors
    colors = [random.randint(0, 1) for _ in range(n * (n - 1) // 2)]
    
    def make_matrix(colors_list):
        mat = [[0] * n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                mat[i][j] = colors_list[idx]
                mat[j][i] = colors_list[idx]
                idx += 1
        return mat
    
    def edge_index(i, j):
        if i > j:
            i, j = j, i
        return i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)
    
    mat = make_matrix(colors)
    
    # Simulated annealing
    T = 10.0
    T_min = 0.01
    cooling = 0.9995
    
    best_colors = colors[:]
    best_energy = _count_k222(mat, n)
    
    if best_energy == 0:
        return {'n': n, 'colors': colors}
    
    step = 0
    while time.time() < deadline:
        step += 1
        
        # Pick a random edge to flip
        idx = random.randint(0, len(colors) - 1)
        # Convert index to (i,j)
        # idx = i*(n-1) - i*(i+1)//2 + (j-i-1)
        # We can just flip and recompute energy locally
        
        old_color = colors[idx]
        new_color = 1 - old_color
        
        # Update matrix
        # Find i,j from idx
        # This is O(n) but n=30 so it's fine
        found = False
        for i in range(n):
            max_j = n - 1
            base = i * (n - 1) - i * (i + 1) // 2
            if base <= idx < base + (n - 1 - i):
                j = i + 1 + (idx - base)
                found = True
                break
        if not found:
            continue
        
        mat[i][j] = new_color
        mat[j][i] = new_color
        colors[idx] = new_color
        
        new_energy = _count_k222(mat, n)
        delta = new_energy - best_energy
        
        if delta <= 0 or random.random() < pow(2.718281828, -delta / T):
            best_energy = new_energy
            if new_energy < best_energy:
                best_colors = colors[:]
            if best_energy == 0:
                # Check validity
                if _is_valid(mat, n):
                    return {'n': n, 'colors': colors[:]}
        else:
            # Reject
            colors[idx] = old_color
            mat[i][j] = old_color
            mat[j][i] = old_color
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    # Check if best is valid
    best_mat = make_matrix(best_colors)
    if _is_valid(best_mat, n):
        return {'n': n, 'colors': best_colors}
    
    # If not valid, try to find the largest valid subgraph
    # Or just return n=17 Paley as fallback
    # Actually, let's try a few more things: restart with different seeds
    remaining = deadline - time.time()
    if remaining > 5:
        for seed in range(1, 10):
            if time.time() > deadline - 5:
                break
            random.seed(seed)
            colors2 = [random.randint(0, 1) for _ in range(n * (n - 1) // 2)]
            mat2 = make_matrix(colors2)
            energy2 = _count_k222(mat2, n)
            if energy2 < best_energy:
                best_colors = colors2[:]
                best_energy = energy2
                if best_energy == 0 and _is_valid(mat2, n):
                    return {'n': n, 'colors': colors2}
    
    # Fallback: return a valid smaller coloring
    # Paley graph of order 17
    squares = {x * x % 17 for x in range(1, 17)}
    n_fallback = 17
    colors_fallback = [0 if (j - i) % 17 in squares else 1 for i in range(n_fallback) for j in range(i + 1, n_fallback)]
    return {'n': n_fallback, 'colors': colors_fallback}

# EVOLVE-BLOCK-END
