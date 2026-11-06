import time
import random
import sys

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 118  # safety margin before 120s

    def edge_index(i, j, n):
        # i < j, lexicographic order
        return i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)

    def build_colors(n, color_matrix):
        """Build flat color list from n x n matrix (color_matrix[i][j] for i<j)."""
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                colors.append(color_matrix[i][j])
        return colors

    def count_c6(n, colors):
        """Count the number of monochromatic 6-cycles in the coloring."""
        # Build adjacency bitmasks per color
        adj = [[0] * n for _ in range(4)]
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[edge_index(i, j, n)]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
        
        c6_count = 0
        # For each 6-tuple, check all 60 distinct cycle orderings
        # A 6-cycle on vertices {v0..v5} is a permutation where the cycle
        # visits all 6 vertices. Number of distinct 6-cycles on 6 labeled vertices = 60.
        # But we can be smarter: for each set of 6 vertices, check if any 6-cycle exists in one color.
        # Actually, let's just enumerate all ordered 6-tuples of distinct vertices and check.
        # That's P(17,6) = 17*16*15*14*13*12 = 5765760. Too many.
        
        # Better: for each color c, for each pair of non-adjacent-in-cycle vertices... 
        # Actually, the standard way: a C6 in color c is a sequence v0,v1,v2,v3,v4,v5 (distinct)
        # where (v0,v1),(v1,v2),(v2,v3),(v3,v4),(v4,v5),(v5,v0) are all color c.
        
        # For n=17, let's use the following: for each color, for each 6-element subset,
        # check if the induced subgraph contains a Hamiltonian cycle.
        # C(17,6) = 12376 subsets, times 4 colors = 49504 checks.
        # Each check: does the 6-vertex induced subgraph have a Hamiltonian cycle?
        # For 6 vertices, we can check all (6-1)!/2 = 60 cycle orderings.
        # Total: 49504 * 60 = ~3M edge checks. Each is a bit test. Should be fast enough.
        
        # Precompute all 6-element subsets
        from itertools import combinations
        subsets6 = list(combinations(range(n), 6))
        
        for c in range(4):
            adj_c = adj[c]
            for subset in subsets6:
                # Check if induced subgraph on subset has a Hamiltonian cycle in color c
                # Try all 60 distinct cycle orderings of the 6 vertices
                # A cycle ordering is a permutation p of the 6 vertices such that
                # p0-p1-p2-p3-p4-p5-p0 are all edges in color c.
                # We fix p0 = subset[0] to reduce symmetry, then permute the other 5.
                # (5-1)!/2 = 12 cycle orderings with p0 fixed? No, (6-1)!/2 = 60 total,
                # with p0 fixed: 5!/2 = 60? No.
                # Total distinct cycles on 6 labeled vertices: (6-1)!/2 = 60.
                # If we fix vertex 0 (subset[0]) to be first, we get 60 cycles still
                # (since each cycle can be rotated to start at any vertex, and reversed).
                # Actually: number of cycles = 60. If we fix which vertex is "first",
                # we get 60/6 = 10? No.
                # Let me just enumerate: fix v0 = subset[0]. Then we need to arrange
                # the other 5 in a cycle back to v0. That's (5-1)! = 24 linear arrangements
                # but since the cycle can go clockwise or counter-clockwise, it's 24/2 = 12? 
                # No: fixing v0, the cycle is v0 -> a -> b -> c -> d -> e -> v0.
                # The number of ways to arrange a,b,c,d,e is 5! = 120, but reversing
                # gives the same cycle, so 120/2 = 60. Wait, that's not right either.
                # 
                # Let me think again. A cycle on 6 vertices has 6!/(2*6) = 60 distinct cycles.
                # If I fix the first vertex to be subset[0], then I'm choosing a cyclic ordering
                # of the remaining 5 vertices. The number of cyclic orderings of 5 elements is
                # (5-1)! = 24. But each cycle through all 6 vertices, when you fix one vertex,
                # corresponds to exactly one cyclic ordering of the other 5 (up to reversal).
                # So 24/2 = 12? 
                # 
                # Actually no. Let's just be safe and check all 60.
                # For performance, let's check: fix v0 = subset[0], then permute the other 5.
                # For each permutation (a,b,c,d,e), check edges:
                # (v0,a), (a,b), (b,c), (c,d), (d,e), (e,v0)
                # This covers each cycle exactly twice (once for each direction), so we'll
                # count each C6 twice. That's fine for counting.
                
                v0 = subset[0]
                rest = subset[1:]
                # Generate permutations of rest
                # For performance, use a manual approach for 5 elements
                # 5! = 120 permutations. Check 6 edges each = 720 bit tests per subset per color.
                # Total: 12376 * 4 * 720 = ~35M bit tests. Might be slow in Python.
                
                # Let's optimize: for the 5 remaining vertices, we need a Hamiltonian path
                # from v0 through all 5 back to v0.
                # Equivalent: the induced subgraph on the 6 vertices must have a Hamiltonian cycle.
                
                # Quick check: each vertex must have degree >= 2 in the induced subgraph.
                degrees = [0] * 6
                for idx1 in range(6):
                    for idx2 in range(idx1 + 1, 6):
                        if (adj_c[subset[idx1]] >> subset[idx2]) & 1:
                            degrees[idx1] += 1
                            degrees[idx2] += 1
                if any(d < 2 for d in degrees):
                    continue
                
                # Now try to find a Hamiltonian cycle
                # Fix v0 = subset[0]. Try all permutations of the other 5.
                # Use itertools.permutations
                found = False
                for perm in __import__('itertools').permutations(rest):
                    # Check edges: v0-perm[0], perm[0]-perm[1], ..., perm[4]-v0
                    if ((adj_c[v0] >> perm[0]) & 1) and \
                       ((adj_c[perm[0]] >> perm[1]) & 1) and \
                       ((adj_c[perm[1]] >> perm[2]) & 1) and \
                       ((adj_c[perm[2]] >> perm[3]) & 1) and \
                       ((adj_c[perm[3]] >> perm[4]) & 1) and \
                       ((adj_c[perm[4]] >> v0) & 1):
                        found = True
                        break
                if found:
                    c6_count += 1  # This counts each C6 once per subset, but a subset
                                    # might have multiple C6s. For our purpose (checking if any exists),
                                    # we just need c6_count > 0.
        
        return c6_count

    def is_valid(n, colors):
        """Check if the coloring has no monochromatic C6."""
        return count_c6(n, colors) == 0

    # --- Fallback: cyclic construction for n=10 ---
    def cyclic_coloring(n):
        """Cyclic coloring: color of (i,j) = min(j-i, n-(j-i)) mod 4, adjusted."""
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                d = (j - i) % n
                d = min(d, n - d)
                c = d % 4
                colors.append(c)
        return colors

    # Verify cyclic coloring for n=10
    n_fallback = 10
    colors_fallback = cyclic_coloring(n_fallback)
    # Verify it's actually valid
    if not is_valid(n_fallback, colors_fallback):
        # Try different constructions
        for offset in range(4):
            colors_try = []
            for i in range(n_fallback):
                for j in range(i + 1, n_fallback):
                    d = (j - i) % n_fallback
                    d = min(d, n_fallback - d)
                    c = (d + offset) % 4
                    colors_try.append(c)
            if is_valid(n_fallback, colors_try):
                colors_fallback = colors_try
                break
        else:
            # Last resort: small n
            n_fallback = 6
            colors_fallback = [0] * (n_fallback * (n_fallback - 1) // 2)

    best_result = {'n': n_fallback, 'colors': colors_fallback}

    # --- Local search for n=17 ---
    def local_search(n, time_limit):
        """
        Local search to find a C6-free 4-coloring of K_n.
        Uses simulated annealing / hill climbing.
        """
        end_time = time.time() + time_limit
        num_edges = n * (n - 1) // 2
        
        # Initialize randomly
        colors = [random.randint(0, 3) for _ in range(num_edges)]
        
        # Build edge list for quick access
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        # We need a fast way to count C6s. Let's use a different approach:
        # Instead of counting all C6s, we can use a penalty-based approach.
        # But counting C6s is expensive. Let's use a smarter method.
        
        # Alternative: use the fact that a C6 exists iff there are two vertices
        # with a common path of length 5. But that's also complex.
        
        # For n=17, let's just do the full C6 count but optimize it.
        # Actually, for local search, we need to evaluate the objective frequently.
        # Let's use a different strategy: start from a known good structure and perturb.
        
        # Strategy: Start with a "blow-up" of a smaller valid coloring.
        # Or use a random start and do hill climbing with the full C6 count.
        
        # Given the time constraints, let's try a random restart local search
        # where we flip one edge's color at a time and accept if it reduces C6 count.
        
        # But the full C6 count is ~35M operations, which might take ~5-10 seconds in Python.
        # That's too slow for many iterations.
        
        # Alternative: use a faster C6 detection.
        # A C6 in color c exists iff there exist vertices v0,v1,...,v5 such that
        # all consecutive pairs (and v5-v0) are color c.
        # 
        # We can check: for each color c, compute the adjacency matrix A_c.
        # A C6 exists iff trace(A_c^6) > 0 (counts closed walks of length 6,
        # which includes C6s but also shorter cycles with backtracking).
        # This overcounts, but if trace(A_c^6) == 0, there's definitely no C6.
        # If trace(A_c^6) > 0, there might be a C6.
        # 
        # For n=17, matrix multiplication of 17x17 matrices is fast.
        # Let's use numpy for this.
        
        import numpy as np
        
        def count_c6_numpy(colors, n):
            """Count C6s using matrix powers (approximate - counts closed walks of length 6)."""
            total = 0
            for c in range(4):
                A = np.zeros((n, n), dtype=np.int64)
                for idx, (i, j) in enumerate(edges):
                    if colors[idx] == c:
                        A[i, j] = 1
                        A[j, i] = 1
                # Compute A^6
                A6 = A.copy()
                for _ in range(5):
                    A6 = A6 @ A
                # trace(A^6) counts closed walks of length 6
                # A C6 contributes 12 to the trace (6 vertices * 2 directions)
                # But shorter cycles with backtracking also contribute.
                # For our purpose, if trace(A^6) == 0, no C6. If > 0, possible C6.
                total += np.trace(A6)
            return total
        
        # Use the numpy-based count for local search, then verify with exact count.
        
        current_colors = colors[:]
        current_penalty = count_c6_numpy(current_colors, n)
        
        best_local = current_colors[:]
        best_local_penalty = current_penalty
        
        if current_penalty == 0 and is_valid(n, current_colors):
            return current_colors
        
        iteration = 0
        while time.time() < end_time:
            iteration += 1
            
            # Pick a random edge to flip
            edge_idx = random.randint(0, num_edges - 1)
            old_color = current_colors[edge_idx]
            new_color = (old_color + random.randint(1, 3)) % 4
            
            current_colors[edge_idx] = new_color
            new_penalty = count_c6_numpy(current_colors, n)
            
            if new_penalty < current_penalty:
                current_penalty = new_penalty
                if new_penalty == 0:
                    # Verify with exact count
                    if is_valid(n, current_colors):
                        return current_colors
                    # False positive from matrix method
                    current_colors[edge_idx] = old_color
                    current_penalty = count_c6_numpy(current_colors, n)
                if new_penalty < best_local_penalty:
                    best_local = current_colors[:]
                    best_local_penalty = new_penalty
            else:
                # Accept with some probability (simulated annealing)
                if new_penalty == current_penalty:
                    # Neutral move, accept with 50% probability
                    if random.random() > 0.5:
                        current_colors[edge_idx] = old_color
                        current_penalty = count_c6_numpy(current_colors, n)
                    # Keep the move
                else:
                    # Reject
                    current_colors[edge_idx] = old_color
            
            # Periodically check if we've found a valid solution
            if iteration % 1000 == 0:
                if best_local_penalty == 0:
                    if is_valid(n, best_local):
                        return best_local
            
            # Random restart if stuck
            if iteration % 5000 == 0:
                if best_local_penalty < current_penalty:
                    current_colors = best_local[:]
                    current_penalty = best_local_penalty
                else:
                    # Full random restart
                    current_colors = [random.randint(0, 3) for _ in range(num_edges)]
                    current_penalty = count_c6_numpy(current_colors, n)
                    if current_penalty < best_local_penalty:
                        best_local = current_colors[:]
                        best_local_penalty = current_penalty
        
        # Return best found
        if best_local_penalty == 0 and is_valid(n, best_local):
            return best_local
        return None

    # Try local search for n=17
    if time.time() < deadline - 5:
        remaining = deadline - time.time() - 5
        # Give most of the remaining time to n=17
        result_17 = local_search(17, remaining)
        if result_17 is not None:
            best_result = {'n': 17, 'colors': result_17}
            return best_result

    # Try n=16 if time remains
    if time.time() < deadline - 5:
        remaining = deadline - time.time() - 5
        result_16 = local_search(16, remaining)
        if result_16 is not None:
            best_result = {'n': 16, 'colors': result_16}
            return best_result

    # Try n=15 if time remains
    if time.time() < deadline - 5:
        remaining = deadline - time.time() - 5
        result_15 = local_search(15, remaining)
        if result_15 is not None:
            best_result = {'n': 15, 'colors': result_15}
            return best_result

    # Try n=14 if time remains
    if time.time() < deadline - 5:
        remaining = deadline - time.time() - 5
        result_14 = local_search(14, remaining)
        if result_14 is not None:
            best_result = {'n': 14, 'colors': result_14}
            return best_result

    # Try n=13 if time remains
    if time.time() < deadline - 5:
        remaining = deadline - time.time() - 5
        result_13 = local_search(13, remaining)
        if result_13 is not None:
            best_result = {'n': 13, 'colors': result_13}
            return best_result

    # Try n=12 if time remains
    if time.time() < deadline - 5:
        remaining = deadline - time.time() - 5
        result_12 = local_search(12, remaining)
        if result_12 is not None:
            best_result = {'n': 12, 'colors': result_12}
            return best_result

    # Try n=11 if time remains
    if time.time() < deadline - 5:
        remaining = deadline - time.time() - 5
        result_11 = local_search(11, remaining)
        if result_11 is not None:
            best_result = {'n': 11, 'colors': result_11}
            return best_result

    # Return best result found
    return best_result

# EVOLVE-BLOCK-END
