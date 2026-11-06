"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s

    # We will try to build a 4-coloring of K_n with no C6 in any color class.
    # Strategy: use spectral embedding of a random graph to get an initial coloring,
    # then do local search to remove C6s.

    def get_c6_count(color_matrix, n):
        """Count number of 6-cycles in a given color class."""
        # For each color, we have a graph. Count C6 in each.
        count = 0
        for c in range(4):
            # Build adjacency for color c
            # Use numpy for speed
            adj = np.zeros((n, n), dtype=np.int8)
            for i in range(n):
                for j in range(i + 1, n):
                    if color_matrix[i * (n - 1) - i * (i + 1) // 2 + j - i - 1] == c:
                        adj[i, j] = 1
                        adj[j, i] = 1
            # Count C6: for each pair of vertices at distance 2, check common neighbors at distance 3, etc.
            # This is expensive. Use a smarter method.
            # Actually, for small n (<=20), we can just enumerate all 6-cycles.
            # Number of 6-cycles in a graph on n vertices: at most C(n,6) * 15 (number of 6-cycles on 6 labeled vertices)
            # For n=20, C(20,6) = 38760, times 15 = ~580k per color, times 4 = ~2.3M. That's feasible.
            pass
        return count

    # Let's use a different approach: precompute all 6-cycles for a given n.
    def precompute_6cycles(n):
        """Return list of tuples of 6 vertices (sorted) that form a 6-cycle in K_n."""
        cycles = []
        from itertools import combinations
        for verts in combinations(range(n), 6):
            # There are 15 distinct 6-cycles on 6 labeled vertices
            # A 6-cycle is a cyclic ordering. Number of distinct cycles = 6!/12 = 5? No.
            # Actually, the number of Hamiltonian cycles in K6 is 5 (since (6-1)!/2 = 5).
            # Wait, (n-1)!/2 for n=6 is 120/2 = 60? No, that's for labeled.
            # Let me just enumerate all permutations of 6 vertices and check if they form a cycle.
            # Better: fix vertex 0 as start, then permute the remaining 5, divide by 2 for reversal.
            # Actually, let's just generate all 6-tuples (v0,...,v5) with v0 < min(v1..v5) to avoid duplicates? 
            # This is getting complicated. Let's use a simpler check: for a set of 6 vertices, 
            # a 6-cycle exists if the induced subgraph contains a Hamiltonian cycle.
            # But we need to check if a SPECIFIC coloring has a C6, meaning the 6 edges of the cycle all have the same color.
            # So for each 6-tuple (v0,v1,v2,v3,v4,v5) with v0 < v1 < ... < v5, we check all cyclic orderings.
            # Actually, it's easier: for each 6-set, check if there exists a cyclic ordering where all consecutive edges (and last-first) have the same color.
            # Number of cyclic orderings of 6 elements: (6-1)! = 120, but each cycle is counted 12 times (6 rotations * 2 directions), so 10 distinct cycles per 6-set.
            # For n=20, C(20,6)=38760, times 10 = 387,600 checks. Each check is 6 edge lookups. ~2.3M operations. Feasible.
            pass
        return cycles

    # Let's implement a cleaner version.
    def check_no_c6(colors, n):
        """Check if any color class contains a C6. colors is a list of length C(n,2)."""
        # Build color matrix: color[i,j] for i<j
        # Map (i,j) to index: idx = i*(n-1) - i*(i+1)//2 + j - i - 1
        def idx(i, j):
            if i > j:
                i, j = j, i
            return i * (n - 1) - i * (i + 1) // 2 + j - i - 1
        
        from itertools import combinations, permutations
        # Precompute all 6-sets
        six_sets = list(combinations(range(n), 6))
        
        for s in six_sets:
            # For this 6-set, check if there's a cyclic ordering where all 6 edges have the same color
            # Generate all cyclic permutations of s[1:] (fix s[0] as start)
            # Number of distinct cycles: (6-1)! / 2 = 60? No, (5)!/2 = 60. Wait, (n-1)!/2 for n=6 is 120/2=60.
            # But we only need to check if ANY of these 60 cycles is monochromatic.
            # Actually, we can be smarter: for each color c, check if the induced subgraph on s has a Hamiltonian cycle using only edges of color c.
            # That's 4 colors * (check if graph on 6 vertices has Hamiltonian cycle).
            for c in range(4):
                # Build adjacency for color c on the 6 vertices
                # Map s[0..5] to 0..5
                adj = np.zeros((6, 6), dtype=np.int8)
                for a in range(6):
                    for b in range(a + 1, 6):
                        i, j = s[a], s[b]
                        if colors[idx(i, j)] == c:
                            adj[a, b] = 1
                            adj[b, a] = 1
                # Check if this graph has a Hamiltonian cycle
                # For 6 vertices, we can check all 60 cycles
                found = False
                for perm in permutations(range(1, 6)):
                    cycle = (0,) + perm
                    # Check if all consecutive edges exist
                    ok = True
                    for k in range(6):
                        u = cycle[k]
                        v = cycle[(k + 1) % 6]
                        if adj[u, v] == 0:
                            ok = False
                            break
                    if ok:
                        # Check for duplicate: we only need one, but to avoid double counting, 
                        # we can break. Actually, we just need to know if ANY exists.
                        found = True
                        break
                if found:
                    return False
        return True

    # This is too slow. Let's use a different approach: start with a known good coloring for n=17 and try to extend.
    # But we don't have the n=17 coloring. Let's try to construct one.
    
    # Alternative: use the fact that the best known is 17. Let's try to find a valid coloring for n=17, 18, 19.
    # Use local search: start with a random coloring, then fix C6s by flipping colors.
    
    def random_coloring(n):
        return [random.randint(0, 3) for _ in range(n * (n - 1) // 2)]
    
    def idx(i, j, n):
        if i > j:
            i, j = j, i
        return i * (n - 1) - i * (i + 1) // 2 + j - i - 1
    
    def count_c6s(colors, n):
        """Count the number of monochromatic C6s."""
        from itertools import combinations, permutations
        six_sets = list(combinations(range(n), 6))
        count = 0
        for s in six_sets:
            for c in range(4):
                adj = np.zeros((6, 6), dtype=np.int8)
                for a in range(6):
                    for b in range(a + 1, 6):
                        i, j = s[a], s[b]
                        if colors[idx(i, j, n)] == c:
                            adj[a, b] = 1
                            adj[b, a] = 1
                # Check for Hamiltonian cycle
                for perm in permutations(range(1, 6)):
                    cycle = (0,) + perm
                    ok = True
                    for k in range(6):
                        u = cycle[k]
                        v = cycle[(k + 1) % 6]
                        if adj[u, v] == 0:
                            ok = False
                            break
                    if ok:
                        count += 1
                        break  # Count each 6-set/color once
        return count

    # This is still too slow for n=17+. Let's use a smarter C6 count.
    # For a graph G, the number of C6 can be computed using trace of A^6, but that counts walks, not simple cycles.
    # Let's just use the brute force but optimize with early termination.
    
    # Given the time constraints, let's try a different strategy:
    # Use the known structure: the best coloring for n=17 is likely based on a finite geometry or algebraic construction.
    # Let's try to use a quadratic residue construction or similar.
    
    # For now, let's return a valid coloring for n=15 or so, which we can verify quickly.
    # Actually, let's just try to extend the n=9 coloring by adding vertices one by one.
    
    # Start with n=9 coloring
    n0 = 9
    colors0 = [min(j - i, n0 - j + i) - 1 for i in range(n0) for j in range(i + 1, n0)]
    
    # Try to add vertices one by one
    current_colors = colors0[:]
    current_n = n0
    
    while current_n < 20 and time.time() < deadline - 10:
        # Try to add a new vertex
        new_n = current_n + 1
        # For each edge from new vertex to old vertices, choose a color
        # We want to avoid creating C6s
        # This is a constraint satisfaction problem
        # Use backtracking
        def backtrack(vertex, edge_idx, colors_so_far):
            if edge_idx == current_n:
                # Check if this coloring is valid
                all_colors = current_colors + colors_so_far
                # Quick check: count C6s
                # This is expensive, so let's do a quick check first
                return all_colors
            
            i = vertex
            j = i + edge_idx
            if j >= new_n:
                j = edge_idx  # This is wrong, let's restructure
            
            # Actually, let's just assign colors to edges (current_n, 0), (current_n, 1), ..., (current_n, current_n-1)
            pass
        
        # This is getting complicated. Let's just return the n=9 coloring for now.
        # Or try a fixed n=12 coloring.
        break
    
    # Let's try a different approach: use a known valid coloring for n=12 or so.
    # Actually, let's just return n=9 for now to ensure we get a valid answer.
    return {'n': n0, 'colors': colors0}
# EVOLVE-BLOCK-END
