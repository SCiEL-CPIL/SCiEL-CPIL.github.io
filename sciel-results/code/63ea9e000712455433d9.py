import time
import random
from itertools import combinations, product

def construct():
    start = time.time()
    deadline = start + 145  # Safety margin before 150s limit

    def has_c6(adj, n):
        """Check if graph with adjacency matrix adj contains a 6-cycle.
        Uses efficient enumeration of 6-subsets with Hamiltonian cycle check."""
        for combo in combinations(range(n), 6):
            # Quick degree check
            local_deg = [0] * 6
            local_adj = [[0] * 6 for _ in range(6)]
            for i in range(6):
                for j in range(i + 1, 6):
                    u, v = combo[i], combo[j]
                    if adj[u][v]:
                        local_adj[i][j] = 1
                        local_adj[j][i] = 1
                        local_deg[i] += 1
                        local_deg[j] += 1
            # Each vertex needs degree >= 2 for a Hamiltonian cycle
            if any(d < 2 for d in local_deg):
                continue
            # Check for Hamiltonian cycle: fix vertex 0, try permutations of 1-5
            # Use backtracking for efficiency
            found = _ham_cycle_check(local_adj)
            if found:
                return True
        return False

    def _ham_cycle_check(adj6):
        """Check if 6-vertex graph has a Hamiltonian cycle. Fix vertex 0."""
        # Try all 5! = 120 permutations of vertices 1-5
        # But use pruning: vertex 0 must connect to first and last
        # Neighbors of 0
        nbrs0 = [i for i in range(1, 6) if adj6[0][i]]
        if len(nbrs0) < 2:
            return False
        for first in nbrs0:
            for last in nbrs0:
                if last == first:
                    continue
                # Need path first -> a -> b -> c -> last using remaining vertices
                remaining = [i for i in range(1, 6) if i != first and i != last]
                if len(remaining) != 3:
                    continue
                # Try all 3! = 6 orderings of remaining
                for a in remaining:
                    if not adj6[first][a]:
                        continue
                    for b in remaining:
                        if b == a or not adj6[a][b]:
                            continue
                        for c in remaining:
                            if c == a or c == b:
                                continue
                            if adj6[b][c] and adj6[c][last]:
                                return True
        return False

    def verify_coloring(n, colors):
        """Verify that no color class contains a C6."""
        num_edges = n * (n - 1) // 2
        if len(colors) != num_edges:
            return False
        for c in range(4):
            adj = [[0] * n for _ in range(n)]
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    if colors[idx] == c:
                        adj[i][j] = 1
                        adj[j][i] = 1
                    idx += 1
            if has_c6(adj, n):
                return False
        return True

    def quadratic_residues_mod_p(p):
        """Return set of quadratic residues mod p (excluding 0)."""
        return {(x * x) % p for x in range(1, p)}

    def construct_finite_field_coloring(n):
        """
        Construct a 4-coloring using finite field / group structure.
        For n = p (prime), use Z_p and partition differences into 4 classes.
        
        Strategy for C6-free: We need each color class to be C6-free.
        A sufficient condition: each color class is a subgraph of a graph with
        no C6. We can use the fact that if a graph has maximum degree d,
        it can still have C6s, so we need more structure.
        
        Better approach: Use the Paley-type construction.
        For p ≡ 1 (mod 4), the Paley graph is self-complementary.
        We partition the non-zero elements of Z_p into 4 sets such that
        each set generates a C6-free graph.
        
        Key insight: If we partition differences into sets S_0, S_1, S_2, S_3
        where each S_i is "C6-free" in the sense that the circulant graph
        with connection set S_i has no C6, then we're done.
        
        For n=17: Z_17, non-zero elements are 1..16.
        QR(17) = {1, 2, 4, 8, 9, 13, 15, 16} (8 elements)
        NQR(17) = {3, 5, 6, 7, 10, 11, 12, 14} (8 elements)
        
        We need to split these 16 elements into 4 sets of 4 each,
        such that each set's circulant graph has no C6.
        
        Let's try: split QR into two sets of 4, NQR into two sets of 4.
        """
        if n != 17:
            return None
        
        # Z_17 structure
        qr = sorted(quadratic_residues_mod_p(17))  # {1,2,4,8,9,13,15,16}
        nqr = sorted(set(range(1, 17)) - set(qr))   # {3,5,6,7,10,11,12,14}
        
        # We need to assign each difference d (1..16) a color 0..3.
        # The edge (i, j) has difference (j-i) mod 17.
        # Color class c contains edges where diff mod 17 is in S_c.
        # For the circulant graph with connection set S_c to be C6-free,
        # we need no 6-cycle in it.
        
        # Let's try various partitions and verify.
        # Start with a structured partition:
        # Group by "order" in the multiplicative group.
        # Generator of Z_17*: 3 is a primitive root mod 17.
        # Powers of 3: 3^1=3, 3^2=9, 3^3=10, 3^4=13, 3^5=5, 3^6=15, 3^7=11,
        #               3^8=16, 3^9=14, 3^10=8, 3^11=7, 3^12=4, 3^13=12, 3^14=2, 3^15=6, 3^16=1
        
        # Assign colors based on which "quarter" of the multiplicative order:
        # Color 0: powers 1-4 (3, 9, 10, 13)
        # Color 1: powers 5-8 (5, 15, 11, 16)
        # Color 2: powers 9-12 (14, 8, 7, 4)
        # Color 3: powers 13-16 (12, 2, 6, 1)
        
        diff_to_color = {}
        powers = [3, 9, 10, 13, 5, 15, 11, 16, 14, 8, 7, 4, 12, 2, 6, 1]
        for i, p in enumerate(powers):
            diff_to_color[p] = i // 4  # 0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3
        
        # Build coloring
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                d = (j - i) % n
                # d is in 1..16
                colors.append(diff_to_color[d])
        
        return colors

    def construct_cyclic_search(n, max_time):
        """Systematic search over cyclic colorings with pruning."""
        half = (n - 1) // 2
        t = time.time()
        
        # For n=17, half=8, 4^8=65536 possible assignments
        # Use smart ordering: try assignments where each color gets ~2 distances
        
        # Precompute: for each assignment, we need to check C6-freeness
        # Speed up by checking each color class separately
        
        # Try structured assignments first
        # Each color gets exactly 2 distances (for n=17, 8 distances / 4 colors = 2 each)
        
        from itertools import permutations
        
        # Generate all ways to assign 8 distances to 4 colors, 2 each
        # This is: choose 2 for color 0, 2 for color 1, 2 for color 2, 2 for color 3
        # Number: 8! / (2!^4) = 40320 / 16 = 2520
        
        best_result = None
        best_n = 0
        
        for n_target in [17, 16, 15]:
            if time.time() > deadline - 10:
                break
            
            half_t = (n_target - 1) // 2
            
            # For even n, handle the n/2 distance specially
            if n_target % 2 == 0:
                # n_target/2 is a special distance
                half_t = n_target // 2  # includes the n/2 distance
                # We have half_t distances to assign
                # For n=16: 8 distances, 4 colors -> 2 each
            else:
                half_t = (n_target - 1) // 2
            
            # Try balanced assignments (each color gets ~half_t/4 distances)
            target_per_color = half_t // 4
            remainder = half_t % 4
            
            # Generate balanced color assignments
            # For efficiency, use recursive generation with pruning
            
            def gen_balanced_assignments(distances, colors, counts, current):
                """Generate assignments where color c gets counts[c] distances."""
                if not distances:
                    if all(cnt == 0 for cnt in counts):
                        yield current[:]
                    return
                d = distances[0]
                for c in range(4):
                    if counts[c] > 0:
                        counts[c] -= 1
                        current.append(c)
                        yield from gen_balanced_assignments(distances[1:], colors, counts, current)
                        current.pop()
                        counts[c] += 1
            
            distances = list(range(1, half_t + 1))
            base_counts = [target_per_color] * 4
            for i in range(remainder):
                base_counts[i] += 1
            
            count = 0
            for assignment in gen_balanced_assignments(distances, [0,1,2,3], base_counts[:], []):
                if time.time() > deadline - 5:
                    return best_result, best_n
                count += 1
                
                # Build coloring for this assignment
                colors_list = []
                for i in range(n_target):
                    for j in range(i + 1, n_target):
                        d = j - i
                        d_min = min(d, n_target - d)
                        # d_min is in 1..half_t
                        colors_list.append(assignment[d_min - 1])
                
                if verify_coloring(n_target, colors_list):
                    return colors_list, n_target
                
                if count >= 5000:
                    break
            
            # Also try the finite field construction for n=17
            if n_target == 17:
                ff_colors = construct_finite_field_coloring(17)
                if ff_colors is not None and verify_coloring(17, ff_colors):
                    return ff_colors, 17
        
        return best_result, best_n

    def local_search_improved(n, max_time):
        """
        Improved local search using simulated annealing.
        Objective: minimize number of C6s across all color classes.
        """
        random.seed(12345)
        num_edges = n * (n - 1) // 2
        
        # Initialize with a structured coloring
        colors = [(i * 7 + j * 3) % 4 for i in range(n) for j in range(i + 1, n)]
        
        def build_adj_matrices(colors):
            adjs = []
            for c in range(4):
                adj = [[0] * n for _ in range(n)]
                idx = 0
                for i in range(n):
                    for j in range(i + 1, n):
                        if colors[idx] == c:
                            adj[i][j] = 1
                            adj[j][i] = 1
                        idx += 1
                adjs.append(adj)
            return adjs
        
        def count_c6s_in_adj(adj, n):
            """Count 6-cycles in a single graph."""
            count = 0
            for combo in combinations(range(n), 6):
                local_adj = [[0] * 6 for _ in range(6)]
                for i in range(6):
                    for j in range(i + 1, 6):
                        u, v = combo[i], combo[j]
                        if adj[u][v]:
                            local_adj[i][j] = 1
                            local_adj[j][i] = 1
                # Quick degree check
                degs = [sum(local_adj[i]) for i in range(6)]
                if any(d < 2 for d in degs):
                    continue
                if _ham_cycle_check(local_adj):
                    count += 1
            return count
        
        def total_c6s(colors):
            adjs = build_adj_matrices(colors)
            return sum(count_c6s_in_adj(adj, n) for adj in adjs)
        
        current_c6s = total_c6s(colors)
        best_colors = colors[:]
        best_c6s = current_c6s
        
        t = time.time()
        temperature = 10.0
        cooling = 0.995
        
        while time.time() - t < max_time:
            # Pick a random edge and change its color
            edge_idx = random.randint(0, num_edges - 1)
            old_c = colors[edge_idx]
            new_c = random.randint(0, 3)
            if new_c == old_c:
                continue
            
            colors[edge_idx] = new_c
            new_c6s = total_c6s(colors)
            
            delta = new_c6s - current_c6s
            
            if delta <= 0 or random.random() < pow(2.718281828, -delta / max(temperature, 0.01)):
                current_c6s = new_c6s
                if new_c6s < best_c6s:
                    best_c6s = new_c6s
                    best_colors = colors[:]
                    if best_c6s == 0:
                        return best_colors
            else:
                colors[edge_idx] = old_c
            
            temperature *= cooling
        
        return best_colors

    # Main strategy:
    # 1. Try the finite field construction for n=17
    # 2. Try systematic cyclic search for n=17, 16
    # 3. Try improved local search for n=17
    # 4. Fallback to smaller n
    
    # Step 1: Finite field construction
    ff_colors = construct_finite_field_coloring(17)
    if ff_colors is not None and verify_coloring(17, ff_colors):
        return {'n': 17, 'colors': ff_colors}
    
    # Step 2: Systematic cyclic search
    cyclic_result, cyclic_n = construct_cyclic_search(17, deadline - time.time() - 30)
    if cyclic_result is not None and cyclic_n >= 15:
        return {'n': cyclic_n, 'colors': cyclic_result}
    
    # Step 3: Local search for n=17
    remaining = deadline - time.time() - 15
    if remaining > 20:
        ls_colors = local_search_improved(17, remaining)
        if ls_colors is not None and verify_coloring(17, ls_colors):
            return {'n': 17, 'colors': ls_colors}
    
    # Step 4: Try n=16 with local search
    remaining = deadline - time.time() - 10
    if remaining > 15:
        ls_colors = local_search_improved(16, remaining)
        if ls_colors is not None and verify_coloring(16, ls_colors):
            return {'n': 16, 'colors': ls_colors}
    
    # Step 5: Fallback - construct a valid coloring for small n
    # n=9: simple cyclic coloring
    n = 9
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            d = j - i
            d_min = min(d, n - d)
            colors.append((d_min - 1) % 4)
    
    # Verify and if valid, return
    if verify_coloring(n, colors):
        return {'n': n, 'colors': colors}
    
    # Even smaller fallback
    n = 8
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            d = j - i
            d_min = min(d, n - d)
            colors.append((d_min - 1) % 4)
    return {'n': n, 'colors': colors}

