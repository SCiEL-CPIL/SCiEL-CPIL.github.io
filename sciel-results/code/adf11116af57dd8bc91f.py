"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations, permutations

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill

    def k9_fallback():
        n = 9
        return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}

    def count_c6_adj(adj, n):
        """Count C6s in a graph given as adjacency list. Returns count."""
        # Use bitsets for speed
        adj_bits = [0] * n
        for i in range(n):
            for j in adj[i]:
                adj_bits[i] |= (1 << j)
        
        count = 0
        # Count paths of length 5 that close to form C6
        # For each ordered pair (a0, a5) where a5 is neighbor of a0,
        # count paths a0-a1-a2-a3-a4-a5 of length 5
        # Then each C6 is counted 12 times (6 rotations * 2 directions)
        # Actually let's just check existence for pruning
        
        # Faster: check if there's any C6 by looking for paths
        for a0 in range(n):
            for a1 in adj[a0]:
                if a1 <= a0:
                    continue
                for a2 in adj[a1]:
                    if a2 == a0:
                        continue
                    for a3 in adj[a2]:
                        if a3 in (a0, a1):
                            continue
                        for a4 in adj[a3]:
                            if a4 in (a0, a1, a2):
                                continue
                            for a5 in adj[a4]:
                                if a5 in (a0, a1, a2, a3):
                                    continue
                                if a0 in adj[a5]:
                                    return True
        return False

    def has_c6(color_edges, n):
        """Check if a set of edges contains a C6."""
        adj = [set() for _ in range(n)]
        for (u, v) in color_edges:
            adj[u].add(v)
            adj[v].add(u)
        return count_c6_adj(adj, n)

    def check_no_c6(color_assignment, n):
        """Check that no colour class contains a C6."""
        for c in range(4):
            edges_c = []
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    if color_assignment[idx] == c:
                        edges_c.append((i, j))
                    idx += 1
            if has_c6(edges_c, n):
                return False
        return True

    def try_cyclic_construction(n):
        """Try cyclic constructions on Z_n."""
        if n % 2 == 0:
            return None
        
        # Partition {1, ..., n-1} into 4 sets
        # For n=17, partition {1,...,16} into 4 sets of 4
        elements = list(range(1, n))
        
        # Try all partitions? Too many. Try random ones.
        for attempt in range(2000):
            random.shuffle(elements)
            sets = [elements[i*4:(i+1)*4] for i in range(4)]
            
            # Check each color class for C6
            valid = True
            for c in range(4):
                S = sets[c]
                # Build adjacency: i connected to i+s and i-s for s in S
                adj = [set() for _ in range(n)]
                for i in range(n):
                    for s in S:
                        j = (i + s) % n
                        k = (i - s) % n
                        adj[i].add(j)
                        adj[i].add(k)
                if count_c6_adj(adj, n):
                    valid = False
                    break
            
            if valid:
                # Build color assignment
                colors = []
                for i in range(n):
                    for j in range(i + 1, n):
                        d = (j - i) % n
                        # Find which set d or n-d is in
                        for c in range(4):
                            if d in sets[c] or (n - d) in sets[c]:
                                colors.append(c)
                                break
                return colors
        return None

    def local_search(n, max_iters=50000):
        """Local search to find C6-free 4-coloring."""
        colors = [random.randint(0, 3) for _ in range(n * (n - 1) // 2)]
        
        def count_conflicts(color_assignment):
            """Count total C6s across all color classes."""
            total = 0
            for c in range(4):
                edges_c = []
                idx = 0
                for i in range(n):
                    for j in range(i + 1, n):
                        if color_assignment[idx] == c:
                            edges_c.append((i, j))
                        idx += 1
                adj = [set() for _ in range(n)]
                for (u, v) in edges_c:
                    adj[u].add(v)
                    adj[v].add(u)
                if count_c6_adj(adj, n):
                    total += 1
            return total
        
        best_colors = colors[:]
        best_conflicts = count_conflicts(colors)
        
        if best_conflicts == 0:
            return colors
        
        current_conflicts = best_conflicts
        temp = 10.0
        cooling = 0.999
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Pick a random edge and flip its color
            pos = random.randint(0, len(colors) - 1)
            old_c = colors[pos]
            new_c = random.randint(0, 3)
            if new_c == old_c:
                continue
            colors[pos] = new_c
            
            new_conflicts = count_conflicts(colors)
            
            if new_conflicts < current_conflicts or random.random() < (temp ** (new_conflicts - current_conflicts)):
                current_conflicts = new_conflicts
                if new_conflicts < best_conflicts:
                    best_conflicts = new_conflicts
                    best_colors = colors[:]
                    if best_conflicts == 0:
                        return best_colors
            else:
                colors[pos] = old_c
            
            temp *= cooling
        
        return best_colors if best_conflicts == 0 else None

    # Try increasing n from 17 downward to find the best valid coloring
    # Start with known best (17) and try to find it
    
    # First, try cyclic construction for n=17
    for n_try in [17, 16, 15, 14, 13, 12, 11, 10]:
        if time.time() > deadline - 10:
            break
        
        # Try cyclic construction
        if n_try % 2 == 1:
            colors = try_cyclic_construction(n_try)
            if colors is not None and check_no_c6(colors, n_try):
                return {'n': n_try, 'colors': colors}
        
        # Try local search
        for restart in range(3):
            if time.time() > deadline - 5:
                break
            colors = local_search(n_try, max_iters=20000)
            if colors is not None and check_no_c6(colors, n_try):
                return {'n': n_try, 'colors': colors}
    
    # Fallback to K9
    return k9_fallback()
# EVOLVE-BLOCK-END
