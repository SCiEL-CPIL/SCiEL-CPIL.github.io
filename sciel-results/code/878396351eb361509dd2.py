import itertools
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    def has_c6_in_color(adj, n):
        """
        Checks if the graph defined by adjacency matrix 'adj' (size n x n)
        contains a 6-cycle (C6).
        """
        # Optimization: If any vertex has degree < 2, no C6 can pass through it.
        # But C6 needs 6 vertices.
        # We check all 6-subsets.
        for combo in itertools.combinations(range(n), 6):
            # Build local adjacency for the 6 vertices
            # Map combo indices to 0..5
            local_adj = [[0]*6 for _ in range(6)]
            for i in range(6):
                for j in range(i+1, 6):
                    u = combo[i]
                    v = combo[j]
                    if adj[u][v]:
                        local_adj[i][j] = 1
                        local_adj[j][i] = 1
            
            # Check degrees in local subgraph
            valid_candidate = True
            for i in range(6):
                deg = sum(local_adj[i])
                if deg < 2:
                    valid_candidate = False
                    break
            
            if not valid_candidate:
                continue
                
            # Check for Hamiltonian cycle in the 6-vertex subgraph
            # Fix vertex 0.
            # Try all permutations of 1..5
            # 5! = 120 permutations
            found = False
            for p in itertools.permutations(range(1, 6)):
                # Cycle: 0 - p[0] - p[1] - p[2] - p[3] - p[4] - 0
                if (local_adj[0][p[0]] and 
                    local_adj[p[0]][p[1]] and 
                    local_adj[p[1]][p[2]] and 
                    local_adj[p[2]][p[3]] and 
                    local_adj[p[3]][p[4]] and 
                    local_adj[p[4]][0]):
                    found = True
                    break
            if found:
                return True
        return False

    def verify_coloring(n, colors):
        """
        Verifies if the given coloring of K_n has no C6 in any color class.
        colors: list of length n*(n-1)//2
        """
        # Reconstruct adjacency for each color
        for c in range(4):
            adj = [[0]*n for _ in range(n)]
            idx = 0
            for i in range(n):
                for j in range(i+1, n):
                    if colors[idx] == c:
                        adj[i][j] = 1
                        adj[j][i] = 1
                    idx += 1
            
            if has_c6_in_color(adj, n):
                return False
        return True

    def generate_coloring_from_partition(n, partition):
        """
        Generates a coloring for K_n based on a partition of distances.
        partition: list of 4 sets, each containing distances 1..(n-1)//2
        """
        colors = []
        max_dist = (n - 1) // 2
        # Map distance to color
        dist_to_color = {}
        for c, dists in enumerate(partition):
            for d in dists:
                if 1 <= d <= max_dist:
                    dist_to_color[d] = c
        
        for i in range(n):
            for j in range(i+1, n):
                d = abs(j - i)
                d = min(d, n - d)
                if d in dist_to_color:
                    colors.append(dist_to_color[d])
                else:
                    # Should not happen if partition covers all distances
                    colors.append(0)
        return colors

    # Try to find a valid coloring for n=17
    n = 17
    max_dist = 8
    all_dists = list(range(1, max_dist + 1))
    
    # List of candidate partitions of {1..8} into 4 subsets
    candidates = [
        [{1, 4}, {2, 5}, {3, 6}, {7, 8}],
        [{1, 2}, {3, 4}, {5, 6}, {7, 8}],
        [{1, 3}, {2, 5}, {4, 6}, {7, 8}],
        [{1, 5}, {2, 6}, {3, 7}, {4, 8}],
        [{1, 6}, {2, 7}, {3, 8}, {4, 5}],
        [{1, 7}, {2, 8}, {3, 5}, {4, 6}],
        [{1, 8}, {2, 4}, {3, 5}, {6, 7}],
        [{1, 2, 3}, {4, 5}, {6, 7}, {8}],
        [{1, 4, 7}, {2, 5, 8}, {3, 6}, set()],
        [{1, 3, 5}, {2, 4, 6}, {7}, {8}],
        [{1, 2, 4}, {3, 5, 7}, {6}, {8}],
        [{1, 3, 6}, {2, 4, 8}, {5}, {7}],
        [{1, 4, 8}, {2, 3, 6}, {5}, {7}],
        [{1, 2, 8}, {3, 4, 7}, {5}, {6}],
        [{1, 5, 8}, {2, 3, 7}, {4}, {6}],
    ]
    
    best_result = None
    
    # Test hardcoded candidates
    for part in candidates:
        if time.time() > deadline - 10:
            break
        colors = generate_coloring_from_partition(n, part)
        if verify_coloring(n, colors):
            best_result = {'n': n, 'colors': colors}
            break

    # If no hardcoded candidate worked, try random partitions
    if best_result is None:
        # Random search over partitions
        # There are 4^8 = 65536 possible assignments of distances to colors
        # We can sample some
        attempts = 0
        while time.time() < deadline - 5 and attempts < 500:
            attempts += 1
            # Randomly assign each distance 1..8 to a color 0..3
            assignment = [random.randint(0, 3) for _ in range(max_dist)]
            part = [set() for _ in range(4)]
            for d, c in zip(all_dists, assignment):
                part[c].add(d)
            
            colors = generate_coloring_from_partition(n, part)
            if verify_coloring(n, colors):
                best_result = {'n': n, 'colors': colors}
                break

    if best_result is not None:
        return best_result

    # Fallback to n=9 which is known to be valid
    n = 9
    colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
    return {'n': n, 'colors': colors}
