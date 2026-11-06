import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    # Step 1: Construct Paley graph on 29 vertices
    q = 29
    squares = set(x * x % q for x in range(1, q))
    
    n_base = q
    # adj[i] is a bitmask of vertices connected to i with color 0
    adj = [0] * n_base
    for i in range(n_base):
        for j in range(i + 1, n_base):
            diff = (j - i) % q
            if diff in squares:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    best_n = n_base
    best_adj = adj[:]
    
    # Helper: check if adding vertex n with given color assignments to existing vertices
    # creates a K2,2,2 in either color class.
    # colors_new[v] = 0 or 1 for the edge (n, v)
    def check_extension(adj, n, colors_new):
        """
        Check if we can add a new vertex n connected to vertices 0..n-1 with colors_new[v].
        Returns True if valid (no K2,2,2 in either color).
        """
        # For each color c in {0, 1}:
        # The new vertex n has some set S_c = {v : colors_new[v] == c}
        # For a K2,2,2 in color c involving the new vertex:
        # The new vertex must be in one of the three parts.
        # Case: n is in part {n, a} where a is NOT in S_c (i.e., colors_new[a] != c)
        # Then the other two parts {b1,b2} and {c1,c2} must all be in S_c.
        # Conditions:
        #   - b1, b2, c1, c2 all in S_c
        #   - b1-b2 not color c, c1-c2 not color c
        #   - all cross edges b_i-c_j are color c
        #   - a is connected to b1,b2,c1,c2 with color c
        #   - a is NOT connected to n with color c (already ensured by a not in S_c)
        
        for c in (0, 1):
            S_c = [v for v in range(n) if colors_new[v] == c]
            if len(S_c) < 4:
                continue
            
            # For each candidate partner a (not in S_c):
            for a in range(n):
                if colors_new[a] == c:
                    continue
                # a must be connected to at least 4 vertices in S_c with color c
                # Find all v in S_c such that edge (a,v) is color c
                neighbors_in_Sc = []
                for v in S_c:
                    if c == 0:
                        if (adj[a] >> v) & 1:
                            neighbors_in_Sc.append(v)
                    else:
                        if not ((adj[a] >> v) & 1):
                            neighbors_in_Sc.append(v)
                
                if len(neighbors_in_Sc) < 4:
                    continue
                
                # Check if there exist 4 vertices in neighbors_in_Sc forming K2,2 in color c
                # i.e., partition into {b1,b2} and {c1,c2} such that:
                #   b1-b2 not color c, c1-c2 not color c, all cross edges color c
                # We can check all 4-subsets of neighbors_in_Sc
                # But that's expensive. Instead, for each pair (b1,b2) in neighbors_in_Sc
                # that is NOT connected in color c, check if there are 2 more vertices
                # in neighbors_in_Sc that are connected to both b1 and b2 in color c
                # and not connected to each other in color c.
                
                # Build a set for fast lookup
                nset = set(neighbors_in_Sc)
                
                # For efficiency, precompute for each vertex in neighbors_in_Sc
                # which other vertices in neighbors_in_Sc it's connected to in color c
                # and which it's not connected to.
                
                # Simpler: iterate over all pairs in neighbors_in_Sc
                m = len(neighbors_in_Sc)
                if m < 4:
                    continue
                
                # For each pair (i,j) in neighbors_in_Sc that is NOT an edge in color c:
                # count how many other vertices k in neighbors_in_Sc are connected to both i and j in color c
                # and not connected to each other... this is getting complex.
                
                # Alternative: for each 4-subset, check if it forms K2,2 in color c.
                # But 4-subsets of up to 28 elements is C(28,4) = 20475, times 29 partners, times 2 colors.
                # That's ~1.2M checks per extension attempt. Too slow for many attempts.
                
                # Better approach: for each pair (b1,b2) not connected in color c,
                # find common neighbors in S_c (connected to both in color c).
                # If there are at least 2 such common neighbors (c1,c2) that are not connected in color c,
                # we have a K2,2,2.
                
                # Precompute adjacency within S_c for color c
                # Actually, let's just do the pair iteration.
                
                for idx1 in range(m):
                    b1 = neighbors_in_Sc[idx1]
                    # Check if b1-b2 is not color c for each b2
                    for idx2 in range(idx1 + 1, m):
                        b2 = neighbors_in_Sc[idx2]
                        # Check if b1-b2 is NOT color c
                        if c == 0:
                            if (adj[b1] >> b2) & 1:
                                continue  # b1-b2 is color 0, skip
                        else:
                            if not ((adj[b1] >> b2) & 1):
                                continue  # b1-b2 is NOT color 1, skip
                        # b1-b2 is not color c, good.
                        # Now find c1, c2 in neighbors_in_Sc \ {b1, b2} such that:
                        #   c1, c2 both connected to b1 and b2 in color c
                        #   c1-c2 not color c
                        common = []
                        for idx3 in range(m):
                            if idx3 == idx1 or idx3 == idx2:
                                continue
                            v = neighbors_in_Sc[idx3]
                            # Check if v is connected to b1 in color c
                            if c == 0:
                                if not ((adj[v] >> b1) & 1):
                                    continue
                            else:
                                if (adj[v] >> b1) & 1:
                                    continue
                            # Check if v is connected to b2 in color c
                            if c == 0:
                                if not ((adj[v] >> b2) & 1):
                                    continue
                            else:
                                if (adj[v] >> b2) & 1:
                                    continue
                            common.append(v)
                        
                        if len(common) < 2:
                            continue
                        
                        # Check all pairs in common
                        for idxa in range(len(common)):
                            c1 = common[idxa]
                            for idxb in range(idxa + 1, len(common)):
                                c2 = common[idxb]
                                # Check c1-c2 not color c
                                if c == 0:
                                    if (adj[c1] >> c2) & 1:
                                        continue
                                else:
                                    if not ((adj[c1] >> c2) & 1):
                                        continue
                                # Found a K2,2,2!
                                return False
        
        return True
    
    # Step 2: Try to extend to 30 vertices
    # We'll try random assignments of colors to the 29 edges from the new vertex.
    # Use local search to minimize violations.
    
    n_new = 30
    # colors_new[v] for v in 0..28
    colors_new = [random.randint(0, 1) for _ in range(n_base)]
    
    # Define violation count: number of K2,2,2 subgraphs in either color involving the new vertex
    def count_violations(adj, n, colors_new):
        """Count K2,2,2 violations involving the new vertex."""
        violations = 0
        for c in (0, 1):
            S_c = [v for v in range(n) if colors_new[v] == c]
            if len(S_c) < 4:
                continue
            for a in range(n):
                if colors_new[a] == c:
                    continue
                neighbors_in_Sc = []
                for v in S_c:
                    if c == 0:
                        if (adj[a] >> v) & 1:
                            neighbors_in_Sc.append(v)
                    else:
                        if not ((adj[a] >> v) & 1):
                            neighbors_in_Sc.append(v)
                if len(neighbors_in_Sc) < 4:
                    continue
                m = len(neighbors_in_Sc)
                for idx1 in range(m):
                    b1 = neighbors_in_Sc[idx1]
                    for idx2 in range(idx1 + 1, m):
                        b2 = neighbors_in_Sc[idx2]
                        if c == 0:
                            if (adj[b1] >> b2) & 1:
                                continue
                        else:
                            if not ((adj[b1] >> b2) & 1):
                                continue
                        common = []
                        for idx3 in range(m):
                            if idx3 == idx1 or idx3 == idx2:
                                continue
                            v = neighbors_in_Sc[idx3]
                            if c == 0:
                                if not ((adj[v] >> b1) & 1):
                                    continue
                                if not ((adj[v] >> b2) & 1):
                                    continue
                            else:
                                if (adj[v] >> b1) & 1:
                                    continue
                                if (adj[v] >> b2) & 1:
                                    continue
                            common.append(v)
                        if len(common) < 2:
                            continue
                        for idxa in range(len(common)):
                            c1 = common[idxa]
                            for idxb in range(idxa + 1, len(common)):
                                c2 = common[idxb]
                                if c == 0:
                                    if (adj[c1] >> c2) & 1:
                                        continue
                                else:
                                    if not ((adj[c1] >> c2) & 1):
                                        continue
                                violations += 1
        return violations
    
    # Local search to minimize violations
    best_violations = count_violations(adj, n_base, colors_new)
    if best_violations == 0:
        # Found a valid 30-vertex coloring!
        # Construct the full adjacency
        adj30 = adj[:] + [0]
        for v in range(n_base):
            if colors_new[v] == 0:
                adj30[n_base] |= (1 << v)
                adj30[v] |= (1 << n_base)
        best_n = 30
        best_adj = adj30
        # Return immediately
        colors = []
        for i in range(best_n):
            for j in range(i + 1, best_n):
                if (best_adj[i] >> j) & 1:
                    colors.append(0)
                else:
                    colors.append(1)
        return {'n': best_n, 'colors': colors}
    
    # Simulated annealing
    T = 1.0
    cooling = 0.999
    current_violations = best_violations
    
    while time.time() < deadline - 5:
        # Pick a random vertex to flip
        v = random.randint(0, n_base - 1)
        old_color = colors_new[v]
        colors_new[v] = 1 - old_color
        
        new_violations = count_violations(adj, n_base, colors_new)
        
        if new_violations < current_violations or random.random() < pow(2.718281828, (current_violations - new_violations) / T):
            current_violations = new_violations
            if current_violations < best_violations:
                best_violations = current_violations
                if best_violations == 0:
                    adj30 = adj[:] + [0]
                    for v2 in range(n_base):
                        if colors_new[v2] == 0:
                            adj30[n_base] |= (1 << v2)
                            adj30[v2] |= (1 << n_base)
                    best_n = 30
                    best_adj = adj30
                    colors = []
                    for i in range(best_n):
                        for j in range(i + 1, best_n):
                            if (best_adj[i] >> j) & 1:
                                colors.append(0)
                            else:
                                colors.append(1)
                    return {'n': best_n, 'colors': colors}
        else:
            colors_new[v] = old_color
        
        T *= cooling
        if T < 0.01:
            T = 0.01
    
    # Return the best known (29-vertex) coloring
    colors = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if (best_adj[i] >> j) & 1:
                colors.append(0)
            else:
                colors.append(1)
    return {'n': best_n, 'colors': colors}
