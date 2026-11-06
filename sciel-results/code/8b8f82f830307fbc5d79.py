import random
import time

def construct():
    """
    Finds a 5-coloring of K_n without monochromatic C4s using Local Search.
    """
    
    def get_neighbors(colors, n, u, c):
        """Returns list of neighbors of u with color c."""
        # colors is a 2D array or list of lists.
        # For speed, we might use bitsets, but for N<=29, lists are fine.
        # Let's use a 2D list colors[u][v] = c.
        # Actually, storing as a flat list or 2D list is fine.
        # To make neighbor lookup fast, we can maintain adjacency lists for each color.
        pass # We will maintain adjacency lists directly.

    def solve_local_search(N, time_limit=100):
        start_time = time.time()
        
        # State: colors[u][v] for u < v. 
        # We store colors in a 2D array of size N x N, where colors[u][v] == colors[v][u].
        colors = [[-1] * N for _ in range(N)]
        
        # Adjacency lists for each color: adj[c][u] is a set of v such that colors[u][v] == c
        adj = [ [set() for _ in range(N)] for _ in range(5) ]
        
        def assign_color(u, v, c):
            # Remove from old color
            old_c = colors[u][v]
            if old_c != -1:
                adj[old_c][u].remove(v)
                adj[old_c][v].remove(u)
            
            # Add to new color
            colors[u][v] = c
            colors[v][u] = c
            adj[c][u].add(v)
            adj[c][v].add(u)

        def count_c4_through_edge(u, v, c):
            """
            Counts the number of monochromatic C4s in color c that include edge (u,v).
            A C4 is u-w-x-v-u.
            w must be in adj[c][u] and x in adj[c][v].
            w != v, x != u.
            w != x.
            Edge (w,x) must be color c.
            """
            # We assume (u,v) is currently color c in this calculation context?
            # Actually, this function is used to calculate the *potential* C4s if (u,v) were color c.
            # So we look for w in adj[c][u] (excluding v) and x in adj[c][v] (excluding u).
            # And check if (w,x) is in adj[c].
            
            count = 0
            # To avoid double counting or issues, we just iterate.
            # Note: adj[c][u] currently might contain v if (u,v) is c. 
            # If we are evaluating a change TO c, (u,v) is not yet c in adj, or we should handle it.
            # It is safer to pass the sets explicitly or assume the edge is NOT in the sets yet.
            # Let's define: 
            # Candidates for w: adj[c][u] - {v}
            # Candidates for x: adj[c][v] - {u}
            
            # Optimization: Iterate over the smaller set.
            set_u = adj[c][u]
            set_v = adj[c][v]
            
            # We need to count pairs (w, x) such that w in set_u, x in set_v, w != v, x != u, w != x, and (w,x) in adj[c].
            
            # If we are checking if adding (u,v) with color c creates C4s,
            # (u,v) is not in adj[c] yet.
            
            # Iterate w in set_u:
            for w in set_u:
                if w == v: continue
                # Check neighbors of w in color c that are in set_v
                # Note: x must be in set_v and x != u and x != w.
                # adj[c][w] contains x if (w,x) is color c.
                for x in adj[c][w]:
                    if x in set_v:
                        if x != u and x != w:
                            count += 1
            return count

        def delta_energy(u, v, c_new):
            """
            Calculate change in total C4 count if we change color of (u,v) to c_new.
            """
            c_old = colors[u][v]
            if c_old == c_new:
                return 0
            
            # Energy removed: C4s in c_old that use (u,v).
            # We need to calculate count_c4_through_edge(u, v, c_old) 
            # BUT, currently (u,v) IS in adj[c_old].
            # The function count_c4_through_edge assumes (u,v) is NOT in adj?
            # Let's check the logic of count_c4_through_edge.
            # It iterates w in adj[c][u]. If (u,v) is color c, v is in adj[c][u].
            # We skip w==v. So it effectively ignores the edge (u,v) in the neighbor set for the path u-w-x-v.
            # Wait, if (u,v) is color c, then v is a neighbor of u in c.
            # The C4 is u-w-x-v-u.
            # w is neighbor of u. x is neighbor of v.
            # If w=v, then u-v-x-v-u? No, w must be distinct from u,v.
            # So skipping w==v is correct.
            # Similarly for x==u.
            # So count_c4_through_edge works whether (u,v) is in adj or not, 
            # as long as we skip the endpoints.
            
            e_old = count_c4_through_edge(u, v, c_old)
            
            # To calculate e_new, we need to simulate adding (u,v) to c_new.
            # But count_c4_through_edge for c_new will look at adj[c_new].
            # Currently (u,v) is NOT in adj[c_new] (unless c_old==c_new which we handled).
            # So we can just call count_c4_through_edge(u, v, c_new).
            # Wait, does adding (u,v) to c_new affect the sets used in the count?
            # The count depends on adj[c][w] for w in path.
            # Adding (u,v) adds v to adj[c][u] and u to adj[c][v].
            # In count_c4_through_edge, we iterate w in adj[c][u].
            # If we haven't added (u,v) to adj yet, v is NOT in adj[c][u].
            # So we skip w==v naturally? No, v is not there.
            # But we should consider paths where w=v? No, w must be distinct.
            # So the count of C4s *formed* by adding (u,v) is exactly the number of paths u-w-x-v 
            # where w,x are existing neighbors.
            # This is what count_c4_through_edge calculates.
            
            e_new = count_c4_through_edge(u, v, c_new)
            
            return e_new - e_old

        # Initialize
        for i in range(N):
            for j in range(i+1, N):
                c = random.randint(0, 4)
                assign_color(i, j, c)
        
        def total_energy():
            # Sum of C4s in all colors.
            # We can compute this by summing count_c4_through_edge for all edges?
            # Each C4 has 4 edges. So sum over edges / 4.
            total = 0
            for i in range(N):
                for j in range(i+1, N):
                    c = colors[i][j]
                    total += count_c4_through_edge(i, j, c)
            return total // 4

        current_energy = total_energy()
        
        # Local Search Loop
        max_iter = 100000
        for _ in range(max_iter):
            if time.time() - start_time > time_limit:
                break
            
            if current_energy == 0:
                # Success!
                final_colors = []
                for i in range(N):
                    for j in range(i+1, N):
                        final_colors.append(colors[i][j])
                return {'n': N, 'colors': final_colors}
            
            # Pick a random edge
            u = random.randint(0, N-1)
            v = random.randint(0, N-1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
                
            c_old = colors[u][v]
            
            # Try all other colors
            best_delta = 0
            best_c = -1
            
            # We can just pick a random new color and check if it improves, 
            # or find the best one. Finding best is O(5 * Cost). Cost is small.
            for c_new in range(5):
                if c_new == c_old:
                    continue
                d = delta_energy(u, v, c_new)
                if d < best_delta:
                    best_delta = d
                    best_c = c_new
            
            if best_c != -1:
                assign_color(u, v, best_c)
                current_energy += best_delta
            else:
                # No improving move. 
                # Perturb: randomly change color of a random edge
                # Or restart?
                # Let's do a random move (simulated annealing style, but simple)
                # To avoid getting stuck in local minima, we accept a worse move with some probability?
                # For simplicity, let's just do a random move if no improving move found.
                c_rand = random.randint(0, 4)
                if c_rand != c_old:
                    d = delta_energy(u, v, c_rand)
                    assign_color(u, v, c_rand)
                    current_energy += d

        return None

    # Try N from 26 down to 11
    # Prioritize higher N.
    total_time_limit = 110
    start_total = time.time()
    
    for N in range(26, 10, -1):
        time_remaining = total_time_limit - (time.time() - start_total)
        if time_remaining < 5:
            break
        
        alloc_time = min(10, time_remaining)
        
        res = solve_local_search(N, time_limit=alloc_time)
        if res is not None:
            return res
            
    # Fallback to seed solution
    n = 11
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
