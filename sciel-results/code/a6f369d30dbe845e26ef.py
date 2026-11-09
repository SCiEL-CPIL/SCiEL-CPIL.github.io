import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 120.0
    
    # 1. Construct F_25 = F_5[x] / (x^2 + 2)
    # Elements are (a, b) representing a + b*x
    # (a,b) + (c,d) = (a+c, b+d) mod 5
    # (a,b) * (c,d) = (ac + 2bd, ad + bc) mod 5
    
    def f25_add(p, q):
        return ((p[0] + q[0]) % 5, (p[1] + q[1]) % 5)
        
    def f25_sub(p, q):
        return ((p[0] - q[0]) % 5, (p[1] - q[1]) % 5)
        
    def f25_mul(p, q):
        a, b = p
        c, d = q
        return ((a*c + 2*b*d) % 5, (a*d + b*c) % 5)
        
    # Generate all 25 elements
    elements = [(a, b) for a in range(5) for b in range(5)]
    elem_to_idx = {e: i for i, e in enumerate(elements)}
    
    # Find a generator of F_25^*
    # Group order is 24.
    # Check each non-zero element
    generator = None
    for e in elements:
        if e == (0, 0):
            continue
        # Check order
        current = (1, 0)
        order = 0
        for _ in range(24):
            current = f25_mul(current, e)
            order += 1
            if current == (1, 0):
                break
        if order == 24:
            generator = e
            break
            
    if generator is None:
        # Fallback to n=11 if something went wrong
        n = 11
        return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}

    # Compute discrete logs
    # pow_gen[k] = generator^k
    pow_gen = [(1, 0)]
    for _ in range(23):
        pow_gen.append(f25_mul(pow_gen[-1], generator))
        
    # Map element to log
    log_map = {}
    for k, e in enumerate(pow_gen):
        log_map[e] = k
        
    # Colors for edges within F_25
    # n_internal = 25
    n_internal = 25
    colors_internal = [0] * (n_internal * (n_internal - 1) // 2)
    
    # Build adjacency lists for each color class in F_25
    # adj[c] is a list of sets of neighbors for each vertex? Or just list of lists
    # We need to check C4 freedom later or during extension.
    # For extension, we need to know N_c(u) for each c and u.
    adj = [set() for _ in range(5)] # adj[c] will hold sets? No, adj[c][u] is set of neighbors
    adj = [ [set() for _ in range(n_internal)] for _ in range(5) ]
    
    idx = 0
    for u in range(n_internal):
        for v in range(u + 1, n_internal):
            diff = f25_sub(elements[v], elements[u])
            # diff is non-zero
            k = log_map[diff]
            c = k % 5
            colors_internal[idx] = c
            adj[c][u].add(v)
            adj[c][v].add(u)
            idx += 1
            
    # Verify internal coloring is C4-free (optional but good for sanity)
    # For each color c, check no C4
    # A C4 in color c exists if there are u, v, w, x such that 
    # v in N_c(u), w in N_c(v), x in N_c(w), u in N_c(x)
    # And u,v,w,x distinct.
    # Equivalent: For any u, v in N_c(u) is not enough.
    # Check: For any pair (u, w), if |N_c(u) \cap N_c(w)| >= 2, then there is a C4?
    # Yes, if u and w share two neighbors v1, v2, then u-v1-w-v2-u is a C4.
    # So condition: For all u != w, |N_c(u) \cap N_c(w)| <= 1.
    
    def is_valid_internal(adj):
        for c in range(5):
            for u in range(n_internal):
                for w in range(u + 1, n_internal):
                    common = adj[c][u].intersection(adj[c][w])
                    if len(common) > 1:
                        return False
        return True

    if not is_valid_internal(adj):
        # This should not happen for the LW construction
        # Fallback
        n = 11
        return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}

    # Now try to extend to n=26
    # Vertex 25 is infinity.
    # We need to assign colors to edges (25, v) for v in 0..24.
    # Let color_inf[v] be the color of edge (25, v).
    # Condition: For each color c, let S_c = {v | color_inf[v] == c}.
    # For any u, w in S_c with u != w, N_c(u) \cap N_c(w) must be empty.
    
    # We can use backtracking to assign colors to 25 vertices.
    # Or random restarts with local search.
    # Given 25 variables, backtracking might be slow if constraints are tight.
    # Let's try a simple randomized greedy with restarts.
    
    def try_find_inf_coloring(adj, max_restarts=5000):
        for _ in range(max_restarts):
            if time.time() > deadline - 10:
                break
            colors_inf = [-1] * n_internal
            # Randomly assign colors
            for v in range(n_internal):
                colors_inf[v] = random.randint(0, 4)
                
            # Check validity
            valid = True
            # We can optimize by checking conflicts as we assign, but full check is 5 * C(25,2) ~ 1500 checks.
            # 1500 checks * 5000 restarts = 7.5M checks. Might be slow in Python.
            # Let's do a local search repair instead.
            
            # Initialize with random
            # Calculate conflicts
            # A conflict is a pair (u, w) with same color c and |N_c(u) \cap N_c(w)| > 0.
            
            # Simpler: Just check if the current assignment is valid.
            # If not, pick a random vertex involved in a conflict and change its color.
            
            def check_validity():
                for c in range(5):
                    S_c = [v for v in range(n_internal) if colors_inf[v] == c]
                    for i in range(len(S_c)):
                        for j in range(i + 1, len(S_c)):
                            u = S_c[i]
                            w = S_c[j]
                            if len(adj[c][u].intersection(adj[c][w])) > 0:
                                return False
                return True

            if check_validity():
                return colors_inf

            # Local search
            for step in range(1000):
                if time.time() > deadline - 5:
                    break
                if check_validity():
                    return colors_inf
                
                # Find a conflict
                # Randomly pick a pair (u, w) that conflicts?
                # Or just pick a random vertex and see if it's in a conflict.
                # Better: Pick a random conflict.
                conflict_found = False
                for c in range(5):
                    S_c = [v for v in range(n_internal) if colors_inf[v] == c]
                    for i in range(len(S_c)):
                        for j in range(i + 1, len(S_c)):
                            u = S_c[i]
                            w = S_c[j]
                            if len(adj[c][u].intersection(adj[c][w])) > 0:
                                # Conflict found. Change color of u or w.
                                # Randomly choose one
                                if random.random() < 0.5:
                                    target = u
                                else:
                                    target = w
                                old_color = colors_inf[target]
                                new_color = random.randint(0, 4)
                                while new_color == old_color:
                                    new_color = random.randint(0, 4)
                                colors_inf[target] = new_color
                                conflict_found = True
                                break
                        if conflict_found:
                            break
                    if conflict_found:
                        break
                if not conflict_found:
                    # No conflict found? Should have returned.
                    return colors_inf

        return None

    colors_inf = try_find_inf_coloring(adj)
    
    if colors_inf is not None:
        # Construct final coloring for n=26
        n = 26
        colors = [0] * (n * (n - 1) // 2)
        
        # Fill internal edges
        idx = 0
        for u in range(n_internal):
            for v in range(u + 1, n_internal):
                colors[idx] = colors_internal[idx]
                idx += 1
                
        # Fill edges to infinity (vertex 25)
        # Lexicographic order: (0,25), (1,25), ..., (24,25)
        # These are the last 25 edges.
        for v in range(n_internal):
            colors[idx] = colors_inf[v]
            idx += 1
            
        return {'n': n, 'colors': colors}
    else:
        # Fall back to n=25
        n = 25
        return {'n': n, 'colors': colors_internal}
