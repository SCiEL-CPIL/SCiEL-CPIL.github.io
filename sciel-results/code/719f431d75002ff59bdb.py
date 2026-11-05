import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n = 20
    a = 11 # Left part size
    b = 9  # Right part size
    
    left = list(range(a))
    right = list(range(a, a+b))
    
    # Matrix M[i][j] is color of edge (left[i], right[j])
    # Initialize randomly
    M = [[random.randint(0, 19) for _ in range(b)] for _ in range(a)]
    
    # Precompute cycles
    # A cycle is defined by (i1, i2, j1, j2) with i1 < i2, j1 < j2
    cycles = []
    for i1 in range(a):
        for i2 in range(i1+1, a):
            for j1 in range(b):
                for j2 in range(j1+1, b):
                    cycles.append((i1, i2, j1, j2))
    
    def count_bad():
        bad = 0
        for i1, i2, j1, j2 in cycles:
            c1 = M[i1][j1]
            c2 = M[i2][j1]
            c3 = M[i2][j2]
            c4 = M[i1][j2]
            if len({c1, c2, c3, c4}) != 4:
                bad += 1
        return bad
        
    # Local Search
    current_bad = count_bad()
    
    # If it's not valid, try to fix
    # We can also try multiple random restarts if the first one fails
    max_restarts = 5
    found_valid = False
    
    for r in range(max_restarts):
        if current_bad == 0:
            found_valid = True
            break
            
        # Local search
        # To speed up, we can track which cycles are bad
        # But for 11x9, total cycles is C(11,2)*C(9,2) = 55 * 36 = 1980.
        # Checking all 1980 is fast.
        
        # Simple Hill Climbing
        no_improve_count = 0
        for step in range(2000):
            if time.time() > deadline - 5:
                break
                
            if current_bad == 0:
                found_valid = True
                break
                
            # Pick a random bad cycle
            # To do this efficiently, we can just pick a random cycle and check if it's bad?
            # Or maintain a list.
            # Given small size, let's just pick a random cycle from the full list.
            # If it's not bad, we wasted a step.
            # Better: iterate through cycles, find one bad one.
            bad_cycle = None
            for cyc in cycles:
                i1, i2, j1, j2 = cyc
                c1 = M[i1][j1]
                c2 = M[i2][j1]
                c3 = M[i2][j2]
                c4 = M[i1][j2]
                if len({c1, c2, c3, c4}) != 4:
                    bad_cycle = cyc
                    break
                    
            if bad_cycle is None:
                # Shouldn't happen if current_bad > 0, but safety
                found_valid = True
                break
                
            i1, i2, j1, j2 = bad_cycle
            # Pick one of the 4 edges to recolor
            # Edges: (i1,j1), (i2,j1), (i2,j2), (i1,j2)
            edges_in_cycle = [(i1, j1), (i2, j1), (i2, j2), (i1, j2)]
            ei, ej = random.choice(edges_in_cycle)
            old_color = M[ei][ej]
            
            best_color = old_color
            best_bad = current_bad
            
            for c in range(20):
                if c == old_color:
                    continue
                M[ei][ej] = c
                # Recalculate bad count?
                # Optimization: Only check cycles involving (ei, ej)?
                # There are C(10,1)*C(8,1) + ... cycles.
                # It's faster to just recount all 1980.
                new_bad = count_bad()
                if new_bad < best_bad:
                    best_bad = new_bad
                    best_color = c
            M[ei][ej] = best_color
            current_bad = best_bad
            
            if current_bad == 0:
                found_valid = True
                break
            elif current_bad == best_bad:
                no_improve_count += 1
            else:
                no_improve_count = 0
                
            if no_improve_count > 100:
                # Restart with new random init
                M = [[random.randint(0, 19) for _ in range(b)] for _ in range(a)]
                current_bad = count_bad()
                no_improve_count = 0
                break
                
    if not found_valid:
        # Fallback: Return a smaller valid graph
        # e.g. C20
        edges = []
        for i in range(20):
            edges.append([i, (i+1)%20, i % 20])
        return edges
        
    # Construct final edge list
    edges = []
    
    # Bipartite edges
    for i in range(a):
        for j in range(b):
            u = left[i]
            v = right[j]
            c = M[i][j]
            edges.append([u, v, c])
            
    # Clique edges on Left (L)
    # Vertices 0..10
    # Assign color 0 to all of them (arbitrary)
    for i in range(a):
        for j in range(i+1, a):
            edges.append([left[i], left[j], 0])
            
    return edges
