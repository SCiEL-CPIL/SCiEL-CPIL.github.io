import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = 140.0 # 20s margin
    
    # Helper to check validity
    def is_valid(n, colors):
        # colors is list of length n*(n-1)//2
        # Map to adjacency matrix per color
        # For each color c, check C4
        # A C4 is (a,b,c,d) with edges (a,b), (b,c), (c,d), (d,a) all color c
        # Equivalent: for any a,b, common neighbors in color c must be < 2
        # Or: for any a,b, count of c such that edge(a,c) and edge(b,c) exist must be < 2?
        # No, common neighbors of a and b in color c.
        # If there are 2 common neighbors u,v, then a-u-b-v-a is a C4?
        # Edges: (a,u), (u,b), (b,v), (v,a).
        # Yes.
        # So for each color c, build adjacency list.
        # For each pair (a,b), count common neighbors.
        
        adj = [set() for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                c = colors[idx]
                adj[c].add((i,j))
                adj[c].add((j,i))
                idx += 1
        
        for c in range(5):
            # Check for C4
            # For each pair (u,v), find common neighbors
            # If any pair has >= 2 common neighbors, C4 exists
            # This is O(n^3) per color
            for u in range(n):
                for v in range(u+1, n):
                    if (u,v) in adj[c]:
                        continue # u,v are adjacent, C4 needs non-adjacent?
                        # Actually C4 is u-x-v-y-u. u,v are opposite.
                        # They don't need to be adjacent.
                        # But if they are adjacent, it's a triangle?
                        # No, C4 is a cycle of length 4.
                        # Vertices u, x, v, y.
                        # Edges (u,x), (x,v), (v,y), (y,u).
                        # u and v are not connected in the cycle.
                        # So we check common neighbors of u and v.
                        # If common neighbors >= 2, say x,y.
                        # Then u-x-v-y-u is a C4.
                        # We need to ensure x != y and u,v,x,y distinct.
                        # Common neighbors of u and v are w such that (u,w) and (v,w) in adj.
                        # If we have w1, w2.
                        # u-w1-v-w2-u.
                        # Distinct: w1!=w2. u!=v. w1!=u (no self loop).
                        # So yes.
                        pass
                    common = adj[c][u] & adj[c][v] # Wait, adj is list of sets?
                    # I used tuples in adj. Let's use sets of neighbors.
                    pass
        
        # Better implementation for is_valid
        return True

    # Correct is_valid
    def check_valid(n, colors):
        adj = [ [set() for _ in range(n)] for _ in range(5) ]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                c = colors[idx]
                adj[c][i].add(j)
                adj[c][j].add(i)
                idx += 1
        
        for c in range(5):
            for u in range(n):
                for v in range(u+1, n):
                    common = adj[c][u] & adj[c][v]
                    if len(common) >= 2:
                        return False
        return True

    def count_violations(n, colors):
        adj = [ [set() for _ in range(n)] for _ in range(5) ]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                c = colors[idx]
                adj[c][i].add(j)
                adj[c][j].add(i)
                idx += 1
        
        viol = 0
        for c in range(5):
            for u in range(n):
                for v in range(u+1, n):
                    common = adj[c][u] & adj[c][v]
                    if len(common) >= 2:
                        viol += len(common) - 1
        return viol

    # Try n=26
    best_n = 11
    best_colors = [min(j - i, 26 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)] # Placeholder
    
    for target_n in [26, 17, 15, 13, 12]:
        if time.time() > deadline - 20:
            break
            
        found = False
        # Try random restarts
        max_restarts = 10 if target_n == 26 else 50
        for r in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            colors = [random.randint(0, 4) for _ in range(target_n*(target_n-1)//2)]
            
            # Local search
            current_viol = count_violations(target_n, colors)
            if current_viol == 0:
                best_n = target_n
                best_colors = colors
                found = True
                break
                
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                # Random move
                idx = random.randint(0, len(colors)-1)
                old_c = colors[idx]
                new_c = random.randint(0, 4)
                if old_c == new_c: continue
                colors[idx] = new_c
                new_viol = count_violations(target_n, colors)
                if new_viol < current_viol:
                    current_viol = new_viol
                    improved = True
                else:
                    colors[idx] = old_c
                    
            if current_viol == 0:
                best_n = target_n
                best_colors = colors
                found = True
                break
                
        if found:
            break
            
    return {'n': best_n, 'colors': best_colors}
