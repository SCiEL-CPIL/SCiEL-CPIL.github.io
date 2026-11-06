import time
import random
import numpy as np

def construct():
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer

    def get_paley_colors(n):
        squares = set()
        for x in range(1, n):
            squares.add((x * x) % n)
        
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                diff = (j - i) % n
                if diff in squares:
                    colors.append(0)
                else:
                    colors.append(1)
        return colors

    def colors_to_adj(colors, n):
        adj = [[0] * n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[i][j] = c
                adj[j][i] = c
                idx += 1
        return adj

    def adj_to_colors(adj, n):
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                colors.append(adj[i][j])
        return colors

    # Partitions of 6 vertices (indices 0-5) into 3 pairs
    # Represented as list of 3 pairs, each pair is tuple of indices
    PARTITIONS = [
        [(0,1), (2,3), (4,5)],
        [(0,2), (1,3), (4,5)],
        [(0,3), (1,2), (4,5)],
        [(0,4), (1,5), (2,3)],
        [(0,5), (1,4), (2,3)],
        [(0,2), (1,4), (3,5)],
        [(0,3), (1,4), (2,5)],
        [(0,4), (1,2), (3,5)],
        [(0,1), (2,4), (3,5)],
        [(0,1), (2,5), (3,4)],
        [(0,2), (1,5), (3,4)],
        [(0,3), (1,5), (2,4)],
        [(0,4), (1,3), (2,5)],
        [(0,5), (1,3), (2,4)],
        [(0,1), (2,3), (4,5)] # Duplicate? 
        # Let's generate them properly to be sure.
    ]
    
    # Correct generation of 15 partitions of 6 items into 3 pairs
    def gen_partitions():
        items = [0,1,2,3,4,5]
        res = []
        def backtrack(remaining, current):
            if not remaining:
                if current:
                    res.append([tuple(sorted(p)) for p in current])
                return
            first = remaining[0]
            for i in range(1, len(remaining)):
                second = remaining[i]
                new_remaining = remaining[1:i] + remaining[i+1:]
                backtrack(new_remaining, current + [(first, second)])
        backtrack(items, [])
        return res

    PARTITIONS = gen_partitions()
    # Verify count is 15
    # assert len(PARTITIONS) == 15

    def is_bad_6set(v, adj):
        # v is a list of 6 vertex indices
        # Check all 15 partitions
        for part in PARTITIONS:
            # part is [(p1a, p1b), (p2a, p2b), (p3a, p3b)]
            p1, p2, p3 = part
            # Edges between p1 and p2
            # v[p1[0]] to v[p2[0]], v[p1[0]] to v[p2[1]], etc.
            
            # We need to check if all 12 edges are 0 or all 12 are 1.
            # Let's collect the colors
            c01 = adj[v[p1[0]]][v[p2[0]]]
            c02 = adj[v[p1[0]]][v[p2[1]]]
            c11 = adj[v[p1[1]]][v[p2[0]]]
            c12 = adj[v[p1[1]]][v[p2[1]]]
            
            c23 = adj[v[p2[0]]][v[p3[0]]]
            c24 = adj[v[p2[0]]][v[p3[1]]]
            c33 = adj[v[p2[1]]][v[p3[0]]]
            c34 = adj[v[p2[1]]][v[p3[1]]]
            
            c03 = adj[v[p1[0]]][v[p3[0]]]
            c04 = adj[v[p1[0]]][v[p3[1]]]
            c13 = adj[v[p1[1]]][v[p3[0]]]
            c14 = adj[v[p1[1]]][v[p3[1]]]
            
            colors_list = [c01, c02, c11, c12, c23, c24, c33, c34, c03, c04, c13, c14]
            
            if all(c == 0 for c in colors_list):
                return True
            if all(c == 1 for c in colors_list):
                return True
        return False

    def count_bad_sets(adj, n):
        count = 0
        for a in range(n-5):
            for b in range(a+1, n-4):
                for c in range(b+1, n-3):
                    for d in range(c+1, n-2):
                        for e in range(d+1, n-1):
                            for f in range(e+1, n):
                                if is_bad_6set([a,b,c,d,e,f], adj):
                                    count += 1
        return count

    def local_search(adj, n, time_limit):
        # Returns True if a valid coloring is found
        # We use a random walk with restarts or just random flips on bad sets
        
        # Initial count
        # We don't want to do a full count every time.
        # We will pick random 6-sets. If we find a bad one, we flip an edge.
        # To ensure we eventually find all bad sets, we can keep a list of "suspects" 
        # or just rely on the randomness. 
        # Given N=30, random sampling should be effective.
        
        # Optimization: If we find a bad set, we know it's bad.
        # We flip one of its edges.
        
        # To avoid getting stuck, we can track the number of flips since last improvement?
        # We don't easily track global improvement without full count.
        
        # Strategy:
        # 1. Full count initially.
        # 2. While time remains:
        #    a. If count == 0, return True.
        #    b. Pick a random 6-set.
        #    c. If bad, flip a random edge in it.
        #    d. Every K iterations, do a full count to see if we are stuck or improving.
        
        count = count_bad_sets(adj, n)
        if count == 0:
            return True
            
        no_improve_count = 0
        last_count = count
        iterations = 0
        check_interval = 5000 # Do full check every 5000 random picks
        
        while time.time() - start_time < time_limit:
            # Pick random 6-set
            verts = random.sample(range(n), 6)
            
            if is_bad_6set(verts, adj):
                # Flip a random edge in this set
                i, j = random.sample(verts, 2)
                adj[i][j] = 1 - adj[i][j]
                adj[j][i] = adj[i][j]
                no_improve_count += 1
            else:
                no_improve_count += 1
            
            iterations += 1
            
            if iterations % check_interval == 0:
                new_count = count_bad_sets(adj, n)
                if new_count == 0:
                    return True
                if new_count < last_count:
                    last_count = new_count
                    no_improve_count = 0
                else:
                    no_improve_count += 1000 # Penalize no improvement
                    
                # If stuck for too long, restart? 
                # Or just continue.
                
        return False

    # Target n=30
    n_target = 30
    found = False
    
    # Try to find a solution for n=30
    # Start with random or Paley(29)+1
    
    # Let's try a few random restarts
    for attempt in range(10):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize adj
        if attempt == 0:
            # Use Paley(29) + random vertex
            colors_29 = get_paley_colors(29)
            adj = colors_to_adj(colors_29, 29)
            # Expand to 30x30
            adj_new = [[0]*30 for _ in range(30)]
            for i in range(29):
                for j in range(30):
                    if i != j:
                        if j < 29:
                            adj_new[i][j] = adj[i][j]
                        else:
                            adj_new[i][j] = random.randint(0, 1)
            # Symmetrize
            for i in range(30):
                for j in range(i+1, 30):
                    adj_new[j][i] = adj_new[i][j]
            adj = adj_new
        else:
            # Random initialization
            adj = [[0]*30 for _ in range(30)]
            for i in range(30):
                for j in range(i+1, 30):
                    c = random.randint(0, 1)
                    adj[i][j] = c
                    adj[j][i] = c
        
        if local_search(adj, 30, time_limit):
            found = True
            break
            
    if found:
        colors = adj_to_colors(adj, 30)
        return {'n': 30, 'colors': colors}
    else:
        # Fallback to n=29
        colors_29 = get_paley_colors(29)
        return {'n': 29, 'colors': colors_29}
