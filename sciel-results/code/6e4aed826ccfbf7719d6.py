"""R_4(C6) - 4-colouring of K_n without monochromatic C6, using efficient local search."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s margin for safety

    def get_edges(n):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))
        return edges

    def solve_for_n(n, time_limit):
        if n <= 1:
            return {'n': n, 'colors': []}
        
        edges = get_edges(n)
        num_edges = len(edges)
        
        # Map from edge index to (u, v)
        edge_to_uv = {}
        for idx, (u, v) in enumerate(edges):
            edge_to_uv[idx] = (u, v)
        
        # Precompute for each edge (u,v) the list of potential C6 paths:
        # A C6 through edge (u,v) is u-a-b-c-d-v (or u-v-d-c-b-a)
        # So we need: u~a, a~b, b~c, c~d, d~v, all in same color, all distinct
        
        # Build adjacency lists per color: adj[c][i] = set of neighbors j such that (i,j) has color c
        def build_adj(colors):
            adj = [ [set() for _ in range(n)] for _ in range(4) ]
            for idx, (u, v) in enumerate(edges):
                c = colors[idx]
                adj[c][u].add(v)
                adj[c][v].add(u)
            return adj

        def count_c6s_containing_edge(adj, n, u, v):
            """
            Count the number of simple 6-cycles in the graph defined by adj
            that contain the edge (u, v).
            adj is a list of lists of sets of neighbors.
            """
            if v not in adj[u]:
                return 0
            
            count = 0
            # For each neighbor a of u (a != v), for each neighbor b of a (b not in {u,v,a}),
            # for each neighbor c of b (c not in {u,v,a,b}), for each neighbor d of c (d not in {u,v,a,b,c}),
            # check if d is adjacent to v
            for a in adj[u]:
                if a == v:
                    continue
                for b in adj[a]:
                    if b in (u, v, a):
                        continue
                    for c in adj[b]:
                        if c in (u, v, a, b):
                            continue
                        for d in adj[c]:
                            if d in (u, v, a, b, c):
                                continue
                            if v in adj[d]:
                                count += 1
            return count

        def total_c6_count(colors, adj):
            total = 0
            for idx, (u, v) in enumerate(edges):
                c = colors[idx]
                total += count_c6s_containing_edge(adj[c], n, u, v)
            return total // 6

        # Local Search with better restart strategy
        best_result = None
        best_sum_found = float('inf')
        
        while time.time() < time_limit:
            # Try structured initial coloring first, then random
            colors = [random.randint(0, 3) for _ in range(num_edges)]
            adj = build_adj(colors)
            
            # Calculate initial total C6 count (sum of counts per edge)
            current_sum = 0
            for idx, (u, v) in enumerate(edges):
                c = colors[idx]
                current_sum += count_c6s_containing_edge(adj[c], n, u, v)
            
            best_sum_found = min(best_sum_found, current_sum)
            
            if current_sum == 0:
                return {'n': n, 'colors': colors}
            
            no_improvement = 0
            max_no_improve = 2000
            
            while no_improvement < max_no_improve and time.time() < time_limit:
                # Find a "bad" edge (one that is in a C6) - try multiple random trials
                bad_idx = -1
                for _ in range(5):
                    trial_idx = random.randint(0, num_edges - 1)
                    tu, tv = edge_to_uv[trial_idx]
                    tc = colors[trial_idx]
                    if count_c6s_containing_edge(adj[tc], n, tu, tv) > 0:
                        bad_idx = trial_idx
                        break
                
                if bad_idx == -1:
                    # No bad edge found in random trials, pick random
                    bad_idx = random.randint(0, num_edges - 1)
                
                idx = bad_idx
                u, v = edge_to_uv[idx]
                old_color = colors[idx]
                
                # Try all new colors, pick best
                best_new_color = old_color
                best_new_sum = current_sum
                
                for new_color in range(4):
                    if new_color == old_color:
                        continue
                    
                    # Temporarily add edge to new_color
                    adj[new_color][u].add(v)
                    adj[new_color][v].add(u)
                    new_count = count_c6s_containing_edge(adj[new_color], n, u, v)
                    adj[new_color][u].remove(v)
                    adj[new_color][v].remove(u)
                    
                    old_count = count_c6s_containing_edge(adj[old_color], n, u, v)
                    
                    delta = new_count - old_count
                    new_sum = current_sum + delta
                    
                    if new_sum < best_new_sum:
                        best_new_sum = new_sum
                        best_new_color = new_color
                
                if best_new_color != old_color:
                    adj[old_color][u].remove(v)
                    adj[old_color][v].remove(u)
                    adj[best_new_color][u].add(v)
                    adj[best_new_color][v].add(u)
                    colors[idx] = best_new_color
                    current_sum = best_new_sum
                    no_improvement = 0
                    
                    if current_sum == 0:
                        return {'n': n, 'colors': colors}
                else:
                    no_improvement += 1
        
        return None

    # Try larger n first
    for n in range(17, 8, -1):
        if time.time() > deadline:
            break
        res = solve_for_n(n, deadline)
        if res:
            return res
    
    # Fallback to n=9
    n = 9
    colors = [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]
    return {'n': n, 'colors': colors}

# EVOLVE-BLOCK-END
