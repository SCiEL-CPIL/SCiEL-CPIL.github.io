"""R_4(C6) - 4-colouring of K_n without monochromatic C6, using Simulated Annealing."""
# EVOLVE-BLOCK-START
import time
import random
import math

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

        # Precompute for each vertex, list of edge indices incident to it
        # vertex_edges[v] = list of (edge_idx, other_vertex)
        vertex_edges = [[] for _ in range(n)]
        for idx, (u, v) in enumerate(edges):
            vertex_edges[u].append((idx, v))
            vertex_edges[v].append((idx, u))

        def build_adj(colors):
            adj = [ [set() for _ in range(n)] for _ in range(4) ]
            for idx, (u, v) in enumerate(edges):
                c = colors[idx]
                adj[c][u].add(v)
                adj[c][v].add(u)
            return adj

        def count_c6s_containing_edge(adj, u, v):
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
                total += count_c6s_containing_edge(adj[c], u, v)
            return total // 6

        # Simulated Annealing
        best_result = None
        best_sum_found = float('inf')
        
        # Initial temperature
        T = 1.0
        cooling_rate = 0.999
        
        while time.time() < time_limit:
            # Try structured initial coloring first, then random
            # A structured init might help, but random is often better for SA
            colors = [random.randint(0, 3) for _ in range(num_edges)]
            adj = build_adj(colors)
            
            # Calculate initial total C6 count (sum of counts per edge)
            current_sum = 0
            for idx, (u, v) in enumerate(edges):
                c = colors[idx]
                current_sum += count_c6s_containing_edge(adj[c], u, v)
            
            best_sum_found = min(best_sum_found, current_sum)
            
            if current_sum == 0:
                return {'n': n, 'colors': colors}
            
            no_improvement = 0
            max_no_improve = 5000
            
            while no_improvement < max_no_improve and time.time() < time_limit:
                # Pick a random edge to change
                idx = random.randint(0, num_edges - 1)
                u, v = edge_to_uv[idx]
                old_color = colors[idx]
                
                # Pick a new color
                new_color = random.randint(0, 3)
                if new_color == old_color:
                    continue
                
                # Calculate delta
                # Remove edge from old_color, add to new_color
                adj[old_color][u].remove(v)
                adj[old_color][v].remove(u)
                adj[new_color][u].add(v)
                adj[new_color][v].add(u)
                
                old_count = count_c6s_containing_edge(adj[old_color], u, v)
                new_count = count_c6s_containing_edge(adj[new_color], u, v)
                
                # Note: changing one edge only affects C6s containing that edge?
                # NO! Changing the color of edge (u,v) affects C6s that contain (u,v) in the OLD color
                # and C6s that contain (u,v) in the NEW color.
                # But does it affect other C6s?
                # A C6 is a set of 6 edges. If we change the color of (u,v),
                # any C6 that previously had (u,v) as part of its monochromatic set in color c
                # is no longer monochromatic in color c (unless the other 5 edges are still color c,
                # but the cycle is broken).
                # Actually, the definition is "No colour class may contain a six-cycle".
                # So we count C6s in each color class.
                # If we move edge (u,v) from color A to color B:
                # 1. C6s in color A that used (u,v) are destroyed.
                # 2. C6s in color B that use (u,v) are created (if the other 5 edges are in B).
                # 3. C6s in other colors are unaffected.
                # 4. C6s in color A that do NOT use (u,v) are unaffected.
                # 5. C6s in color B that do NOT use (u,v) are unaffected.
                
                # So the change in total C6 count is:
                # - (number of C6s in color A containing (u,v) BEFORE removal)
                # + (number of C6s in color B containing (u,v) AFTER addition)
                
                # We calculated old_count AFTER removal? No, we removed it first.
                # So `old_count` is the count in A *after* removal, which is 0 for C6s containing (u,v) because (u,v) is gone.
                # Wait, `count_c6s_containing_edge` counts C6s that *contain* the edge.
                # If the edge is removed from the graph, `count_c6s_containing_edge` will return 0.
                # So I need the count BEFORE removal.
                
                # Let's swap the logic:
                # 1. Count C6s in old_color containing (u,v) BEFORE removal.
                # 2. Remove (u,v) from old_color.
                # 3. Add (u,v) to new_color.
                # 4. Count C6s in new_color containing (u,v) AFTER addition.
                
                # Undo the changes to recalculate correctly
                adj[old_color][u].add(v)
                adj[old_color][v].add(u)
                adj[new_color][u].remove(v)
                adj[new_color][v].remove(u)
                
                old_count_before = count_c6s_containing_edge(adj[old_color], u, v)
                
                # Now perform the swap
                adj[old_color][u].remove(v)
                adj[old_color][v].remove(u)
                adj[new_color][u].add(v)
                adj[new_color][v].add(u)
                
                new_count_after = count_c6s_containing_edge(adj[new_color], u, v)
                
                delta = new_count_after - old_count_before
                new_sum = current_sum + delta
                
                # Acceptance criterion
                if new_sum < current_sum:
                    accept = True
                else:
                    # Boltzmann factor
                    if T > 1e-10:
                        exp_arg = -delta / T
                        if exp_arg < -700: # prevent underflow
                            accept = False
                        else:
                            accept = random.random() < math.exp(exp_arg)
                    else:
                        accept = False
                
                if accept:
                    colors[idx] = new_color
                    current_sum = new_sum
                    no_improvement = 0
                    if current_sum == 0:
                        return {'n': n, 'colors': colors}
                else:
                    # Revert
                    adj[old_color][u].add(v)
                    adj[old_color][v].add(u)
                    adj[new_color][u].remove(v)
                    adj[new_color][v].remove(u)
                    no_improvement += 1
                
                # Cool down
                T *= cooling_rate
        
        return None

    # Try larger n first
    # Start with n=17 as it's the known best. If we can't find it, we might try 16, etc.
    # But the task says "Maximize n".
    # The seed program tries 17 down to 9.
    # Let's stick to that strategy but with SA.
    
    best_res = None
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
