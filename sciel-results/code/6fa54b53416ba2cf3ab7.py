import random
import time
import numpy as np

def construct():
    target_ns = [26, 25, 24, 23, 22, 21, 20]
    start_time = time.time()
    time_limit = 115.0

    for n in target_ns:
        if time.time() - start_time > time_limit:
            break

        # Precompute edge list
        edges = []
        edge_idx_map = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))
                edge_idx_map[(i, j)] = idx
                idx += 1
        m = len(edges)

        # Partition vertices into 5 sets
        vertices = list(range(n))
        random.shuffle(vertices)
        sets = [[] for _ in range(5)]
        for i, v in enumerate(vertices):
            sets[i % 5].append(v)
        
        # Map vertex to set index
        v_set = [0] * n
        for c in range(5):
            for v in sets[c]:
                v_set[v] = c

        # Generate Base Graphs (C4-free) for each set
        # base_adj[c][u] is a set of neighbors of u in the base graph for color c
        # We want to maximize edges in base graph to reduce reliance on hash
        base_adj = [ [set() for _ in range(n)] for _ in range(5) ]
        
        def is_c4_free_add(adj, u, v, n):
            # Check if adding (u,v) creates a C4 in adj
            # C4: u-v-x-y-u. 
            # New edge (u,v). Common neighbors of u and v in adj?
            # If they have 2 common neighbors, adding (u,v) creates a C4?
            # No, a C4 is u-v-x-y-u.
            # If we add (u,v), we create C4s if there exist x,y such that 
            # x is neighbor of u, y is neighbor of v, and x-y is an edge?
            # Actually, if u and v have two common neighbors x and y, and x-y is NOT an edge, 
            # then u-x-v-y-u is a C4? No, u-x and x-v are edges. 
            # The cycle is u-x-v-y-u.
            # Edges: (u,x), (x,v), (v,y), (y,u).
            # If we add (u,v), does it create a C4?
            # A C4 containing (u,v) must be u-v-x-y-u.
            # This requires (v,x), (x,y), (y,u).
            # So we need to check if there is a path of length 2 from v to u in adj \ {(u,v)}?
            # No, if (u,v) is added, a C4 is formed if there is a path of length 3 between u and v in the existing graph?
            # u-v-x-y-u. 
            # (u,v) is new. (v,x), (x,y), (y,u) must exist.
            # So we need to check if there exists x such that (v,x) is in adj, and y such that (x,y) is in adj and (y,u) is in adj.
            # This is equivalent to checking if the number of common neighbors of u and v is >= 2?
            # No. If u and v have common neighbors x, y.
            # Then u-x-v is a path.
            # If x and y are connected, u-x-y-u is a triangle? No.
            # If u and v have common neighbors x and y, and (x,y) is NOT an edge, then u-x-v-y-u is a C4?
            # Edges: (u,x), (x,v), (v,y), (y,u).
            # Yes.
            # So adding (u,v) creates a C4 if u and v have at least 2 common neighbors in the current graph.
            
            common = adj[u] & adj[v]
            return len(common) < 2

        for c in range(5):
            # Try to add edges greedily to base_adj[c] within sets[c]
            # We want a C4-free graph.
            # We can just use a random ordering and add if valid.
            verts = sets[c]
            random.shuffle(verts)
            for i_idx in range(len(verts)):
                u = verts[i_idx]
                for j_idx in range(i_idx + 1, len(verts)):
                    v = verts[j_idx]
                    if is_c4_free_add(base_adj[c], u, v, n):
                        base_adj[c][u].add(v)
                        base_adj[c][v].add(u)

        # Initialize colors
        colors = [0] * m
        adj = [ [set() for _ in range(n)] for _ in range(5) ]
        
        # Deterministic Hash for inter-subset and non-base intra-subset
        def hash_color(u, v):
            # Simple deterministic hash
            return (u * 31 + v * 17) % 5

        for k, (i, j) in enumerate(edges):
            c_u = v_set[i]
            c_j = v_set[j]
            
            if c_u == c_j:
                # Intra-subset
                if j in base_adj[c_u][i]:
                    colors[k] = c_u
                else:
                    # Use hash, but maybe bias towards c_u?
                    # The directive says "edges within each subset are assigned distinct colors based on... base graph".
                    # This implies non-base edges get other colors.
                    colors[k] = hash_color(i, j)
            else:
                # Inter-subset
                colors[k] = hash_color(i, j)
            
            c = colors[k]
            adj[c][i].add(j)
            adj[c][j].add(i)

        # Count C4s
        def count_c4s():
            total = 0
            for c in range(5):
                for u in range(n):
                    for v in adj[c][u]:
                        if u < v:
                            # Count common neighbors of u and v in adj[c]
                            # Each C4 is counted 4 times (once per edge? No, once per pair of opposite vertices?)
                            # If we count pairs (u,v) with 2+ common neighbors, each C4 has 2 pairs of opposite vertices.
                            # So total // 2?
                            # Let's just count "bad pairs".
                            common = adj[c][u] & adj[c][v]
                            if len(common) >= 2:
                                total += len(common) * (len(common) - 1) // 2
            return total

        current_energy = count_c4s()
        best_energy = current_energy
        best_colors = colors.copy()

        # SA Parameters
        T = 5.0
        cooling = 0.999
        min_T = 0.01
        steps = 0
        max_steps = 500000
        
        while T > min_T:
            if time.time() - start_time > time_limit:
                break
                
            # Pick random edge
            k = random.randint(0, m - 1)
            i, j = edges[k]
            
            c_old = colors[k]
            c_new = random.randint(0, 4)
            if c_old == c_new: continue
            
            # Delta calc
            # Remove (i,j) from c_old
            # Add (i,j) to c_new
            
            # Count C4s involving (i,j) in c_old
            # A C4 involving (i,j) in color c is a pair (u,v) such that 
            # u is neighbor of i, v is neighbor of j, and u-v is edge?
            # No, the C4 is i-j-u-v-i.
            # Edges: (i,j), (j,u), (u,v), (v,i).
            # So we need to count pairs (u,v) where u in N_c(i) \ {j}, v in N_c(j) \ {i}, 
            # and (u,v) is an edge in c.
            # This is equivalent to counting edges in the bipartite graph between N(i) and N(j)?
            # No, u and v must be connected.
            
            # Faster delta:
            # When we change color of (i,j) from c_old to c_new:
            # 1. In c_old, we remove edge (i,j).
            #    This breaks C4s that used (i,j).
            #    A C4 using (i,j) is i-j-u-v-i.
            #    So u is in N_c_old(i) \ {j}, v is in N_c_old(j) \ {i}, and (u,v) is in c_old.
            #    Number of such C4s = number of edges between N_c_old(i) and N_c_old(j) in c_old.
            #    Let S_i = N_c_old(i) - {j}, S_j = N_c_old(j) - {i}.
            #    Count = sum(1 for u in S_i for v in S_j if v in adj[c_old][u])
            
            cnt_old = 0
            S_i = adj[c_old][i] - {j}
            S_j = adj[c_old][j] - {i}
            # Iterate over smaller set
            if len(S_i) < len(S_j):
                for u in S_i:
                    for v in S_j:
                        if v in adj[c_old][u]:
                            cnt_old += 1
            else:
                for v in S_j:
                    for u in S_i:
                        if u in adj[c_old][v]:
                            cnt_old += 1

            cnt_new = 0
            S_i_new = adj[c_new][i] - {j} # Note: j is not in adj[c_new][i] yet
            S_j_new = adj[c_new][j] - {i}
            if len(S_i_new) < len(S_j_new):
                for u in S_i_new:
                    for v in S_j_new:
                        if v in adj[c_new][u]:
                            cnt_new += 1
            else:
                for v in S_j_new:
                    for u in S_i_new:
                        if u in adj[c_new][v]:
                            cnt_new += 1

            delta_E = cnt_new - cnt_old
            
            if delta_E < 0 or random.random() < pow(2.718, -delta_E / T):
                colors[k] = c_new
                adj[c_old][i].remove(j)
                adj[c_old][j].remove(i)
                adj[c_new][i].add(j)
                adj[c_new][j].add(i)
                
                current_energy += delta_E
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_colors = colors.copy()
                    if best_energy == 0:
                        return {'n': n, 'colors': best_colors}
                        
            T *= cooling
            steps += 1
            
        if best_energy == 0:
            return {'n': n, 'colors': best_colors}

    # Fallback
    n = 11
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
