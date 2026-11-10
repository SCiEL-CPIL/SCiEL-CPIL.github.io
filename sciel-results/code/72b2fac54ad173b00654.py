import time
import random

def construct():
    start_time = time.time()
    DEADLINE = start_time + 145.0  # Safety margin before 150s kill

    def gen_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [0] * n
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.append((u, v))
        return adj, edges

    def bronk_max_clique(P, X, adj, deadline):
        """
        Find a maximum clique in the graph defined by adj, restricted to vertices in P.
        Returns a bitmask of the vertices in the max clique found.
        Uses Bron-Kerbosch with pivot.
        """
        best_clique = 0
        best_size = 0
        
        def bk(P, X, R):
            nonlocal best_clique, best_size
            
            if time.time() > deadline:
                return
            
            if not P and not X:
                if R.bit_count() > best_size:
                    best_size = R.bit_count()
                    best_clique = R
                return
            
            # Choose pivot u from P ∪ X to minimize branching
            union = P | X
            if not union:
                return
            
            # Find pivot
            pivot = 0
            max_common = -1
            temp = union
            while temp:
                u = (temp & -temp).bit_length() - 1
                temp &= temp - 1
                common = P & adj[u]
                count = common.bit_count()
                if count > max_common:
                    max_common = count
                    pivot = u
            
            # Branch on vertices in P \ N(pivot)
            candidates = P & ~adj[pivot]
            
            temp_cand = candidates
            while temp_cand:
                if time.time() > deadline:
                    return
                
                v = (temp_cand & -temp_cand).bit_length() - 1
                temp_cand &= temp_cand - 1
                
                # Recurse
                bk(P & adj[v], X, R | (1 << v))
                
                P &= ~(1 << v)
                X |= (1 << v)

        bk(P, X, 0)
        return best_clique

    def solve_instance(n, p, seed, deadline):
        adj, edges = gen_graph(n, p, seed)
        if not edges:
            return []
        
        best_cliques = [[u, v] for u, v in edges]
        best_count = len(best_cliques)
        
        rng_master = random.Random(12345)
        
        elapsed = time.time() - start_time
        remaining_time = deadline - start_time - elapsed
        
        if n <= 32:
            num_restarts = min(200, max(20, int(remaining_time / 0.1)))
        else:
            num_restarts = min(100, max(10, int(remaining_time / 0.3)))
            
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            rng = random.Random(rng_master.randint(0, 2**31))
            
            rem_adj = adj[:]
            cliques = []
            
            while True:
                if time.time() > deadline:
                    break
                
                # Check if any edges remain
                has_edge = False
                for i in range(n):
                    if rem_adj[i]:
                        has_edge = True
                        break
                if not has_edge:
                    break
                
                # Pick a starting point: alternate between Max Degree and Min Common
                if rng.random() < 0.5:
                    # Max Degree Vertex
                    max_deg = -1
                    best_v = -1
                    for v in range(n):
                        d = rem_adj[v].bit_count()
                        if d > max_deg:
                            max_deg = d
                            best_v = v
                    
                    if best_v == -1:
                        break
                        
                    P = rem_adj[best_v]
                    X = 0
                    K = bronk_max_clique(P, X, rem_adj, deadline)
                    clique_mask = K | (1 << best_v)
                else:
                    # Min Common Neighbors Edge
                    min_common = float('inf')
                    best_u = -1
                    best_v = -1
                    
                    for u in range(n):
                        if not rem_adj[u]:
                            continue
                        temp = rem_adj[u] & ((1 << n) - 1) & ~((1 << (u+1)) - 1)
                        while temp:
                            v = (temp & -temp).bit_length() - 1
                            temp &= temp - 1
                            common = rem_adj[u] & rem_adj[v]
                            count = common.bit_count()
                            if count < min_common:
                                min_common = count
                                best_u = u
                                best_v = v
                                if count == 0:
                                    break
                        if min_common == 0:
                            break
                            
                    if best_u == -1:
                        break
                        
                    P = rem_adj[best_u] & rem_adj[best_v]
                    X = 0
                    K = bronk_max_clique(P, X, rem_adj, deadline)
                    clique_mask = K | (1 << best_u) | (1 << best_v)
                
                # Extract clique vertices
                clique_vertices = []
                cm = clique_mask
                while cm:
                    v = (cm & -cm).bit_length() - 1
                    cm &= cm - 1
                    clique_vertices.append(v)
                    
                if len(clique_vertices) < 2:
                    found = False
                    for u in range(n):
                        if rem_adj[u]:
                            v = (rem_adj[u] & -rem_adj[u]).bit_length() - 1
                            clique_vertices = [u, v]
                            clique_mask = (1 << u) | (1 << v)
                            found = True
                            break
                    if not found:
                        break

                cliques.append(clique_vertices)
                
                # Remove edges within clique
                for i in range(len(clique_vertices)):
                    ui = clique_vertices[i]
                    for j in range(i+1, len(clique_vertices)):
                        vj = clique_vertices[j]
                        rem_adj[ui] &= ~(1 << vj)
                        rem_adj[vj] &= ~(1 << ui)
            
            if cliques and len(cliques) < best_count:
                best_count = len(cliques)
                best_cliques = cliques
        
        return best_cliques

    # Solve both instances with time allocation
    deadline1 = start_time + 50.0
    deadline2 = DEADLINE
    
    result1 = solve_instance(32, 0.6, 601, deadline1)
    
    if time.time() > DEADLINE:
        adj2, edges2 = gen_graph(64, 0.55, 602)
        result2 = [[u, v] for u, v in edges2]
    else:
        result2 = solve_instance(64, 0.55, 602, DEADLINE)
    
    return [result1, result2]
