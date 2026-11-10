import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = start_time + 110.0

    def gen_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def build_adj(n, edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def popcount(x):
        return x.bit_count()

    def max_clique_tomita(n, adj, rng=None):
        best_clique = []
        best_size = 0
        
        def expand(R, P):
            nonlocal best_clique, best_size
            if not P:
                if len(R) > best_size:
                    best_clique = R[:]
                    best_size = len(R)
                return
            
            if len(R) + popcount(P) <= best_size:
                return
            
            # Pivot selection
            u = -1
            max_deg = -1
            P_copy = P
            # Randomize pivot selection if rng provided
            candidates = []
            while P_copy:
                v_bit = P_copy & (-P_copy)
                v = v_bit.bit_length() - 1
                P_copy ^= v_bit
                deg = popcount(P & adj[v])
                if deg > max_deg:
                    max_deg = deg
                    candidates = [v]
                elif deg == max_deg:
                    candidates.append(v)
            
            if not candidates:
                return
                
            if rng:
                u = rng.choice(candidates)
            else:
                u = candidates[0]

            Q = P & ~adj[u]
            
            while Q:
                # Pick v from Q
                # Randomize v selection
                Q_list = []
                Q_copy = Q
                while Q_copy:
                    v_bit = Q_copy & (-Q_copy)
                    v = v_bit.bit_length() - 1
                    Q_copy ^= v_bit
                    Q_list.append(v)
                
                if rng:
                    rng.shuffle(Q_list)
                
                for v in Q_list:
                    if not (Q & (1 << v)):
                        continue
                    new_R = R + [v]
                    new_P = P & adj[v]
                    expand(new_R, new_P)
                    
                    P &= ~(1 << v)
                    Q = Q & P

        expand([], (1 << n) - 1)
        return best_clique

    def greedy_clique_partition(n, edges, adj, rng=None):
        rem_adj = [0] * n
        for u, v in edges:
            rem_adj[u] |= (1 << v)
            rem_adj[v] |= (1 << u)
        
        cliques = []
        remaining_edges = len(edges)
        
        while remaining_edges > 0:
            if time.time() > DEADLINE:
                return None
            
            clique = max_clique_tomita(n, rem_adj, rng)
            
            if len(clique) < 2:
                found = False
                for u in range(n):
                    if rem_adj[u]:
                        v_bit = rem_adj[u] & (-rem_adj[u])
                        v = v_bit.bit_length() - 1
                        if (rem_adj[u] >> v) & 1:
                            clique = [u, v]
                            found = True
                            break
                if not found:
                    break
            
            edges_removed = 0
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    u, v = clique[i], clique[j]
                    if (rem_adj[u] >> v) & 1:
                        rem_adj[u] &= ~(1 << v)
                        rem_adj[v] &= ~(1 << u)
                        edges_removed += 1
            
            if edges_removed == 0:
                break
            
            cliques.append(clique)
            remaining_edges -= edges_removed
        
        return cliques

    def solve_instance(n, p, seed):
        edges = gen_graph(n, p, seed)
        adj = build_adj(n, edges)
        
        baseline = [[u, v] for u, v in edges]
        best_cliques = baseline
        best_count = len(baseline)
        
        # Try multiple random restarts
        # Estimate time per run
        # If n=64, one run might take 0.1s?
        # Let's try 50 restarts.
        max_restarts = 50 if n < 50 else 20
        
        for i in range(max_restarts):
            if time.time() > DEADLINE - 5.0:
                break
            rng = random.Random(seed + i * 12345)
            result = greedy_clique_partition(n, edges, adj, rng)
            if result is not None:
                count = len(result)
                if count < best_count:
                    best_count = count
                    best_cliques = result
        
        return best_cliques

    result1 = solve_instance(32, 0.6, 601)
    
    if time.time() > DEADLINE - 10.0:
        edges2 = gen_graph(64, 0.55, 602)
        result2 = [[u, v] for u, v in edges2]
    else:
        result2 = solve_instance(64, 0.55, 602)
    
    return [result1, result2]
