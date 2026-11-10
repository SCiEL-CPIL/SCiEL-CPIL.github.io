import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = start_time + 110.0  # safety margin before 150s kill

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
        """Count set bits using Python's int.bit_count() (Python 3.10+) or fallback."""
        return x.bit_count() if hasattr(int, 'bit_count') else bin(x).count('1')

    def max_clique_tomita(n, adj):
        """
        Find a maximum clique in the graph defined by adj (list of bitmasks).
        Returns the list of vertices in the clique.
        """
        best_clique = []
        best_size = 0
        
        def expand(R, P):
            nonlocal best_clique, best_size
            if not P:
                if len(R) > best_size:
                    best_clique = R[:]
                    best_size = len(R)
                return
            
            # Pruning: if |R| + |P| <= best_size, no improvement possible
            if len(R) + popcount(P) <= best_size:
                return
            
            # Pivot selection: choose u in P with max degree in P
            u = -1
            max_deg = -1
            P_copy = P
            while P_copy:
                v_bit = P_copy & (-P_copy)
                v = v_bit.bit_length() - 1
                P_copy ^= v_bit
                deg = popcount(P & adj[v])
                if deg > max_deg:
                    max_deg = deg
                    u = v
            
            if u == -1:
                return

            # Q = P \ N(u)
            Q = P & ~adj[u]
            
            while Q:
                # Pick a vertex v from Q with highest degree in P
                v = -1
                max_v_deg = -1
                Q_copy = Q
                while Q_copy:
                    v_bit = Q_copy & (-Q_copy)
                    v = v_bit.bit_length() - 1
                    Q_copy ^= v_bit
                    deg = popcount(P & adj[v])
                    if deg > max_v_deg:
                        max_v_deg = deg
                
                if v == -1:
                    break
                
                new_R = R + [v]
                new_P = P & adj[v]
                expand(new_R, new_P)
                
                P &= ~(1 << v)
                Q = Q & P  # Update Q to reflect P changes
                
        expand([], (1 << n) - 1)
        return best_clique

    def greedy_clique_partition(n, edges, adj, max_time):
        """
        Greedy partition: repeatedly find a large clique using a heuristic,
        remove its edges, and continue.
        """
        rem_adj = [0] * n
        for u, v in edges:
            rem_adj[u] |= (1 << v)
            rem_adj[v] |= (1 << u)
        
        cliques = []
        remaining_edges = len(edges)
        
        while remaining_edges > 0:
            if time.time() > max_time:
                return None  # timeout
            
            # Find a good clique: use Tomita on remaining graph
            clique = max_clique_tomita(n, rem_adj)
            
            if len(clique) < 2:
                # Fallback: take an edge
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
            
            # Remove edges of clique
            edges_removed = 0
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    u, v = clique[i], clique[j]
                    if (rem_adj[u] >> v) & 1:
                        rem_adj[u] &= ~(1 << v)
                        rem_adj[v] &= ~(1 << u)
                        edges_removed += 1
            
            if edges_removed == 0:
                # Should not happen if clique is valid
                break
            
            cliques.append(clique)
            remaining_edges -= edges_removed
        
        return cliques

    def solve_instance(n, p, seed):
        edges = gen_graph(n, p, seed)
        adj = build_adj(n, edges)
        
        # Baseline: each edge as its own clique
        baseline = [[u, v] for u, v in edges]
        best_cliques = baseline
        best_count = len(baseline)
        
        # Use greedy max clique partition
        result = greedy_clique_partition(n, edges, adj, DEADLINE)
        if result is not None:
            count = len(result)
            if count < best_count:
                best_count = count
                best_cliques = result
        
        return best_cliques

    # Solve both instances
    result1 = solve_instance(32, 0.6, 601)
    
    if time.time() > DEADLINE:
        # Fallback: return baseline for second
        edges2 = gen_graph(64, 0.55, 602)
        result2 = [[u, v] for u, v in edges2]
    else:
        result2 = solve_instance(64, 0.55, 602)
    
    return [result1, result2]
