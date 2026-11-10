"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    def random_graph(n, p, seed):
        rng = Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def _greedy_partition_fast(n, edges, seed_heur, mode="max_common"):
        """Fast greedy clique partition using bitmasks for n<=64."""
        neighbors = [0] * n  # bitmask of neighbors
        for u, v in edges:
            neighbors[u] |= (1 << v)
            neighbors[v] |= (1 << u)
        
        remaining_edges = set(edges)
        cliques = []
        rng_heur = Random(seed_heur)
        
        while remaining_edges:
            if mode == "max_common":
                best_edge = None
                best_score = -1
                edge_list = list(remaining_edges)
                rng_heur.shuffle(edge_list)
                
                for u, v in edge_list:
                    common = neighbors[u] & neighbors[v]
                    score = bin(common).count('1')
                    if score > best_score:
                        best_score = score
                        best_edge = (u, v)
            
            elif mode == "min_common":
                best_edge = None
                best_score = float('inf')
                edge_list = list(remaining_edges)
                rng_heur.shuffle(edge_list)
                
                for u, v in edge_list:
                    common = neighbors[u] & neighbors[v]
                    score = bin(common).count('1')
                    if score < best_score:
                        best_score = score
                        best_edge = (u, v)
            elif mode == "max_deg":
                # Pick edge incident to highest degree vertex, then max common among those
                best_edge = None
                best_score = -1
                edge_list = list(remaining_edges)
                rng_heur.shuffle(edge_list)
                
                for u, v in edge_list:
                    deg_u = bin(neighbors[u]).count('1')
                    deg_v = bin(neighbors[v]).count('1')
                    common = neighbors[u] & neighbors[v]
                    score = common.__class__.__call__(common).bit_count() if hasattr(common, 'bit_count') else bin(common).count('1')
                    # Use degree sum as primary, common as tiebreaker
                    total_score = (deg_u + deg_v) * 100 + score
                    if total_score > best_score:
                        best_score = total_score
                        best_edge = (u, v)
            else:
                best_edge = None
                best_score = -1
                edge_list = list(remaining_edges)
                rng_heur.shuffle(edge_list)
                for u, v in edge_list:
                    common = neighbors[u] & neighbors[v]
                    score = bin(common).count('1')
                    if score > best_score:
                        best_score = score
                        best_edge = (u, v)

            if best_edge is None:
                break
                
            u, v = best_edge
            C = [u, v]
            P = neighbors[u] & neighbors[v]
            
            # Greedily expand the clique
            while P:
                # Find w in P with max degree in P
                max_deg = -1
                best_w = -1
                P_list = []
                temp = P
                while temp:
                    lsb = temp & (-temp)
                    w = lsb.bit_length() - 1
                    P_list.append(w)
                    temp ^= lsb
                
                for w in P_list:
                    deg = bin(neighbors[w] & P).count('1')
                    if deg > max_deg:
                        max_deg = deg
                        best_w = w
                
                if best_w < 0:
                    break
                    
                C.append(best_w)
                P = P & neighbors[best_w]
            
            # Remove edges of C from remaining graph
            for i in range(len(C)):
                for j in range(i + 1, len(C)):
                    a, b = C[i], C[j]
                    e = (a, b) if a < b else (b, a)
                    if e in remaining_edges:
                        remaining_edges.remove(e)
                        neighbors[a] &= ~(1 << b)
                        neighbors[b] &= ~(1 << a)
            
            cliques.append(C)
            
        return cliques

    def _local_search(n, edges, cliques, seed_ls, time_limit):
        """Local search: try to merge two cliques if their union is a clique, or swap vertices."""
        if not cliques:
            return cliques
        
        # Build edge-to-clique mapping
        edge_to_clique = {}
        for idx, C in enumerate(cliques):
            for i in range(len(C)):
                for j in range(i + 1, len(C)):
                    a, b = C[i], C[j]
                    e = (a, b) if a < b else (b, a)
                    edge_to_clique[e] = idx
        
        # Build adjacency
        neighbors = [0] * n
        for u, v in edges:
            neighbors[u] |= (1 << v)
            neighbors[v] |= (1 << u)
        
        rng_ls = Random(seed_ls)
        improved = True
        current_cliques = [list(C) for C in cliques]
        
        while improved and time.time() < time_limit:
            improved = False
            # Try merging pairs of cliques
            m = len(current_cliques)
            pairs = [(i, j) for i in range(m) for j in range(i + 1, m)]
            rng_ls.shuffle(pairs)
            
            for i, j in pairs:
                if time.time() > time_limit:
                    break
                C_i = set(current_cliques[i])
                C_j = set(current_cliques[j])
                union = C_i | C_j
                # Check if union is a clique
                is_clique = True
                for u in union:
                    for v in union:
                        if u < v and not (neighbors[u] & (1 << v)):
                            is_clique = False
                            break
                    if not is_clique:
                        break
                if is_clique and len(union) >= 2:
                    # Merge
                    current_cliques[i] = list(union)
                    current_cliques.pop(j)
                    improved = True
                    break
        
        return current_cliques

    def solve_instance(n, p, seed, time_limit):
        edges = random_graph(n, p, seed)
        if not edges:
            return []
        
        best_cliques = None
        best_count = float('inf')
        
        attempt = 0
        modes = ["max_common", "min_common", "max_deg"]
        
        while time.time() < time_limit - 10:  # Leave 10s for local search
            heur_seed = seed + attempt * 1000 + 42
            mode = modes[attempt % len(modes)]
            cliques = _greedy_partition_fast(n, edges, heur_seed, mode)
            
            if len(cliques) < best_count:
                best_count = len(cliques)
                best_cliques = cliques
            
            attempt += 1
                
        if best_cliques is None:
            # Fallback: each edge is a clique
            best_cliques = [[u, v] for u, v in edges]
        else:
            # Apply local search to the best solution
            ls_time = min(time.time(), time_limit)
            best_cliques = _local_search(n, edges, best_cliques, seed + 99999, time_limit)
        
        return best_cliques

    # Allocate time: 40s for n=32, rest for n=64
    t1_limit = start_time + 40.0
    res1 = solve_instance(32, 0.6, 601, t1_limit)
    
    t2_limit = deadline
    res2 = solve_instance(64, 0.55, 602, t2_limit)
    
    return [res1, res2]
# EVOLVE-BLOCK-END
