"""Biclique partition solver using bitmask greedy with randomized restarts and local search."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

N = P['n']
ALL_EDGE_SET = frozenset(frozenset(e) for e in P['edges'])
NUM_EDGES = len(ALL_EDGE_SET)

# Precompute adjacency as bitmasks
ADJ = [0] * N
for u, v in P['edges']:
    ADJ[u] |= (1 << v)
    ADJ[v] |= (1 << u)

def bits_to_list(mask):
    """Convert a bitmask to a list of vertex indices."""
    return [i for i in range(N) if (mask >> i) & 1]

def is_valid_partition(bicliques):
    """Check if the partition is valid: all edges covered exactly once, all cross pairs are host edges."""
    covered = set()
    for L, R in bicliques:
        Ls = set(L)
        Rs = set(R)
        if Ls & Rs:
            return False
        for u in Ls:
            for v in Rs:
                if frozenset((u, v)) not in ALL_EDGE_SET:
                    return False
                covered.add(frozenset((u, v)))
    return covered == set(ALL_EDGE_SET)

def greedy_once(deadline, seed=None):
    """One greedy pass with randomized ordering."""
    if seed is not None:
        random.seed(seed)
    
    remaining_mask = (1 << NUM_EDGES) - 1
    edge_list = list(ALL_EDGE_SET)
    edge_to_idx = {e: i for i, e in enumerate(edge_list)}
    
    bicliques = []
    
    while remaining_mask != 0:
        if time.time() > deadline:
            break
        
        best_biclique = None
        best_edges_covered = 0
        
        # Try each vertex as potential left anchor
        for u in range(N):
            u_incident_edges = []
            for e in edge_list:
                idx = edge_to_idx[e]
                if (remaining_mask >> idx) & 1:
                    es = set(e)
                    if u in es:
                        u_incident_edges.append(e)
            
            if not u_incident_edges:
                continue
            
            R_mask = 0
            for e in u_incident_edges:
                es = list(e)
                other = es[0] if es[1] == u else es[1]
                R_mask |= (1 << other)
            
            if R_mask == 0:
                continue
            
            L_mask = (1 << u)
            
            # Extend L
            extended = True
            while extended and time.time() < deadline:
                extended = False
                for w in range(N):
                    if (L_mask >> w) & 1 or (R_mask >> w) & 1:
                        continue
                    can_add = True
                    for v in bits_to_list(R_mask):
                        e = frozenset((w, v))
                        if e not in ALL_EDGE_SET:
                            can_add = False
                            break
                        idx = edge_to_idx.get(e, -1)
                        if idx < 0 or not ((remaining_mask >> idx) & 1):
                            can_add = False
                            break
                    if can_add:
                        L_mask |= (1 << w)
                        extended = True
                        break
            
            # Extend R
            extended_r = True
            while extended_r and time.time() < deadline:
                extended_r = False
                for w in range(N):
                    if (L_mask >> w) & 1 or (R_mask >> w) & 1:
                        continue
                    can_add = True
                    for u2 in bits_to_list(L_mask):
                        e = frozenset((u2, w))
                        if e not in ALL_EDGE_SET:
                            can_add = False
                            break
                        idx = edge_to_idx.get(e, -1)
                        if idx < 0 or not ((remaining_mask >> idx) & 1):
                            can_add = False
                            break
                    if can_add:
                        R_mask |= (1 << w)
                        extended_r = True
                        break
            
            L_list = bits_to_list(L_mask)
            R_list = bits_to_list(R_mask)
            edges_covered = 0
            for u2 in L_list:
                for v2 in R_list:
                    e = frozenset((u2, v2))
                    idx = edge_to_idx.get(e, -1)
                    if idx >= 0 and (remaining_mask >> idx) & 1:
                        edges_covered += 1
            
            if edges_covered > best_edges_covered:
                best_edges_covered = edges_covered
                best_biclique = (L_list, R_list)
        
        if best_biclique is None:
            uncovered_indices = [i for i in range(NUM_EDGES) if (remaining_mask >> i) & 1]
            if not uncovered_indices:
                break
            idx = random.choice(uncovered_indices)
            e = edge_list[idx]
            u, v = list(e)
            best_biclique = ([u], [v])
        
        L_list, R_list = best_biclique
        bicliques.append((L_list, R_list))
        
        for u2 in L_list:
            for v2 in R_list:
                e = frozenset((u2, v2))
                idx = edge_to_idx.get(e, -1)
                if idx >= 0:
                    remaining_mask &= ~(1 << idx)
    
    return bicliques

def local_search(bicliques, deadline):
    """Try to improve the partition by merging."""
    bicliques = [list(b) for b in bicliques]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        n = len(bicliques)
        for i in range(n):
            if time.time() > deadline:
                return bicliques
            for j in range(i + 1, n):
                L1, R1 = bicliques[i]
                L2, R2 = bicliques[j]
                L1s, R1s = set(L1), set(R1)
                L2s, R2s = set(L2), set(R2)
                
                if L1s & L2s or R1s & R2s or L1s & R2s or L2s & R1s:
                    continue
                
                valid = True
                for u in L1s:
                    for v in R2s:
                        if frozenset((u, v)) not in ALL_EDGE_SET:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    for u in L2s:
                        for v in R1s:
                            if frozenset((u, v)) not in ALL_EDGE_SET:
                                valid = False
                                break
                        if not valid:
                            break
                
                if not valid:
                    continue
                
                merged_L = L1s | L2s
                merged_R = R1s | R2s
                merged_edges = set()
                for u in merged_L:
                    for v in merged_R:
                        merged_edges.add(frozenset((u, v)))
                
                conflict = False
                for k in range(n):
                    if k == i or k == j:
                        continue
                    Lk, Rk = bicliques[k]
                    for u in Lk:
                        for v in Rk:
                            if frozenset((u, v)) in merged_edges:
                                conflict = True
                                break
                        if conflict:
                            break
                    if conflict:
                        break
                
                if conflict:
                    continue
                
                new_bicliques = []
                for k in range(n):
                    if k != i and k != j:
                        new_bicliques.append(bicliques[k])
                new_bicliques.append([list(merged_L), list(merged_R)])
                bicliques = new_bicliques
                improved = True
                break
            if improved:
                break
    
    return bicliques

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    best_bicliques = None
    best_count = float('inf')
    
    num_restarts = 25
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        seed = restart * 12345 + 678
        bicliques = greedy_once(deadline - 5, seed=seed)
        
        if not bicliques:
            continue
        
        bicliques = local_search(bicliques, deadline - 2)
        
        if is_valid_partition(bicliques):
            count = len(bicliques)
            if count < best_count:
                best_count = count
                best_bicliques = bicliques
                if count <= 8:
                    break
    
    if best_bicliques is None:
        best_bicliques = [[list(e)] for e in ALL_EDGE_SET]
    
    result = []
    for L, R in best_bicliques:
        result.append([list(int(x) for x in L), list(int(x) for x in R)])
    
    return result
# EVOLVE-BLOCK-END
