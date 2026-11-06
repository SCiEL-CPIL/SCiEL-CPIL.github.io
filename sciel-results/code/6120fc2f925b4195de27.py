import time
import random
from itertools import combinations, permutations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n = 16
    
    # Construct Clebsch graph (48 edges, Q3-free since girth 5)
    def construct_clebsch():
        edges = []
        for i in range(16):
            for j in range(i+1, 16):
                xor = i ^ j
                if bin(xor).count('1') == 2:
                    edges.append((i, j))
        return edges
    
    clebsch_edges = set(construct_clebsch())
    
    # All possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    candidate_edges = [e for e in all_edges if e not in clebsch_edges]
    
    # Fast Q3 check using precomputed Q3 embeddings
    # Precompute all 8-subsets and for each, the Q3 edge patterns
    # For efficiency, precompute once
    
    # Q3 bipartite structure
    # Q3 vertices 0-7, bipartition: even parity {0,3,5,6}, odd parity {1,2,4,7}
    # Edges between parts:
    q3_left = [0, 3, 5, 6]
    q3_right = [1, 2, 4, 7]
    q3_edges_bip = [(0,1),(0,2),(0,4),(3,1),(3,2),(3,7),(5,1),(5,4),(5,7),(6,2),(6,4),(6,7)]
    # Map to indices in left/right arrays
    left_idx = {v:i for i,v in enumerate(q3_left)}
    right_idx = {v:i for i,v in enumerate(q3_right)}
    q3_pat = [[0]*4 for _ in range(4)]
    for u,v in q3_edges_bip:
        q3_pat[left_idx[u]][right_idx[v]] = 1
    
    # Precompute all 8-subsets with their 4+4 splits
    # For each 8-subset S and each 4-subset L of S, store (S, L, R)
    # This is 12870 * 70 = ~900k entries, too many
    
    # Instead, do on-the-fly check but optimize
    
    def is_q3_free(adj):
        """adj: list of 16 integers, bitmask of neighbors"""
        # For each 8-vertex subset, check if it contains Q3
        # Optimize: only check subsets with >= 12 internal edges
        for S in combinations(range(16), 8):
            S_list = list(S)
            # Count internal edges
            internal_edges = 0
            for i in range(8):
                for j in range(i+1, 8):
                    if adj[S_list[i]] & (1 << S_list[j]):
                        internal_edges += 1
            if internal_edges < 12:
                continue
            
            # Try all 4+4 splits
            for left_idx_set in combinations(range(8), 4):
                left = [S_list[i] for i in left_idx_set]
                right = [S_list[i] for i in range(8) if i not in left_idx_set]
                
                # Check degrees
                ok = True
                for u in left:
                    cnt = sum(1 for v in right if adj[u] & (1 << v))
                    if cnt < 3:
                        ok = False
                        break
                if not ok:
                    continue
                for v in right:
                    cnt = sum(1 for u in left if adj[u] & (1 << v))
                    if cnt < 3:
                        ok = False
                        break
                if not ok:
                    continue
                
                # Build 4x4 bipartite adjacency
                bip = [[0]*4 for _ in range(4)]
                for i, u in enumerate(left):
                    for j, v in enumerate(right):
                        bip[i][j] = 1 if (adj[u] & (1 << v)) else 0
                
                # Check if q3_pat is a subgraph of bip
                found = False
                for pi in permutations(range(4)):
                    for sigma in permutations(range(4)):
                        ok2 = True
                        for i in range(4):
                            for j in range(4):
                                if q3_pat[i][j]:
                                    if not bip[pi[i]][sigma[j]]:
                                        ok2 = False
                                        break
                            if not ok2:
                                break
                        if ok2:
                            found = True
                            break
                    if found:
                        break
                if found:
                    return False
        return True
    
    current_edges = set(clebsch_edges)
    
    # Greedily add edges
    random.seed(123)
    random.shuffle(candidate_edges)
    
    for edge in candidate_edges:
        if time.time() > deadline - 15:
            break
        if edge in current_edges:
            continue
        test_edges = current_edges | {edge}
        
        # Build adjacency
        adj = [0] * 16
        for u, v in test_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        if is_q3_free(adj):
            current_edges = test_edges
    
    # Local search: try to add more edges
    for _ in range(10):
        if time.time() > deadline - 5:
            break
        improved = False
        random.shuffle(candidate_edges)
        for edge in candidate_edges:
            if edge in current_edges:
                continue
            test_edges = current_edges | {edge}
            adj = [0] * 16
            for u, v in test_edges:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            if is_q3_free(adj):
                current_edges = test_edges
                improved = True
                break
        if not improved:
            break
    
    return [list(e) for e in current_edges]

