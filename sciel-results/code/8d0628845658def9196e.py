import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Precompute Q3 bipartite adjacency patterns
    M_canon = [
        [1, 1, 1, 0],
        [1, 1, 0, 1],
        [1, 0, 1, 1],
        [0, 1, 1, 1]
    ]
    
    def mat_to_hash(M):
        h = 0
        for i in range(4):
            for j in range(4):
                if M[i][j]:
                    h |= (1 << (i*4 + j))
        return h
    
    valid_hashes = set()
    for perm_r in itertools.permutations(range(4)):
        for perm_c in itertools.permutations(range(4)):
            M_perm = [[0]*4 for _ in range(4)]
            for i in range(4):
                for j in range(4):
                    M_perm[i][j] = M_canon[perm_r[i]][perm_c[j]]
            valid_hashes.add(mat_to_hash(M_perm))
            
    def check_8_set(v_list, adj_mask):
        # v_list is a tuple of 8 vertices
        for i in range(8):
            for j in range(i+1, 8):
                for k in range(j+1, 8):
                    for l in range(k+1, 8):
                        A_idx = [i, j, k, l]
                        B_idx = [x for x in range(8) if x not in A_idx]
                        
                        A = [v_list[idx] for idx in A_idx]
                        B = [v_list[idx] for idx in B_idx]
                        
                        # Check internal edges
                        valid = True
                        for idx1 in range(4):
                            for idx2 in range(idx1+1, 4):
                                if (adj_mask[A[idx1]] >> A[idx2]) & 1:
                                    valid = False
                                    break
                                if (adj_mask[B[idx1]] >> B[idx2]) & 1:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            continue
                        
                        M = [[0]*4 for _ in range(4)]
                        for i_a in range(4):
                            u = A[i_a]
                            for i_b in range(4):
                                v = B[i_b]
                                if (adj_mask[u] >> v) & 1:
                                    M[i_a][i_b] = 1
                                    
                        if mat_to_hash(M) in valid_hashes:
                            return True
        return False

    def is_safe_edge(u, v, adj_mask):
        N_u = [w for w in range(16) if w != u and w != v and (adj_mask[u] >> w) & 1]
        N_v = [w for w in range(16) if w != u and w != v and (adj_mask[v] >> w) & 1]
        
        if len(N_u) < 2 or len(N_v) < 2:
            return True
        
        for i in range(len(N_u)):
            for j in range(i+1, len(N_u)):
                x1, x2 = N_u[i], N_u[j]
                for k in range(len(N_v)):
                    for l in range(k+1, len(N_v)):
                        y1, y2 = N_v[k], N_v[l]
                        
                        if not ((adj_mask[x1] >> y1) & 1):
                            continue
                        if not ((adj_mask[x2] >> y2) & 1):
                            continue
                        
                        cand_z1 = []
                        for w in range(16):
                            if w in (u, v, x1, x2, y1, y2):
                                continue
                            if (adj_mask[x1] >> w) & 1 and (adj_mask[y2] >> w) & 1:
                                cand_z1.append(w)
                                
                        cand_z2 = []
                        for w in range(16):
                            if w in (u, v, x1, x2, y1, y2):
                                continue
                            if (adj_mask[x2] >> w) & 1 and (adj_mask[y1] >> w) & 1:
                                cand_z2.append(w)
                                
                        for z1 in cand_z1:
                            for z2 in cand_z2:
                                if z1 == z2:
                                    continue
                                if (adj_mask[z1] >> z2) & 1:
                                    return False
        
        return True

    best_edges = set()
    best_count = -1
    
    for restart in range(200):
        if time.time() > deadline - 10:
            break
        
        # Start with random graph density 0.6
        edges = set()
        adj_mask = [0] * 16
        for u in range(16):
            for v in range(u+1, 16):
                if random.random() < 0.6:
                    edges.add((u, v))
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
        
        # Remove Q3s
        found_q3 = True
        while found_q3 and time.time() < deadline - 20:
            found_q3 = False
            for _ in range(50):
                if time.time() > deadline - 20:
                    break
                subset = random.sample(range(16), 8)
                if check_8_set(tuple(subset), adj_mask):
                    found_q3 = True
                    sub_edges = [(u, v) for u in subset for v in subset if u < v and (adj_mask[u] >> v) & 1]
                    if sub_edges:
                        e = random.choice(sub_edges)
                        edges.remove(e)
                        adj_mask[e[0]] &= ~(1 << e[1])
                        adj_mask[e[1]] &= ~(1 << e[0])
                    break
        
        # Polish
        non_edges = set(itertools.combinations(range(16), 2)) - edges
        non_edges_list = list(non_edges)
        random.shuffle(non_edges_list)
        
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            for e in non_edges_list:
                if time.time() > deadline - 5:
                    break
                u, v = e
                if e in edges:
                    continue
                if is_safe_edge(u, v, adj_mask):
                    edges.add(e)
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
                    changed = True
                    non_edges_list.remove(e)
                    if len(edges) > best_count:
                        best_count = len(edges)
                        best_edges = edges.copy()
                        
    return [list(e) for e in best_edges]
