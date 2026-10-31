import numpy as np
import time
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 140.0
    
    D = 11
    supports = list(itertools.combinations(range(D), 4))
    n_supports = len(supports)
    
    all_vectors = []
    for s in supports:
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, sign in zip(s, signs):
                v[pos] = sign
            all_vectors.append(v)
    
    all_vectors_np = np.array(all_vectors, dtype=np.int8)
    n_total = len(all_vectors_np)
    
    # Build conflict graph
    tri_to_tet = defaultdict(list)
    for i, s in enumerate(supports):
        for tri in itertools.combinations(s, 3):
            tri_to_tet[tri].append(i)
    
    # Use numpy boolean matrix for conflicts (memory efficient enough)
    conflict = np.zeros((n_total, n_total), dtype=bool)
    
    for tri, tets in tri_to_tet.items():
        for u in range(len(tets)):
            for v in range(u + 1, len(tets)):
                s1 = tets[u]
                s2 = tets[v]
                base1 = s1 * 16
                base2 = s2 * 16
                v1 = all_vectors_np[base1:base1+16]
                v2 = all_vectors_np[base2:base2+16]
                d = v1 @ v2.T
                idx = np.argwhere(d > 2)
                for i, j in idx:
                    conflict[base1+i, base2+j] = True
                    conflict[base2+j, base1+i] = True
    
    # Convert to list of sets for faster neighborhood queries
    neighbors = [set(np.where(conflict[i])[0]) for i in range(n_total)]
    deg = np.array([len(neighbors[i]) for i in range(n_total)], dtype=np.int32)
    
    rng = np.random.default_rng(42)
    best_k = 0
    best_selected = None
    
    def greedy():
        avail = np.ones(n_total, dtype=bool)
        deg_cur = deg.copy()
        selected = np.zeros(n_total, dtype=bool)
        
        while avail.any():
            avail_idx = np.where(avail)[0]
            if len(avail_idx) == 0:
                break
            d = deg_cur[avail_idx]
            min_d = np.min(d)
            min_rel = np.where(d == min_d)[0]
            pick = min_rel[rng.integers(len(min_rel))]
            u = avail_idx[pick]
            selected[u] = True
            avail[u] = False
            for v in neighbors[u]:
                if avail[v]:
                    avail[v] = False
                    for w in neighbors[v]:
                        if avail[w]:
                            deg_cur[w] -= 1
        return selected
    
    def refine(selected):
        """Local search: try to add vertices, then try swaps"""
        sel_set = set(np.where(selected)[0])
        improved = True
        while improved and time.time() - start_time < deadline - 15:
            improved = False
            # Try adding
            for i in range(n_total):
                if i not in sel_set:
                    if not (neighbors[i] & sel_set):
                        sel_set.add(i)
                        improved = True
                        break
            # Try 1-for-2 swaps
            if len(sel_set) >= 2:
                sel_list = list(sel_set)
                for rem in sel_list:
                    if time.time() - start_time > deadline - 15:
                        break
                    rem_neigh = neighbors[rem]
                    # Candidates: not in sel_set, not rem, not neighbor of any remaining
                    remaining = sel_set - {rem}
                    cand = []
                    for i in range(n_total):
                        if i in sel_set or i == rem:
                            continue
                        if not (neighbors[i] & remaining):
                            cand.append(i)
                    # Try pairs
                    for a in range(len(cand)):
                        ca = cand[a]
                        for b in range(a+1, len(cand)):
                            cb = cand[b]
                            if ca in neighbors[cb]:
                                continue
                            if not (neighbors[ca] & remaining) and not (neighbors[cb] & remaining):
                                sel_set.discard(rem)
                                sel_set.add(ca)
                                sel_set.add(cb)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
        return sel_set
    
    for restart in range(300):
        if time.time() - start_time > deadline - 5:
            break
        sel = greedy()
        k = np.sum(sel)
        sel_set = refine(sel)
        k = len(sel_set)
        if k > best_k:
            best_k = k
            best_selected = np.zeros(n_total, dtype=bool)
            for i in sel_set:
                best_selected[i] = True
    
    if best_selected is None:
        return np.array([], dtype=np.int64).reshape(0, D)
    return all_vectors_np[best_selected].astype(np.int64)
