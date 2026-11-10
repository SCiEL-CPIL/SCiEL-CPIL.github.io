import numpy as np
import time
from itertools import combinations, product

def kissing_number11() -> np.ndarray:
    d = 11
    deadline = time.time() + 570
    
    # Weight-4 vectors with precomputed conflict structure
    four_positions = list(combinations(range(d), 4))
    n_pos = len(four_positions)  # 330
    
    # Create all candidate vectors
    candidates = []
    cand_support = []
    for pos in four_positions:
        for signs in product([-1, 1], repeat=4):
            v = np.zeros(d, dtype=np.int64)
            for i, p in enumerate(pos):
                v[p] = signs[i]
            candidates.append(v)
            cand_support.append(tuple(pos))
    
    n_cand = len(candidates)  # 5280
    
    # Group candidates by support
    support_to_cands = {}
    for i, s in enumerate(cand_support):
        if s not in support_to_cands:
            support_to_cands[s] = []
        support_to_cands[s].append(i)
    
    # Precompute same-support conflicts: conflict if Hamming distance on support <= 1
    same_support_conflict = {}
    for s, cands in support_to_cands.items():
        conflict_map = {i: set() for i in cands}
        for i in range(len(cands)):
            for j in range(i+1, len(cands)):
                vi = candidates[cands[i]]
                vj = candidates[cands[j]]
                ham = sum(1 for p in s if vi[p] != vj[p])
                if ham <= 1:
                    conflict_map[cands[i]].add(cands[j])
                    conflict_map[cands[j]].add(cands[i])
        same_support_conflict[s] = conflict_map
    
    # Precompute supports with intersection size 3
    support_list = list(support_to_cands.keys())
    support_to_idx = {s: i for i, s in enumerate(support_list)}
    n_supports = len(support_list)
    
    # For each pair of supports with intersection 3, precompute conflicting candidate pairs
    inter3_conflict = {}
    for i in range(n_supports):
        s1_set = set(support_list[i])
        for j in range(i+1, n_supports):
            s2_set = set(support_list[j])
            inter = s1_set & s2_set
            if len(inter) == 3:
                inter_tuple = tuple(sorted(inter))
                cands1 = support_to_cands[support_list[i]]
                cands2 = support_to_cands[support_list[j]]
                conflict_set = set()
                for i1 in cands1:
                    v1 = candidates[i1]
                    for i2 in cands2:
                        v2 = candidates[i2]
                        # Conflict if signs agree on intersection
                        if all(v1[p] == v2[p] for p in inter_tuple):
                            conflict_set.add((i1, i2))
                            conflict_set.add((i2, i1))
                inter3_conflict[(i, j)] = conflict_set
    
    # Greedy with precomputed conflicts
    best_points = []
    best_size = 0
    
    rng = np.random.default_rng(42)
    
    num_restarts = 0
    while time.time() < deadline - 30:
        num_restarts += 1
        
        order = rng.permutation(n_cand)
        selected = []
        selected_set = set()
        
        for idx in order:
            s_idx = support_to_idx[cand_support[idx]]
            ok = True
            for sel in selected:
                sel_s_idx = support_to_idx[cand_support[sel]]
                # Check same support
                if s_idx == sel_s_idx:
                    if sel in same_support_conflict[cand_support[idx]][idx]:
                        ok = False
                        break
                else:
                    key = (min(s_idx, sel_s_idx), max(s_idx, sel_s_idx))
                    if key in inter3_conflict:
                        if (idx, sel) in inter3_conflict[key]:
                            ok = False
                            break
            if ok:
                selected.append(idx)
                selected_set.add(idx)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_points = selected[:]
        
        if best_size >= 120:
            break
    
    if len(best_points) < 2:
        points = [np.ones(d, dtype=np.int64), -np.ones(d, dtype=np.int64)]
    else:
        points = [candidates[i] for i in best_points]
    
    pts_array = np.array(points, dtype=np.int64)
    return np.around(pts_array).astype(np.int64)
