import numpy as np
import itertools
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    D = 11
    indices = list(range(D))
    
    # Generate Type A vectors: one non-zero entry +/-2
    vectors_A = []
    for i in range(D):
        v1 = [0] * D
        v1[i] = 2
        vectors_A.append(v1)
        v2 = [0] * D
        v2[i] = -2
        vectors_A.append(v2)
    vectors_A = np.array(vectors_A, dtype=np.int64)
    
    # Generate all weight-4 supports
    supports = list(combinations(indices, 4))
    n_sup = len(supports)  # 330
    
    # Build support conflict graph: two supports conflict if they share >= 3 indices
    # (since for weight-4 vectors, dot product >= 3 means conflict when both have norm 2)
    sup_adj = [set() for _ in range(n_sup)]
    for i in range(n_sup):
        for j in range(i + 1, n_sup):
            inter = set(supports[i]) & set(supports[j])
            if len(inter) >= 3:
                sup_adj[i].add(j)
                sup_adj[j].add(i)
    
    # Greedy MIS on support graph with multiple orderings
    best_sup_set = []
    best_sup_size = 0
    
    # Degree-based ordering
    sup_degrees = np.array([len(sup_adj[i]) for i in range(n_sup)])
    sup_order = np.argsort(sup_degrees)
    
    selected_mask = np.zeros(n_sup, dtype=bool)
    selected = []
    for idx in sup_order:
        if selected_mask[idx]:
            continue
        if not any(j in selected_mask for j in sup_adj[idx]):
            selected.append(idx)
            selected_mask[idx] = True
    
    best_sup_size = len(selected)
    best_sup_set = selected[:]
    
    # Random restarts for support MIS
    num_sup_restarts = 30
    for r in range(num_sup_restarts):
        if time.time() > deadline - 20:
            break
        random_order = np.random.permutation(n_sup)
        selected_mask = np.zeros(n_sup, dtype=bool)
        selected = []
        for idx in random_order:
            if selected_mask[idx]:
                continue
            if not any(j in selected_mask for j in sup_adj[idx]):
                selected.append(idx)
                selected_mask[idx] = True
        if len(selected) > best_sup_size:
            best_sup_size = len(selected)
            best_sup_set = selected[:]
    
    # For each selected support, we need to pick signs from 16 possibilities
    # such that the resulting vectors have pairwise dot <= 2
    # Two vectors with same support: dot = sum of sign products = 4 - 2*(number of sign disagreements)
    #   dot values: 4 (identical), 2 (2 disagreements), 0 (4 disagreements), -2, -4
    #   Conflict if dot >= 3, so dot=4 is conflict (identical), dot<=2 is OK
    #   So we can pick at most one vector per support? No - we can pick multiple with dot <= 2
    #   dot=2 means exactly 2 sign positions differ. That's allowed.
    
    # For different supports (which share at most 2 indices since they're in the MIS):
    # dot = sum over shared indices of sign products
    # Max dot = 2 (if all shared signs agree)
    # So cross-support conflicts are automatically satisfied!
    
    # Therefore: for each selected support, we can include ALL sign combinations
    # except we need to avoid identical vectors (dot=4)
    # But wait - we can include multiple vectors from the same support as long as they're not identical
    # dot between two distinct vectors on same support: 2, 0, -2, -4
    # All <= 2, so all are fine!
    
    # So we can include ALL 16 sign combinations for each selected support!
    
    selected_vectors_D = []
    for idx in best_sup_set:
        support = supports[idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, sign in zip(support, signs):
                v[pos] = sign
            selected_vectors_D.append(v)
    
    selected_vectors_D = np.array(selected_vectors_D, dtype=np.int64) if selected_vectors_D else np.empty((0, D), dtype=np.int64)
    
    # Now check Type A vectors against selected Type D vectors
    # Type A vector has norm 2, Type D has norm 2
    # Need ||u-v||^2 >= 4, i.e., u.v <= 2
    # Type A: e_i * 2 or -e_i * 2
    # Type D: has entries +/-1 at 4 positions
    # dot = 2 * (sign at position i if i is in support, else 0)
    # dot can be: 2, -2, or 0
    # All <= 2, so all Type A vectors are compatible with all Type D vectors!
    
    # So we can include ALL Type A vectors!
    
    # Let's also try to add Type B vectors: weight-2, +/-1
    # Type B: exactly 2 non-zero entries, each +/-1, norm sqrt(2)
    # But the constraint is min pairwise distance >= max norm
    # If we mix norms, we need: ||u-v|| >= max(||u||, ||v||)
    # For Type A (norm 2) and Type B (norm sqrt(2)): ||u-v|| >= 2
    #   ||u-v||^2 = ||u||^2 + ||v||^2 - 2*u.v = 4 + 2 - 2*u.v = 6 - 2*u.v
    #   Need 6 - 2*u.v >= 4, so u.v <= 1
    # For Type B (norm sqrt(2)) and Type D (norm 2): ||u-v|| >= 2
    #   ||u-v||^2 = 2 + 4 - 2*u.v = 6 - 2*u.v
    #   Need 6 - 2*u.v >= 4, so u.v <= 1
    # For Type B and Type B: ||u-v|| >= sqrt(2)
    #   ||u-v||^2 = 2 + 2 - 2*u.v = 4 - 2*u.v
    #   Need 4 - 2*u.v >= 2, so u.v <= 1
    
    # Let's generate Type B vectors and select a compatible subset
    vectors_B_all = []
    for i in range(D):
        for j in range(i+1, D):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    vectors_B_all.append(v)
    vectors_B_all = np.array(vectors_B_all, dtype=np.int64)
    n_B = len(vectors_B_all)  # 220
    
    # Check compatibility of Type B with existing vectors
    # With Type A: u.v <= 1
    # With Type D: u.v <= 1
    
    # Compute max dot of each Type B with any Type A
    # Type A: 22 vectors
    if len(vectors_A) > 0 and n_B > 0:
        dots_BA = vectors_B_all @ vectors_A.T  # (220, 22)
        max_dot_BA = dots_BA.max(axis=1)
        valid_B_vs_A = max_dot_BA <= 1
    else:
        valid_B_vs_A = np.ones(n_B, dtype=bool)
    
    # With Type D
    if len(selected_vectors_D) > 0 and n_B > 0:
        dots_BD = vectors_B_all @ selected_vectors_D.T  # (220, up to 5280)
        max_dot_BD = dots_BD.max(axis=1)
        valid_B_vs_D = max_dot_BD <= 1
    else:
        valid_B_vs_D = np.ones(n_B, dtype=bool)
    
    valid_B_mask = valid_B_vs_A & valid_B_vs_D
    valid_B_indices = np.where(valid_B_mask)[0]
    
    # Now select a subset of valid Type B vectors with pairwise dot <= 1
    # Conflict if dot > 1, i.e., dot >= 2
    # Two Type B vectors: dot can be -2, -1, 0, 1, 2
    # dot=2 means identical vectors
    # So we just need to avoid duplicates? Let's check:
    # u = e_i + e_j, v = e_i + e_j: dot = 2 (identical)
    # u = e_i + e_j, v = e_i - e_j: dot = 1 - 1 = 0
    # u = e_i + e_j, v = -e_i + e_j: dot = -1 + 1 = 0
    # u = e_i + e_j, v = -e_i - e_j: dot = -1 - 1 = -2
    # u = e_i + e_j, v = e_k + e_l (disjoint supports): dot = 0
    # u = e_i + e_j, v = e_i + e_k (share i): dot = 1
    # So the only conflict is identical vectors (dot=2)
    # Since our generation doesn't produce duplicates, all valid Type B vectors are mutually compatible!
    
    selected_B = vectors_B_all[valid_B_mask]
    
    # Combine all vectors
    all_parts = [vectors_A]
    if len(selected_vectors_D) > 0:
        all_parts.append(selected_vectors_D)
    if len(selected_B) > 0:
        all_parts.append(selected_B)
    
    final_vectors = np.vstack(all_parts)
    
    return final_vectors
