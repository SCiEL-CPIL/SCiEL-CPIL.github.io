"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
from collections import defaultdict

def construct():
    """
    Construct nonzero integer vectors in R^11 whose minimum pairwise distance
    is at least the maximum vector norm.
    
    Strategy: Greedy maximum independent set on the weight-4 vector conflict graph,
    using the algebraic characterization of conflicts for weight-4 vectors.
    Additionally include weight-2 vectors that don't conflict with selected weight-4 vectors.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    # Generate weight-4 vectors: four ±1's, rest 0 (norm^2 = 4)
    # Represent as (support_tuple, sign_tuple) for efficient conflict checking
    w4_supports = list(itertools.combinations(range(D), 4))
    
    # For the set-packing approach, we need to check conflicts between weight-4 vectors.
    # Two weight-4 vectors u, v conflict iff <u,v> > 2 (since 4+4-2<u,v> >= 4 → <u,v> <= 2)
    # <u,v> for weight-4 vectors = sum over positions where both have nonzero entries.
    # If supports intersect in k positions, <u,v> = (# same sign) - (# opposite sign) over those k positions.
    # Maximum <u,v> when supports are identical: 4 (all same sign) → conflict
    # <u,v> = 3: 3 same, 1 opposite → conflict
    # <u,v> = 2: 2 same, 2 opposite → OK (boundary)
    # <u,v> = 1: OK
    # etc.
    
    # Conflict between two weight-4 vectors with supports S1, S2:
    # Let I = S1 ∩ S2, |I| = k
    # <u,v> = sum_{i in I} u_i * v_i
    # This ranges from -k to k in steps of 2.
    # Conflict iff <u,v> > 2, i.e., <u,v> >= 3.
    # For k=4: <u,v> can be 4, 2, 0, -2, -4. Conflict if 4 or 2? No: conflict if > 2, so only 4.
    #   Wait: <u,v> = 4 means all 4 same sign. <u,v> = 2 means 3 same 1 opp. 2 is NOT > 2, so OK.
    #   Actually let me recheck: dist_sq = 4+4-2*<u,v> >= 4 means <u,v> <= 2.
    #   So conflict iff <u,v> > 2, i.e., <u,v> >= 3.
    #   For k=4: possible values are 4, 2, 0, -2, -4. Only 4 > 2. So conflict iff all 4 same sign.
    # For k=3: possible values are 3, 1, -1, -3. Conflict iff 3 > 2, so iff all 3 same sign.
    # For k=2: possible values are 2, 0, -2. None > 2. Never conflict.
    # For k=1: possible values are 1, -1. Never conflict.
    # For k=0: <u,v> = 0. Never conflict.
    
    # So the conflict condition for two weight-4 vectors is:
    # - If supports are identical (k=4): conflict iff all 4 signs are the same
    # - If |intersection| = 3: conflict iff the 3 overlapping positions have all same signs
    # - If |intersection| <= 2: never conflict
    
    # Generate all weight-4 vectors as (support, signs) pairs
    # For each support, generate 16 sign patterns
    w4_vectors = []  # list of (support_tuple, sign_tuple)
    for support in w4_supports:
        for signs in itertools.product([1, -1], repeat=4):
            w4_vectors.append((support, signs))
    
    n_w4_total = len(w4_vectors)
    
    # Build conflict structure using the algebraic characterization
    # For efficiency, group by support
    support_to_indices = defaultdict(list)
    for idx, (sup, sig) in enumerate(w4_vectors):
        support_to_indices[sup].append(idx)
    
    # For each pair of supports with |intersection| >= 3, determine conflicts
    # |intersection| = 4: same support, conflict iff same signs
    # |intersection| = 3: different supports sharing 3 positions, conflict iff those 3 signs match
    
    # Build adjacency as a set of frozensets for the conflict graph
    # Actually, let's build a bitset-based conflict representation
    # n_w4_total can be up to C(11,4)*16 = 330*16 = 5280
    
    # Build conflict list for each vector
    # For vector i with (sup_i, sig_i):
    #   Conflict with j if:
    #   - sup_j == sup_i and sig_j == sig_i (same vector, excluded by diagonal)
    #     Actually same vector: dist=0 < 4, so it IS a conflict, but we exclude self.
    #     Same support, same signs: dist_sq = 0 < 4 → conflict
    #     Same support, different signs: depends on how many signs differ
    #       If 1 sign differs: <u,v> = 2 (3 same, 1 diff) → dist_sq = 4+4-4 = 4 → OK
    #       If 2 signs differ: <u,v> = 0 → dist_sq = 8 → OK
    #       If 3 signs differ: <u,v> = -2 → dist_sq = 12 → OK
    #       If 4 signs differ: <u,v> = -4 → dist_sq = 16 → OK
    #     So same support: conflict ONLY if identical (all 4 same sign)
    #   - |sup_i ∩ sup_j| = 3: conflict iff the 3 shared positions have matching signs
    #   - |sup_i ∩ sup_j| <= 2: no conflict
    
    # Let's build the conflict graph efficiently
    # For each pair of supports with intersection size 3 or 4, enumerate sign conflicts
    
    # Precompute for each pair of supports their intersection
    support_list = list(support_to_indices.keys())
    n_sup = len(support_list)
    
    # For efficiency, build conflict as a list of lists
    conflicts = [set() for _ in range(n_w4_total)]
    
    # Case 1: Same support (intersection = 4)
    # Only identical sign patterns conflict
    for sup, indices in support_to_indices.items():
        # All indices in this group have the same support
        # They conflict with each other only if same signs
        # Since each (sup, sig) is unique, no two different indices have same (sup, sig)
        # So actually, within same support, NO two distinct vectors conflict!
        # Wait, that's wrong. Let me recheck.
        # Same support, different signs:
        #   1 sign diff: <u,v> = 3-1 = 2, dist_sq = 8-4 = 4 >= 4 → OK
        #   So no conflict within same support for distinct vectors.
        pass  # No intra-support conflicts for distinct vectors
    
    # Case 2: Different supports with |intersection| = 3
    # For supports S1, S2 with |S1∩S2| = 3:
    # Let shared = S1 ∩ S2 (3 positions)
    # For u in S1 with signs sig1, v in S2 with signs sig2:
    # <u,v> = sum over shared positions of u_i * v_i
    # This equals (number of same signs in shared) - (number of opposite signs in shared)
    # = 2*(# same) - 3
    # Possible values: -3, -1, 1, 3
    # Conflict iff <u,v> > 2, i.e., <u,v> = 3, i.e., all 3 same sign.
    
    # Build mapping from support to its index in support_list
    sup_to_idx = {sup: i for i, sup in enumerate(support_list)}
    
    # For each pair of supports with intersection size 3
    for i in range(n_sup):
        S1 = support_list[i]
        for j in range(i + 1, n_sup):
            S2 = support_list[j]
            inter = set(S1) & set(S2)
            if len(inter) != 3:
                continue
            
            # Get indices for S1 and S2
            idx1_list = support_to_indices[S1]
            idx2_list = support_to_indices[S2]
            
            # For each sign pattern in S1 and S2, check if shared positions have all same signs
            # Map shared positions to their indices in S1 and S2
            inter_sorted = sorted(inter)
            pos_in_S1 = [S1.index(p) for p in inter_sorted]
            pos_in_S2 = [S2.index(p) for p in inter_sorted]
            
            for idx1 in idx1_list:
                sig1 = w4_vectors[idx1][1]
                for idx2 in idx2_list:
                    sig2 = w4_vectors[idx2][1]
                    # Check if all 3 shared positions have same sign
                    all_same = True
                    for p1, p2 in zip(pos_in_S1, pos_in_S2):
                        if sig1[p1] != sig2[p2]:
                            all_same = False
                            break
                    if all_same:
                        conflicts[idx1].add(idx2)
                        conflicts[idx2].add(idx1)
    
    # Now we have the conflict graph for weight-4 vectors.
    # Additionally, we need to consider weight-2 vectors and weight-1 doubled vectors.
    # But let's first maximize the weight-4 set, then try to add others.
    
    # Greedy maximum independent set on the weight-4 conflict graph
    # Using degree-based heuristic with multiple restarts
    
    np.random.seed(42)
    best_w4_selected = []
    best_w4_size = 0
    
    # Initial degrees
    initial_degrees = np.array([len(c) for c in conflicts], dtype=np.int32)
    
    # Multiple restarts with different random seeds
    max_restarts = 500
    
    for attempt in range(max_restarts):
        if time.time() > deadline - 20:
            break
        
        remaining = np.ones(n_w4_total, dtype=bool)
        degrees = initial_degrees.copy()
        selected = []
        
        while remaining.any():
            if time.time() > deadline - 20:
                break
            
            rem_indices = np.where(remaining)[0]
            if len(rem_indices) == 0:
                break
            
            rem_degrees = degrees[rem_indices]
            min_deg = np.min(rem_degrees)
            min_mask = rem_degrees == min_deg
            min_indices_local = np.where(min_mask)[0]
            
            chosen_local = np.random.choice(min_indices_local)
            chosen_global = rem_indices[chosen_local]
            
            selected.append(chosen_global)
            
            # Remove chosen and its neighbors
            to_remove = set(conflicts[chosen_global])
            to_remove.add(chosen_global)
            
            # Update remaining and degrees
            for r in to_remove:
                if remaining[r]:
                    remaining[r] = False
                    # Update degrees of remaining neighbors
                    for nb in conflicts[r]:
                        if remaining[nb]:
                            degrees[nb] -= 1
        
        if len(selected) > best_w4_size:
            best_w4_size = len(selected)
            best_w4_selected = selected.copy()
    
    # Now try to add weight-2 and weight-1 doubled vectors
    # Weight-2 vectors: two ±1's, norm^2 = 2
    # Weight-1 doubled: one ±2, norm^2 = 4
    
    # For a weight-2 vector w (positions i,j with signs si,sj) and weight-4 vector u:
    # dist_sq = 2 + 4 - 2<u,w> >= 4 → <u,w> <= 1
    # <u,w> = sum over positions where both nonzero.
    # If {i,j} ⊆ support(u): <u,w> = si*u_i + sj*u_j, which is in {-2, 0, 2}
    #   Conflict iff <u,w> > 1, i.e., <u,w> = 2 (both same sign)
    # If |{i,j} ∩ support(u)| = 1: <u,w> = ±1, never > 1
    # If |{i,j} ∩ support(u)| = 0: <u,w> = 0
    
    # For weight-1 doubled d (position i, value ±2) and weight-4 vector u:
    # dist_sq = 4 + 4 - 2<u,d> >= 4 → <u,d> <= 2
    # <u,d> = ±2 * u_i (if i in support(u), else 0)
    # If i in support(u): <u,d> = ±2, conflict iff |<u,d>| > 2, never. So always OK?
    #   Wait: <u,d> = 2*d_i*u_i where d_i = ±2. So <u,d> = ±2*u_i = ±2.
    #   dist_sq = 8 - 2*(±2) = 8∓4 = 4 or 12. Both >= 4. OK.
    # If i not in support(u): <u,d> = 0. OK.
    # So weight-1 doubled vectors NEVER conflict with weight-4 vectors!
    
    # For two weight-2 vectors w1, w2:
    # dist_sq = 2+2-2<w1,w2> >= 4 → <w1,w2> <= 0
    # If supports are same: <w1,w2> = ±2 or 0. Conflict if 2 (same signs).
    # If |intersection| = 1: <w1,w2> = ±1. OK.
    # If |intersection| = 0: 0. OK.
    
    # For weight-2 w and weight-1 doubled d:
    # dist_sq = 2+4-2<w,d> >= 4 → <w,d> <= 1
    # <w,d> = d_i * w_i if i in support(w), else 0. d_i = ±2.
    # If i in support(w): <w,d> = ±2. Conflict iff > 1, i.e., = 2.
    # So conflict iff d and w share a position and have same sign.
    
    # For two weight-1 doubled d1, d2:
    # dist_sq = 4+4-2<d1,d2> >= 4 → <d1,d2> <= 2
    # If same position: <d1,d2> = ±4. Conflict if 4.
    # If different positions: 0. OK.
    
    # Strategy: Add all weight-1 doubled vectors (22 of them) since they don't conflict with w4.
    # Then try to add weight-2 vectors that don't conflict with selected w4 and each other.
    
    # Build the selected weight-4 vectors as numpy array
    selected_w4 = np.array([w4_vectors[i] for i in best_w4_selected])
    
    # Convert to integer vectors
    w4_int_vectors = np.zeros((best_w4_size, D), dtype=np.int64)
    for i, (sup, sig) in enumerate(selected_w4):
        for pos, sign in zip(sup, sig):
            w4_int_vectors[i, pos] = sign
    
    # Add weight-1 doubled vectors
    w1_vectors = []
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s * 2
            w1_vectors.append(v)
    w1_vectors = np.array(w1_vectors, dtype=np.int64)
    
    # Try to add weight-2 vectors
    # Generate all weight-2 vectors
    w2_candidates = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                w2_candidates.append((i, j, si, sj))
    
    # For each weight-2 candidate, check if it conflicts with any selected w4
    # Conflict with w4 iff they share both positions AND have same signs at both
    # (i.e., <w2, w4> = 2)
    
    # Build a lookup: for each position pair (i,j), which selected w4 vectors have both i,j in support
    # and what are their signs at i and j?
    
    # More efficient: for each selected w4, mark its support
    # For a w2 candidate (i, j, si, sj), it conflicts with w4 vector u iff
    #   i ∈ support(u) and j ∈ support(u) and u_i = si and u_j = sj
    
    # Build a set of (i, j, si, sj) tuples for each selected w4
    w4_blocked_w2 = set()
    for i, (sup, sig) in enumerate(selected_w4):
        for a, b in itertools.combinations(range(4), 2):
            pos_a, pos_b = sup[a], sup[b]
            sig_a, sig_b = sig[a], sig[b]
            if pos_a > pos_b:
                pos_a, pos_b = pos_b, pos_a
                sig_a, sig_b = sig_b, sig_a
            w4_blocked_w2.add((pos_a, pos_b, sig_a, sig_b))
    
    # Also, w2 vectors conflict with w1 vectors if they share a position with same sign
    # And w2 vectors conflict with each other if same support and same signs
    
    # Greedy selection of w2 vectors
    w2_selected = []
    w2_selected_set = set()  # (i, j, si, sj) for selected w2
    
    for (i, j, si, sj) in w2_candidates:
        if time.time() > deadline - 10:
            break
        # Check conflict with w4
        if (i, j, si, sj) in w4_blocked_w2:
            continue
        # Check conflict with already selected w2
        # Conflict iff same support and same signs
        if (i, j, si, sj) in w2_selected_set:
            continue
        # Check conflict with w1 vectors: w1 at position i with sign si, or position j with sign sj
        # w1 vector at pos i with sign s conflicts with w2 (i,j,si,sj) iff s == si (and i in support)
        # Actually: <w1, w2> = s * si if w1 is at position i. Conflict iff <w1,w2> > 1, i.e., = 2, i.e., s*si = 2, i.e., s=si.
        # So w2 (i,j,si,sj) conflicts with w1 at (i, si) and w1 at (j, sj).
        # But we're adding ALL w1 vectors, so this check is automatic.
        # Wait, we add all w1 vectors. So w2 (i,j,si,sj) conflicts with w1(i,si) and w1(j,sj).
        # Since all w1 are added, we must skip w2 that conflict with any w1.
        # w2 (i,j,si,sj) conflicts with w1(i,si) → always true since w1(i,si) is in our set.
        # So NO weight-2 vector can be added if all weight-1 doubled vectors are present!
        
        # Let me reconsider. Maybe we should not add all w1 vectors, or handle this differently.
        # Actually, the conflict between w2 and w1: dist_sq = 2+4-2<w1,w2> >= 4 → <w1,w2> <= 1.
        # <w1,w2> = ±2 if they share a position. So conflict iff they share a position with same sign.
        # Since we add both w1(i,1) and w1(i,-1), every w2 that has position i will conflict with one of them.
        # So indeed, if we add all 22 w1 vectors, no w2 can be added.
        
        # Alternative: skip w1 vectors that block many w2's, or just accept the trade-off.
        # 22 w1 vectors vs potentially many w2 vectors.
        # Max w2 without w1 conflicts: we need to select w2 vectors such that no two share a position with same sign,
        # and they don't conflict with w4.
        
        w2_selected.append((i, j, si, sj))
        w2_selected_set.add((i, j, si, sj))
    
    # Given the conflict between w2 and w1, let's decide:
    # Option A: Include all w1 (22 vectors), no w2.
    # Option B: Include some w1 and some w2.
    # Option C: No w1, maximize w2.
    
    # Let's compute Option C: maximize w2 without w1, subject to w4 conflicts.
    # w2 vectors conflict with each other only if same (i,j,si,sj) — but we have unique candidates.
    # Actually two different w2 candidates can conflict if they have the same support and same signs.
    # Since each (i,j,si,sj) is unique in our candidate list, no two distinct candidates conflict.
    # Wait, that's not right. w2 (i,j,1,1) and w2 (i,j,1,1) would be the same, but we only have one.
    # What about w2 (i,j,1,1) and w2 (j,i,1,1)? We use combinations so i<j always, no duplicates.
    # So all w2 candidates are mutually non-conflicting!
    # The only constraint is the w4 conflict.
    
    # So Option C: all w2 candidates that don't conflict with selected w4.
    w2_option_c = []
    for (i, j, si, sj) in w2_candidates:
        if (i, j, si, sj) not in w4_blocked_w2:
            w2_option_c.append((i, j, si, sj))
    
    # Option A: all w1, no w2
    # Option C: no w1, w2_option_c
    
    # Choose the better option
    if len(w2_option_c) >= len(w1_vectors):
        # Use Option C
        aux_vectors = []
        for (i, j, si, sj) in w2_option_c:
            v = np.zeros(D, dtype=np.int64)
            v[i] = si
            v[j] = sj
            aux_vectors.append(v)
        aux_vectors = np.array(aux_vectors, dtype=np.int64) if aux_vectors else np.empty((0, D), dtype=np.int64)
    else:
        # Use Option A
        aux_vectors = w1_vectors
    
    # Combine
    all_vectors = np.vstack([w4_int_vectors, aux_vectors])
    
    return all_vectors
# EVOLVE-BLOCK-END
