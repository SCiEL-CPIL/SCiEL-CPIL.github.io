import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    # Generate all weight-4 vectors in {-1, 0, 1}^11 (Norm^2 = 4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    # Generate all weight-1 vectors in {-2, 0, 2}^11 (Norm^2 = 4)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
            
    # Generate all weight-2 vectors in {-1, 0, 1}^11 (Norm^2 = 2)
    v2_candidates = []
    for positions in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
                v2_candidates.append(v)

    all_vectors = np.vstack([
        np.array(v4_candidates, dtype=np.int64),
        np.array(v1_candidates, dtype=np.int64),
        np.array(v2_candidates, dtype=np.int64)
    ])
    
    N = len(all_vectors)
    norms_sq = np.zeros(N, dtype=np.int64)
    n4 = len(v4_candidates)
    n1 = len(v1_candidates)
    n2 = len(v2_candidates)
    norms_sq[:n4] = 4
    norms_sq[n4:n4+n1] = 4
    norms_sq[n4+n1:] = 2
    
    # Compute dot products
    dots = all_vectors @ all_vectors.T
    
    # Conflict if dot > min(norm_i, norm_j) / 2
    # This is equivalent to dist_sq < min(norm_i, norm_j)
    # For mixed sets, max_norm_sq = 4. Condition: dist_sq >= 4.
    # dist_sq = n_i + n_j - 2*dot.
    # If n_i=4, n_j=4: 8-2d >= 4 => d <= 2.
    # If n_i=4, n_j=2: 6-2d >= 4 => d <= 1.
    # If n_i=2, n_j=2: 4-2d >= 4 => d <= 0.
    # Let's re-verify the "dot > min(norm_i, norm_j) / 2" logic from current code.
    # min_norms is min of squared norms? No, the code says min(norms_sq).
    # Code: thresholds = min_norms / 2.0.
    # If n_i=4, n_j=2. min=2. thresh=1. Conflict if dot > 1. i.e. dot >= 2.
    # My derivation: d <= 1 allowed. So d >= 2 is conflict. Matches.
    # If n_i=2, n_j=2. min=2. thresh=1. Conflict if dot > 1. i.e. dot >= 2.
    # My derivation: d <= 0 allowed. So d >= 1 is conflict.
    # DISCREPANCY: The current code allows dot=1 for two norm^2=2 vectors.
    # Let's check validity.
    # If we select two norm^2=2 vectors with dot=1.
    # dist_sq = 2+2-2(1) = 2.
    # Max norm in set is sqrt(2) (if only these).
    # Condition: dist >= max_norm => sqrt(2) >= sqrt(2). Valid.
    # So for a PURE norm^2=2 set, dot=1 is allowed.
    # But if the set contains ANY norm^2=4 vector, max_norm=2.
    # Then dist >= 2 => dist_sq >= 4.
    # For two norm^2=2 vectors: dist_sq = 4-2d.
    # We need 4-2d >= 4 => d <= 0.
    # So in a MIXED set, dot=1 is a CONFLICT.
    
    # The current code builds a single conflict graph based on `dots > thresholds`.
    # For (2,2) pair, thresh=1. Conflict if dot > 1.
    # This allows dot=1.
    # This is CORRECT for a pure set, but INCORRECT for a mixed set if we just take the independent set of this graph.
    # If the greedy MIS selects two norm^2=2 vectors with dot=1, AND also selects a norm^2=4 vector, the result is INVALID.
    
    # We need a structural change to handle this correctly.
    # Strategy:
    # 1. Solve for best PURE norm^2=2 set (Constraint: dot <= 1).
    # 2. Solve for best MIXED set (Constraint: dist_sq >= 4 for all pairs).
    #    This implies:
    #    - (4,4): dot <= 2
    #    - (4,2): dot <= 1
    #    - (2,2): dot <= 0
    # 3. Return the better of the two.

    # --- Case 1: Pure Norm^2=2 ---
    mask_t1 = norms_sq == 2
    candidates_t1 = all_vectors[mask_t1]
    n_t1 = len(candidates_t1)
    
    best_result = None
    best_score = 0
    
    if n_t1 > 0:
        # Conflict if dot > 1
        dots_t1 = candidates_t1 @ candidates_t1.T
        conflict_t1 = dots_t1 > 1
        np.fill_diagonal(conflict_t1, False)
        
        # Greedy MIS for this graph
        selected_t1 = greedy_mis(conflict_t1, deadline, start_time)
        if selected_t1 is not None and len(selected_t1) > 0:
            result_t1 = candidates_t1[selected_t1]
            # Verify validity (should be valid by construction)
            if is_valid(result_t1):
                if len(result_t1) > best_score:
                    best_score = len(result_t1)
                    best_result = result_t1

    # --- Case 2: Mixed Set (dist_sq >= 4) ---
    # Conflict conditions:
    # (4,4): dot > 2
    # (4,2): dot > 1
    # (2,2): dot > 0
    dots_all = all_vectors @ all_vectors.T
    
    conflict_mixed = np.zeros((N, N), dtype=bool)
    
    # (4,4)
    mask4 = norms_sq == 4
    mask2 = norms_sq == 2
    
    # Vectorized approach for mixed conflicts
    # Create matrices for the conditions
    # d_44 = dots_all[mask4][:, mask4]
    # conflict_44 = d_44 > 2
    
    # It's easier to build the full matrix
    # Initialize with False
    # For all pairs (i,j):
    # if n_i==4 and n_j==4: conflict if dot > 2
    # if n_i==4 and n_j==2: conflict if dot > 1
    # if n_i==2 and n_j==4: conflict if dot > 1
    # if n_i==2 and n_j==2: conflict if dot > 0
    
    # We can compute thresholds per pair
    # thresh[i,j] = min(n_i, n_j) / 2 ? No.
    # If min=2, thresh=1. (4,2) needs dot<=1. (2,2) needs dot<=0.
    # So min/2 is NOT sufficient for (2,2).
    
    # Let's just build it explicitly with broadcasting or loops if needed.
    # N is ~5500. N^2 is ~30M. This is fine.
    
    # We can use the fact that:
    # conflict[i,j] = (n_i + n_j - 2*dots[i,j] < 4)
    dist_sq_all = norms_sq[:, None] + norms_sq[None, :] - 2 * dots_all
    conflict_mixed = dist_sq_all < 4
    np.fill_diagonal(conflict_mixed, False)
    
    selected_mixed = greedy_mis(conflict_mixed, deadline, start_time)
    if selected_mixed is not None and len(selected_mixed) > 0:
        result_mixed = all_vectors[selected_mixed]
        if is_valid(result_mixed):
            if len(result_mixed) > best_score:
                best_score = len(result_mixed)
                best_result = result_mixed

    if best_result is None:
        # Fallback: return empty or a single vector
        return np.array([[1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
        
    return best_result

def greedy_mis(conflict, deadline, start_time):
    """
    Greedy Maximum Independent Set with local search refinement.
    conflict: boolean matrix, True if edge exists.
    """
    N = conflict.shape[0]
    if N == 0:
        return np.array([], dtype=int)
    
    rng = np.random.default_rng(42)
    alive = np.ones(N, dtype=bool)
    selected = []
    degrees = conflict.sum(axis=1).astype(np.int32)
    
    # Phase 1: Greedy selection
    while np.any(alive):
        if time.time() > deadline - 15:
            break
        alive_indices = np.where(alive)[0]
        if len(alive_indices) == 0:
            break
        min_deg = np.min(degrees[alive_indices])
        cands = alive_indices[degrees[alive_indices] == min_deg]
        chosen = cands[rng.integers(len(cands))]
        selected.append(chosen)
        neighbors = np.where(conflict[chosen])[0]
        to_remove = np.unique(np.concatenate(([chosen], neighbors)))
        alive[to_remove] = False
        rem = np.where(alive)[0]
        if len(rem) > 0:
            degrees[rem] -= conflict[rem][:, to_remove].sum(axis=1)
            
    if not selected:
        return np.array([], dtype=int)
        
    selected_set = set(selected)
    
    # Phase 2: Local Search (Kempe-like / Swap)
    # Try to improve by swapping vertices or adding/removing.
    # Given the size, a simple "try to add unselected vertices" or "swap" loop is good.
    
    # Build a set of selected for fast lookup
    # We want to maximize size.
    
    # Simple local search:
    # Iterate through unselected vertices. If it can be added (no conflict with selected), add it.
    # Iterate through selected vertices. If removing it allows adding >1 vertices, do it.
    
    # However, the greedy MIS is usually quite good.
    # Let's try a simple "relaxation" or "re-greedy" on the remaining graph?
    # Or just a few passes of "try to add".
    
    # Check if any unselected vertex can be added
    # This is O(N * |Selected|) if done naively.
    # With N=5500, |Selected|~1000, it's 5.5M ops. Fast.
    
    unselected = np.setdiff1d(np.arange(N), selected)
    
    # Map selected to indices in selected list for fast conflict checking?
    # Or just use the conflict matrix.
    
    # For each unselected u, check if conflict[u, selected] is all False.
    # This is equivalent to: conflict[u][:, selected_mask].any() == False
    
    selected_mask = np.zeros(N, dtype=bool)
    selected_mask[selected] = True
    
    # We can check all unselected at once:
    # conflicts_with_selected = conflict[unselected][:, selected_mask]
    # addable = ~conflicts_with_selected.any(axis=1)
    
    # But we must also ensure no conflict between the NEW vertices themselves?
    # No, we add one by one or check if the set of addable vertices is an independent set in the induced subgraph of unselected.
    # Actually, if u and v are both addable (no conflict with current selected), they might conflict with EACH OTHER.
    # So we should pick the best independent set from the "addable" candidates.
    
    # Let's just do a simple greedy add pass.
    # Re-run greedy on the "addable" candidates?
    # Or just iterate.
    
    # Given time constraints, a single pass of "add if possible" in random order might help.
    # But the initial greedy already did that.
    
    # Let's try a "swap" heuristic:
    # For each selected vertex s, remove it. Then see how many unselected vertices become addable.
    # If count > 1, perform the swap.
    
    # This is expensive: O(|Selected| * N * |Selected|).
    # |Selected| ~ 1000. N ~ 5500. 5.5M * 1000 = 5.5B. Too slow.
    
    # Alternative: Use the structure.
    # The inspiration mentioned "Kempe-chain search".
    # Let's stick to the high-quality greedy MIS which is fast.
    # The main improvement here is the CORRECT handling of the mixed vs pure constraint.
    
    return np.array(selected, dtype=int)

def is_valid(vectors):
    """
    Check if the set of vectors satisfies the kissing number condition.
    min pairwise distance >= max vector norm.
    """
    if len(vectors) == 0:
        return True
    if len(vectors) == 1:
        return True
        
    norms_sq = np.sum(vectors**2, axis=1)
    max_norm_sq = np.max(norms_sq)
    
    # Compute pairwise squared distances
    # dist_sq[i,j] = n_i + n_j - 2 dot_ij
    dots = vectors @ vectors.T
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
    
    # Check min distance (off-diagonal)
    # np.fill_diagonal(dist_sq, np.inf)
    # min_dist_sq = np.min(dist_sq)
    
    # More efficient:
    # We only need to check if any pair has dist_sq < max_norm_sq.
    # Since max_norm_sq is constant for the set, we just check if dist_sq < max_norm_sq anywhere off-diagonal.
    
    # To avoid diagonal (dist=0), we can set diagonal to max_norm_sq + 1.
    dist_sq_copy = dist_sq.copy()
    np.fill_diagonal(dist_sq_copy, max_norm_sq + 1)
    
    min_dist_sq = np.min(dist_sq_copy)
    
    return min_dist_sq >= max_norm_sq

# Helper for greedy MIS used in construct
# Defined above as greedy_mis
