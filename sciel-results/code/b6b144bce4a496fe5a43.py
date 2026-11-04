"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def _cost(a, b, p):
    """Compute the exact QAP cost for permutation p."""
    Bp = b[np.ix_(p, p)]
    return int(np.sum(a * Bp))


def _delta_swap(a, b, p, i, j):
    """
    Compute the change in cost if we swap facilities i and j in the assignment p.
    """
    pi = p[i]
    pj = p[j]
    n = len(p)

    mask = np.ones(n, dtype=bool)
    mask[i] = False
    mask[j] = False

    diff1 = b[pj, p[mask]] - b[pi, p[mask]]
    diff2 = b[p[mask], pj] - b[p[mask], pi]

    a_diff_row = a[i, mask] - a[j, mask]
    a_diff_col = a[mask, i] - a[mask, j]

    delta_k = np.sum(a_diff_row * diff1) + np.sum(a_diff_col * diff2)

    delta_self = (a[i, j] * (b[pj, pi] - b[pi, pj])
                  + a[j, i] * (b[pi, pj] - b[pj, pi])
                  + a[i, i] * (b[pj, pj] - b[pi, pi])
                  + a[j, j] * (b[pi, pi] - b[pj, pj]))

    return int(delta_k + delta_self)


def _delta_or(a, b, p, start, length, new_pos):
    """
    Compute the change in cost if we move a contiguous block of facilities
    from position [start, start+length) to new_pos in the permutation.
    """
    n = len(p)
    if length <= 1:
        return 0

    block_fac = p[start:start+length]
    
    # Determine the new positions after moving
    # Case 1: new_pos < start (move block to the left)
    # Case 2: new_pos > start + length (move block to the right)
    # Case 3: invalid or same position
    
    if new_pos == start:
        return 0

    # Build the new permutation to check validity, but we want to compute delta efficiently
    # Let's just compute cost of new permutation and subtract old.
    # For n=100, this is 10k ops. If we do this for O(n^2) moves, it's 100M ops.
    # In Python with NumPy overhead, this might be slow if done naively in a loop.
    # But _delta_swap is also vectorized.
    # Let's try a direct calculation similar to _delta_swap but for blocks.
    
    # The cost change comes from interactions between the block facilities and the rest,
    # and changes in interactions among the block facilities (which is 0 since internal order is preserved? No, wait.)
    # Or-opt moves a block. The relative order of the block facilities is preserved.
    # So a[i,j] for i,j in block stays associated with the same pair of locations?
    # Wait. p[i] is location of facility i.
    # If we move facilities {f1, f2} from positions (1, 2) to (5, 6).
    # Before: p[1]=L1, p[2]=L2.
    # After: p[5]=L1, p[6]=L2.
    # The pair (f1, f2) was at (L1, L2). Now at (L1, L2).
    # So the term a[f1, f2] * b[L1, L2] is unchanged.
    # The term a[f1, f1] * b[L1, L1] is unchanged.
    # So internal block interactions are UNCHANGED.
    
    # Only interactions between block facilities and non-block facilities change.
    # Let B = set of block facilities.
    # Let R = set of rest facilities.
    
    # Change in cost = Sum_{i in B, j in R} a[i,j] * (b[new_loc(i), p[j]] - b[old_loc(i), p[j]])
    #                + Sum_{i in R, j in B} a[i,j] * (b[p[i], new_loc(j)] - b[p[i], old_loc(j)])
    
    # new_loc(i) for i in B:
    # If moving left: new_loc(f_k) = new_pos + k (where k is index in block)
    # If moving right: new_loc(f_k) = new_pos + k
    
    # We need to be careful about which positions are "new" and "old".
    
    # Let's construct the new permutation array to get new locations.
    new_p = p.copy()
    
    if new_pos < start:
        # Move left
        # Block goes to new_pos.
        # Elements between new_pos and start shift right.
        block = new_p[start:start+length]
        new_p[new_pos:new_pos+length] = block
        # Wait, simple slice assignment in numpy doesn't shift correctly like list insert.
        # p = [A, B, C, D, E]
        # Move block [C] (start=2, len=1) to new_pos=0.
        # Result: [C, A, B, D, E]
        # new_p[0:1] = [C] -> [C, B, C, D, E] (WRONG)
        
        # We need to do this manually or use list conversion.
        # For n=100, list conversion is fast enough?
        # 100 elements. Yes.
        
        pass # Use list logic below
        
    elif new_pos > start + length:
        # Move right
        # Block goes to new_pos.
        # Elements between start+length and new_pos shift left.
        pass
        
    else:
        # Overlapping or invalid?
        # If new_pos is within (start, start+length), it's a rotation, not a simple move.
        # Or-opt usually implies cutting and pasting.
        # If new_pos < start or new_pos >= start + length, it's a standard move.
        # If they overlap, it's more complex. Let's restrict to non-overlapping moves.
        if new_pos >= start and new_pos < start + length:
            return 0 # Skip overlapping moves for simplicity, or handle specially
        
    # Let's implement the list-based shift to get new_p correctly.
    p_list = p.tolist()
    block_list = p_list[start:start+length]
    del p_list[start:start+length]
    p_list[new_pos:new_pos] = block_list
    new_p = np.array(p_list, dtype=np.int64)
    
    # Now compute delta
    # B_mask: indices in p that are in the block
    # R_mask: indices in p that are not in the block
    
    # We need to map facility indices to their old and new locations.
    # old_loc[fac] = p[fac_index]
    # new_loc[fac] = new_p[fac_index]
    
    # The cost is sum_{i,j} a[i,j] b[p[i], p[j]]
    # Delta = sum_{i,j} a[i,j] (b[new_p[i], new_p[j]] - b[p[i], p[j]])
    
    # This is just _cost(a, b, new_p) - _cost(a, b, p).
    # Since _cost is O(n^2) = 10k ops, and we are doing this inside a loop...
    # How many times?
    # In local search, we try many swaps.
    # If we try Or-opt moves, we need to be efficient.
    
    # Can we vectorize the delta calculation without building new_p?
    # Yes.
    
    # Let B_idx = [start, ..., start+length-1]
    # Let R_idx = all other indices.
    
    # For i in B_idx, j in R_idx:
    #   term = a[i, j] * (b[new_loc(i), p[j]] - b[p[i], p[j]])
    # For i in R_idx, j in B_idx:
    #   term = a[i, j] * (b[p[i], new_loc(j)] - b[p[i], p[j]])
    
    # We need new_loc(i) for i in B_idx.
    # new_loc(i) is the location assigned to facility i in the new permutation.
    # In new_p, facility i is at position pos_new(i).
    # new_loc(i) = new_p[pos_new(i)].
    
    # How to get new_loc for the block?
    # The block facilities are p[start], ..., p[start+length-1].
    # They are moved to positions new_pos, ..., new_pos+length-1.
    # So the facility that was at p[start] is now at position new_pos.
    # The facility that was at p[start+1] is now at position new_pos+1.
    # ...
    # So, new_loc for the k-th facility in the block (index start+k) is new_p[new_pos+k].
    # Wait. p[start+k] is the facility ID.
    # In new_p, the facility ID p[start+k] is at position new_pos+k.
    # So new_loc(p[start+k]) = new_p[new_pos+k].
    
    # Let's verify.
    # p = [10, 20, 30, 40]
    # Move block [20, 30] (start=1, len=2) to new_pos=3.
    # p_list = [10, 40, 20, 30] -> new_p = [10, 40, 20, 30]
    # Block facilities: 20, 30.
    # Old locations: p[1]=20, p[2]=30.
    # New locations: 20 is at index 2, so new_loc(20)=new_p[2]=20.
    #                30 is at index 3, so new_loc(30)=new_p[3]=30.
    # Formula: new_loc(p[1+k]) = new_p[3+k].
    # k=0: new_loc(20) = new_p[3] = 30. MISMATCH.
    
    # Why?
    # In new_p = [10, 40, 20, 30].
    # Facility 20 is at index 2.
    # Facility 30 is at index 3.
    # My formula said new_loc(20) = new_p[3] = 30.
    # But actually new_loc(20) should be the LOCATION value, which is 20?
    # No, p[i] is the location.
    # "Facility i is placed at location p[i]".
    # So p[1] = 20 means Facility 1 is at Location 20.
    # If we move the "entry" for Facility 1, we are moving its location.
    
    # Let's re-read the problem statement.
    # "Facility i is placed at location p[i]".
    # So p is an array where index is facility, value is location.
    # Swapping p[i] and p[j] means Facility i takes Location p[j] and Facility j takes Location p[i].
    
    # Or-opt: "move a contiguous block of facilities to a new position".
    # This implies reordering the *assignment* array p.
    # So we are moving the *values* (locations) in the array p.
    # If we move p[start:start+length] to new_pos, we are taking the locations assigned to facilities start..start+length-1
    # and assigning them to facilities new_pos..new_pos+length-1.
    # This changes which facility gets which location.
    
    # So:
    # Old: Facility start got location p[start].
    # New: Facility new_pos gets location p[start].
    
    # This is effectively a permutation of the locations among the facilities.
    
    # Let's check the delta calculation with this interpretation.
    # B_fac: Facilities in the source range [start, start+length).
    # R_fac: Facilities in the rest.
    # The locations associated with B_fac are moved to R_fac (specifically the destination range).
    # The locations associated with the destination range are moved to the source range (if overlapping) or stay?
    
    # Actually, it's simpler:
    # We have a vector of locations L = p.
    # We permute L.
    # Cost depends on a and b[L[i], L[j]].
    
    # If we just swap two elements p[i] and p[j], we are swapping locations for facilities i and j.
    # If we move a block, we are shuffling locations.
    
    # The vectorized delta for swap works because only 2 facilities change locations.
    # For Or-opt, a block of facilities change locations, and another block (the "gap" fillers) change locations.
    
    # Let S = set of facilities whose location changes.
    # In a move of block [start, start+len) to new_pos:
    # The facilities in [start, start+len) get new locations.
    # The facilities in [new_pos, new_pos+len) (if distinct) get the old locations of the block.
    # If they overlap, it's a rotation.
    
    # Let's assume non-overlapping for now.
    # Src = [start, start+len)
    # Dst = [new_pos, new_pos+len)
    
    # Facilities in Src get locations from Dst (in the new permutation, the values at Dst are the old values from Src? No.)
    
    # Let's trace:
    # p = [A, B, C, D]
    # Move block [B, C] (indices 1,2) to index 0.
    # New p = [B, C, A, D]
    # Facility 0: Old A, New B.
    # Facility 1: Old B, New C.
    # Facility 2: Old C, New A.
    # Facility 3: Old D, New D.
    
    # So Facility 0 gets old location of Facility 1.
    # Facility 1 gets old location of Facility 2.
    # Facility 2 gets old location of Facility 0.
    # This is a 3-cycle of locations!
    
    # So the set of facilities involved is Src ∪ Dst.
    # In this example, Src={1,2}, Dst={0,1}. Union = {0,1,2}.
    # The locations are permuted among these 3 facilities.
    
    # To compute delta:
    # Sum over i in Involved, j in All of a[i,j] * (b[new_loc(i), new_loc(j)] - b[old_loc(i), old_loc(j)])
    # This includes i,j both in Involved, and one in Involved one not.
    
    # If |Involved| is small (e.g. length 2 or 3), we can just compute the cost for the sub-matrix?
    # No, because b is global.
    
    # However, if we precompute b, we can just do:
    # old_sub = b[np.ix_(old_locs, old_locs)] -> No, b is location-location.
    # Cost = sum a[i,j] b[p[i], p[j]].
    
    # If we build new_p, we can just call _cost(a, b, new_p) - current_cost.
    # Is this too slow?
    # _cost is ~10k ops.
    # If we do this for 1000 Or-opt moves, that's 10M ops.
    # In Python, this is ~0.1-0.5s.
    # We have 120s.
    # Local search with 2-swaps does ~5000 swaps/pass * 10 passes = 50k swaps.
    # Each swap delta is vectorized O(n). 50k * 100 = 5M ops. Very fast.
    # Or-opt with full cost recomputation:
    # If we check O(n^2) Or-opt moves (n positions * n lengths?), that's 10k moves.
    # 10k * 10k ops = 100M ops. ~1-2s.
    # This is feasible if done occasionally, not every single pass.
    
    # Strategy:
    # 1. Standard 2-swap local search.
    # 2. If stuck (no improvement in 2-swaps), try Or-opt moves.
    #    For Or-opt, iterate over possible block lengths (2, 3) and positions.
    #    Compute new_p, compute cost. If better, accept.
    
    return int(_cost(a, b, new_p) - _cost(a, b, p))


def _local_search(a, b, p, deadline, max_passes=100, use_or_opt=False):
    """
    First-improvement local search with 2-opt swaps, and optionally Or-opt moves.
    Returns (p, cost).
    """
    n = len(p)
    cost = _cost(a, b, p)

    for _ in range(max_passes):
        if time.time() > deadline:
            break
        improved = False
        
        # 2-swap pass
        for i in range(n):
            if time.time() > deadline:
                return p, cost
            for j in range(i + 1, n):
                delta = _delta_swap(a, b, p, i, j)
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    cost += delta
                    improved = True
                    break  # First improvement per i, then move to next i
        
        # Or-opt pass (if enabled and no 2-swap improvement)
        if use_or_opt and not improved:
            or_improved = False
            # Try block lengths 2 and 3
            for length in [2, 3]:
                if time.time() > deadline:
                    return p, cost
                for start in range(n - length + 1):
                    for new_pos in range(n - length + 1):
                        if new_pos == start:
                            continue
                        # Check overlap
                        if new_pos < start + length and new_pos + length > start:
                            continue # Skip overlapping for simplicity in this vectorized check
                            
                        delta = _delta_or(a, b, p, start, length, new_pos)
                        if delta < 0:
                            # Apply the move
                            p_list = p.tolist()
                            block = p_list[start:start+length]
                            del p_list[start:start+length]
                            p_list[new_pos:new_pos] = block
                            p = np.array(p_list, dtype=np.int64)
                            cost += delta
                            or_improved = True
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            # If Or-opt improved, restart the pass logic (loop continues)
            # If not, we are at a local minimum for both 2-swap and Or-opt.

        if not improved:
            break

    return p, cost


def _build_surrogate(a, b, p):
    """
    Build a surrogate cost matrix C[i, l] approximating the marginal cost
    of assigning facility i to location l, holding other assignments fixed.
    """
    n = len(p)
    term1 = a @ b[:, p].T
    term2 = a.T @ b[p, :]
    C = term1 + term2
    return C


def _surrogate_perturb(a, b, p, current_cost, rng, n):
    """
    Surrogate-guided perturbation.
    """
    perturbed_p = p.copy()

    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)

    contrib = a_row_sums * b_row_sums[p]

    k = 3 + int(rng.integers(0, 4))
    top_idx = np.argsort(-contrib)[:k]

    for idx in top_idx:
        if rng.random() < 0.5:
            j = rng.integers(0, n)
            if j != idx:
                perturbed_p[idx], perturbed_p[j] = perturbed_p[j], perturbed_p[idx]
        else:
            low_idx = np.argsort(contrib)[:k]
            j = low_idx[rng.integers(0, k)]
            if j != idx:
                perturbed_p[idx], perturbed_p[j] = perturbed_p[j], perturbed_p[idx]

    return perturbed_p


def _spectral_init(a, b, n, rng):
    """
    Spectral initialization.
    """
    try:
        a_sym = (a + a.T) / 2.0
        b_sym = (b + b.T) / 2.0
        evals_a, evecs_a = np.linalg.eigh(a_sym)
        evals_b, evecs_b = np.linalg.eigh(b_sym)
    except np.linalg.LinAlgError:
        return rng.permutation(n).astype(np.int64)

    k = min(10, n)
    idx_a = np.argsort(-np.abs(evals_a))[:k]
    fac_emb = evecs_a[:, idx_a]

    idx_b = np.argsort(-np.abs(evals_b))[:k]
    loc_emb = evecs_b[:, idx_b]

    fac_norm = np.linalg.norm(fac_emb, axis=1, keepdims=True)
    fac_norm[fac_norm == 0] = 1
    fac_emb = fac_emb / fac_norm

    loc_norm = np.linalg.norm(loc_emb, axis=1, keepdims=True)
    loc_norm[loc_norm == 0] = 1
    loc_emb = loc_emb / loc_norm

    cost_matrix = -fac_emb @ loc_emb.T
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind

    if rng.random() < 0.3:
        subset_size = rng.integers(5, 20)
        subset = rng.choice(n, size=subset_size, replace=False)
        p[subset] = p[subset][::-1]

    return p


def _quantum_perturb(p, rng, n, temperature=0.5):
    """
    Quantum-inspired perturbation.
    """
    perturbed_p = p.copy()

    base_swaps = 2
    max_swaps = 8
    n_swaps = base_swaps + int(temperature * (max_swaps - base_swaps))
    n_swaps = max(1, min(n_swaps, n // 2))

    for _ in range(n_swaps):
        i = rng.integers(0, n)
        j = rng.integers(0, n)
        if i != j:
            perturbed_p[i], perturbed_p[j] = perturbed_p[j], perturbed_p[i]

    if rng.random() < 0.1 * temperature:
        block_len = rng.integers(3, min(10, n // 3))
        start = rng.integers(0, n - block_len + 1)
        perturbed_p[start:start + block_len] = perturbed_p[start:start + block_len][::-1]

    return perturbed_p


def _best_improvement_pass(a, b, p, deadline):
    """
    One pass of best-improvement for 2-swaps.
    """
    n = len(p)
    cost = _cost(a, b, p)
    best_delta = 0
    best_i = -1
    best_j = -1

    for i in range(n):
        if time.time() > deadline:
            break
        for j in range(i + 1, n):
            delta = _delta_swap(a, b, p, i, j)
            if delta < best_delta:
                best_delta = delta
                best_i = i
                best_j = j

    if best_i >= 0:
        p[best_i], p[best_j] = p[best_j], p[best_i]
        cost += best_delta

    return p, cost, best_delta


def solve(a, b):
    n = len(a)
    t_start = time.time()
    deadline = t_start + 115.0  # safety margin before 150s kill

    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)

    best_p = None
    best_cost = np.inf

    rng = np.random.default_rng(12345)

    # Heuristic 1: Rank-matching
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p1 = np.empty(n, dtype=np.int64)
    p1[heavy] = central
    p1, c1 = _local_search(a, b, p1.copy(), deadline)
    if c1 < best_cost:
        best_cost = c1
        best_p = p1.copy()

    # Heuristic 2: Reverse rank-matching
    heavy2 = np.argsort(a_row_sums, kind="stable")
    central2 = np.argsort(-b_row_sums, kind="stable")
    p2 = np.empty(n, dtype=np.int64)
    p2[heavy2] = central2
    p2, c2 = _local_search(a, b, p2.copy(), deadline)
    if c2 < best_cost:
        best_cost = c2
        best_p = p2.copy()

    # Heuristic 3: Spectral initialization
    p3 = _spectral_init(a, b, n, rng)
    p3, c3 = _local_search(a, b, p3, deadline)
    if c3 < best_cost:
        best_cost = c3
        best_p = p3.copy()

    # Heuristic 4: Random
    p4 = rng.permutation(n).astype(np.int64)
    p4, c4 = _local_search(a, b, p4, deadline)
    if c4 < best_cost:
        best_cost = c4
        best_p = p4.copy()

    # --- Phase 2: Surrogate-guided ---
    if time.time() < deadline - 15.0:
        surrogate = _build_surrogate(a, b, best_p)
        row_ind, col_ind = linear_sum_assignment(surrogate)
        p_surr = np.empty(n, dtype=np.int64)
        p_surr[row_ind] = col_ind
        p_surr, c_surr = _local_search(a, b, p_surr, deadline - 10.0)
        if c_surr < best_cost:
            best_cost = c_surr
            best_p = p_surr.copy()

    # --- Phase 3: Iterated Local Search with Or-opt ---
    current_p = best_p.copy()
    current_cost = best_cost

    ilse_deadline = deadline - 10.0
    iteration = 0
    total_iterations_expected = 30
    
    while time.time() < ilse_deadline:
        iteration += 1
        progress = min(1.0, iteration / total_iterations_expected)
        temperature = 1.0 - 0.7 * progress

        r = rng.random()
        if r < 0.4:
            perturbed_p = _quantum_perturb(current_p, rng, n, temperature)
        elif r < 0.7:
            perturbed_p = _surrogate_perturb(a, b, current_p, current_cost, rng, n)
        else:
            k = 2 + int(temperature * 4)
            perturbed_p = current_p.copy()
            for _ in range(k):
                i, j = rng.integers(0, n, size=2)
                if i != j:
                    perturbed_p[i], perturbed_p[j] = perturbed_p[j], perturbed_p[i]

        # Use Or-opt in local search for perturbed solutions to escape 2-swap minima
        use_or = (iteration % 2 == 0)  # Use Or-opt every other iteration
        
        # Adjust deadline for Or-opt as it is more expensive
        ls_deadline = ilse_deadline - (5.0 if use_or else 0.0)
        if time.time() > ls_deadline:
            ls_deadline = time.time()

        perturbed_p, perturbed_cost = _local_search(a, b, perturbed_p, ls_deadline, use_or_opt=use_or)

        if perturbed_cost < best_cost:
            best_cost = perturbed_cost
            best_p = perturbed_p.copy()
            current_p = perturbed_p
            current_cost = perturbed_cost
        elif perturbed_cost < current_cost:
            delta = perturbed_cost - current_cost
            accept_prob = np.exp(-delta / (temperature * 1000.0))
            if rng.random() < accept_prob:
                current_p = perturbed_p
                current_cost = perturbed_cost
        elif perturbed_cost == current_cost:
            if rng.random() < 0.1:
                current_p = perturbed_p
                current_cost = perturbed_cost

    # --- Phase 4: Aggressive final polish with best-improvement ---
    if time.time() < deadline - 3.0:
        polish_deadline = deadline - 1.0
        polish_p = best_p.copy()
        polish_cost = best_cost
        for _ in range(30):
            if time.time() > polish_deadline:
                break
            polish_p, polish_cost, bd = _best_improvement_pass(a, b, polish_p, polish_deadline)
            if bd == 0:
                break
        if polish_cost < best_cost:
            best_cost = polish_cost
            best_p = polish_p.copy()

    # --- Phase 5: Final Or-opt polish ---
    if time.time() < deadline - 0.5:
        final_p = best_p.copy()
        final_p, final_cost = _local_search(a, b, final_p, deadline - 0.1, max_passes=5, use_or_opt=True)
        if final_cost < best_cost:
            best_cost = final_cost
            best_p = final_p.copy()

    return best_p.tolist()
# EVOLVE-BLOCK-END
