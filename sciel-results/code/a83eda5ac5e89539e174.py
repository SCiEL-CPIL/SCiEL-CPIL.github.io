import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    # ---------- Precomputation ----------
    points = np.arange(v)
    all_t_subs = list(itertools.combinations(range(v), t))
    num_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    all_k_subs = list(itertools.combinations(range(v), k))
    num_k = len(all_k_subs)

    # Precompute for each k-subset: list of t-subset indices it covers
    # Also build a sparse matrix cov_matrix[num_k, num_t] (0/1)
    cov_lists = []
    for k_sub in all_k_subs:
        idxs = [t_sub_to_idx[tsub] for tsub in itertools.combinations(k_sub, t)]
        cov_lists.append(idxs)

    # Build coverage matrix as numpy array (num_k x num_t) of uint8
    # num_k ~ 50388, num_t = 3876 -> ~195 MB as uint8, acceptable
    cov_matrix = np.zeros((num_k, num_t), dtype=np.uint8)
    for i, idxs in enumerate(cov_lists):
        for j in idxs:
            cov_matrix[i, j] = 1

    # ---------- Greedy Construction ----------
    uncovered = np.ones(num_t, dtype=np.uint8)
    selected = []
    selected_set = set()

    while uncovered.any():
        if time.time() > deadline:
            break
        # Vectorized gain computation
        gains = (cov_matrix * uncovered).sum(axis=1)
        # Among candidates not yet selected, pick max gain
        # To avoid O(num_k) Python loop, use numpy
        # Mask out already-selected
        # Simple approach: set gains of selected to -1
        for s in selected_set:
            gains[s] = -1
        best = int(np.argmax(gains))
        if gains[best] <= 0:
            break
        selected.append(best)
        selected_set.add(best)
        uncovered = uncovered & (1 - cov_matrix[best])

    # ---------- Local Search ----------
    # Work with list of indices
    sel = list(selected)

    def recompute_counts(sel_list):
        counts = np.zeros(num_t, dtype=np.int16)
        for idx in sel_list:
            counts += cov_matrix[idx]
        return counts

    def check_valid(sel_list):
        if not sel_list:
            return num_t == 0
        counts = np.zeros(num_t, dtype=np.int16)
        for idx in sel_list:
            counts += cov_matrix[idx]
        return (counts > 0).all()

    improved = True
    while improved and time.time() < deadline:
        improved = False

        # Recompute counts
        counts = recompute_counts(sel)

        # --- Phase 1: Remove redundant blocks ---
        redundant = []
        for idx in sel:
            # Block is redundant if for all t-subsets it covers, count > 1
            covered = cov_matrix[idx]
            if (counts[covered > 0] > 1).all():
                redundant.append(idx)

        if redundant:
            # Remove one
            idx_rm = redundant[0]
            sel.remove(idx_rm)
            improved = True
            continue

        # --- Phase 2: 1-exchange swaps ---
        # For each block in sel, find a replacement block (not in sel) that covers
        # all the "unique" t-subsets of the current block, and that might enable
        # further reductions.
        if time.time() > deadline - 15:
            break

        sel_set = set(sel)
        # Shuffle for diversity
        order = list(range(len(sel)))
        random.shuffle(order)

        swap_done = False
        for pos in order:
            if time.time() > deadline - 10:
                break
            idx = sel[pos]
            # Unique subsets: covered by idx with count == 1
            covered_mask = cov_matrix[idx] > 0
            unique_mask = covered_mask & (counts == 1)
            unique_count = unique_mask.sum()

            if unique_count == 0:
                # Should have been caught by redundancy, but remove anyway
                sel.pop(pos)
                improved = True
                swap_done = True
                break

            # Find replacement: a block not in sel that covers all unique subsets
            # and ideally has high overlap with other uncovered needs
            # Vectorized: for each candidate j, check if cov_matrix[j] covers unique_mask
            # i.e., cov_matrix[j][unique_mask].all()
            # This is expensive to do for all j. Instead, filter candidates.
            # A candidate must cover all unique subsets.
            # Precompute: for each unique subset, which blocks cover it?
            # That's expensive. Instead, iterate candidates but use numpy.

            # Get unique subset indices
            unique_idxs = np.where(unique_mask)[0]
            if len(unique_idxs) == 0:
                sel.pop(pos)
                improved = True
                swap_done = True
                break

            # For each candidate not in sel, check if it covers all unique_idxs
            # cov_matrix[:, unique_idxs] gives shape (num_k, len(unique_idxs))
            # We need all ones in those columns
            # To save time, only check candidates that cover at least the first unique subset
            first_unique = unique_idxs[0]
            # Blocks covering first_unique:
            candidates = np.where(cov_matrix[:, first_unique] == 1)[0]
            # Filter out those in sel
            candidates = [c for c in candidates if c not in sel_set]

            best_candidate = -1
            best_score = -1
            for c in candidates:
                if time.time() > deadline - 8:
                    break
                # Check if c covers all unique subsets
                if (cov_matrix[c, unique_idxs] == 1).all():
                    # Score: prefer candidates that cover more of the "tight" subsets
                    # or that have high total coverage to enable future reductions
                    # Simple: prefer candidates with more total coverage
                    total_cov = cov_matrix[c].sum()
                    # Also penalize if candidate is very similar to others in sel
                    # Use total_cov as proxy
                    if total_cov > best_score:
                        best_score = total_cov
                        best_candidate = c

            if best_candidate != -1:
                sel[pos] = best_candidate
                sel_set.discard(idx)
                sel_set.add(best_candidate)
                improved = True
                swap_done = True
                break

        if swap_done:
            continue

        # --- Phase 3: 2-exchange swaps (if time allows) ---
        if time.time() > deadline - 20:
            break

        # Try to replace two blocks with two different blocks
        # This is more expensive. Try a limited number of random pairs.
        if len(sel) >= 2:
            # Randomly pick a pair of positions
            num_tries = 50
            for _ in range(num_tries):
                if time.time() > deadline - 5:
                    break
                p1, p2 = random.sample(range(len(sel)), 2)
                idx1, idx2 = sel[p1], sel[p2]

                # Unique subsets for idx1 and idx2 (considering both are present)
                # A subset is "unique to idx1" if covered by idx1, not by idx2, and count==1
                cov1 = cov_matrix[idx1]
                cov2 = cov_matrix[idx2]
                unique1 = (cov1 > 0) & (cov2 == 0) & (counts == 1)
                unique2 = (cov2 > 0) & (cov1 == 0) & (counts == 1)

                # Combined unique mask: subsets that must be covered by the replacements
                # Actually, we need replacements c1, c2 such that:
                # - c1 covers all unique1 subsets (or c2 does, but let's keep it simple)
                # - c2 covers all unique2 subsets
                # - c1 != c2, c1 not in sel, c2 not in sel
                # - c1 and c2 together cover all unique1 and unique2

                u1_idxs = np.where(unique1)[0]
                u2_idxs = np.where(unique2)[0]

                if len(u1_idxs) == 0 and len(u2_idxs) == 0:
                    # Both are redundant together? Remove one
                    sel.pop(p2)
                    improved = True
                    break

                # Find candidates for c1: must cover u1_idxs
                if len(u1_idxs) > 0:
                    first_u1 = u1_idxs[0]
                    cand1 = np.where(cov_matrix[:, first_u1] == 1)[0]
                    cand1 = [c for c in cand1 if c not in sel_set]
                    # Filter: must cover all u1_idxs
                    cand1 = [c for c in cand1 if (cov_matrix[c, u1_idxs] == 1).all()]
                else:
                    cand1 = [c for c in range(num_k) if c not in sel_set]

                if len(u2_idxs) > 0:
                    first_u2 = u2_idxs[0]
                    cand2 = np.where(cov_matrix[:, first_u2] == 1)[0]
                    cand2 = [c for c in cand2 if c not in sel_set]
                    cand2 = [c for c in cand2 if (cov_matrix[c, u2_idxs] == 1).all()]
                else:
                    cand2 = [c for c in range(num_k) if c not in sel_set]

                # Limit candidates for speed
                cand1 = cand1[:100]
                cand2 = cand2[:100]

                found = False
                for c1 in cand1:
                    if time.time() > deadline - 3:
                        break
                    for c2 in cand2:
                        if c1 == c2:
                            continue
                        # Check combined coverage of unique subsets
                        # c1 covers u1, c2 covers u2, but what if u1 has subsets only c2 covers?
                        # Actually, we defined u1 as subsets covered by idx1 but not idx2.
                        # The replacement pair must cover all of u1 and u2.
                        # Let's check: (cov_matrix[c1] | cov_matrix[c2]) covers u1_idxs and u2_idxs
                        combined = cov_matrix[c1] | cov_matrix[c2]
                        if len(u1_idxs) > 0 and not (combined[u1_idxs] == 1).all():
                            continue
                        if len(u2_idxs) > 0 and not (combined[u2_idxs] == 1).all():
                            continue
                        # Also ensure we don't break coverage of other subsets
                        # The subsets covered by idx1 or idx2 that are not unique must still be covered
                        # This is complex. For now, accept and rely on subsequent redundancy check.
                        sel[p1] = c1
                        sel[p2] = c2
                        sel_set.discard(idx1)
                        sel_set.discard(idx2)
                        sel_set.add(c1)
                        sel_set.add(c2)
                        improved = True
                        found = True
                        break
                    if found:
                        break
                if found:
                    break

    # ---------- Final Validation and Return ----------
    # Ensure validity
    if sel:
        final_counts = np.zeros(num_t, dtype=np.int16)
        for idx in sel:
            final_counts += cov_matrix[idx]
        if not (final_counts > 0).all():
            # Fallback: add blocks greedily to cover missing subsets
            missing = np.where(final_counts == 0)[0]
            while len(missing) > 0:
                if time.time() > deadline:
                    break
                # Find block covering most missing
                gains = (cov_matrix[:, missing]).sum(axis=1)
                for s in sel:
                    gains[s] = -1
                best = int(np.argmax(gains))
                if gains[best] <= 0:
                    break
                sel.append(best)
                final_counts += cov_matrix[best]
                still_missing = np.where(final_counts == 0)[0]
                missing = still_missing

    # Convert to blocks
    final_blocks = [list(all_k_subs[i]) for i in sel]

    # Deduplicate just in case
    seen = set()
    unique_blocks = []
    for b in final_blocks:
        key = tuple(b)
        if key not in seen:
            seen.add(key)
            unique_blocks.append(b)

    return unique_blocks
