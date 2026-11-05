"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    # Enumerate all t-subsets and assign indices
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    # Enumerate all k-subsets (candidate blocks)
    all_k_subs = list(itertools.combinations(range(v), k))
    n_k = len(all_k_subs)
    
    # Build lookup for k-subsets to indices early to fix NameError
    ksub_to_idx = {sub: i for i, sub in enumerate(all_k_subs)}

    # Build incidence matrix: block i covers t-subset j if t_sub j is subset of k_sub i
    # For each t-subset, find which blocks contain it
    # A block (k-subset) contains a t-subset iff all t elements are in the k-subset
    # Precompute: for each t-subset, the set of blocks that contain it
    # This is expensive for large v,k,t but for v=19,k=8,t=5:
    # C(19,5)=11628 t-subsets, C(19,8)=75582 k-subsets
    # For each t-subset, the number of k-subsets containing it is C(19-5, 8-5) = C(14,3) = 364

    # Build a mapping: for each t-subset index, list of block indices that cover it
    # And for each block index, list of t-subset indices it covers
    # Use numpy for vectorized operations where possible

    # For each block, compute which t-subsets it covers
    # block_cov[i] = set of t-subset indices covered by block i
    # t_block_cov[j] = set of block indices covering t-subset j

    # Build t_subset to block coverage mapping
    # For each t-subset, find all k-subsets that contain it
    # A k-subset contains a t-subset iff it contains all t elements plus (k-t) more from the remaining v-t
    t_sub_blocks = [[] for _ in range(n_t)]
    for j, tsub in enumerate(all_t_subs):
        tset = set(tsub)
        remaining = [p for p in range(v) if p not in tset]
        for extra in itertools.combinations(remaining, k - t):
            ksub = tuple(sorted(tsub + extra))
            # Need to find index of ksub in all_k_subs
            # This is slow; let's use a different approach

    # Better approach: build incidence matrix directly using bit masks
    # Represent each subset as a bitmask (v <= 64 fits in uint64)
    # t_sub_masks[j] = bitmask of t-subset j
    # k_sub_masks[i] = bitmask of k-subset i
    # Block i covers t-subset j iff (k_sub_masks[i] & t_sub_masks[j]) == t_sub_masks[j]

    t_sub_masks = np.zeros(n_t, dtype=np.uint64)
    for j, sub in enumerate(all_t_subs):
        mask = 0
        for p in sub:
            mask |= (1 << p)
        t_sub_masks[j] = mask

    k_sub_masks = np.zeros(n_k, dtype=np.uint64)
    for i, sub in enumerate(all_k_subs):
        mask = 0
        for p in sub:
            mask |= (1 << p)
        k_sub_masks[i] = mask

    # Incidence: inc[i, j] = 1 if block i covers t-subset j
    # For v=19, n_t=11628, n_k=75582: this is 75582*11628 ≈ 878M entries - too large for dense
    # Use sparse representation or compute on-the-fly

    # Alternative: for each t-subset j, precompute list of blocks that cover it
    # C(14,3) = 364 blocks per t-subset on average
    # Total edges: 11628 * 364 ≈ 4.2M - manageable

    # Build t_sub_to_blocks: for each t-subset, list of block indices
    t_sub_to_blocks = [[] for _ in range(n_t)]
    # For each block, determine which t-subsets it covers
    block_to_tsubs = [[] for _ in range(n_k)]

    # For each block i, the t-subsets it covers are all t-subsets of the k-subset
    # Number of t-subsets per block: C(8,5) = 56
    for i, ksub in enumerate(all_k_subs):
        for tsub in itertools.combinations(ksub, t):
            j = t_sub_to_idx[tsub]
            block_to_tsubs[i].append(j)
            t_sub_to_blocks[j].append(i)

    # Convert to numpy arrays for fast access
    block_to_tsubs_np = [np.array(lst, dtype=np.int32) for lst in block_to_tsubs]
    t_sub_to_blocks_np = [np.array(lst, dtype=np.int32) for lst in t_sub_to_blocks]

    # Precompute fixed-size block->tsubs flat array for vectorized gain computation
    max_tsubs_per_block = max(len(lst) for lst in block_to_tsubs)
    block_tsubs_flat = np.full((n_k, max_tsubs_per_block), -1, dtype=np.int32)
    block_tsubs_lens = np.zeros(n_k, dtype=np.int32)
    for i in range(n_k):
        lst = block_to_tsubs[i]
        L = len(lst)
        block_tsubs_flat[i, :L] = lst
        block_tsubs_lens[i] = L
    valid_mask = block_tsubs_flat >= 0
    safe_flat = np.where(block_tsubs_flat >= 0, block_tsubs_flat, 0)

    def compute_gains(cov_count):
        """Vectorized: gain[i] = # of t-subsets in block i that are uncovered."""
        uncovered_flag = (cov_count == 0).astype(np.int32)
        uncovered_vals = uncovered_flag[safe_flat]
        uncovered_vals = np.where(valid_mask, uncovered_vals, 0)
        return uncovered_vals.sum(axis=1)

    def greedy_cover_vectorized(seed=None, temperature=0.0):
        if seed is not None:
            random.seed(seed)

        covered = np.zeros(n_t, dtype=np.uint8)
        blocks_used = []
        used_blocks_set = set()
        cov_count = np.zeros(n_t, dtype=np.int32)

        while True:
            uncovered_idx = np.where(covered == 0)[0]
            if len(uncovered_idx) == 0:
                break
            if time.time() > deadline:
                break

            # Vectorized gain over all candidate blocks
            gains = compute_gains(cov_count)
            for bi in used_blocks_set:
                gains[bi] = 0

            max_gain = int(gains.max())
            if max_gain <= 0:
                break

            if temperature > 0.0:
                # Probabilistic selection: sample proportional to gain^2 (soft-max / inverse-frequency flavor)
                w = gains.astype(np.float64) ** 2
                total = w.sum()
                if total <= 0:
                    best_block = int(np.argmax(gains))
                else:
                    probs = w / total
                    best_block = int(np.random.choice(n_k, p=probs))
                    if gains[best_block] == 0:
                        # Fallback to argmax if we sampled a zero-gain block
                        best_block = int(np.argmax(gains))
            else:
                best_block = int(np.argmax(gains))

            blocks_used.append(list(all_k_subs[best_block]))
            used_blocks_set.add(best_block)

            for j in block_to_tsubs[best_block]:
                cov_count[j] += 1
                if cov_count[j] == 1:
                    covered[j] = 1

        return blocks_used, cov_count

    def repair_phase(blocks, cov_count, temperature=0.0):
        """Dynamic repair: while any t-subset is uncovered, force blocks covering the
        most-frequently-uncovered t-subsets. Escapes local minima of the pure greedy."""
        used_blocks_set = set()
        block_ids = []
        for block in blocks:
            bid = ksub_to_idx[tuple(block)]
            if bid not in used_blocks_set:
                used_blocks_set.add(bid)
                block_ids.append(bid)
        
        # Precompute boolean mask of used blocks for fast filtering
        used_mask = np.zeros(n_k, dtype=bool)
        for bid in used_blocks_set:
            used_mask[bid] = True

        while True:
            uncovered_idx = np.where(cov_count == 0)[0]
            if len(uncovered_idx) == 0:
                break
            if time.time() > deadline:
                break

            # Select a random uncovered t-subset to repair
            # Random selection is faster than finding the "easiest" one and works well for coverage
            best_tsub = uncovered_idx[np.random.randint(len(uncovered_idx))]
            
            # Find unused blocks covering this t-subset
            cand_blocks = t_sub_to_blocks_np[best_tsub]
            # Filter out used blocks
            valid_cand = cand_blocks[~used_mask[cand_blocks]]
            
            if len(valid_cand) == 0:
                # This t-subset cannot be covered by any new block? 
                # This shouldn't happen if the initial set of all k-subsets is available.
                # If it does, break to avoid infinite loop.
                break

            # Compute gains for these candidates
            # We can compute gains for all blocks and index, or just for candidates.
            # Given the small number of candidates per t-subset (C(14,3)=364), 
            # and that we only do this for one t-subset at a time, 
            # computing global gains is O(n_k * C(k,t)) which is expensive.
            # Instead, calculate gain for each candidate block individually.
            # Gain = number of uncovered t-subsets in this block.
            
            best_block = -1
            best_gain = -1
            
            # Optimization: If temperature is 0, just pick the one with max gain.
            # We can vectorize the gain calculation for the candidates.
            # But candidates are a list of block IDs. 
            # Let's just loop through candidates, it's small (~300).
            
            for bid in valid_cand:
                # Count how many t-subsets in this block are currently uncovered (cov_count == 0)
                # block_to_tsubs[bid] is an array of t-subset indices
                tsubs_in_block = block_to_tsubs[bid]
                # Check which ones are uncovered
                gains = np.sum(cov_count[tsubs_in_block] == 0)
                
                if gains > best_gain:
                    best_gain = gains
                    best_block = bid
                elif gains == best_gain and temperature > 0:
                    # Probabilistic tie-breaking
                    if np.random.rand() < 0.5:
                        best_block = bid

            if best_block == -1:
                break

            blocks.append(list(all_k_subs[best_block]))
            used_blocks_set.add(best_block)
            used_mask[best_block] = True
            block_ids.append(best_block)
            
            for j in block_to_tsubs[best_block]:
                cov_count[j] += 1

        return blocks

    def is_complete(blocks):
        cov = np.zeros(n_t, dtype=np.uint8)
        for block in blocks:
            for sub in itertools.combinations(block, t):
                cov[t_sub_to_idx[sub]] = 1
        return np.all(cov == 1)

    # Run multiple probabilistic greedy + repair restarts; keep the best complete result
    blocks = None
    for seed in range(0, 30):
        if time.time() > deadline - 15:
            break
        # temperature: probabilistic sampling proportional to gain^2 (soft-max / inverse-frequency)
        temp = 0.5
        b, cc = greedy_cover_vectorized(seed=seed, temperature=temp)
        # Repair any uncovered t-subsets
        b = repair_phase(b, cc, temperature=0.3)
        if is_complete(b):
            if blocks is None or len(b) < len(blocks):
                blocks = b
        elif blocks is None:
            blocks = b
    if blocks is None:
        b, cc = greedy_cover_vectorized(seed=0, temperature=0.0)
        blocks = repair_phase(b, cc, temperature=0.0)

    # Reduction pass: remove redundant blocks using coverage counts
    def try_reduce(blocks):
        if not blocks:
            return blocks

        cov_count = np.zeros(n_t, dtype=np.int32)
        block_ids = []
        for block in blocks:
            bid = ksub_to_idx[tuple(block)]
            block_ids.append(bid)
            for j in block_to_tsubs[bid]:
                cov_count[j] += 1

        changed = True
        while changed:
            changed = False
            for i in range(len(block_ids)):
                if time.time() > deadline - 2:
                    break
                bid = block_ids[i]
                # A block is redundant iff none of its t-subsets is covered only by it
                redundant = True
                for j in block_to_tsubs[bid]:
                    if cov_count[j] == 1:
                        redundant = False
                        break
                if redundant:
                    for j in block_to_tsubs[bid]:
                        cov_count[j] -= 1
                    block_ids.pop(i)
                    blocks.pop(i)
                    changed = True
                    break  # restart

        return blocks

    if time.time() < deadline - 10:
        blocks = try_reduce(blocks)

    # Final verification
    if not is_complete(blocks):
        # Emergency fallback: repair from current state
        cov_count = np.zeros(n_t, dtype=np.int32)
        for block in blocks:
            bid = ksub_to_idx[tuple(block)]
            for j in block_to_tsubs[bid]:
                cov_count[j] += 1
        blocks = repair_phase(blocks, cov_count, temperature=0.0)
        
        # If still not complete, force a very aggressive repair or fresh start
        if not is_complete(blocks):
             # Fresh deterministic greedy + repair
             b, cc = greedy_cover_vectorized(seed=0, temperature=0.0)
             blocks = repair_phase(b, cc, temperature=0.0)
             
             # If STILL not complete, there might be a bug in logic or time ran out.
             # As a last resort, ensure we return *something* valid if possible.
             # The task requires a valid covering.
             # If we can't complete it, we might fail the score, but we must return a list.
             # Let's try one more repair pass with temperature 0.
             if not is_complete(blocks) and time.time() < deadline:
                 cov_count = np.zeros(n_t, dtype=np.int32)
                 for block in blocks:
                     bid = ksub_to_idx[tuple(block)]
                     for j in block_to_tsubs[bid]:
                         cov_count[j] += 1
                 blocks = repair_phase(blocks, cov_count, temperature=0.0)

    return blocks
# EVOLVE-BLOCK-END
