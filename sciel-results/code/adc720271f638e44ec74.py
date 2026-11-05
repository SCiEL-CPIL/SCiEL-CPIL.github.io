import itertools
import numpy as np
from math import comb
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill

    # 1. Precomputation
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {trip: i for i, trip in enumerate(all_triples)}
    n_triples = len(all_triples)

    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    k_t = comb(k, t)

    # Build block -> triple index mapping
    block_triple_indices = np.empty((n_blocks, k_t), dtype=np.int32)
    for i, block in enumerate(all_blocks):
        for j, sub in enumerate(itertools.combinations(block, t)):
            block_triple_indices[i, j] = triple_to_idx[sub]

    # Triple occurrence frequency
    triple_occurrence = np.zeros(n_triples, dtype=np.int32)
    for i in range(n_blocks):
        triple_occurrence[block_triple_indices[i]] += 1
    inv_freq = 1.0 / (triple_occurrence + 1).astype(np.float64)

    def greedy_cover(uncovered_mask, random_seed, use_weighting=True):
        """Greedy cover starting from given uncovered_mask state."""
        rng = np.random.RandomState(random_seed)
        uncovered = uncovered_mask.copy()
        selected_blocks = []
        remaining = int(np.sum(uncovered))

        while remaining > 0:
            if time.time() > deadline - 3:
                break

            # Calculate gains: number of uncovered triples in each block
            gains = np.sum(uncovered[block_triple_indices], axis=1)
            max_gain = int(np.max(gains))
            if max_gain == 0:
                break

            candidates = np.where(gains == max_gain)[0]

            if len(candidates) == 1:
                best_idx = candidates[0]
            else:
                if use_weighting:
                    cand_triples = block_triple_indices[candidates]
                    cand_uncovered = uncovered[cand_triples]
                    weights = np.sum(inv_freq[cand_triples] * cand_uncovered, axis=1)
                    weights += rng.random(len(candidates)) * 0.01
                    best_local = int(np.argmax(weights))
                    best_idx = candidates[best_local]
                else:
                    best_idx = candidates[int(rng.randint(len(candidates)))]

            block_idx = block_triple_indices[best_idx]
            uncovered[block_idx] = False
            remaining -= max_gain
            selected_blocks.append(best_idx)

        return selected_blocks, uncovered

    def lp_guided_fill(uncovered_mask, max_blocks=200):
        """Use LP relaxation to guide selection for hard-to-cover triples."""
        uncovered_idx = np.where(uncovered_mask)[0]
        if len(uncovered_idx) == 0:
            return []

        # Find candidate blocks that cover at least one uncovered triple
        # A block is relevant if it covers >= 1 uncovered triple
        gains = np.sum(uncovered_mask[block_triple_indices], axis=1)
        candidate_mask = gains > 0
        candidate_blocks = np.where(candidate_mask)[0]
        candidate_gains = gains[candidate_blocks]

        if len(candidate_blocks) == 0:
            return []

        # If too many candidates, restrict to top-gain blocks
        if len(candidate_blocks) > max_blocks:
            top_idx = np.argsort(-candidate_gains)[:max_blocks]
            candidate_blocks = candidate_blocks[top_idx]

        # Build incidence: for each uncovered triple, which candidate blocks cover it
        # uncovered_idx has length U, candidate_blocks has length B
        U = len(uncovered_idx)
        B = len(candidate_blocks)

        # incidence[u, b] = 1 if candidate_blocks[b] covers uncovered_idx[u]
        # For each candidate block, find which uncovered triples it covers
        incidence = np.zeros((U, B), dtype=np.float64)
        for b_idx, blk in enumerate(candidate_blocks):
            blk_triples = block_triple_indices[blk]
            # Check which of uncovered_idx are in blk_triples
            # Use set intersection
            blk_set = set(blk_triples.tolist())
            for u_idx, tri in enumerate(uncovered_idx):
                if tri in blk_set:
                    incidence[u_idx, b_idx] = 1.0

        # Solve LP: minimize sum(x_b) subject to incidence @ x >= 1, 0 <= x <= 1
        # Use scipy linprog
        from scipy.optimize import linprog

        c = np.ones(B)  # minimize sum x
        # Constraints: -incidence @ x <= -1
        A_ub = -incidence
        b_ub = -np.ones(U)
        bounds = [(0, 1)] * B

        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0 and res.x is not None:
                # Select blocks where x_b is high
                x_vals = res.x
                # Greedy select based on LP values
                selected = []
                remaining_uncovered = set(uncovered_idx.tolist())
                order = np.argsort(-x_vals)
                for b_idx in order:
                    if len(selected) >= 50:
                        break
                    if time.time() > deadline - 3:
                        break
                    blk = candidate_blocks[b_idx]
                    blk_triples = block_triple_indices[blk]
                    new_cover = [t for t in blk_triples if t in remaining_uncovered]
                    if new_cover:
                        selected.append(blk)
                        for t in new_cover:
                            remaining_uncovered.discard(t)
                        if not remaining_uncovered:
                            break
                return selected
        except Exception:
            pass

        # Fallback: pure greedy on candidates
        selected = []
        remaining_uncovered = set(uncovered_idx.tolist())
        for b_idx in np.argsort(-candidate_gains):
            if time.time() > deadline - 3:
                break
            blk = candidate_blocks[b_idx]
            blk_triples = block_triple_indices[blk]
            new_cover = [t for t in blk_triples if t in remaining_uncovered]
            if new_cover:
                selected.append(blk)
                for t in new_cover:
                    remaining_uncovered.discard(t)
                if not remaining_uncovered:
                    break
        return selected

    def local_search_polish(selected_block_indices, max_iters=300):
        """Try to remove redundant blocks."""
        if not selected_block_indices:
            return selected_block_indices

        n_sel = len(selected_block_indices)
        triple_count = np.zeros(n_triples, dtype=np.int32)
        for bi in selected_block_indices:
            triple_count[block_triple_indices[bi]] += 1

        improved = True
        iters = 0
        while improved and iters < max_iters:
            if time.time() > deadline - 3:
                break
            improved = False
            iters += 1
            order = list(range(n_sel))
            np.random.RandomState(iters * 12345 + 7).shuffle(order)
            for pos in order:
                if pos >= n_sel:
                    continue
                bi = selected_block_indices[pos]
                bi_triples = block_triple_indices[bi]
                only_covered = int(np.sum(triple_count[bi_triples] == 1))
                if only_covered == 0:
                    triple_count[bi_triples] -= 1
                    selected_block_indices = [x for x in selected_block_indices if x != bi]
                    n_sel -= 1
                    improved = True
                    break
        return selected_block_indices

    def swap_local_search(selected_block_indices, max_iters=100):
        """Try swapping one block for another to reduce total count."""
        if len(selected_block_indices) < 2:
            return selected_block_indices

        n_sel = len(selected_block_indices)
        triple_count = np.zeros(n_triples, dtype=np.int32)
        for bi in selected_block_indices:
            triple_count[block_triple_indices[bi]] += 1

        improved = True
        iters = 0
        while improved and iters < max_iters:
            if time.time() > deadline - 3:
                break
            improved = False
            iters += 1

            # Try removing each block and see if we can re-cover with one fewer
            order = list(range(n_sel))
            np.random.RandomState(iters * 999 + 3).shuffle(order)

            for pos in order:
                if pos >= n_sel:
                    continue
                bi = selected_block_indices[pos]
                bi_triples = block_triple_indices[bi]
                only_covered = int(np.sum(triple_count[bi_triples] == 1))
                if only_covered == 0:
                    # Can remove entirely
                    triple_count[bi_triples] -= 1
                    selected_block_indices = [x for x in selected_block_indices if x != bi]
                    n_sel -= 1
                    improved = True
                    break

        return selected_block_indices

    # Main search
    best_blocks = None
    best_len = float('inf')

    n_restarts = 40

    for i in range(n_restarts):
        if time.time() > deadline - 8:
            break

        # Phase 1: Greedy to get close
        blocks_idx, uncovered = greedy_cover(np.ones(n_triples, dtype=bool), i, use_weighting=(i % 2 == 0))

        # Phase 2: If triples remain, use LP-guided fill
        if np.sum(uncovered) > 0 and time.time() < deadline - 5:
            lp_blocks = lp_guided_fill(uncovered)
            blocks_idx = blocks_idx + lp_blocks

        # Phase 3: Local search polish
        if time.time() < deadline - 3:
            polished = local_search_polish(blocks_idx)
            if len(polished) < len(blocks_idx):
                blocks_idx = polished

        if len(blocks_idx) < best_len:
            best_len = len(blocks_idx)
            best_blocks = blocks_idx

    if best_blocks is None:
        best_blocks, _ = greedy_cover(np.ones(n_triples, dtype=bool), 0, use_weighting=False)
        best_len = len(best_blocks)

    # Final polish
    if time.time() < deadline - 3:
        polished = local_search_polish(best_blocks)
        if len(polished) < best_len:
            best_blocks = polished
            best_len = len(best_blocks)

    # Convert indices back to actual blocks
    return [list(all_blocks[i]) for i in best_blocks]
