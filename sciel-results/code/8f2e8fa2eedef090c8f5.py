import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0

    points = list(range(v))
    triples = list(itertools.combinations(points, t))
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    num_triples = len(triples)

    all_k_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_k_blocks)

    # Precompute: for each block, the set of triples it covers (as a Python int bitmask)
    # And for each triple, the set of blocks that cover it (as a Python int bitmask)
    block_triple_mask = []  # list of int bitmasks, one per block
    for block in all_k_blocks:
        mask = 0
        for triple in itertools.combinations(block, t):
            mask |= (1 << triple_to_idx[triple])
        block_triple_mask.append(mask)

    triple_block_mask = [0] * num_triples
    for b_idx, mask in enumerate(block_triple_mask):
        # Extract set bits
        m = mask
        while m:
            low = m & (-m)
            t_idx = low.bit_length() - 1
            triple_block_mask[t_idx] |= (1 << b_idx)
            m ^= low

    def greedy_cover_bitset(seed, time_budget):
        """Greedy set cover using bitmasks. Returns list of block indices."""
        rng = random.Random(seed)
        uncovered = (1 << num_triples) - 1  # all triples uncovered
        selected = []
        
        while uncovered:
            if time.time() > time_budget:
                break
            # Find block with maximum gain (most newly covered triples)
            best_b = -1
            best_gain = 0
            # Sample blocks for speed when num_blocks is large
            if num_blocks > 5000:
                sample_size = min(num_blocks, 3000)
                candidates = rng.sample(range(num_blocks), sample_size)
            else:
                candidates = range(num_blocks)
            
            for b_idx in candidates:
                gain = (block_triple_mask[b_idx] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_b = b_idx
                elif gain == best_gain and gain > 0:
                    # Tie-breaking: random choice among ties
                    if rng.random() < 0.5:
                        best_b = b_idx
            
            if best_b == -1 or best_gain == 0:
                break
            
            selected.append(best_b)
            uncovered &= ~block_triple_mask[best_b]
        
        return selected

    def complete_cover_bitset(selected, deadline):
        """Complete the cover if greedy broke early."""
        # Compute what's still uncovered
        covered = 0
        for b_idx in selected:
            covered |= block_triple_mask[b_idx]
        uncovered = ((1 << num_triples) - 1) & ~covered
        
        while uncovered:
            if time.time() > deadline:
                break
            best_b = -1
            best_gain = 0
            if num_blocks > 5000:
                sample_size = min(num_blocks, 3000)
                candidates = random.sample(range(num_blocks), sample_size)
            else:
                candidates = range(num_blocks)
            
            for b_idx in candidates:
                gain = (block_triple_mask[b_idx] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_b = b_idx
            
            if best_b == -1 or best_gain == 0:
                break
            selected.append(best_b)
            covered |= block_triple_mask[best_b]
            uncovered = ((1 << num_triples) - 1) & ~covered
        
        return selected

    def reduce_cover_bitset(selected):
        """Remove redundant blocks from the cover."""
        if not selected:
            return selected
        
        # Compute coverage count for each triple
        covered_mask = 0
        for b_idx in selected:
            covered_mask |= block_triple_mask[b_idx]
        
        # For each block, check if all its triples are covered by other blocks
        active = [True] * len(selected)
        changed = True
        while changed:
            changed = False
            for i in range(len(selected)):
                if not active[i]:
                    continue
                b_idx = selected[i]
                # What would be the coverage without this block?
                others_cover = 0
                for j in range(len(selected)):
                    if j != i and active[j]:
                        others_cover |= block_triple_mask[selected[j]]
                # If all triples of this block are covered by others, remove it
                if (block_triple_mask[b_idx] & ~others_cover) == 0:
                    active[i] = False
                    changed = True
        
        return [selected[i] for i in range(len(selected)) if active[i]]

    def local_search_bitset(selected, deadline):
        """Local search: try to replace blocks to reduce count."""
        if len(selected) < 2:
            return selected
        selected = list(selected)
        
        rng = random.Random(42)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            order = list(range(len(selected)))
            rng.shuffle(order)
            
            for i in order:
                if time.time() > deadline - 5:
                    break
                b_idx = selected[i]
                
                # Compute coverage without this block
                others_cover = 0
                for j in range(len(selected)):
                    if j != i:
                        others_cover |= block_triple_mask[selected[j]]
                
                # Uncovered triples if we remove this block
                uncovered = block_triple_mask[b_idx] & ~others_cover
                if uncovered == 0:
                    # Block is redundant
                    selected.pop(i)
                    improved = True
                    break
                
                # Try to find a single block that covers all uncovered triples
                # AND is not already in selected
                # Find intersection of all blocks covering each uncovered triple
                # Get the uncovered triple indices
                unc_triples = []
                m = uncovered
                while m:
                    low = m & (-m)
                    t_idx = low.bit_length() - 1
                    unc_triples.append(t_idx)
                    m ^= low
                
                if len(unc_triples) > 20:
                    # Too many to easily find a single replacement
                    continue
                
                # Find candidate blocks: intersection of triple_block_mask for each uncovered triple
                cand_mask = (1 << num_blocks) - 1
                for t_idx in unc_triples:
                    cand_mask &= triple_block_mask[t_idx]
                
                # Remove already-selected blocks from candidates
                selected_mask = 0
                for j in range(len(selected)):
                    if j != i:
                        selected_mask |= (1 << selected[j])
                cand_mask &= ~selected_mask
                
                if cand_mask:
                    # Pick a candidate
                    m = cand_mask
                    low = m & (-m)
                    new_b = low.bit_length() - 1
                    selected.pop(i)
                    selected.append(new_b)
                    improved = True
                    break
        
        return reduce_cover_bitset(selected)

    best_cover = None
    best_len = float('inf')

    # Phase 1: Multiple greedy restarts
    seed = 0
    while time.time() < deadline - 30:
        cover = greedy_cover_bitset(seed, deadline - 25)
        cover = complete_cover_bitset(cover, deadline - 20)
        cover = reduce_cover_bitset(cover)
        
        if len(cover) < best_len:
            best_len = len(cover)
            best_cover = cover
        
        seed += 1

    if best_cover is None:
        best_cover = greedy_cover_bitset(0, deadline - 10)
        best_cover = complete_cover_bitset(best_cover, deadline - 5)
        best_cover = reduce_cover_bitset(best_cover)
        best_len = len(best_cover)

    # Phase 2: Local search on the best cover
    best_cover = local_search_bitset(best_cover, deadline - 5)
    best_len = len(best_cover)

    # Phase 3: If time remains, try more restarts
    while time.time() < deadline - 5:
        seed = random.randint(1000, 100000)
        cover = greedy_cover_bitset(seed, deadline - 10)
        cover = complete_cover_bitset(cover, deadline - 8)
        cover = reduce_cover_bitset(cover)
        if len(cover) < best_len:
            best_len = len(cover)
            best_cover = cover
            # Do local search on the new best
            best_cover = local_search_bitset(best_cover, deadline - 5)
            best_len = len(best_cover)

    # Final safety: ensure validity
    covered_mask = 0
    for b_idx in best_cover:
        covered_mask |= block_triple_mask[b_idx]
    if covered_mask != ((1 << num_triples) - 1):
        best_cover = complete_cover_bitset(best_cover, deadline)

    return [list(all_k_blocks[i]) for i in best_cover]
