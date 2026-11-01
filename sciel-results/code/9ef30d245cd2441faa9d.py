import itertools
import time
import random
from collections import defaultdict

# Precomputation for specific v=19, k=8, t=5 is not strictly necessary 
# if we generate on the fly, but precomputing t-subset IDs helps.

def _precompute(v, k, t):
    """
    Precompute mappings for efficient set cover operations.
    Returns:
    - tsubs: list of all t-subsets as tuples
    - tsub_id: dict mapping t-subset tuple to index
    - block_tsubs: dict mapping k-block tuple to list of t-subset indices it covers
    """
    # Generate all t-subsets
    tsubs = list(itertools.combinations(range(v), t))
    tsub_id = {ts: i for i, ts in enumerate(tsubs)}
    
    # Generate all k-blocks and their covered t-subsets
    # This is expensive if done for all blocks, but we can do it on demand or precompute for all.
    # Number of blocks: C(19, 8) = 75582.
    # Number of t-subsets per block: C(8, 5) = 56.
    # Total operations: ~4.2M. This is fast enough to precompute once.
    
    block_tsubs = {}
    for block in itertools.combinations(range(v), k):
        covered = []
        for ts in itertools.combinations(block, t):
            covered.append(tsub_id[ts])
        block_tsubs[block] = covered
        
    return tsubs, tsub_id, block_tsubs

def _greedy_cover(v, k, t, rng, tsubs, tsub_id, block_tsubs):
    """
    Greedy set cover using randomized selection of uncovered t-subsets.
    """
    all_ids = set(range(len(tsubs)))
    uncovered = all_ids.copy()
    blocks = []
    
    # Map t-subset index to list of blocks containing it
    # We can build this on the fly or precompute. 
    # Precomputing tsub_to_blocks is useful for the "rarest" strategy.
    tsub_to_blocks = defaultdict(list)
    for block, covered in block_tsubs.items():
        for ts_id in covered:
            tsub_to_blocks[ts_id].append(block)
            
    while uncovered:
        # Pick a random uncovered t-subset
        # To make it "rarest", we could pick one with fewest covering blocks, 
        # but all blocks are potential candidates. 
        # Random picking from uncovered is a good heuristic to avoid bias.
        ts_id = rng.choice(list(uncovered))
        
        # Get all blocks containing this t-subset
        candidates = tsub_to_blocks[ts_id]
        
        # Find the block that covers the most uncovered t-subsets
        best_block = None
        best_gain = -1
        best_gain_set = None
        
        for block in candidates:
            gain = 0
            # Count how many of this block's t-subsets are in uncovered
            # We can do this by intersecting block_tsubs[block] with uncovered
            # Since uncovered is a set, `len(block_tsubs[block] & uncovered)` is fast if block_tsubs[block] is a set.
            # Let's store block_tsubs as sets for faster intersection.
            # Wait, I stored as lists. Let's convert to sets or use a different check.
            # Actually, iterating 56 elements is fast.
            for ts in block_tsubs[block]:
                if ts in uncovered:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block = block
                # Store the set of covered IDs to remove them later
                best_gain_set = block_tsubs[block]
                
        if best_block is None:
            # Should not happen if uncovered is not empty
            break
            
        blocks.append(best_block)
        # Remove covered t-subsets
        for ts in best_gain_set:
            uncovered.discard(ts)
            
    return blocks

def _local_search(blocks, v, k, t, deadline, tsubs, tsub_id, block_tsubs):
    """
    Local search: Removal, 1-Block Replacement, 2-Block Merge.
    """
    blocks = [list(b) for b in blocks]
    
    def build_coverage(blocks):
        """Map t_sub_id -> set of block indices covering it."""
        coverage = defaultdict(set)
        for bi, block in enumerate(blocks):
            # Ensure block is a tuple for lookup
            b_tuple = tuple(block)
            if b_tuple in block_tsubs:
                for ts_id in block_tsubs[b_tuple]:
                    coverage[ts_id].add(bi)
        return coverage

    coverage = build_coverage(blocks)
    
    def get_unique_sets(bi, blocks, coverage):
        """Get t-subsets covered ONLY by block bi."""
        b_tuple = tuple(blocks[bi])
        unique = []
        if b_tuple not in block_tsubs:
            return unique
        for ts_id in block_tsubs[b_tuple]:
            if len(coverage.get(ts_id, set())) == 1 and bi in coverage.get(ts_id, set()):
                unique.append(ts_id)
        return unique

    def try_remove(bi, blocks, coverage):
        """Remove block bi if it has no unique t-subsets."""
        unique = get_unique_sets(bi, blocks, coverage)
        if not unique:
            # Safe to remove
            b_tuple = tuple(blocks.pop(bi))
            if b_tuple in block_tsubs:
                for ts_id in block_tsubs[b_tuple]:
                    if ts_id in coverage:
                        coverage[ts_id].discard(bi)
                        if not coverage[ts_id]:
                            del coverage[ts_id]
            # Re-index coverage
            for ts_id in coverage:
                new_set = set()
                for idx in coverage[ts_id]:
                    if idx < bi:
                        new_set.add(idx)
                    else:
                        new_set.add(idx - 1)
                coverage[ts_id] = new_set
            return True
        return False

    def try_merge(bi, bj, blocks, coverage):
        """Try to merge blocks bi and bj into a single block."""
        unique_i = get_unique_sets(bi, blocks, coverage)
        unique_j = get_unique_sets(bj, blocks, coverage)
        
        if not unique_i and not unique_j:
            # Both redundant, just remove one?
            # try_remove handles this.
            return False
            
        # The new block must cover all unique sets of i and j
        # Union of all elements in unique_i and unique_j
        union_elems = set()
        for ts_id in unique_i:
            union_elems.update(tsubs[ts_id])
        for ts_id in unique_j:
            union_elems.update(tsubs[ts_id])
            
        if len(union_elems) > k:
            return False
            
        # Find a block containing union_elems
        # We need to extend union_elems to size k
        # Elements in union_elems are fixed.
        remaining = [x for x in range(v) if x not in union_elems]
        needed = k - len(union_elems)
        
        found = False
        # If needed is 0, the block is exactly union_elems
        if needed == 0:
            candidate = tuple(sorted(union_elems))
            if candidate in block_tsubs:
                # Check if it covers all unique_i and unique_j
                # By construction, it contains the elements, so it covers the sets.
                # Replace bi and bj with candidate
                # Remove bi and bj
                b_i = tuple(blocks[bi])
                b_j = tuple(blocks[bj])
                
                # We will remove the higher index first to keep indices stable for the lower one
                if bi > bj:
                    bi, bj = bj, bi
                    
                # Remove bj
                blocks.pop(bj)
                if b_j in block_tsubs:
                    for ts_id in block_tsubs[b_j]:
                        if ts_id in coverage:
                            coverage[ts_id].discard(bj)
                            if not coverage[ts_id]:
                                del coverage[ts_id]
                # Remove bi (index might have shifted if bj < bi, but we swapped so bi < bj originally? 
                # No, we swapped so bi is the smaller index in the pair, but we popped bj first.
                # If original bi < bj, we swapped to bi=bj_orig, bj=bi_orig? 
                # Let's just rebuild coverage after modification to be safe.
                
                # Insert candidate at position bi
                blocks.insert(bi, list(candidate))
                
                # Rebuild coverage is safer than manual update after two pops/inserts
                coverage.clear()
                for idx, b in enumerate(blocks):
                    b_t = tuple(b)
                    if b_t in block_tsubs:
                        for ts_id in block_tsubs[b_t]:
                            coverage[ts_id].add(idx)
                return True
            return False
        else:
            # Need to choose 'needed' elements from 'remaining'
            # Number of combinations: C(len(remaining), needed)
            # This can be large. 
            # Heuristic: Just try a few random extensions? Or exhaustive if small.
            # For now, if needed > 0, we skip complex merge to save time, 
            # unless needed is very small.
            if needed <= 2:
                for ext in itertools.combinations(remaining, needed):
                    candidate_elems = union_elems.union(ext)
                    candidate = tuple(sorted(candidate_elems))
                    if candidate in block_tsubs:
                        # Verify coverage
                        # It contains union_elems, so it covers unique_i and unique_j.
                        # Perform replacement
                        b_i = tuple(blocks[bi])
                        b_j = tuple(blocks[bj])
                        
                        if bi > bj:
                            bi, bj = bj, bi
                            
                        blocks.pop(bj)
                        blocks.pop(bi)
                        blocks.insert(bi, list(candidate))
                        
                        coverage.clear()
                        for idx, b in enumerate(blocks):
                            b_t = tuple(b)
                            if b_t in block_tsubs:
                                for ts_id in block_tsubs[b_t]:
                                    coverage[ts_id].add(idx)
                        return True
            return False

    improved = True
    while improved and time.time() < deadline:
        improved = False
        n_blocks = len(blocks)
        
        # Pass 1: Removal
        for bi in range(n_blocks):
            if time.time() >= deadline:
                break
            if try_remove(bi, blocks, coverage):
                improved = True
                n_blocks -= 1
                # Re-index not needed if we rebuild? 
                # try_remove updates coverage indices.
                # But if we remove, indices shift. 
                # My try_remove re-indexes.
                break # Restart loop to be safe
        
        if improved:
            continue

        # Pass 2: Merging
        for bi in range(len(blocks)):
            if time.time() >= deadline:
                break
            for bj in range(bi + 1, len(blocks)):
                if time.time() >= deadline:
                    break
                if try_merge(bi, bj, blocks, coverage):
                    improved = True
                    break
            if improved:
                break
                
    return blocks

def construct(v, k, t):
    deadline = time.time() + 145  # Safety margin
    rng = random.Random(42)
    
    # Precompute
    tsubs, tsub_id, block_tsubs = _precompute(v, k, t)
    
    best_blocks = None
    best_count = float('inf')
    
    # Phase 1: Multiple Greedy Restarts
    # Greedy is fast. We can do many restarts.
    while time.time() < deadline - 60:
        blocks = _greedy_cover(v, k, t, rng, tsubs, tsub_id, block_tsubs)
        if len(blocks) < best_count:
            best_blocks = blocks
            best_count = len(blocks)
            
    # Phase 2: Local Search on the best block
    if best_blocks is not None and time.time() < deadline - 5:
        best_blocks = _local_search(best_blocks, v, k, t, deadline - 2, tsubs, tsub_id, block_tsubs)
        
    return best_blocks
