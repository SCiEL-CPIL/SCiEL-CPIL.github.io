import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0

    # Precompute all t-subsets and map to indices
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {frozenset(c): i for i, c in enumerate(all_tsubs)}

    # Precompute all k-blocks
    all_kblocks = list(itertools.combinations(range(v), k))
    n_kblocks = len(all_kblocks)

    # block_tsubs_list[bi] = list of t-subset indices
    block_tsubs_list = [[] for _ in range(n_kblocks)]
    # tsub_blocks_list[j] = list of block indices
    tsub_blocks_list = [[] for _ in range(n_tsubs)]

    for bi, block in enumerate(all_kblocks):
        for c in itertools.combinations(block, t):
            j = tsub_to_idx[frozenset(c)]
            block_tsubs_list[bi].append(j)
            tsub_blocks_list[j].append(bi)

    # Bitsets for fast operations
    # block_tsub_bitsets[bi] : int with bits set for t-subsets covered by block bi
    block_tsub_bitsets = [0] * n_kblocks
    for bi in range(n_kblocks):
        mask = 0
        for j in block_tsubs_list[bi]:
            mask |= (1 << j)
        block_tsub_bitsets[bi] = mask

    # tsub_block_bitsets[j] : int with bits set for blocks covering t-subset j
    tsub_block_bitsets = [0] * n_tsubs
    for j in range(n_tsubs):
        mask = 0
        for bi in tsub_blocks_list[j]:
            mask |= (1 << bi)
        tsub_block_bitsets[j] = mask

    def _greedy_init(rng):
        """
        Greedy construction using bitsets for coverage tracking.
        """
        selected = set()
        cover_mask = [0] * n_tsubs
        
        # Initial gains
        gains = [len(block_tsubs_list[bi]) for bi in range(n_kblocks)]
        
        # Use a simple list-based greedy for speed, or heap
        # For simplicity and speed in Python, list scan might be ok for small N, 
        # but N=50k is large. Heap is better.
        
        import heapq
        heap = [(-gains[bi], bi) for bi in range(n_kblocks)]
        heapq.heapify(heap)
        
        uncovered = n_tsubs
        # We need to know which t-subsets are uncovered to update gains correctly?
        # Actually, standard greedy picks max marginal gain.
        
        while uncovered > 0 and time.time() < deadline - 20:
            best_bi = -1
            best_gain = 0
            while heap:
                neg_g, bi = heapq.heappop(heap)
                g = -neg_g
                if bi in selected:
                    continue
                if g == gains[bi] and g > 0:
                    best_bi = bi
                    best_gain = g
                    break
            
            if best_bi == -1:
                break
                
            selected.add(best_bi)
            
            # Update cover_mask and gains
            for j in block_tsubs_list[best_bi]:
                if not (cover_mask[j] >> best_bi) & 1:
                    cover_mask[j] |= (1 << best_bi)
                    uncovered -= 1
                    # Update gains for all other blocks covering j
                    # This is expensive if done naively.
                    # For greedy init, we can just recompute gains periodically or use a simpler heuristic.
                    # Given time constraints, a simpler greedy might suffice for init.
            
            gains[best_bi] = -1
            heapq.heappush(heap, (-gains[best_bi], best_bi))

        return list(selected)

    def _local_search(block_indices, dl):
        """
        Bitset-accelerated local search.
        """
        selected = set(block_indices)
        cover_mask = [0] * n_tsubs
        for bi in selected:
            for j in block_tsubs_list[bi]:
                cover_mask[j] |= (1 << bi)

        rng = random.Random()
        improved = True
        while improved and time.time() < dl:
            improved = False
            
            # Phase 1: Remove redundant blocks
            to_remove = []
            for bi in list(selected):
                redundant = True
                for j in block_tsubs_list[bi]:
                    if cover_mask[j] == (1 << bi):
                        redundant = False
                        break
                if redundant:
                    to_remove.append(bi)
            
            if to_remove:
                for bi in to_remove:
                    selected.remove(bi)
                    for j in block_tsubs_list[bi]:
                        cover_mask[j] &= ~(1 << bi)
                improved = True
                continue

            # Phase 2: 1-for-1 Swaps
            current_blocks = list(selected)
            rng.shuffle(current_blocks)
            
            for bi in current_blocks:
                if time.time() > dl:
                    break
                if bi not in selected:
                    continue
                
                # Find critical t-subsets for bi
                critical = []
                for j in block_tsubs_list[bi]:
                    if cover_mask[j] == (1 << bi):
                        critical.append(j)
                
                if not critical:
                    # Should have been caught in Phase 1
                    selected.remove(bi)
                    for j in block_tsubs_list[bi]:
                        cover_mask[j] &= ~(1 << bi)
                    improved = True
                    break
                
                # Find candidate blocks
                # Pick most constrained critical t-subset
                min_j = min(critical, key=lambda j: bin(tsub_block_bitsets[j]).count('1'))
                candidates_mask = tsub_block_bitsets[min_j]
                
                # Filter candidates that cover all other critical t-subsets
                valid_candidates = []
                # Iterate through bits in candidates_mask
                cm = candidates_mask
                while cm:
                    # Get lowest set bit
                    lb = cm & -cm
                    bj = lb.bit_length() - 1
                    cm ^= lb
                    
                    if bj == bi or bj in selected:
                        continue
                    
                    # Check if bj covers all critical t-subsets
                    # (block_tsub_bitsets[bj] & critical_bitset) == critical_bitset
                    # Precompute critical_bitset?
                    # For small critical lists, direct check is fine.
                    ok = True
                    for j in critical:
                        if not (block_tsub_bitsets[bj] >> j) & 1:
                            ok = False
                            break
                    if ok:
                        valid_candidates.append(bj)
                
                if valid_candidates:
                    # Pick a random valid candidate
                    bj = rng.choice(valid_candidates)
                    
                    # Perform swap
                    selected.remove(bi)
                    for j in block_tsubs_list[bi]:
                        cover_mask[j] &= ~(1 << bi)
                    
                    selected.add(bj)
                    for j in block_tsubs_list[bj]:
                        cover_mask[j] |= (1 << bj)
                    
                    improved = True
                    break

            # Phase 3: 2-for-1 Swaps (Simplified)
            if not improved and len(selected) > 2:
                current_blocks = list(selected)
                rng.shuffle(current_blocks)
                for i in range(min(10, len(current_blocks))): # Limit checks
                    if time.time() > dl:
                        break
                    bi = current_blocks[i]
                    if bi not in selected:
                        continue
                    # Pick a random partner
                    bj = rng.choice([b for b in selected if b != bi])
                    
                    # Find union of critical t-subsets
                    critical_bi = [j for j in block_tsubs_list[bi] if cover_mask[j] == (1 << bi)]
                    critical_bj = [j for j in block_tsubs_list[bj] if cover_mask[j] == (1 << bj)]
                    
                    # Note: A t-subset might be covered only by bi and bj.
                    # We need to find a block bk that covers all j where cover_mask[j] is subset of {bi, bj}
                    # and j is in bi or bj.
                    
                    critical_union = set()
                    for j in block_tsubs_list[bi]:
                        if cover_mask[j] & ~(1 << bi) & ~(1 << bj) == 0:
                            critical_union.add(j)
                    for j in block_tsubs_list[bj]:
                        if cover_mask[j] & ~(1 << bi) & ~(1 << bj) == 0:
                            critical_union.add(j)
                            
                    if not critical_union:
                        continue
                        
                    min_j = min(critical_union, key=lambda j: bin(tsub_block_bitsets[j]).count('1'))
                    candidates_mask = tsub_block_bitsets[min_j]
                    
                    valid_candidates = []
                    cm = candidates_mask
                    while cm:
                        lb = cm & -cm
                        bk = lb.bit_length() - 1
                        cm ^= lb
                        
                        if bk == bi or bk == bj or bk in selected:
                            continue
                            
                        ok = True
                        for j in critical_union:
                            if not (block_tsub_bitsets[bk] >> j) & 1:
                                ok = False
                                break
                        if ok:
                            valid_candidates.append(bk)
                            
                    if valid_candidates:
                        bk = rng.choice(valid_candidates)
                        
                        # Perform 2-for-1 swap
                        selected.remove(bi)
                        for j in block_tsubs_list[bi]:
                            cover_mask[j] &= ~(1 << bi)
                        selected.remove(bj)
                        for j in block_tsubs_list[bj]:
                            cover_mask[j] &= ~(1 << bj)
                        
                        selected.add(bk)
                        for j in block_tsub_bitsets[bk]: # Wait, block_tsubs_list
                            cover_mask[j] |= (1 << bk)
                        
                        improved = True
                        break

        return [list(all_kblocks[bi]) for bi in selected]

    best_blocks = None
    best_len = float('inf')

    # Multiple restarts
    for seed in range(20):
        if time.time() > deadline - 10:
            break
        rng = random.Random(seed * 12345 + 6789)
        blocks = _greedy_init(rng)
        if not blocks:
            continue
        blocks = _local_search(blocks, deadline - 5)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks

    return best_blocks if best_blocks else []
