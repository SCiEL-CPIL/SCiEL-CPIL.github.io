import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    # Allow up to 110 seconds to be safe (limit is 120s)
    timeout = 110.0
    
    points = list(range(v))
    all_triples = list(itertools.combinations(points, t))
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    # Precompute all blocks and their covered triples
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Map block tuple to index for O(1) lookup
    block_to_idx = {b: i for i, b in enumerate(all_blocks)}
    
    # block_triples[i] = list of triple indices covered by block i
    block_triples = []
    # triple_to_blocks[j] = list of block indices covering triple j
    triple_to_blocks = [[] for _ in range(num_triples)]
    
    for i, block in enumerate(all_blocks):
        covered = [triple_to_idx[tup] for tup in itertools.combinations(block, t)]
        block_triples.append(covered)
        for tr_idx in covered:
            triple_to_blocks[tr_idx].append(i)
            
    best_blocks = None
    best_count = float('inf')
    
    def greedy_cover(random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
            
        uncovered = set(range(num_triples))
        blocks_idx = []
        
        while uncovered:
            # Pick a random uncovered triple
            target_triple_idx = random.choice(list(uncovered))
            
            # Get candidate blocks covering this triple
            candidates = triple_to_blocks[target_triple_idx]
            
            best_gain = -1
            best_candidates = []
            
            # Evaluate gain for each candidate
            for bi in candidates:
                gain = 0
                for tr_idx in block_triples[bi]:
                    if tr_idx in uncovered:
                        gain += 1
                        if gain == 35: 
                            break
                            
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [bi]
                    if best_gain == 35:
                        break
                elif gain == best_gain:
                    best_candidates.append(bi)
            
            if best_candidates:
                # Pick randomly among the best candidates to diversify
                best_block_idx = random.choice(best_candidates)
                blocks_idx.append(best_block_idx)
                # Update uncovered
                for tr_idx in block_triples[best_block_idx]:
                    uncovered.discard(tr_idx)
            else:
                break
                
        return blocks_idx

    def local_search(blocks_idx):
        if not blocks_idx:
            return blocks_idx
        
        # triple_count[j] = number of blocks covering triple j
        triple_count = [0] * num_triples
        
        # Initialize counts
        for bi in blocks_idx:
            for tr_idx in block_triples[bi]:
                triple_count[tr_idx] += 1
        
        active_blocks = list(blocks_idx)
        active_set = set(active_blocks)
        
        # Helper to check if block is redundant
        def is_redundant(bi):
            for tr_idx in block_triples[bi]:
                if triple_count[tr_idx] == 1:
                    return False
            return True

        # Phase 1: Remove redundant blocks
        changed = True
        while changed:
            changed = False
            to_remove = None
            for bi in active_blocks:
                if is_redundant(bi):
                    to_remove = bi
                    break
            
            if to_remove is not None:
                active_blocks.remove(to_remove)
                active_set.remove(to_remove)
                for tr_idx in block_triples[to_remove]:
                    triple_count[tr_idx] -= 1
                changed = True
                
        # Phase 2: Swap Search
        # Try to swap blocks to potentially allow further deletions
        max_swaps = 200
        swaps_done = 0
        
        while swaps_done < max_swaps and time.time() - start_time < timeout - 10:
            random.shuffle(active_blocks)
            swapped = False
            for bi in active_blocks:
                # Find exclusive triples of bi
                exclusive_triples = []
                for tr_idx in block_triples[bi]:
                    if triple_count[tr_idx] == 1:
                        exclusive_triples.append(tr_idx)
                
                if not exclusive_triples:
                    # Redundant
                    active_blocks.remove(bi)
                    active_set.remove(bi)
                    for tr_idx in block_triples[bi]:
                        triple_count[tr_idx] -= 1
                    swapped = True
                    break
                
                # Find candidate blocks B' such that B' covers all exclusive_triples
                candidates = set(triple_to_blocks[exclusive_triples[0]])
                candidates.discard(bi)
                
                for tr_idx in exclusive_triples[1:]:
                    if not candidates:
                        break
                    candidates &= set(triple_to_blocks[tr_idx])
                
                candidates -= active_set
                
                if candidates:
                    new_bi = random.choice(list(candidates))
                    
                    # Perform Swap
                    for tr_idx in block_triples[bi]:
                        triple_count[tr_idx] -= 1
                    active_blocks.remove(bi)
                    active_set.remove(bi)
                    
                    for tr_idx in block_triples[new_bi]:
                        triple_count[tr_idx] += 1
                    active_blocks.append(new_bi)
                    active_set.add(new_bi)
                    
                    swaps_done += 1
                    swapped = True
                    break
            
            if not swapped:
                break
        
        # Final Phase 1 cleanup
        changed = True
        while changed:
            changed = False
            to_remove = None
            for bi in active_blocks:
                if is_redundant(bi):
                    to_remove = bi
                    break
            
            if to_remove is not None:
                active_blocks.remove(to_remove)
                active_set.remove(to_remove)
                for tr_idx in block_triples[to_remove]:
                    triple_count[tr_idx] -= 1
                changed = True
        
        return active_blocks

    # Main Loop
    attempts = 0
    while time.time() - start_time < timeout:
        seed = attempts
        blocks_idx = greedy_cover(seed)
        
        if time.time() - start_time > timeout - 5.0:
            break
            
        improved = local_search(blocks_idx)
        
        if len(improved) < best_count:
            best_count = len(improved)
            best_blocks = improved
            
        attempts += 1
        
        if time.time() - start_time > timeout - 5.0:
            break

    if best_blocks is None:
        blocks_idx = greedy_cover(0)
        best_blocks = local_search(blocks_idx)
        
    return [list(all_blocks[bi]) for bi in best_blocks]
