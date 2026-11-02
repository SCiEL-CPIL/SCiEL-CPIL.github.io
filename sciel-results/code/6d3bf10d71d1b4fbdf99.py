import itertools
import random
import time

def construct(v, k, t):
    # Precomputation
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    block_masks = [0] * num_blocks
    blocks_containing_triple = [[] for _ in range(num_triples)]
    
    for b_idx, block in enumerate(all_blocks):
        mask = 0
        for triple in itertools.combinations(block, t):
            idx = triple_to_idx[triple]
            mask |= (1 << idx)
            blocks_containing_triple[idx].append(b_idx)
        block_masks[b_idx] = mask
        
    best_blocks = None
    best_count = float('inf')
    
    # Run multiple times
    for run in range(100):
        random.seed(run)
        uncovered_mask = (1 << num_triples) - 1
        uncovered_list = list(range(num_triples))
        blocks = []
        
        while uncovered_list:
            seed_idx = random.choice(uncovered_list)
            
            best_b_idx = -1
            max_covered = -1
            candidates = []
            
            for b_idx in blocks_containing_triple[seed_idx]:
                covered = block_masks[b_idx] & uncovered_mask
                count = covered.bit_count()
                if count > max_covered:
                    max_covered = count
                    candidates = [b_idx]
                elif count == max_covered:
                    candidates.append(b_idx)
            
            if not candidates:
                break
                
            chosen_b_idx = random.choice(candidates)
            blocks.append(list(all_blocks[chosen_b_idx]))
            
            # Update uncovered
            new_mask = uncovered_mask & ~block_masks[chosen_b_idx]
            # Update list: keep indices where bit is still set
            uncovered_list = [i for i in uncovered_list if (new_mask >> i) & 1]
            uncovered_mask = new_mask
            
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
    # Local search improvement
    if best_blocks is not None:
        # Try to remove redundant blocks
        for run in range(50):
            random.seed(run + 1000)
            current_blocks = [list(b) for b in best_blocks]
            
            # Build coverage count
            triple_count = {}
            for block in current_blocks:
                for triple in itertools.combinations(block, t):
                    triple_count[triple] = triple_count.get(triple, 0) + 1
            
            improved = True
            while improved:
                improved = False
                # Try removing each block
                for i in range(len(current_blocks)):
                    # Check if removing block i leaves all triples covered
                    can_remove = True
                    for triple in itertools.combinations(current_blocks[i], t):
                        if triple_count[triple] <= 1:
                            can_remove = False
                            break
                    if can_remove:
                        # Remove block i
                        for triple in itertools.combinations(current_blocks[i], t):
                            triple_count[triple] -= 1
                        current_blocks.pop(i)
                        improved = True
                        break
            
            if len(current_blocks) < best_count:
                best_count = len(current_blocks)
                best_blocks = current_blocks
    
    return best_blocks
