import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    # Allow up to 110 seconds to be safe (limit is 150s, keep margin)
    timeout = 110.0
    
    points = list(range(v))
    all_triples = list(itertools.combinations(points, t))
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    # Precompute all blocks and their covered triples
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # block_triples[i] = list of triple indices covered by block i
    block_triples = []
    # triple_to_blocks[j] = list of block indices covering triple j
    triple_to_blocks = [[] for _ in range(num_triples)]
    
    for i, block in enumerate(all_blocks):
        covered = [triple_to_idx[tup] for tup in itertools.combinations(block, t)]
        block_triples.append(covered)
        for tr_idx in covered:
            triple_to_blocks[tr_idx].append(i)
    
    # Precompute triple degree (number of blocks covering each triple)
    triple_degree = [len(triple_to_blocks[j]) for j in range(num_triples)]
    
    # Bitmask representation: each block's covered triples as a Python int bitmask
    # num_triples can be up to C(24,3)=2024, so we need big ints
    block_bitmasks = []
    for tr_list in block_triples:
        mask = 0
        for tr_idx in tr_list:
            mask |= (1 << tr_idx)
        block_bitmasks.append(mask)
    
    best_block_indices = None
    best_count = float('inf')
    
    def greedy_cover_weighted(random_seed=None):
        """Weighted greedy: prioritize covering rare uncovered triples. Returns list of block indices."""
        if random_seed is not None:
            random.seed(random_seed)
            
        # uncovered as bitmask
        uncovered_mask = (1 << num_triples) - 1
        block_indices = []
        
        while uncovered_mask:
            # Find uncovered triples and pick one randomly
            uncovered_indices = []
            temp = uncovered_mask
            idx = 0
            while temp:
                if temp & 1:
                    uncovered_indices.append(idx)
                temp >>= 1
                idx += 1
            
            if not uncovered_indices:
                break
                
            # Weighted random selection: prefer triples with lower degree (rarer)
            weights = [1.0 / triple_degree[tr] for tr in uncovered_indices]
            total_w = sum(weights)
            r = random.random() * total_w
            cum = 0.0
            target_triple_idx = uncovered_indices[0]
            for tr, w in zip(uncovered_indices, weights):
                cum += w
                if r <= cum:
                    target_triple_idx = tr
                    break
            
            # Get candidate blocks covering this triple
            candidates = triple_to_blocks[target_triple_idx]
            
            best_gain = -1
            best_block_idx = -1
            
            # Evaluate gain using bitmask AND
            for bi in candidates:
                gain = (block_bitmasks[bi] & uncovered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = bi
                    if best_gain == 35:  # C(7,3)
                        break
            
            if best_block_idx != -1:
                block_indices.append(best_block_idx)
                # Update uncovered
                uncovered_mask &= ~block_bitmasks[best_block_idx]
            else:
                break
                
        return block_indices

    def peel(block_indices):
        """Remove redundant blocks. Returns list of block indices."""
        if not block_indices:
            return block_indices
            
        # Count coverage of each triple
        counts = [0] * num_triples
        for bi in block_indices:
            for tr_idx in block_triples[bi]:
                counts[tr_idx] += 1
        
        new_indices = []
        for bi in block_indices:
            removable = True
            for tr_idx in block_triples[bi]:
                if counts[tr_idx] == 1:
                    removable = False
                    break
            if not removable:
                new_indices.append(bi)
            else:
                for tr_idx in block_triples[bi]:
                    counts[tr_idx] -= 1
                    
        return new_indices

    def local_search_remove(block_indices):
        """Try to remove redundant blocks iteratively. Returns list of block indices."""
        if not block_indices:
            return block_indices
        
        counts = [0] * num_triples
        for bi in block_indices:
            for tr_idx in block_triples[bi]:
                counts[tr_idx] += 1
        
        improved = True
        while improved:
            improved = False
            for i, bi in enumerate(block_indices):
                # Check if block bi is removable
                removable = True
                for tr_idx in block_triples[bi]:
                    if counts[tr_idx] == 1:
                        removable = False
                        break
                if removable:
                    for tr_idx in block_triples[bi]:
                        counts[tr_idx] -= 1
                    block_indices.pop(i)
                    improved = True
                    break
        
        return block_indices

    # Main Loop
    attempts = 0
    while time.time() - start_time < timeout:
        seed = attempts
        block_indices = greedy_cover_weighted(seed)
        block_indices = peel(block_indices)
        block_indices = local_search_remove(block_indices)
        
        if len(block_indices) < best_count:
            best_count = len(block_indices)
            best_block_indices = list(block_indices)
            
        attempts += 1
        
        # If we are getting close to timeout, break early to ensure return
        if time.time() - start_time > timeout - 8.0:
            break

    if best_block_indices is None:
        block_indices = greedy_cover_weighted(0)
        best_block_indices = peel(block_indices)
        
    return [list(all_blocks[bi]) for bi in best_block_indices]
