"""Covering design C(24,7,3): simulated annealing with bitmask coverage and 2-for-2 exchange."""
import itertools
import time
import random
import numpy as np

# Precompute all triples and their bit positions
_ALL_TRIPLES = list(itertools.combinations(range(24), 3))
_NUM_TRIPLES = len(_ALL_TRIPLES)  # 2024
_TRIPLE_INDEX = {ts: i for i, ts in enumerate(_ALL_TRIPLES)}

def _block_triple_mask(block):
    """Return a numpy uint64 bitmask of triples covered by this 7-block."""
    mask = np.zeros(_NUM_TRIPLES // 64 + 1, dtype=np.uint64)
    for ts in itertools.combinations(block, 3):
        idx = _TRIPLE_INDEX[ts]
        mask[idx // 64] |= (1 << (idx % 64))
    return mask

def _mask_popcount(mask):
    """Count bits set in a multi-word bitmask."""
    return sum(bin(w).count('1') for w in mask)

def _masks_cover_all(masks):
    """Check if union of masks covers all triples."""
    union = np.zeros_like(masks[0])
    for m in masks:
        union |= m
    # Check all 2024 bits are set
    for i in range(len(union)):
        expected = (1 << 64) - 1 if i < _NUM_TRIPLES // 64 else ((1 << (_NUM_TRIPLES % 64)) - 1 if _NUM_TRIPLES % 64 > 0 else 0)
        if union[i] != expected:
            return False
    return True

def _fast_greedy_init(v, k, t, rng):
    """Fast greedy to get an initial covering."""
    uncovered = np.ones(_NUM_TRIPLES, dtype=bool)
    blocks = []
    
    # Precompute triple masks for all C(24,7) blocks? Too many. Instead, greedy.
    for _ in range(200):  # max blocks
        if not uncovered.any():
            break
        
        # Pick random uncovered triple as seed
        unc_idx = np.where(uncovered)[0]
        if len(unc_idx) == 0:
            break
        seed_idx = rng.choice(unc_idx)
        seed_triple = _ALL_TRIPLES[seed_idx]
        
        # Build a block containing the seed triple, maximizing new coverage
        block = list(seed_triple)
        block_set = set(block)
        
        # Greedily add points
        for _ in range(k - t):
            best_x = -1
            best_gain = 0
            current_mask = _block_triple_mask(block)
            
            for x in range(v):
                if x in block_set:
                    continue
                cand = block + [x]
                cand_set = set(cand)
                # Count how many uncovered triples are in this candidate block
                gain = 0
                for ts in itertools.combinations(cand, t):
                    idx = _TRIPLE_INDEX[ts]
                    if uncovered[idx]:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x == -1:
                break
            block.append(best_x)
            block_set.add(best_x)
        
        if len(block) < k:
            continue
        
        block.sort()
        blocks.append(block)
        
        # Update uncovered
        bm = _block_triple_mask(block)
        for i in range(_NUM_TRIPLES):
            word = i // 64
            bit = i % 64
            if bm[word] & (1 << bit):
                uncovered[i] = False
    
    return blocks

def _validate_and_get_masks(blocks, v, k, t):
    """Return list of masks if valid covering, else None."""
    if len(blocks) == 0:
        return None
    
    masks = [_block_triple_mask(b) for b in blocks]
    
    union = np.zeros_like(masks[0])
    for m in masks:
        union |= m
    
    # Verify full coverage
    for i in range(len(union)):
        if i < _NUM_TRIPLES // 64:
            expected = (1 << 64) - 1
        else:
            rem = _NUM_TRIPLES % 64
            expected = (1 << rem) - 1 if rem > 0 else 0
        if union[i] != expected:
            return None
    
    return masks

def _anneal_search(blocks, v, k, t, rng, time_limit):
    """
    Simulated annealing with:
    - Move 1: Replace one block with a random one (keep size same)
    - Move 2: 2-for-2 exchange: remove 2 blocks, add 2 new blocks (keep size same)
    - Move 3: Try to remove one block if redundant (decrease size)
    """
    deadline = time.time() + time_limit
    blocks = [list(b) for b in blocks]
    n = len(blocks)
    
    if n == 0:
        return blocks
    
    # Initial state
    masks = _validate_and_get_masks(blocks, v, k, t)
    if masks is None:
        return blocks
    
    current_size = n
    best_blocks = [list(b) for b in blocks]
    best_size = n
    
    temperature = 5.0
    cooling = 0.995
    
    # Precompute all possible 7-subsets? C(24,7) = 346104. Too many to precompute masks.
    # Instead, generate random blocks on the fly.
    
    def random_block():
        return sorted(rng.sample(range(v), k))
    
    def block_covers_triple(block, triple):
        return all(x in block for x in triple)
    
    while time.time() < deadline:
        move_type = rng.random()
        
        if move_type < 0.3 and current_size > 1:
            # Try to remove a redundant block
            idx = rng.randint(0, current_size - 1)
            # Check if block idx is redundant
            redundant = True
            for ts in itertools.combinations(blocks[idx], t):
                # Check if any other block covers this triple
                found = False
                for j in range(current_size):
                    if j == idx:
                        continue
                    if all(x in blocks[j] for x in ts):
                        found = True
                        break
                if not found:
                    redundant = False
                    break
            
            if redundant:
                new_blocks = blocks[:idx] + blocks[idx+1:]
                new_size = current_size - 1
                # Accept (always improves)
                blocks = new_blocks
                current_size = new_size
                masks = _validate_and_get_masks(blocks, v, k, t)
                if masks is not None and current_size < best_size:
                    best_size = current_size
                    best_blocks = [list(b) for b in blocks]
                continue
        
        elif move_type < 0.7:
            # Single block replacement
            idx = rng.randint(0, current_size - 1)
            old_block = blocks[idx]
            new_block = random_block()
            
            if new_block == old_block:
                continue
            
            # Check if replacing old with new maintains coverage
            # A triple is uncovered if it was only covered by old_block and not by new_block
            valid = True
            for ts in itertools.combinations(old_block, t):
                # Is ts covered by new_block?
                if all(x in new_block for x in ts):
                    continue
                # Is ts covered by any other block?
                found = False
                for j in range(current_size):
                    if j == idx:
                        continue
                    if all(x in blocks[j] for x in ts):
                        found = True
                        break
                if not found:
                    valid = False
                    break
            
            if valid:
                blocks[idx] = new_block
                masks = _validate_and_get_masks(blocks, v, k, t)
                if masks is None:
                    # Revert
                    blocks[idx] = old_block
                else:
                    if current_size < best_size:
                        best_size = current_size
                        best_blocks = [list(b) for b in blocks]
            else:
                # Penalty-based acceptance (Metropolis)
                # Count how many triples would be uncovered
                penalty = 0
                for ts in itertools.combinations(old_block, t):
                    if all(x in new_block for x in ts):
                        continue
                    found = False
                    for j in range(current_size):
                        if j == idx:
                            continue
                        if all(x in blocks[j] for x in ts):
                            found = True
                            break
                    if not found:
                        penalty += 1
                
                if penalty > 0:
                    # Accept with probability exp(-penalty / temperature)
                    if rng.random() < np.exp(-penalty / max(temperature, 0.01)):
                        blocks[idx] = new_block
                        # Don't update best (invalid state)
        
        else:
            # 2-for-2 exchange
            if current_size < 2:
                continue
            
            idx1, idx2 = rng.sample(range(current_size), 2)
            old1, old2 = blocks[idx1], blocks[idx2]
            new1, new2 = random_block(), random_block()
            
            if new1 == new2:
                continue
            if new1 == old1 and new2 == old2:
                continue
            if new1 == old2 and new2 == old1:
                continue
            
            # Check coverage
            valid = True
            # Triples in old1 not in new1 or new2 must be covered by other blocks
            for ts in itertools.combinations(old1, t):
                if all(x in new1 for x in ts) or all(x in new2 for x in ts):
                    continue
                found = False
                for j in range(current_size):
                    if j in (idx1, idx2):
                        continue
                    if all(x in blocks[j] for x in ts):
                        found = True
                        break
                if not found:
                    valid = False
                    break
            
            if valid:
                # Also check triples in old2
                for ts in itertools.combinations(old2, t):
                    if all(x in new1 for x in ts) or all(x in new2 for x in ts):
                        continue
                    found = False
                    for j in range(current_size):
                        if j in (idx1, idx2):
                            continue
                        if all(x in blocks[j] for x in ts):
                            found = True
                            break
                    if not found:
                        valid = False
                        break
            
            if valid:
                blocks[idx1] = new1
                blocks[idx2] = new2
                if current_size < best_size:
                    best_size = current_size
                    best_blocks = [list(b) for b in blocks]
            else:
                # Penalty acceptance
                penalty = 0
                for ts in itertools.combinations(old1, t):
                    if all(x in new1 for x in ts) or all(x in new2 for x in ts):
                        continue
                    found = False
                    for j in range(current_size):
                        if j in (idx1, idx2):
                            continue
                        if all(x in blocks[j] for x in ts):
                            found = True
                            break
                    if not found:
                        penalty += 1
                for ts in itertools.combinations(old2, t):
                    if all(x in new1 for x in ts) or all(x in new2 for x in ts):
                        continue
                    found = False
                    for j in range(current_size):
                        if j in (idx1, idx2):
                            continue
                        if all(x in blocks[j] for x in ts):
                            found = True
                            break
                    if not found:
                        penalty += 1
                
                if penalty > 0 and rng.random() < np.exp(-penalty / max(temperature, 0.01)):
                    blocks[idx1] = new1
                    blocks[idx2] = new2
        
        temperature *= cooling
    
    return best_blocks

def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using simulated annealing with bitmask coverage.
    """
    start_time = time.time()
    total_time_limit = 140  # Leave 10s margin before 150s limit
    
    best_blocks = None
    best_len = float('inf')
    
    rng = random.Random(12345)
    
    # Phase 1: Generate multiple greedy initial solutions
    num_init = 6
    init_time_each = (total_time_limit * 0.3) / num_init
    
    for i in range(num_init):
        if time.time() - start_time > total_time_limit * 0.3:
            break
        
        blocks = _fast_greedy_init(v, k, t, rng)
        
        # Validate
        masks = _validate_and_get_masks(blocks, v, k, t)
        if masks is not None and len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = [list(b) for b in blocks]
    
    if best_blocks is None:
        # Fallback
        blocks = _fast_greedy_init(v, k, t, rng)
        return blocks
    
    # Phase 2: Simulated annealing on best solution
    remaining = total_time_limit - (time.time() - start_time)
    if remaining > 5:
        annealed = _anneal_search(best_blocks, v, k, t, rng, remaining * 0.7)
        if len(annealed) < best_len:
            best_len = len(annealed)
            best_blocks = annealed
    
    # Phase 3: Final redundant block removal
    remaining = total_time_limit - (time.time() - start_time)
    if remaining > 2:
        blocks = [list(b) for b in best_blocks]
        improved = True
        while improved and time.time() - start_time < total_time_limit:
            improved = False
            for i in range(len(blocks)):
                if time.time() - start_time > total_time_limit:
                    break
                redundant = True
                for ts in itertools.combinations(blocks[i], t):
                    found = False
                    for j in range(len(blocks)):
                        if j == i:
                            continue
                        if all(x in blocks[j] for x in ts):
                            found = True
                            break
                    if not found:
                        redundant = False
                        break
                if redundant:
                    blocks.pop(i)
                    improved = True
                    break
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
    
    return best_blocks
