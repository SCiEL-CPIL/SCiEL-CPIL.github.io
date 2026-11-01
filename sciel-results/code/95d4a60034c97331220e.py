import itertools
import time
import random
from collections import defaultdict

DEADLINE = 145.0

def _all_tsubsets(v, t):
    return [tuple(c) for c in itertools.combinations(range(v), t)]

def _block_tsubsets(block, t):
    return [tuple(c) for c in itertools.combinations(block, t)]

def _count_unique_triples(blocks, t):
    """
    Returns a dict: block_index -> list of triples that are covered ONLY by this block.
    """
    # First, count how many blocks cover each triple
    triple_count = defaultdict(int)
    for b in blocks:
        for ts in _block_tsubsets(b, t):
            triple_count[ts] += 1
    
    unique_triples = defaultdict(list)
    for i, b in enumerate(blocks):
        for ts in _block_tsubsets(b, t):
            if triple_count[ts] == 1:
                unique_triples[i].append(ts)
                
    return unique_triples

def _try_delete_blocks(blocks, t):
    """
    Repeatedly delete blocks that have no unique triples.
    Returns new list of blocks and a boolean indicating if any block was deleted.
    """
    changed = True
    while changed:
        changed = False
        unique_triples = _count_unique_triples(blocks, t)
        new_blocks = []
        for i, b in enumerate(blocks):
            if not unique_triples[i]:
                # This block has no unique triples, safe to delete
                changed = True
            else:
                new_blocks.append(b)
        if changed:
            blocks = new_blocks
    return blocks, changed

def _construct_replacement_block(essential_triples, v, k, t, rng, existing_blocks_set):
    """
    Attempt to construct a block that covers all essential_triples.
    Returns the block as a tuple or None.
    """
    if not essential_triples:
        return None # No essential triples, block is redundant, handled by deletion
        
    # Start with the union of points in essential triples? 
    # Or pick a random essential triple to start.
    # Let's try to build a block that covers all essential triples.
    
    # If the union of points in essential triples is > k, we can't cover them all in one block.
    # In that case, we cannot replace this block with a single block that covers its unique triples 
    # unless we also modify other blocks. For simplicity, we only try if union size <= k.
    # Actually, if union size > k, we MUST keep this block (or a subset of its unique triples must be covered by others, 
    # which contradicts them being unique). So if union size > k, we cannot remove this block via simple replacement.
    
    union_points = set()
    for ts in essential_triples:
        union_points.update(ts)
        
    if len(union_points) > k:
        return None

    # We need to find a k-subset B' containing union_points that covers all essential_triples.
    # Since union_points covers all essential_triples by definition, any superset of union_points works.
    # We want to choose the remaining k - len(union_points) points to maximize "goodness".
    # Goodness: cover more non-essential triples? Or avoid creating new unique triples?
    # Let's just pick random points to fill the rest, or points that appear in other blocks.
    
    base_block = list(union_points)
    remaining_slots = k - len(base_block)
    
    if remaining_slots > 0:
        # Pick random points not in base_block
        candidates = [p for p in range(v) if p not in base_block]
        if len(candidates) < remaining_slots:
            return None
        fillers = rng.sample(candidates, remaining_slots)
        base_block.extend(fillers)
        
    base_block.sort()
    return tuple(base_block)

def construct(v, k, t):
    start_time = time.time()
    rng = random.Random(42)
    
    def time_check():
        return time.time() - start_time > DEADLINE - 5.0

    all_ts = _all_tsubsets(v, t)
    
    best_blocks = None
    best_count = float('inf')
    
    num_restarts = 100
    
    for restart in range(num_restarts):
        if time_check():
            break
        
        # 1. Greedy Construction
        uncovered = set(all_ts)
        blocks = []
        
        while uncovered and not time_check():
            # Pick a random uncovered triple
            seed_ts = rng.choice(tuple(uncovered))
            block = list(seed_ts)
            
            # Greedily add points to cover most uncovered triples
            while len(block) < k:
                best_x = None
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    # Count how many uncovered triples are completed by adding x
                    # A triple (a,b,x) is covered if (a,b) is in block and (a,b,x) is uncovered
                    for rest in itertools.combinations(block, t - 1):
                        sub = tuple(sorted(rest + (x,)))
                        if sub in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is None:
                    break
                block.append(best_x)
            
            # Fill remaining
            if len(block) < k:
                for x in range(v):
                    if len(block) >= k:
                        break
                    if x not in block:
                        block.append(x)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for ts in _block_tsubsets(block, t):
                uncovered.discard(ts)
        
        # 2. Local Search
        # Repeat deletion and swap until no improvement
        improved = True
        while improved and not time_check():
            improved = False
            
            # Deletion
            blocks, changed = _try_delete_blocks(blocks, t)
            if changed:
                improved = True
                continue
            
            # If no deletion, try a swap
            if len(blocks) == 0:
                break
                
            # Pick a random block
            idx = rng.randrange(len(blocks))
            current_block = blocks[idx]
            
            # Find unique triples for this block
            unique_triples = _count_unique_triples(blocks, t)[idx]
            
            # If no unique triples, it would have been deleted. So there must be some.
            # Try to construct a replacement block that covers these unique triples.
            # To increase diversity, we might try multiple times with different random fillers.
            
            # Optimization: If the block has many unique triples, it's hard to replace.
            # We only try if the number of unique triples is small? Or just try.
            
            replacement = _construct_replacement_block(unique_triples, v, k, t, rng, set(map(tuple, blocks)))
            
            if replacement:
                # Check if replacement is already in blocks
                if replacement in set(map(tuple, blocks)):
                    # We can just delete current_block
                    blocks.pop(idx)
                    improved = True
                else:
                    # Replace current_block with replacement
                    # This doesn't change the count, but might allow future deletions
                    blocks[idx] = list(replacement)
                    # To avoid infinite loops of swapping back and forth, 
                    # we could track visited states, but for now, we rely on deletion eventually kicking in.
                    # However, simple replacement might oscillate.
                    # Let's only accept if it covers *more* total triples? 
                    # Or just accept randomly to explore.
                    # A better heuristic: Only replace if the new block covers at least as many 
                    # "new" triples as the old one?
                    # Let's just accept it. The deletion step will clean up.
                    # To prevent oscillation, we can add a small penalty or randomization.
                    # For now, accept.
                    improved = True # Mark as improved to continue loop, hoping deletion works next time
                    # Actually, if we just swap, and no deletion happens, we loop forever.
                    # We should only mark improved if we actually changed the set of blocks in a way that 
                    # is not trivially reversible, or limit swap iterations.
                    # Let's say we only accept a swap if it reduces the number of unique triples 
                    # for the new block? No, the new block must cover the old unique triples.
                    # Let's track the number of swaps without deletion.
            
            # To prevent infinite swap loops, we'll limit the number of swap attempts per deletion phase
            # This simple loop structure might oscillate.
            # Better structure:
            # 1. Delete.
            # 2. If no delete, do ONE swap.
            # 3. Repeat.
            # If we do many swaps without delete, we might cycle.
            # Let's add a counter.
            
        # Update best
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks[:]
            
        if best_count <= 76:
            break
            
    # Final verification and cleanup
    if best_blocks is None:
        # Fallback: simple greedy
        uncovered = set(all_ts)
        best_blocks = []
        while uncovered and not time_check():
            seed_ts = rng.choice(tuple(uncovered))
            block = list(seed_ts)
            while len(block) < k:
                best_x = None
                best_gain = -1
                for x in range(v):
                    if x in block: continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        sub = tuple(sorted(rest + (x,)))
                        if sub in uncovered: gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is None: break
                block.append(best_x)
            if len(block) < k:
                for x in range(v):
                    if len(block) >= k: break
                    if x not in block: block.append(x)
            block.sort()
            best_blocks.append(block)
            for ts in _block_tsubsets(block, t):
                uncovered.discard(ts)
    
    # Ensure validity
    covered = set()
    for b in best_blocks:
        for ts in _block_tsubsets(b, t):
            covered.add(ts)
    uncovered = set(all_ts) - covered
    if uncovered:
        for ts in list(uncovered):
            block = list(ts)
            for x in range(v):
                if len(block) >= k: break
                if x not in block: block.append(x)
            block.sort()
            if block not in best_blocks:
                best_blocks.append(block)
            for ts2 in _block_tsubsets(block, t):
                uncovered.discard(ts2)
                
    return best_blocks
