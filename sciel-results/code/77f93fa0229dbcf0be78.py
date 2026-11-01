import itertools
import random
import time
import math

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115  # Leave 5s buffer
    
    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    all_tsubs_set = set(all_tsubs)
    
    def block_triples(block):
        return set(itertools.combinations(block, t))

    # Phase 1: Fast Greedy Initialization
    random.seed(42)
    uncovered = set(all_tsubs)
    blocks = []
    
    # Precompute candidate blocks sorted by "potential" is hard, so we just pick greedily.
    # To speed up greedy, we can pick the block that covers the most uncovered triples.
    # C(v,k) is large, so we sample candidates or use a heuristic.
    # For v=24, k=7, C(24,7) ~ 346k. Iterating all is too slow per step.
    # We will use a randomized greedy: pick random uncovered triple, extend to block.
    
    while uncovered:
        # Pick a random uncovered triple
        start_triple = random.choice(list(uncovered))
        block = list(start_triple)
        random.shuffle(block)
        
        while len(block) < k:
            best_x = -1
            best_gain = -1
            candidates = [x for x in range(v) if x not in block]
            random.shuffle(candidates)
            # Limit candidates for speed? No, v=24 is small.
            for x in candidates:
                gain = 0
                for rest in itertools.combinations(block, t-1):
                    ts = tuple(sorted(rest + (x,)))
                    if ts in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            block.append(best_x)
        
        block.sort()
        blocks.append(block)
        uncovered.difference_update(block_triples(block))
        
        if time.time() - start_time > time_limit * 0.2:
            break

    # Finish greedy if needed
    if uncovered:
        while uncovered:
            start_triple = min(uncovered)
            block = list(start_triple)
            while len(block) < k:
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t-1):
                        ts = tuple(sorted(rest + (x,)))
                        if ts in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                block.append(best_x)
            block.sort()
            blocks.append(block)
            uncovered.difference_update(block_triples(block))

    # Phase 2: Simulated Annealing
    # State: blocks, coverage counts
    coverage = {}
    for ts in all_tsubs:
        coverage[ts] = 0
    for block in blocks:
        for ts in block_triples(block):
            coverage[ts] += 1
            
    def get_uncovered_count():
        return sum(1 for ts in all_tsubs if coverage[ts] == 0)
    
    current_uncovered = get_uncovered_count()
    current_cost = len(blocks) * 1000 + current_uncovered
    
    best_blocks = list(blocks)
    best_cost = current_cost
    
    T = 10.0
    cooling_rate = 0.995
    min_T = 0.1
    
    while T > min_T and time.time() - start_time < time_limit:
        # Generate a move
        move_type = random.random()
        
        if move_type < 0.3:
            # Try to delete a block
            if not blocks:
                T *= cooling_rate
                continue
            idx = random.randint(0, len(blocks) - 1)
            block = blocks[idx]
            # Check if deletion is valid (no triple becomes uncovered)
            can_delete = True
            for ts in block_triples(block):
                if coverage[ts] <= 1:
                    can_delete = False
                    break
            
            if can_delete:
                # Apply deletion
                for ts in block_triples(block):
                    coverage[ts] -= 1
                blocks.pop(idx)
                new_uncovered = get_uncovered_count()
                new_cost = len(blocks) * 1000 + new_uncovered
                
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_blocks = list(blocks)
                
                current_uncovered = new_uncovered
                current_cost = new_cost
            else:
                # Deletion not possible, skip or try other move? 
                # In SA, we could try to "break" it, but let's stick to valid moves for deletion.
                pass

        elif move_type < 0.5:
            # Try to add a block
            # Generate a random block or a greedy block
            # Random block
            new_block = sorted(random.sample(range(v), k))
            
            # Calculate cost change
            # Adding a block increases block count by 1 (cost +1000)
            # It may decrease uncovered count
            gain = 0
            for ts in block_triples(new_block):
                if coverage[ts] == 0:
                    gain += 1
            
            delta_cost = 1000 - gain
            
            # Acceptance
            if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
                # Apply addition
                blocks.append(new_block)
                for ts in block_triples(new_block):
                    coverage[ts] += 1
                
                new_uncovered = get_uncovered_count()
                new_cost = len(blocks) * 1000 + new_uncovered
                
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_blocks = list(blocks)
                
                current_uncovered = new_uncovered
                current_cost = new_cost

        else:
            # Try to mutate a block (swap a point)
            if not blocks:
                T *= cooling_rate
                continue
            idx = random.randint(0, len(blocks) - 1)
            old_block = blocks[idx]
            old_point = random.choice(old_block)
            new_point = random.choice([x for x in range(v) if x not in old_block])
            
            new_block = sorted([new_point] + [p for p in old_block if p != old_point])
            
            # Calculate cost change
            # Block count stays same
            # Uncovered count may change
            old_triples = block_triples(old_block)
            new_triples = block_triples(new_block)
            
            lost_uncovered = 0
            gained_uncovered = 0
            # Triples that were covered only by this block and are lost
            for ts in old_triples:
                if ts not in new_triples:
                    if coverage[ts] == 1:
                        lost_uncovered += 1
            # Triples that are newly covered
            for ts in new_triples:
                if ts not in old_triples:
                    if coverage[ts] == 0:
                        gained_uncovered += 1
            
            delta_cost = lost_uncovered - gained_uncovered
            
            if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
                # Apply mutation
                for ts in old_triples:
                    if ts not in new_triples:
                        coverage[ts] -= 1
                for ts in new_triples:
                    if ts not in old_triples:
                        coverage[ts] += 1
                blocks[idx] = new_block
                
                new_uncovered = get_uncovered_count()
                new_cost = len(blocks) * 1000 + new_uncovered
                
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_blocks = list(blocks)
                
                current_uncovered = new_uncovered
                current_cost = new_cost

        T *= cooling_rate

    # Ensure best_blocks is valid
    # If best_blocks is not valid, we need to repair it
    # Check validity
    final_coverage = {}
    for ts in all_tsubs:
        final_coverage[ts] = 0
    for block in best_blocks:
        for ts in block_triples(block):
            final_coverage[ts] += 1
            
    if any(final_coverage[ts] == 0 for ts in all_tsubs):
        # Repair by adding greedy blocks for uncovered triples
        uncovered = {ts for ts in all_tsubs if final_coverage[ts] == 0}
        while uncovered:
            start_triple = min(uncovered)
            block = list(start_triple)
            while len(block) < k:
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t-1):
                        ts = tuple(sorted(rest + (x,)))
                        if ts in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                block.append(best_x)
            block.sort()
            best_blocks.append(block)
            uncovered.difference_update(block_triples(block))

    return best_blocks
