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
    
    # Helper to get triples in a block
    def block_triples(block):
        return set(itertools.combinations(block, t))
    
    # Phase 1: Randomized Greedy to get initial covering
    # We will run this a few times to get a good start
    def greedy_construct(seed):
        random.seed(seed)
        uncovered = set(all_tsubs)
        blocks = []
        
        while uncovered:
            # Pick a random uncovered triple to start
            start_triple = random.choice(list(uncovered))
            block = list(start_triple)
            random.shuffle(block)
            
            while len(block) < k:
                # Find point that covers most uncovered triples
                best_x = -1
                best_gain = -1
                candidates = [x for x in range(v) if x not in block]
                random.shuffle(candidates)
                for x in candidates:
                    # Count how many uncovered triples would be covered
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
            
        return blocks

    # Run greedy a few times to find a good initial solution
    initial_blocks = None
    best_initial_len = float('inf')
    
    for seed in [42, 123, 456, 789]:
        if time.time() - start_time > time_limit * 0.2:
            break
        candidate_blocks = greedy_construct(seed)
        if len(candidate_blocks) < best_initial_len:
            best_initial_len = len(candidate_blocks)
            initial_blocks = candidate_blocks
            
    blocks = initial_blocks
    if not blocks:
        # Fallback if greedy failed or timed out
        blocks = greedy_construct(42)

    # Phase 2: Simulated Annealing Local Search
    # Maintain coverage counts
    coverage = {}
    for ts in all_tsubs:
        coverage[ts] = 0
    for block in blocks:
        for ts in block_triples(block):
            coverage[ts] += 1
    
    def is_valid():
        # Check if any triple has count 0
        # Optimization: keep track of uncovered_count
        # For now, just check min
        return all(coverage[ts] >= 1 for ts in all_tsubs)

    def try_delete_block(idx):
        """Try to delete block at index idx. Returns True if successful."""
        block = blocks[idx]
        # Check if all triples in this block are covered by other blocks
        for ts in block_triples(block):
            if coverage[ts] <= 1:
                return False
        # Delete it
        for ts in block_triples(block):
            coverage[ts] -= 1
        blocks.pop(idx)
        return True

    def try_swap_point(idx, old_x, new_x):
        """Try to swap old_x with new_x in block idx. Returns True if successful."""
        if new_x in blocks[idx]:
            return False
        block = blocks[idx]
        other_points = [p for p in block if p != old_x]
        
        old_block_triples = block_triples(block)
        new_block = sorted([new_x] + other_points)
        new_block_triples = block_triples(new_block)
        
        # Check all triples that would lose coverage
        for ts in old_block_triples:
            if coverage[ts] <= 1 and ts not in new_block_triples:
                return False
        
        # Apply swap
        for ts in old_block_triples:
            coverage[ts] -= 1
        for ts in new_block_triples:
            coverage[ts] += 1
        blocks[idx] = new_block
        return True

    # Simulated Annealing Loop
    best_blocks = [list(b) for b in blocks]
    best_count = len(blocks)
    
    # Temperature parameters
    T = 1.0
    cooling_rate = 0.995
    min_T = 0.01
    
    while time.time() - start_time < time_limit:
        if T < min_T:
            T = 1.0 # Restart if cooled down too much, or break
            # Actually, let's just keep going with a lower T or restart
            break
            
        # 1. Try to delete blocks (always good if possible)
        deleted_any = False
        indices = list(range(len(blocks)))
        random.shuffle(indices)
        for idx in indices:
            if try_delete_block(idx):
                deleted_any = True
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_blocks = [list(b) for b in blocks]
        
        # 2. If no deletion, try swaps
        if not deleted_any:
            # Try a few random swaps
            for _ in range(20):
                if time.time() - start_time > time_limit:
                    break
                if len(blocks) == 0:
                    break
                idx = random.randint(0, len(blocks)-1)
                old_x = random.choice(blocks[idx])
                new_x = random.randint(0, v-1)
                if new_x == old_x or new_x in blocks[idx]:
                    continue
                
                # Backup state
                old_blocks = [list(b) for b in blocks]
                old_coverage = dict(coverage)
                
                if try_swap_point(idx, old_x, new_x):
                    # Check if this swap allows any deletion
                    improved = False
                    indices2 = list(range(len(blocks)))
                    random.shuffle(indices2)
                    for idx2 in indices2:
                        if try_delete_block(idx2):
                            improved = True
                            break
                    
                    if improved:
                        if len(blocks) < best_count:
                            best_count = len(blocks)
                            best_blocks = [list(b) for b in blocks]
                    else:
                        # Accept with probability exp(-delta / T)
                        # Here "delta" is 0 in terms of block count, but we accept to explore
                        # We can just accept it to change the landscape
                        if random.random() < math.exp(-1.0 / T):
                            # Keep the change
                            pass
                        else:
                            # Rollback
                            blocks = old_blocks
                            coverage = old_coverage
                else:
                    # Swap was invalid, do nothing
                    pass
                    
        T *= cooling_rate

    return best_blocks
