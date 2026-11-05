import itertools
import random
import time
import math

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Precomputation
    points = list(range(v))
    triples = list(itertools.combinations(points, t))
    triple_to_id = {tr: i for i, tr in enumerate(triples)}
    num_triples = len(triples)
    
    # Map block to list of triple IDs
    # We don't precompute all blocks, we generate candidates on the fly.
    
    def get_block_triple_ids(block):
        # block is a tuple of k points
        # return list of IDs of t-subsets of block
        return [triple_to_id[tup] for tup in itertools.combinations(block, t)]

    # 1. Randomized Greedy
    def greedy_construction():
        uncovered = set(range(num_triples))
        blocks = []
        while uncovered:
            # Pick random uncovered triple
            tid = random.choice(list(uncovered))
            target_triple = triples[tid]
            
            # Find all k-blocks containing target_triple
            # target_triple has t points.
            # Need k-t more points from v-t remaining.
            remaining_points = [p for p in points if p not in target_triple]
            candidates = []
            # Generate combinations of k-t from remaining_points
            # And union with target_triple
            for combo in itertools.combinations(remaining_points, k - t):
                block = tuple(sorted(target_triple + combo))
                # Calculate gain
                # We can optimize by checking intersection with uncovered
                # But 5985 candidates * 35 triples is small.
                block_triples = get_block_triple_ids(block)
                gain = len(set(block_triples).intersection(uncovered))
                if gain > 0:
                    candidates.append((gain, block))
            
            if not candidates:
                # Should not happen if uncovered is not empty
                break
            
            # Pick best, tie-break random
            max_gain = max(g for g, b in candidates)
            best_blocks = [b for g, b in candidates if g == max_gain]
            chosen_block = random.choice(best_blocks)
            
            blocks.append(chosen_block)
            # Update uncovered
            covered_ids = set(get_block_triple_ids(chosen_block))
            uncovered.difference_update(covered_ids)
            
        return blocks

    # 2. Local Search
    def local_search(blocks, deadline):
        # Initialize cover counts
        cover_count = [0] * num_triples
        for b in blocks:
            ids = get_block_triple_ids(b)
            for tid in ids:
                cover_count[tid] += 1
        
        best_blocks = blocks
        best_len = len(blocks)
        
        T = 1.0 # Temperature
        
        while time.time() < deadline:
            # Reduction Phase
            changed = False
            # Iterate until no change
            while True:
                removal_idx = -1
                for i, b in enumerate(blocks):
                    ids = get_block_triple_ids(b)
                    redundant = True
                    for tid in ids:
                        if cover_count[tid] == 1:
                            redundant = False
                            break
                    if redundant:
                        removal_idx = i
                        break
                
                if removal_idx != -1:
                    # Remove block
                    b = blocks.pop(removal_idx)
                    ids = get_block_triple_ids(b)
                    for tid in ids:
                        cover_count[tid] -= 1
                    changed = True
                else:
                    break
            
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = blocks.copy()
            
            # Perturbation Phase
            if not blocks:
                break
                
            i = random.randint(0, len(blocks) - 1)
            old_block = blocks[i]
            old_ids = get_block_triple_ids(old_block)
            
            # Remove
            blocks.pop(i)
            for tid in old_ids:
                cover_count[tid] -= 1
            
            # Find uncovered
            uncovered = [tid for tid in range(num_triples) if cover_count[tid] == 0]
            
            if not uncovered:
                # This should be handled by reduction, but if it happens, we saved a block
                # Actually, if reduction was complete, this implies old_block was redundant.
                # We accept this state.
                continue
            
            # Repair: Greedily add blocks to cover uncovered
            added_blocks = []
            current_uncovered = set(uncovered)
            
            while current_uncovered:
                tid = random.choice(list(current_uncovered))
                target_triple = triples[tid]
                remaining_points = [p for p in points if p not in target_triple]
                
                candidates = []
                for combo in itertools.combinations(remaining_points, k - t):
                    block = tuple(sorted(target_triple + combo))
                    block_triples = get_block_triple_ids(block)
                    gain = len(set(block_triples).intersection(current_uncovered))
                    if gain > 0:
                        candidates.append((gain, block))
                
                if not candidates:
                    break
                
                max_gain = max(g for g, b in candidates)
                best_blocks_c = [b for g, b in candidates if g == max_gain]
                chosen_block = random.choice(best_blocks_c)
                
                added_blocks.append(chosen_block)
                covered_ids = set(get_block_triple_ids(chosen_block))
                current_uncovered.difference_update(covered_ids)
                
                # Early stop if we already added too many
                if len(added_blocks) >= 2: 
                    # If we need 2 or more, it's likely a bad move unless SA accepts
                    # We can stop generating and just evaluate
                    # But we need to finish covering to know the exact delta?
                    # No, if we add 2, delta is +1. If we add 1, delta is 0.
                    # If we add 0, delta is -1.
                    # If we break early with >1 added, we know it's at least +1.
                    # Let's just finish the greedy repair to be precise, it's fast.
                    pass
            
            new_len = len(blocks) + len(added_blocks)
            old_len = len(blocks) + 1
            
            delta = new_len - old_len
            
            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                accept = True # Swap
            else:
                # delta > 0
                if random.random() < math.exp(-delta / T):
                    accept = True
            
            if accept:
                blocks.extend(added_blocks)
                for b in added_blocks:
                    ids = get_block_triple_ids(b)
                    for tid in ids:
                        cover_count[tid] += 1
            else:
                # Restore
                blocks.insert(i, old_block)
                for tid in old_ids:
                    cover_count[tid] += 1
            
            # Cool down
            T *= 0.99
            
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = blocks.copy()
                
        return best_blocks

    # Main Loop: Multiple Restarts
    best_result = None
    best_size = float('inf')
    
    while time.time() < deadline:
        # Greedy
        blocks = greedy_construction()
        
        # Local Search
        # Pass deadline to local search to let it run until time is up
        # But we want to try multiple restarts.
        # Let's give each restart a time slice.
        # Or just run local search until deadline if it's the last one.
        
        # Strategy: Run greedy + short local search.
        # If time allows, run another.
        
        # Let's allocate 5s for local search per restart?
        # Or just run local search until deadline for the first good one?
        # Better: Run greedy, then local search with a time limit (e.g., 10s).
        # If we find < 76, keep improving.
        
        ls_deadline = min(deadline, time.time() + 10.0)
        improved = local_search(blocks, ls_deadline)
        
        if len(improved) < best_size:
            best_size = len(improved)
            best_result = improved
            
        # If we found 69 (lower bound) or 70, we can stop?
        # Lower bound is 69. Best known is 76.
        # If we find 70, it's a huge improvement.
        if best_size <= 70:
            break
            
    # Return as list of lists
    if best_result:
        return [list(b) for b in best_result]
    else:
        # Fallback: simple greedy
        blocks = greedy_construction()
        return [list(b) for b in blocks]
