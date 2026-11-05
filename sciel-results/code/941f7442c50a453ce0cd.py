import itertools
import random
import time
import sys

def construct(v, k, t):
    # For this specific task, v, k, t are fixed by the evaluator to 24, 7, 3
    # but we should handle general inputs or at least be robust.
    # The prompt says "evaluator passes this task's parameters", so v=24, k=7, t=3.
    
    # Precompute all t-subsets
    all_subs = list(itertools.combinations(range(v), t))
    sub_to_idx = {s: i for i, s in enumerate(all_subs)}
    n_subs = len(all_subs)
    
    # Precompute for each block (k-subset) the list of t-subsets it covers
    # This is expensive for general v, k, but for v=24, k=7, t=3 it's manageable.
    # Number of blocks is C(24,7) = 346104.
    # Each block covers C(7,3) = 35 triples.
    # Total edges in incidence: 346104 * 35 ~ 12M. This might be too much memory/time to precompute fully.
    # Better to compute on the fly or use a smarter representation.
    
    # Let's use a lazy evaluation approach for the coverage counts.
    
    def get_block_triples(block):
        return list(itertools.combinations(block, t))
        
    def get_block_idx_triples(block):
        return [sub_to_idx[tr] for tr in get_block_triples(block)]

    # 1. Initial Greedy Construction
    # We want to cover all triples.
    # uncovered: set of indices of uncovered triples
    uncovered = set(range(n_subs))
    blocks = []
    
    # Heuristic for greedy: 
    # Pick an uncovered triple. 
    # Build a block around it.
    # To build the block, pick points that maximize coverage of uncovered triples.
    
    # Optimization: Precompute for each point and each (t-1) subset, which triples are formed.
    # This is complex. Let's stick to a simpler greedy first, then optimize.
    
    # Simpler Greedy:
    # While uncovered:
    #   Pick a random uncovered triple (or the one with min potential blocks)
    #   Start block with this triple
    #   While len(block) < k:
    #       For each candidate point x not in block:
    #           Count how many uncovered triples are covered by adding x to block
    #       Pick x with max count
    #   Add block, update uncovered
    
    # To make it faster, we can randomize the starting triple selection or the tie-breaking.
    # Let's run multiple greedy passes and keep the best.
    
    best_blocks = None
    best_count = float('inf')
    
    start_time = time.time()
    time_limit = 100 # Leave 20s for cleanup and final checks
    
    # Run several iterations of Greedy + Local Search
    for iter_idx in range(50):
        if time.time() - start_time > time_limit:
            break
            
        # Greedy Construction
        uncovered = set(range(n_subs))
        blocks = []
        
        # Shuffle triples to avoid bias in greedy
        subs_list = list(all_subs)
        random.shuffle(subs_list)
        
        # We process uncovered triples. 
        # Instead of picking "first", we can pick a random one from uncovered to diversify.
        # But standard greedy picks the one that is "hardest" or just the first.
        # Let's pick a random uncovered triple to start a block.
        
        while uncovered:
            # Pick a random uncovered triple
            triple_idx = random.choice(list(uncovered))
            triple = all_subs[triple_idx]
            
            block = list(triple)
            
            # Greedily add points
            while len(block) < k:
                best_x = -1
                max_gain = -1
                # Candidates are points not in block
                for x in range(v):
                    if x in block:
                        continue
                    # Calculate gain: number of uncovered triples covered by adding x
                    # Triples covered are formed by (t-1) elements from block + x
                    gain = 0
                    for rest in itertools.combinations(block, t-1):
                        new_triple = tuple(sorted(rest + (x,)))
                        idx = sub_to_idx[new_triple]
                        if idx in uncovered:
                            gain += 1
                    if gain > max_gain:
                        max_gain = gain
                        best_x = x
                        # Early break if we cover all possible? No, just max.
                
                if best_x == -1:
                    # Should not happen if v > k
                    break
                block.append(best_x)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for tr in get_block_triples(block):
                idx = sub_to_idx[tr]
                uncovered.discard(idx)
        
        # Local Search Phase
        # Maintain cover counts
        cover_counts = [0] * n_subs
        for b in blocks:
            for tr in get_block_triples(b):
                cover_counts[sub_to_idx[tr]] += 1
        
        # Verify validity
        if any(c == 0 for c in cover_counts):
            continue # Invalid, shouldn't happen with greedy
            
        # Hill Climbing / SA
        # We try to remove blocks. If a block is removed, we must ensure no triple drops to 0.
        # We also try swaps.
        
        current_blocks = blocks[:]
        current_count = len(current_blocks)
        
        # Simple Local Search:
        # 1. Try to remove any redundant block
        # 2. If none, try to swap a block with a better one to enable future removals
        
        improved = True
        while improved and (time.time() - start_time < time_limit):
            improved = False
            
            # Step 1: Remove redundant blocks
            for i in range(len(current_blocks) - 1, -1, -1):
                b = current_blocks[i]
                # Check if any triple in b has count == 1
                is_redundant = True
                for tr in get_block_triples(b):
                    if cover_counts[sub_to_idx[tr]] == 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    # Remove it
                    for tr in get_block_triples(b):
                        cover_counts[sub_to_idx[tr]] -= 1
                    current_blocks.pop(i)
                    current_count -= 1
                    improved = True
                    break # Restart loop to re-evaluate indices
            
            # Step 2: If no removal, try a swap
            if not improved:
                # Pick a random block to potentially swap
                if current_blocks:
                    i = random.randint(0, len(current_blocks) - 1)
                    b_old = current_blocks[i]
                    
                    # Find critical triples (count == 1) in b_old
                    critical_triples = []
                    for tr in get_block_triples(b_old):
                        if cover_counts[sub_to_idx[tr]] == 1:
                            critical_triples.append(tr)
                    
                    if not critical_triples:
                        # This block is redundant? Why didn't step 1 catch it?
                        # Maybe step 1 iterated differently. Let's just remove it.
                        for tr in get_block_triples(b_old):
                            cover_counts[sub_to_idx[tr]] -= 1
                        current_blocks.pop(i)
                        current_count -= 1
                        improved = True
                        continue
                    
                    # We need a new block b_new that covers all critical_triples
                    # And ideally covers other well-covered triples to maintain robustness
                    
                    # Generate candidate blocks? 
                    # A block covering a specific set of triples is constrained.
                    # If critical_triples is large, finding a single block covering all is hard.
                    # For t=3, a block of size k=7 covers 35 triples.
                    # If we have 2 critical triples, they must be subset of the new block.
                    
                    # Heuristic: 
                    # If len(critical_triples) is small, we can try to find a block containing them.
                    # If len(critical_triples) is large, swap might be impossible without adding blocks.
                    
                    # Let's only attempt swap if len(critical_triples) <= 2 or 3.
                    # Actually, if we pick a random block, it's likely to have several unique triples.
                    # A more effective move is "Add-Remove": Add a block that covers many critical triples, then remove the old one?
                    # That increases count by 1. Not good for minimization directly, but might unlock removals.
                    
                    # Let's try a different local search move: 
                    # "Block Replacement": 
                    # Identify the block with the highest "redundancy" (most triples with count > 1).
                    # Try to replace it with a block that covers its unique triples but has higher overall coverage.
                    
                    # Given time, let's stick to simple removals and restarts.
                    # The greedy is decent. The removals will clean up a lot.
                    pass
        
        if current_count < best_count:
            best_count = current_count
            best_blocks = current_blocks[:]
            
    # If best_blocks is None, return the last generated one (should not happen)
    if best_blocks is None:
        # Fallback: return the blocks from the last iteration
        # We need to ensure we have a valid set
        # Re-run greedy once more if needed
        uncovered = set(range(n_subs))
        blocks = []
        while uncovered:
            triple_idx = random.choice(list(uncovered))
            triple = all_subs[triple_idx]
            block = list(triple)
            while len(block) < k:
                best_x = -1
                max_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t-1):
                        new_triple = tuple(sorted(rest + (x,)))
                        idx = sub_to_idx[new_triple]
                        if idx in uncovered:
                            gain += 1
                    if gain > max_gain:
                        max_gain = gain
                        best_x = x
                block.append(best_x)
            block.sort()
            blocks.append(block)
            for tr in get_block_triples(block):
                uncovered.discard(sub_to_idx[tr])
        return blocks
        
    return best_blocks

# Ensure we don't exceed time
# The loop in construct breaks based on time.
