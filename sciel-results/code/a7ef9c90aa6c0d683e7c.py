"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import sys

def _greedy_build(v, k, t, uncovered, seed, deadline):
    """Greedy set-cover construction with deadline awareness."""
    rng = random.Random(seed)
    blocks = []
    start_time = time.time()
    
    while uncovered:
        if time.time() > deadline:
            break
        
        # Pick an uncovered t-subset (randomly among a few candidates to add diversity)
        candidates = list(uncovered)
        if len(candidates) > 20:
            target = rng.sample(candidates, 20)
            target_t = rng.choice(target)
        else:
            target_t = rng.choice(candidates)
        
        block = set(target_t)
        
        while len(block) < k:
            if time.time() > deadline:
                return blocks, False
            
            best_x = None
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    new_t = tuple(sorted(rest + (x,)))
                    if new_t in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_gain <= 0:
                # No point helps, just pick randomly
                remaining = [x for x in range(v) if x not in block]
                if remaining:
                    best_x = rng.choice(remaining)
                else:
                    break
            
            block.add(best_x)
        
        if len(block) == k:
            block = sorted(block)
            blocks.append(block)
            # Remove covered t-subsets
            for comb in itertools.combinations(block, t):
                uncovered.discard(tuple(comb))
    
    return blocks, len(uncovered) == 0

def _deletion_pass(v, k, t, blocks, deadline):
    """Try to remove blocks that are redundant."""
    if not blocks:
        return blocks
    
    # Build a map of which t-subsets each block covers
    all_t = set(itertools.combinations(range(v), t))
    
    # For each t-subset, track which blocks cover it
    t_to_blocks = {t_sub: [] for t_sub in all_t}
    for i, block in enumerate(blocks):
        for comb in itertools.combinations(block, t):
            t_to_blocks[comb].append(i)
    
    # Try removing each block
    kept = [True] * len(blocks)
    removed_count = 0
    
    for i in range(len(blocks)):
        if not kept[i]:
            continue
        if time.time() > deadline:
            break
        
        # Check if all t-subsets in this block are covered by other kept blocks
        can_remove = True
        for comb in itertools.combinations(blocks[i], t):
            other_blocks = [j for j in t_to_blocks[comb] if j != i and kept[j]]
            if not other_blocks:
                can_remove = False
                break
        
        if can_remove:
            kept[i] = False
            removed_count += 1
            for comb in itertools.combinations(blocks[i], t):
                # Update references (not strictly needed but good for accuracy)
                pass
    
    return [blocks[i] for i in range(len(blocks)) if kept[i]]

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s safety margin for 120s limit
    
    # Initial greedy construction with a few random seeds
    best_blocks = None
    best_count = float('inf')
    
    # Run greedy construction with different seeds
    num_greedy = 0
    seed = 42
    while time.time() < deadline:
        uncovered = set(itertools.combinations(range(v), t))
        blocks, complete = _greedy_build(v, k, t, uncovered, seed, deadline)
        
        if blocks:
            # Do a deletion pass
            blocks = _deletion_pass(v, k, t, blocks, deadline)
            
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks[:]
        
        num_greedy += 1
        seed += 1
        
        # After first few, reduce frequency of new greedy runs and focus on deletion
        if num_greedy > 5 and time.time() > start_time + 60:
            # Do more deletion passes on best found so far
            if best_blocks and time.time() < deadline:
                improved = _deletion_pass(v, k, t, best_blocks, deadline)
                if len(improved) < best_count:
                    best_count = len(improved)
                    best_blocks = improved
            # Occasionally try another greedy
            if num_greedy % 3 == 0:
                seed += random.randint(1, 100)
            else:
                continue
    
    # Final verification and return
    if best_blocks is None:
        # Fallback: pure greedy without time limit awareness (shouldn't happen)
        uncovered = set(itertools.combinations(range(v), t))
        best_blocks = []
        while uncovered:
            block = list(min(uncovered))
            while len(block) < k:
                def gain(x):
                    return sum(1 for rest in itertools.combinations(block, t - 1)
                               if tuple(sorted(rest + (x,))) in uncovered)
                block.append(max((x for x in range(v) if x not in block), key=gain))
            block.sort()
            best_blocks.append(block)
            uncovered.difference_update(itertools.combinations(block, t))
    
    # Verify coverage
    if time.time() < deadline - 1:
        all_t = set(itertools.combinations(range(v), t))
        covered = set()
        for block in best_blocks:
            for comb in itertools.combinations(block, t):
                covered.add(comb)
        if covered != all_t:
            # If incomplete, do one more greedy pass to fill gaps
            uncovered = all_t - covered
            extra_blocks, _ = _greedy_build(v, k, t, uncovered, 12345, deadline)
            best_blocks.extend(extra_blocks)
    
    return best_blocks
# EVOLVE-BLOCK-END
