"""Structural redesign: maximum-entropy importance sampling + block-composition local search."""
import itertools
import random
import time
import sys

# EVOLVE-BLOCK-START

def _greedy_seed(v, k, t, rng, time_limit):
    """Fast greedy seed: always pick the element that covers the most uncovered t-sets."""
    deadline = time.time() + time_limit
    all_tsets = list(itertools.combinations(range(v), t))
    uncovered = set(all_tsets)
    blocks = []
    
    while uncovered and time.time() < deadline:
        first = min(uncovered)
        block = list(first)
        while len(block) < k and time.time() < deadline:
            candidates = [x for x in range(v) if x not in block]
            if not candidates:
                break
            best_x = None
            best_gain = -1
            for x in candidates:
                gain = 0
                for S in itertools.combinations(block, t - 1):
                    if x not in S:
                        tset = tuple(sorted(S + (x,)))
                        if tset in uncovered:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)
        if len(block) < k:
            break
        block.sort()
        blocks.append(block)
        covered = set(itertools.combinations(block, t))
        uncovered -= covered
    return blocks

def _importance_sample_blocks(v, k, t, rng, n_samples, deadline):
    """
    Draw blocks via importance sampling: weight each point by inverse of its marginal
    contribution to coverage (i.e., points that appear in fewer blocks are more likely).
    This concentrates search on low-redundancy, high-impact blocks.
    """
    all_tsets = list(itertools.combinations(range(v), t))
    uncovered = set(all_tsets)
    blocks = []
    
    # Track marginal contribution: how many times each point has been "used"
    point_usage = [0] * v
    
    for _ in range(n_samples):
        if time.time() >= deadline:
            break
        if not uncovered:
            break
        
        # Compute sampling weights: inverse of (1 + usage), normalized
        weights = [1.0 / (1.0 + point_usage[x]) for x in range(v)]
        total_w = sum(weights)
        if total_w == 0:
            break
        probs = [w / total_w for w in weights]
        
        # Sample k distinct points with replacement-free via weighted sampling
        block = []
        available = list(range(v))
        avail_weights = list(probs)
        
        while len(block) < k and available:
            # Weighted random pick
            r = rng.random() * sum(avail_weights)
            cum = 0.0
            chosen = len(available) - 1
            for i, w in enumerate(avail_weights):
                cum += w
                if r <= cum:
                    chosen = i
                    break
            block.append(available[chosen])
            point_usage[available[chosen]] += 1
            available.pop(chosen)
            avail_weights.pop(chosen)
        
        if len(block) < k:
            break
        block.sort()
        
        # Only keep block if it covers at least one uncovered t-set
        covered_this = set(itertools.combinations(block, t))
        new_coverage = covered_this & uncovered
        if new_coverage:
            blocks.append(block)
            uncovered -= covered_this
    
    return blocks

def _entropy_refine(v, k, t, blocks, rng, time_limit):
    """
    Iteratively refine via importance resampling:
    - Maintain a distribution over candidate blocks weighted by their marginal gain.
    - Resample: discard low-gain blocks, generate new ones via entropy-weighted sampling.
    - Accept if it reduces total block count while maintaining coverage.
    """
    deadline = time.time() + time_limit
    all_tsets = list(itertools.combinations(range(v), t))
    
    # Build coverage map
    tset_count = {ts: 0 for ts in all_tsets}
    for b in blocks:
        for ts in itertools.combinations(b, t):
            tset_count[ts] += 1
    
    def coverage_valid(tset_count):
        return all(c > 0 for c in tset_count.values())
    
    current_blocks = [list(b) for b in blocks]
    best_blocks = list(current_blocks)
    best_len = len(best_blocks)
    
    # Iteration loop
    while time.time() < deadline and best_len > 69:  # 69 is theoretical lower bound
        # Compute marginal gain for each block (how many t-sets it uniquely covers)
        gains = []
        for i, b in enumerate(current_blocks):
            gain = 0
            for ts in itertools.combinations(b, t):
                if tset_count[ts] == 1:
                    gain += 1
            gains.append(gain)
        
        # Identify blocks with zero or very low marginal gain (redundant candidates)
        low_gain_indices = [i for i, g in enumerate(gains) if g <= 1]
        
        if not low_gain_indices:
            break
        
        # Try removing one low-gain block at a time
        improved = False
        for idx in random.sample(low_gain_indices, min(len(low_gain_indices), 5)):
            if time.time() >= deadline:
                break
            
            # Remove block idx temporarily
            removed_block = current_blocks[idx]
            for ts in itertools.combinations(removed_block, t):
                tset_count[ts] -= 1
            
            # Check validity
            if coverage_valid(tset_count):
                # Accept removal
                current_blocks.pop(idx)
                if len(current_blocks) < best_len:
                    best_len = len(current_blocks)
                    best_blocks = [list(b) for b in current_blocks]
                improved = True
            else:
                # Restore
                for ts in itertools.combinations(removed_block, t):
                    tset_count[ts] += 1
        
        if not improved:
            # Try adding a new block via importance sampling to enable future removals
            if len(current_blocks) < best_len + 2:  # Don't grow too much
                new_blocks = _importance_sample_blocks(
                    v, k, t, rng, 3, 
                    time.time() + 0.5
                )
                for nb in new_blocks:
                    if time.time() >= deadline:
                        break
                    # Check if this block covers any t-set with count == 0 (shouldn't happen if valid)
                    # Or covers t-sets that would allow removal of a redundant block
                    added = False
                    for ts in itertools.combinations(nb, t):
                        tset_count[ts] += 1
                    # Now try to remove a previously redundant block
                    for j in range(len(current_blocks)):
                        if time.time() >= deadline:
                            break
                        gain_j = 0
                        for ts in itertools.combinations(current_blocks[j], t):
                            if tset_count[ts] == 1:
                                gain_j += 1
                        if gain_j <= 1:
                            # Try removing block j
                            for ts in itertools.combinations(current_blocks[j], t):
                                tset_count[ts] -= 1
                            if coverage_valid(tset_count):
                                current_blocks.pop(j)
                                current_blocks.append(nb)
                                if len(current_blocks) < best_len:
                                    best_len = len(current_blocks)
                                    best_blocks = [list(b) for b in current_blocks]
                                added = True
                                break
                            else:
                                for ts in itertools.combinations(current_blocks[j], t):
                                    tset_count[ts] += 1
                                break
                    if not added:
                        # Roll back
                        for ts in itertools.combinations(nb, t):
                            tset_count[ts] -= 1
    
    return best_blocks

def _block_composition_search(v, k, t, blocks, rng, time_limit):
    """
    Local search operating on block compositions: for each block, try replacing it
    with a composition that has higher marginal contribution. Uses a neighborhood
    defined by single-point substitutions.
    """
    deadline = time.time() + time_limit
    all_tsets = list(itertools.combinations(range(v), t))
    tset_count = {ts: 0 for ts in all_tsets}
    
    def build_counts(blk_list):
        for ts in all_tsets:
            tset_count[ts] = 0
        for b in blk_list:
            for ts in itertools.combinations(b, t):
                tset_count[ts] += 1
    
    def is_valid():
        return all(c > 0 for c in tset_count.values())
    
    current_blocks = [list(b) for b in blocks]
    build_counts(current_blocks)
    best_blocks = list(current_blocks)
    best_len = len(best_blocks)
    
    while time.time() < deadline:
        n = len(current_blocks)
        if n == 0:
            break
        
        # Pick a random block to modify
        idx = rng.randint(0, n - 1)
        old_block = list(current_blocks[idx])
        
        # Compute current marginal gain of this block
        old_gain = sum(1 for ts in itertools.combinations(old_block, t) if tset_count[ts] == 1)
        
        # Try single-point substitution
        improved = False
        for pos in range(len(old_block)):
            if time.time() >= deadline:
                break
            old_point = old_block[pos]
            candidates = [x for x in range(v) if x not in old_block]
            if not candidates:
                continue
            
            # Try a few random replacements
            for _ in range(3):
                if time.time() >= deadline:
                    break
                new_point = rng.choice(candidates)
                new_block = old_block.copy()
                new_block[pos] = new_point
                new_block.sort()
                
                # Compute new marginal gain
                # Remove old block's contribution
                for ts in itertools.combinations(old_block, t):
                    tset_count[ts] -= 1
                # Add new block's contribution
                for ts in itertools.combinations(new_block, t):
                    tset_count[ts] += 1
                
                new_gain = sum(1 for ts in itertools.combinations(new_block, t) if tset_count[ts] == 1)
                
                if new_gain > old_gain or (new_gain == old_gain and rng.random() < 0.2):
                    # Accept
                    current_blocks[idx] = new_block
                    old_block = new_block
                    old_gain = new_gain
                    improved = True
                else:
                    # Roll back
                    for ts in itertools.combinations(new_block, t):
                        tset_count[ts] -= 1
                    for ts in itertools.combinations(old_block, t):
                        tset_count[ts] += 1
        
        # After modifications, check for redundancies
        for j in range(len(current_blocks)):
            if time.time() >= deadline:
                break
            gain_j = sum(1 for ts in itertools.combinations(current_blocks[j], t) if tset_count[ts] == 1)
            if gain_j == 0:
                # Try removing
                for ts in itertools.combinations(current_blocks[j], t):
                    tset_count[ts] -= 1
                if is_valid():
                    current_blocks.pop(j)
                    if len(current_blocks) < best_len:
                        best_len = len(current_blocks)
                        best_blocks = [list(b) for b in current_blocks]
                    improved = True
                else:
                    for ts in itertools.combinations(current_blocks[j], t):
                        tset_count[ts] += 1
        
        if not improved and len(current_blocks) >= best_len:
            # Stagnation: try a perturbation
            if len(current_blocks) > 0:
                idx = rng.randint(0, len(current_blocks) - 1)
                # Randomly reassign one point
                pos = rng.randint(0, len(current_blocks[idx]) - 1)
                old_point = current_blocks[idx][pos]
                for ts in itertools.combinations(current_blocks[idx], t):
                    tset_count[ts] -= 1
                candidates = [x for x in range(v) if x not in current_blocks[idx]]
                if candidates:
                    new_point = rng.choice(candidates)
                    current_blocks[idx][pos] = new_point
                    current_blocks[idx].sort()
                    for ts in itertools.combinations(current_blocks[idx], t):
                        tset_count[ts] += 1
    
    return best_blocks

def _verify_covering(blocks, v, k, t):
    """Verify every t-subset is covered."""
    covered = set()
    for b in blocks:
        for ts in itertools.combinations(b, t):
            covered.add(ts)
    total = sum(1 for _ in itertools.combinations(range(v), t))
    return len(covered) == total

def construct(v, k, t):
    start_time = time.time()
    total_budget = 135.0  # Safety margin for 150s kill
    
    best_blocks = None
    best_len = float('inf')
    
    # Phase 1: Generate diverse seeds
    seed_deadline = start_time + total_budget * 0.25
    num_seeds = 4
    per_seed_time = (seed_deadline - start_time) / num_seeds
    
    for i in range(num_seeds):
        if time.time() >= seed_deadline:
            break
        rng = random.Random(100 + i * 7919)
        blocks = _greedy_seed(v, k, t, rng, per_seed_time)
        if blocks and _verify_covering(blocks, v, k, t):
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = blocks
    
    # Phase 2: Importance sampling for additional candidates
    is_deadline = start_time + total_budget * 0.45
    if time.time() < is_deadline and best_blocks is not None:
        rng = random.Random(4242)
        is_blocks = _importance_sample_blocks(v, k, t, rng, 200, is_deadline)
        if is_blocks:
            # Combine with best seed and prune
            combined = best_blocks + is_blocks
            # Deduplicate
            seen = set()
            unique = []
            for b in combined:
                key = tuple(sorted(b))
                if key not in seen:
                    seen.add(key)
                    unique.append(list(b))
            if _verify_covering(unique, v, k, t) and len(unique) < best_len:
                best_len = len(unique)
                best_blocks = unique
    
    # Phase 3: Entropy refinement
    ent_deadline = start_time + total_budget * 0.70
    if time.time() < ent_deadline and best_blocks is not None:
        rng = random.Random(999)
        refined = _entropy_refine(v, k, t, best_blocks, rng, ent_deadline - time.time())
        if refined and _verify_covering(refined, v, k, t):
            if len(refined) < best_len:
                best_len = len(refined)
                best_blocks = refined
    
    # Phase 4: Block composition local search
    comp_deadline = start_time + total_budget * 0.95
    if time.time() < comp_deadline and best_blocks is not None:
        rng = random.Random(7777)
        refined = _block_composition_search(v, k, t, best_blocks, rng, comp_deadline - time.time())
        if refined and _verify_covering(refined, v, k, t):
            if len(refined) < best_len:
                best_len = len(refined)
                best_blocks = refined
    
    # Final safety
    if time.time() - start_time > total_budget - 5.0:
        if best_blocks is None:
            rng = random.Random(0)
            best_blocks = _greedy_seed(v, k, t, rng, 5.0)
        return best_blocks
    
    return best_blocks

# EVOLVE-BLOCK-END
