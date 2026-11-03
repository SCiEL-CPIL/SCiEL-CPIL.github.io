"""Optimized greedy + local search for covering design C(24,7,3)."""
import itertools
import random
import time
import numpy as np


def _precompute(v, k, t):
    """Precompute all t-subsets and block coverage maps."""
    all_triples = list(itertools.combinations(range(v), t))
    n_triples = len(all_triples)
    triple_to_idx = {frozenset(s): i for i, s in enumerate(all_triples)}
    
    # Precompute for each pair (a,b), the list of third points c such that (a,b,c) is a triple
    # pair_to_thirds[(a,b)] = list of c values
    pair_to_thirds = {}
    for a in range(v):
        for b in range(a + 1, v):
            third_list = []
            for c in range(v):
                if c == a or c == b:
                    continue
                third_list.append(c)
            pair_to_thirds[(a, b)] = third_list
    
    return all_triples, triple_to_idx, pair_to_thirds, n_triples


def _block_triple_indices(block, triple_to_idx):
    """Return list of triple indices covered by a block."""
    indices = []
    for c in itertools.combinations(block, 3):
        idx = triple_to_idx[frozenset(c)]
        indices.append(idx)
    return indices


def _greedy_construct(v, k, t, rng, triple_to_idx, n_triples, pair_to_thirds):
    """Greedy set cover construction."""
    uncovered = set(range(n_triples))
    blocks = []
    block_set = set()
    
    while uncovered:
        # Pick the uncovered triple with highest "potential" (random tie-break)
        seed_triple = random.choice(list(uncovered))
        seed_points = list(seed_triple)
        
        block = list(seed_points)
        block_set_local = set(block)
        
        while len(block) < k:
            best_x = None
            best_gain = -1
            
            cands = [x for x in range(v) if x not in block_set_local]
            random.shuffle(cands)
            
            for x in cands:
                gain = 0
                # For each pair in current block, check (pair, x) coverage
                for a_idx in range(len(block)):
                    for b_idx in range(a_idx + 1, len(block)):
                        a, b = block[a_idx], block[b_idx]
                        # Check triple (a, b, x)
                        fs = frozenset((a, b, x))
                        if fs in triple_to_idx:
                            idx = triple_to_idx[fs]
                            if idx in uncovered:
                                gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x is None:
                break
            block.append(best_x)
            block_set_local.add(best_x)
        
        block.sort()
        b_tuple = tuple(block)
        if b_tuple not in block_set:
            block_set.add(b_tuple)
            blocks.append(b_tuple)
            for c in itertools.combinations(block, t):
                idx = triple_to_idx[frozenset(c)]
                uncovered.discard(idx)
        
        if not uncovered:
            break
    
    return blocks


def construct(v, k, t):
    rng = random.Random(42)
    start_time = time.time()
    time_limit = 118.0  # safety margin before 120s
    
    # Precompute
    all_triples, triple_to_idx, pair_to_thirds, n_triples = _precompute(v, k, t)
    
    # Multiple greedy restarts to find good starting point
    best_blocks = None
    best_count = float('inf')
    
    n_restarts = 15
    for restart in range(n_restarts):
        if time.time() - start_time > time_limit - 10:
            break
        r = random.Random(restart * 12345 + 7)
        blocks = _greedy_construct(v, k, t, r, triple_to_idx, n_triples, pair_to_thirds)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [list(b) for b in blocks]
    
    blocks = best_blocks
    
    # Build coverage data structures
    block_covered = [_block_triple_indices(b, triple_to_idx) for b in blocks]
    counts = np.zeros(n_triples, dtype=np.int32)
    for covered in block_covered:
        for idx in covered:
            counts[idx] += 1
    
    best_count = len(blocks)
    best_blocks = [list(b) for b in blocks]
    
    # Local search
    no_improve = 0
    max_no_improve = 200
    
    while time.time() - start_time < time_limit:
        iteration_done = False
        
        # 1. Try to remove redundant blocks
        removed = False
        indices = list(range(len(blocks)))
        rng.shuffle(indices)
        for i in indices:
            if counts[block_covered[i]].min() > 1:
                for idx in block_covered[i]:
                    counts[idx] -= 1
                blocks.pop(i)
                block_covered.pop(i)
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_blocks = [list(b) for b in blocks]
                    no_improve = 0
                removed = True
                break
        
        if removed:
            iteration_done = True
        else:
            # 2. Try to replace a block with a better one
            if len(blocks) > 0:
                i = rng.randrange(len(blocks))
                old_block = blocks[i]
                old_covered = block_covered[i]
                
                # Find critical triples (count == 1)
                critical_mask = (counts[old_covered] == 1)
                critical_count = critical_mask.sum()
                
                if critical_count == 0:
                    # Can remove this block
                    for idx in old_covered:
                        counts[idx] -= 1
                    blocks.pop(i)
                    block_covered.pop(i)
                    if len(blocks) < best_count:
                        best_count = len(blocks)
                        best_blocks = [list(b) for b in blocks]
                        no_improve = 0
                    iteration_done = True
                    continue
                
                # Collect required points from critical triples
                critical_triples = [all_triples[idx] for idx in old_covered if counts[idx] == 1]
                points_needed = set()
                for ct in critical_triples:
                    points_needed.update(ct)
                
                if len(points_needed) > k:
                    no_improve += 1
                    continue
                
                # Build new block
                new_block_set = set(points_needed)
                cands = [x for x in range(v) if x not in new_block_set]
                rng.shuffle(cands)
                
                for x in cands:
                    if len(new_block_set) >= k:
                        break
                    # Gain: how many new triples does x cover?
                    gain = 0
                    block_list = list(new_block_set)
                    for a in block_list:
                        for b in block_list:
                            if a >= b:
                                continue
                            fs = frozenset((a, b, x))
                            if fs in triple_to_idx:
                                idx = triple_to_idx[fs]
                                if counts[idx] == 0:
                                    gain += 100
                                elif counts[idx] == 1:
                                    gain += 1
                    if gain > 0 or len(new_block_set) < k:
                        new_block_set.add(x)
                
                if len(new_block_set) == k:
                    new_block = tuple(sorted(new_block_set))
                    if new_block != tuple(old_block):
                        new_covered = _block_triple_indices(new_block, triple_to_idx)
                        
                        # Check validity: no critical triple loses coverage
                        valid = True
                        for idx in old_covered:
                            if counts[idx] == 1 and idx not in set(new_covered):
                                valid = False
                                break
                        
                        if valid:
                            for idx in old_covered:
                                counts[idx] -= 1
                            for idx in new_covered:
                                counts[idx] += 1
                            blocks[i] = new_block
                            block_covered[i] = new_covered
                            iteration_done = True
                else:
                    no_improve += 1
        
        # 3. Try merge: replace 2 blocks with 1
        if not iteration_done and len(blocks) >= 2 and rng.random() < 0.1:
            i, j = rng.sample(range(len(blocks)), 2)
            crit_mask_i = (counts[block_covered[i]] == 1)
            crit_mask_j = (counts[block_covered[j]] == 1)
            
            if not crit_mask_i.any() and not crit_mask_j.any():
                # Remove one
                for idx in block_covered[i]:
                    counts[idx] -= 1
                blocks.pop(i)
                block_covered.pop(i)
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_blocks = [list(b) for b in blocks]
                    no_improve = 0
                iteration_done = True
                continue
            
            all_crit = set()
            for idx in block_covered[i]:
                if counts[idx] == 1:
                    all_crit.add(idx)
            for idx in block_covered[j]:
                if counts[idx] == 1:
                    all_crit.add(idx)
            
            points_needed = set()
            for idx in all_crit:
                points_needed.update(all_triples[idx])
            
            if len(points_needed) <= k:
                new_block_set = set(points_needed)
                cands = [x for x in range(v) if x not in new_block_set]
                rng.shuffle(cands)
                
                for x in cands:
                    if len(new_block_set) >= k:
                        break
                    new_block_set.add(x)
                
                if len(new_block_set) == k:
                    new_block = tuple(sorted(new_block_set))
                    new_covered = _block_triple_indices(new_block, triple_to_idx)
                    
                    valid = True
                    for idx in all_crit:
                        if idx not in set(new_covered):
                            valid = False
                            break
                    
                    if valid:
                        for idx in block_covered[i]:
                            counts[idx] -= 1
                        for idx in block_covered[j]:
                            counts[idx] -= 1
                        for idx in new_covered:
                            counts[idx] += 1
                        
                        if i < j:
                            blocks.pop(j)
                            block_covered.pop(j)
                            blocks.pop(i)
                            block_covered.pop(i)
                        else:
                            blocks.pop(i)
                            block_covered.pop(i)
                            blocks.pop(j)
                            block_covered.pop(j)
                        
                        blocks.append(new_block)
                        block_covered.append(new_covered)
                        
                        if len(blocks) < best_count:
                            best_count = len(blocks)
                            best_blocks = [list(b) for b in blocks]
                            no_improve = 0
                        iteration_done = True
        
        # 4. Perturbation: random block replacement
        if not iteration_done and len(blocks) > 0 and no_improve > 50:
            i = rng.randrange(len(blocks))
            old_block = blocks[i]
            old_covered = block_covered[i]
            
            # Random new block
            new_block = tuple(sorted(rng.sample(range(v), k)))
            new_covered = _block_triple_indices(new_block, triple_to_idx)
            
            # Check if valid
            valid = True
            for idx in old_covered:
                if counts[idx] == 1 and idx not in set(new_covered):
                    valid = False
                    break
            
            if valid:
                for idx in old_covered:
                    counts[idx] -= 1
                for idx in new_covered:
                    counts[idx] += 1
                blocks[i] = new_block
                block_covered[i] = new_covered
        
        if not iteration_done:
            no_improve += 1
        else:
            no_improve = 0
        
        # Reset after many failures
        if no_improve > max_no_improve:
            no_improve = 0
            # Try a fresh greedy restart
            r = random.Random(rng.randint(0, 10**9))
            new_blocks = _greedy_construct(v, k, t, r, triple_to_idx, n_triples, pair_to_thirds)
            if len(new_blocks) < len(blocks):
                blocks = [list(b) for b in new_blocks]
                block_covered = [_block_triple_indices(b, triple_to_idx) for b in blocks]
                counts = np.zeros(n_triples, dtype=np.int32)
                for covered in block_covered:
                    for idx in covered:
                        counts[idx] += 1
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_blocks = [list(b) for b in blocks]
    
    # Final cleanup
    changed = True
    while changed:
        changed = False
        for i in range(len(blocks) - 1, -1, -1):
            if counts[block_covered[i]].min() > 1:
                for idx in block_covered[i]:
                    counts[idx] -= 1
                blocks.pop(i)
                block_covered.pop(i)
                changed = True
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_blocks = [list(b) for b in blocks]
    
    return best_blocks
