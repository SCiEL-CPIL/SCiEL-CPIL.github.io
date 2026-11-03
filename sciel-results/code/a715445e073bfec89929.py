"""Optimized covering C(24,7,3) using Information-Theoretic (Entropy Reduction) Greedy and ILS."""
import itertools
import random
import time
import sys
from collections import defaultdict

# Precompute combinations for efficiency
# v=24, k=7, t=3
# Total triples = 2024
# We need a mapping from triple index to triple tuple and vice versa for fast array access.

def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using an Information-Theoretic approach:
    1. Entropy-Reduction Greedy: Prioritize covering 'scarce' or 'isolated' triples.
    2. Iterated Local Search: Perturb and repair to escape local minima.
    """
    # Map triples to indices for O(1) access in counts array
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    # Pre-compute for each triple, the list of blocks that contain it?
    # No, that's too big. 
    # Instead, for the greedy step, we pick a "seed" triple, 
    # then iterate over all blocks containing that seed triple.
    
    # Pre-compute: for each point, list of triples containing it?
    # Or better: for each triple, list of points NOT in it?
    
    # Let's build a helper to get candidate blocks containing a specific triple.
    # To speed up, we can pre-compute the list of all blocks? 
    # C(24,7) is 346104. Storing them is fine.
    # But generating them on the fly for a specific seed triple is faster.
    
    # Let's use the ILS approach.
    
    # 1. Initial Solution via Greedy
    blocks = _greedy_construct(v, k, t, all_triples, triple_to_idx)
    
    # 2. Local Search / ILS
    blocks = _ils_optimize(v, k, t, all_triples, triple_to_idx, blocks)
    
    return blocks

def _greedy_construct(v, k, t, all_triples, triple_to_idx):
    """
    Greedy construction maximizing information gain.
    Gain = sum of weights of newly covered triples.
    Weight of a triple is 1 / (number of blocks that could potentially cover it).
    Since we are building sequentially, we can approximate "potential" by 
    the total number of k-subsets containing the triple, which is constant.
    However, to be more "information-theoretic", we can weight triples by 
    how "isolated" they are in the uncovered set.
    
    Optimization: 
    1. Find the uncovered triple with the highest "scarcity score".
       Scarcity score is inversely proportional to the number of other uncovered triples 
       it intersects with? No, that's expensive.
       
    Let's use a standard weighted greedy:
    Weight of triple = 1.0
    Gain of block B = number of new triples covered by B.
    This is standard greedy. To improve, we can use "Lookahead" or "Dynamic Weights".
    
    Dynamic Weight Idea:
    If a triple is covered by few *remaining* potential blocks, it is more critical.
    Since we don't know the remaining blocks, we can use a proxy:
    Prioritize triples that share points with few other uncovered triples.
    """
    uncovered = set(all_triples)
    blocks = []
    
    # Pre-compute point incidence for uncovered set to find "isolated" triples
    # This is O(N * t) per step. N=2024. 2024*3*76 ~ 450k ops. Fast enough.
    
    while uncovered:
        # Calculate weights for uncovered triples
        # Heuristic: A triple is "hard" if its points are "rare" in the uncovered set.
        point_freq = [0] * v
        for triple in uncovered:
            for p in triple:
                point_freq[p] += 1
        
        # Score each uncovered triple
        # We want to pick a seed triple that is "critical".
        # Let's pick the triple with the minimum sum of point frequencies (most isolated).
        best_seed = None
        best_seed_score = float('inf')
        
        # Optimization: Instead of checking all, check a random subset? 
        # No, 2024 is small. Check all.
        for triple in uncovered:
            score = point_freq[triple[0]] + point_freq[triple[1]] + point_freq[triple[2]]
            if score < best_seed_score:
                best_seed_score = score
                best_seed = triple
        
        if best_seed is None:
            break
            
        # Generate all candidate blocks containing best_seed
        # best_seed is a tuple of 3 points.
        # We need to choose k-t = 4 more points from the remaining v-t = 21 points.
        seed_set = set(best_seed)
        remaining_points = [p for p in range(v) if p not in seed_set]
        
        best_block = None
        best_gain = -1
        
        # Iterate over combinations of 4 points from remaining_points
        # C(21, 4) = 5985 candidates.
        # For each candidate, calculate gain.
        # Gain = number of triples in the block that are in `uncovered`.
        
        # To speed up gain calculation:
        # A block B contains C(7,3)=35 triples.
        # We can pre-compute the triples for the candidate block?
        # Or just iterate.
        
        # Let's optimize the inner loop.
        # Instead of generating all 5985 and checking, we can be smarter?
        # Probably not needed, 5985 * 35 * 76 steps is ~ 15M ops. Might be tight in Python.
        
        # Let's try to prune candidates.
        # We can estimate gain by checking a few key triples.
        
        # Let's just do the full check but optimized.
        
        # Pre-compute the triples of the seed part?
        # The block is seed_set U add_set.
        # Triples are:
        # 1. Triples entirely in seed_set (1 triple: best_seed itself).
        # 2. Triples with 2 from seed, 1 from add.
        # 3. Triples with 1 from seed, 2 from add.
        # 4. Triples entirely in add (C(4,3)=4 triples).
        
        # We can pre-compute the "base" triples involving the seed.
        # seed = (a, b, c)
        # Triples with 2 from seed: (a,b,x), (a,c,x), (b,c,x) for x in add.
        # Triples with 1 from seed: (a,x,y), (b,x,y), (c,x,y) for x,y in add.
        
        # Let's iterate over add_sets.
        
        for add_combo in itertools.combinations(remaining_points, k - t):
            add_set = set(add_combo)
            full_block = seed_set.union(add_set)
            
            gain = 0
            # Check all triples in full_block
            # To avoid sorting and set lookups for every triple, we can check membership directly.
            # triple = tuple(sorted(...)) is slow.
            
            # We can iterate through the specific structures.
            
            # 1. Seed itself
            if best_seed in uncovered:
                gain += 1
            
            # 2. 2 from seed, 1 from add
            # Pairs in seed: (a,b), (a,c), (b,c)
            seed_list = list(seed_set)
            pairs = [(seed_list[0], seed_list[1]), (seed_list[0], seed_list[2]), (seed_list[1], seed_list[2])]
            for p1, p2 in pairs:
                for x in add_set:
                    # Create triple tuple
                    # Sort is needed for dict lookup
                    # We can optimize by pre-sorting or using a canonical form
                    # Since v=24, we can pack into an integer?
                    # Or just use tuple(sorted((p1,p2,x)))
                    # Let's try to avoid sort by checking order?
                    # No, just use sorted. It's small.
                    tr = tuple(sorted((p1, p2, x)))
                    if tr in uncovered:
                        gain += 1
            
            # 3. 1 from seed, 2 from add
            for s in seed_set:
                for combo2 in itertools.combinations(add_set, 2):
                    tr = tuple(sorted((s, combo2[0], combo2[1])))
                    if tr in uncovered:
                        gain += 1
                        
            # 4. 0 from seed, 3 from add
            for combo3 in itertools.combinations(add_set, 3):
                tr = tuple(sorted(combo3))
                if tr in uncovered:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_block = sorted(full_block)
                
        if best_block is None:
            # Fallback: pick random block containing seed
            add_combo = random.sample(remaining_points, k-t)
            best_block = sorted(seed_set.union(set(add_combo)))
            
        blocks.append(best_block)
        
        # Update uncovered
        for tr in itertools.combinations(best_block, t):
            uncovered.discard(tr)
            
    return blocks

def _ils_optimize(v, k, t, all_triples, triple_to_idx, initial_blocks, time_limit=100):
    """
    Iterated Local Search.
    1. Remove redundant blocks.
    2. Perturb: Remove a random block.
    3. Repair: Greedily add blocks to cover gaps.
    4. Accept if better.
    """
    current_blocks = list(initial_blocks)
    best_blocks = list(current_blocks)
    best_len = len(current_blocks)
    
    start_time = time.time()
    random.seed(42)
    
    def get_coverage_counts(blocks):
        counts = [0] * len(all_triples)
        for b in blocks:
            for tr in itertools.combinations(b, t):
                idx = triple_to_idx[tr]
                counts[idx] += 1
        return counts
        
    # Initial cleanup
    current_blocks = _remove_redundant(v, k, t, all_triples, triple_to_idx, current_blocks)
    if len(current_blocks) < best_len:
        best_blocks = list(current_blocks)
        best_len = len(current_blocks)
        
    while time.time() - start_time < time_limit:
        # Perturbation
        perturbed = list(current_blocks)
        if perturbed:
            # Remove a random block
            idx = random.randint(0, len(perturbed) - 1)
            removed_block = perturbed.pop(idx)
            
            # Repair
            # Find uncovered triples
            counts = get_coverage_counts(perturbed)
            uncovered = [all_triples[i] for i, c in enumerate(counts) if c == 0]
            
            if not uncovered:
                # We removed a redundant block effectively?
                # This shouldn't happen if we cleaned first, but maybe.
                pass
            else:
                # Greedily add blocks to cover uncovered
                # Use a simple greedy: pick block covering most uncovered
                while uncovered:
                    # Pick most critical uncovered triple
                    # Just pick the first one for speed in repair?
                    # Or pick one with min count? All are 0.
                    # Let's pick a random uncovered triple to vary the repair
                    seed_triple = random.choice(uncovered)
                    
                    # Find best block covering seed_triple and other uncovered
                    seed_set = set(seed_triple)
                    remaining_points = [p for p in range(v) if p not in seed_set]
                    
                    best_block = None
                    best_gain = -1
                    
                    # Limit search space for repair speed
                    # Sample candidates?
                    # C(21,4) is 5985. 
                    # We can sample 100 random candidates to speed up repair.
                    candidates = []
                    for _ in range(50):
                        add_combo = random.sample(remaining_points, k-t)
                        candidates.append(sorted(seed_set.union(set(add_combo))))
                    
                    # Also add the removed_block if it covers seed?
                    if removed_block and seed_triple in itertools.combinations(removed_block, t):
                         candidates.append(removed_block)
                    
                    for cand in candidates:
                        gain = 0
                        for tr in itertools.combinations(cand, t):
                            if tr in uncovered:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_block = cand
                            
                    if best_block is None:
                        # Fallback
                        add_combo = random.sample(remaining_points, k-t)
                        best_block = sorted(seed_set.union(set(add_combo)))
                        
                    perturbed.append(best_block)
                    
                    # Update uncovered
                    for tr in itertools.combinations(best_block, t):
                        if tr in uncovered:
                            uncovered.remove(tr)
                            # Note: removing from list is O(N). 
                            # For repair, N is small (number of uncovered triples).
                            # If we removed 1 block, ~35 triples become uncovered.
                            # So this is fast.
                            
        # Clean up redundant blocks
        perturbed = _remove_redundant(v, k, t, all_triples, triple_to_idx, perturbed)
        
        if len(perturbed) < best_len:
            best_len = len(perturbed)
            best_blocks = list(perturbed)
            current_blocks = list(perturbed)
            
    return best_blocks

def _remove_redundant(v, k, t, all_triples, triple_to_idx, blocks):
    """Remove blocks that are redundant."""
    counts = [0] * len(all_triples)
    for b in blocks:
        for tr in itertools.combinations(b, t):
            counts[triple_to_idx[tr]] += 1
            
    new_blocks = []
    for b in blocks:
        # Check if any triple in b has count > 1
        # If all triples in b have count > 1, it's redundant?
        # No, if count > 1, it means other blocks cover them.
        # If we remove b, the count drops by 1.
        # It is redundant if for ALL triples in b, count > 1.
        redundant = True
        for tr in itertools.combinations(b, t):
            if counts[triple_to_idx[tr]] == 1:
                redundant = False
                break
                
        if not redundant:
            new_blocks.append(b)
            # If we keep it, we don't update counts here because 
            # we are checking against the *original* counts.
            # This is a "parallel" removal check.
            # To be safe and allow chaining, we should update counts 
            # or do it in a loop.
            
    # The above loop is a single pass. It might miss blocks that become redundant 
    # after another block is removed.
    # Let's do a while loop.
    
    changed = True
    while changed:
        changed = False
        # Recompute counts for current new_blocks? 
        # That's expensive.
        # Let's just do one pass first, then check again.
        
        counts = [0] * len(all_triples)
        for b in new_blocks:
            for tr in itertools.combinations(b, t):
                counts[triple_to_idx[tr]] += 1
                
        to_remove = []
        for i, b in enumerate(new_blocks):
            redundant = True
            for tr in itertools.combinations(b, t):
                if counts[triple_to_idx[tr]] == 1:
                    redundant = False
                    break
            if redundant:
                to_remove.append(i)
                
        if to_remove:
            changed = True
            # Remove from end to start to preserve indices
            for i in reversed(to_remove):
                new_blocks.pop(i)
                
    return new_blocks
