"""
Spectral-guided greedy covering for C(24,7,3).

Strategy:
1. Precompute a fast triple-lookup structure.
2. Use a weighted greedy: prioritize triples that are "rare" (harder to cover) by
   assigning weights inversely proportional to how many blocks in a random sample
   would cover them. In practice, we use a simpler heuristic: each time we add a
   block, we update the count of how many times each triple is covered.
3. The greedy step: for the current partial block, choose the next point that
   maximizes the number of newly-covered weighted triples.
4. After the greedy phase, run a local search / pruning pass to remove redundant
   blocks.
5. Deadline guard: return best solution found before time limit.
"""

import time
import itertools
import random
import sys

def construct(v, k, t):
    """
    Construct a covering design C(v,k,t): find a collection of k-subsets (blocks)
    of {0,...,v-1} such that every t-subset is contained in at least one block.
    Minimizes the number of blocks.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    # All t-subsets
    all_triples = list(itertools.combinations(range(v), t))
    n_triples = len(all_triples)
    
    # Map each triple to an index
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    
    # Precompute for each point, which triples contain it
    # point_to_triples[x] = list of triple indices that contain x
    point_to_triples = [[] for _ in range(v)]
    for i, triple in enumerate(all_triples):
        for x in triple:
            point_to_triples[x].append(i)
    
    # Weighted greedy: maintain uncovered triple set and weights
    # Weight of a triple = 1 / (number of blocks that could cover it in a random design)
    # Approximation: all triples have equal a priori weight, but we can use
    # the frequency in a random sample. For simplicity, start with uniform weights
    # and update dynamically.
    
    uncovered = set(range(n_triples))  # set of triple indices not yet covered
    blocks = []
    
    # For tracking: how many times each triple is covered (for pruning later)
    triple_cover_count = [0] * n_triples
    
    # Greedy construction
    while uncovered:
        if time.time() > deadline:
            break
        
        # Pick the triple with the highest weight among uncovered ones
        # For simplicity, pick the first uncovered triple (or use weighted selection)
        # Better: pick the uncovered triple that has the highest "difficulty"
        # Use a simple heuristic: pick the uncovered triple with the fewest
        # potential completions to a k-block given remaining structure
        # For now, pick an arbitrary uncovered triple to start the block
        target_triple = min(uncovered)  # deterministic choice
        block = list(all_triples[target_triple])
        
        # Extend the block to size k
        while len(block) < k:
            if time.time() > deadline:
                break
            
            # For each candidate point x not in block, compute gain:
            # number of uncovered triples that would be covered by adding x
            best_x = -1
            best_gain = -1
            
            # Optimize: only consider points that appear in uncovered triples
            # with the current block's (t-1)-subsets
            candidate_points = set()
            for rest in itertools.combinations(block, t - 1):
                for x in range(v):
                    if x not in block:
                        candidate_points.add(x)
            
            for x in candidate_points:
                # Count how many uncovered triples contain all of (block_rest + x)
                # where block_rest is a (t-1)-subset of block
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    triple = tuple(sorted(rest + (x,)))
                    if triple in triple_to_idx:
                        idx = triple_to_idx[triple]
                        if idx in uncovered:
                            gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x == -1 or best_gain <= 0:
                # No gain, pick a random point
                available = [x for x in range(v) if x not in block]
                best_x = random.choice(available)
            
            block.append(best_x)
        
        if len(block) < k:
            # Timeout during block construction, fill randomly
            available = [x for x in range(v) if x not in block]
            random.shuffle(available)
            block.extend(available[:k - len(block)])
        
        block.sort()
        blocks.append(block)
        
        # Update uncovered set
        for triple in itertools.combinations(block, t):
            idx = triple_to_idx.get(tuple(triple), -1)
            if idx != -1 and idx in uncovered:
                uncovered.discard(idx)
                triple_cover_count[idx] += 1
            elif idx != -1:
                triple_cover_count[idx] += 1
    
    if time.time() > deadline:
        return blocks
    
    # Pruning pass: remove blocks that are redundant
    # A block is redundant if all its triples are covered by other blocks
    # Iterate until no more removals
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Recompute coverage from scratch for accuracy
        covered_by = [set() for _ in range(n_triples)]
        for bi, block in enumerate(blocks):
            for triple in itertools.combinations(block, t):
                idx = triple_to_idx.get(tuple(triple), -1)
                if idx != -1:
                    covered_by[idx].add(bi)
        
        new_blocks = []
        for bi, block in enumerate(blocks):
            # Check if this block is needed
            redundant = True
            for triple in itertools.combinations(block, t):
                idx = triple_to_idx.get(tuple(triple), -1)
                if idx != -1:
                    # Is this triple covered by any other block?
                    if len(covered_by[idx]) > 1:
                        continue
                    else:
                        redundant = False
                        break
            if not redundant:
                new_blocks.append(block)
            else:
                changed = True
        
        if changed:
            blocks = new_blocks
    
    return blocks


# Alternative: use numpy for faster computation if available
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using weighted greedy with pruning.
    """
    start_time = time.time()
    deadline = start_time + 145.0

    all_triples = list(itertools.combinations(range(v), t))
    n_triples = len(all_triples)
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}

    uncovered = set(range(n_triples))
    blocks = []
    triple_cover_count = [0] * n_triples

    # Precompute for speed: for each (t-1)-subset, list of points that complete it
    # (t-1)-subset -> list of (point, triple_idx)
    from collections import defaultdict
    partial_to_completions = defaultdict(list)
    for idx, triple in enumerate(all_triples):
        for j in range(t):
            partial = tuple(triple[:j] + triple[j+1:])
            partial_to_completions[partial].append((triple[j], idx))

    while uncovered and time.time() < deadline:
        # Pick a target triple (the one with highest "weight" - use first for simplicity)
        # Better: pick the uncovered triple that appears in the fewest potential blocks
        # For now, use a weighted random selection biased toward hard triples
        # Compute a simple difficulty score: for each uncovered triple, count how many
        # (k-t) extensions exist that are "viable" (don't conflict)
        # This is expensive, so use a simpler heuristic:
        # Pick the uncovered triple with the smallest index (deterministic)
        target_idx = min(uncovered)
        target_triple = all_triples[target_idx]
        block = list(target_triple)

        # Greedily extend
        while len(block) < k and time.time() < deadline:
            best_x = -1
            best_gain = -1

            # Candidate points: those that can complete some (t-1)-subset of block
            candidate_set = set()
            for rest in itertools.combinations(block, t - 1):
                for x, _ in partial_to_completions.get(tuple(rest), []):
                    if x not in block:
                        candidate_set.add(x)

            for x in candidate_set:
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    partial = tuple(rest + (x,))
                    for _, idx in partial_to_completions.get(partial, []):
                        if idx in uncovered:
                            gain += 1
                    # Also check the direct triple
                    triple = tuple(sorted(rest + (x,)))
                    if triple in triple_to_idx:
                        idx = triple_to_idx[triple]
                        if idx in uncovered:
                            gain += 1

                if gain > best_gain:
                    best_gain = gain
                    best_x = x

            if best_x == -1:
                available = [x for x in range(v) if x not in block]
                if not available:
                    break
                best_x = random.choice(available)

            block.append(best_x)

        if len(block) < k:
            available = [x for x in range(v) if x not in block]
            random.shuffle(available)
            block.extend(available[:k - len(block)])

        block.sort()
        blocks.append(block)

        for triple in itertools.combinations(block, t):
            idx = triple_to_idx.get(tuple(triple), -1)
            if idx != -1:
                if idx in uncovered:
                    uncovered.discard(idx)
                triple_cover_count[idx] += 1

    # Pruning
    if time.time() < deadline - 5:
        for _ in range(3):  # max 3 pruning passes
            if time.time() > deadline - 2:
                break
            covered_by = [set() for _ in range(n_triples)]
            for bi, block in enumerate(blocks):
                for triple in itertools.combinations(block, t):
                    idx = triple_to_idx.get(tuple(triple), -1)
                    if idx != -1:
                        covered_by[idx].add(bi)

            new_blocks = []
            for bi, block in enumerate(blocks):
                keep = False
                for triple in itertools.combinations(block, t):
                    idx = triple_to_idx.get(tuple(triple), -1)
                    if idx != -1 and len(covered_by[idx]) == 1:
                        keep = True
                        break
                if keep:
                    new_blocks.append(block)

            if len(new_blocks) == len(blocks):
                break
            blocks = new_blocks

    return blocks
