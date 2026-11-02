"""Optimized covering design search for C(24,7,3)."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np

# Precompute all 3-subsets of 24 points and map them to indices
_ALL_TRIPLES = list(itertools.combinations(range(24), 3))
_NUM_TRIPLES = len(_ALL_TRIPLES)
_TRIPLE_TO_IDX = {t: i for i, t in enumerate(_ALL_TRIPLES)}

# For each point, precompute the list of triple-indices that contain it
_POINT_TRIPLES = [[] for _ in range(24)]
for idx, triple in enumerate(_ALL_TRIPLES):
    for p in triple:
        _POINT_TRIPLES[p].append(idx)

# Precompute for each pair of points, the list of triple-indices containing both
_PAIR_TRIPLES = {}
for i in range(24):
    for j in range(i + 1, 24):
        # Triples containing both i and j: the third element can be any other point
        triples = []
        for k in range(24):
            if k != i and k != j:
                t = tuple(sorted((i, j, k)))
                triples.append(_TRIPLE_TO_IDX[t])
        _PAIR_TRIPLES[(i, j)] = triples


def _block_triples(block):
    """Return the set of triple-indices covered by a 7-element block."""
    covered = set()
    for combo in itertools.combinations(block, 3):
        covered.add(_TRIPLE_TO_IDX[combo])
    return covered


def _greedy_covering(v=24, k=7, t=3, seed=0, time_limit=10):
    """Greedy covering with randomized tie-breaking."""
    rng = random.Random(seed)
    uncovered = np.ones(_NUM_TRIPLES, dtype=np.uint8)
    blocks = []
    start_time = time.time()
    
    while np.any(uncovered):
        if time.time() - start_time > time_limit:
            break
        
        # Find the first uncovered triple (or a random one for diversity)
        uncovered_indices = np.nonzero(uncovered)[0]
        if not len(uncovered_indices):
            break
        # Pick a random uncovered triple to start the block
        start_triple_idx = rng.choice(uncovered_indices)
        start_triple = _ALL_TRIPLES[start_triple_idx]
        
        block = list(start_triple)
        block_set = set(block)
        
        # Greedily add points
        while len(block) < k:
            if time.time() - start_time > time_limit:
                break
            
            # For each candidate point, compute how many new triples it covers
            # given the current block
            best_point = -1
            best_gain = -1
            best_ties = []
            
            for x in range(v):
                if x in block_set:
                    continue
                
                # Count uncovered triples that would be covered by adding x
                # A triple (a, b, x) is covered if a,b are in block
                # and the triple was uncovered
                gain = 0
                for a, b in itertools.combinations(block, 2):
                    pair_key = (min(a, b), max(a, b))
                    for t_idx in _PAIR_TRIPLES[pair_key]:
                        if uncovered[t_idx] and x in _ALL_TRIPLES[t_idx]:
                            gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
                    best_ties = [x]
                elif gain == best_gain and gain > 0:
                    best_ties.append(x)
            
            if best_point == -1:
                # No gain, just pick a random point
                candidates = [x for x in range(v) if x not in block_set]
                if not candidates:
                    break
                best_point = rng.choice(candidates)
            elif len(best_ties) > 1:
                best_point = rng.choice(best_ties)
            
            block.append(best_point)
            block_set.add(best_point)
        
        if len(block) < k:
            break
        
        block.sort()
        blocks.append(tuple(block))
        
        # Update uncovered
        for t_idx in _block_triples(block):
            uncovered[t_idx] = 0
    
    return blocks


def _is_valid_covering(blocks, v=24, k=7, t=3):
    """Check if blocks form a valid covering."""
    if not blocks:
        return False
    covered = np.zeros(_NUM_TRIPLES, dtype=np.uint8)
    for block in blocks:
        if len(block) != k:
            return False
        if len(set(block)) != k:
            return False
        for combo in itertools.combinations(block, t):
            covered[_TRIPLE_TO_IDX[combo]] = 1
    return np.all(covered)


def _remove_redundant(blocks):
    """Remove blocks that don't cover any unique triple."""
    if not blocks:
        return blocks
    
    n = len(blocks)
    # Count how many blocks cover each triple
    triple_count = np.zeros(_NUM_TRIPLES, dtype=np.int32)
    block_triples_list = []
    for block in blocks:
        t_set = _block_triples(block)
        block_triples_list.append(t_set)
        for t_idx in t_set:
            triple_count[t_idx] += 1
    
    # A block is redundant if all its triples are covered by other blocks
    essential = [True] * n
    for i in range(n):
        for t_idx in block_triples_list[i]:
            if triple_count[t_idx] == 1:
                break
        else:
            essential[i] = False
    
    if all(essential):
        return blocks
    
    new_blocks = [blocks[i] for i in range(n) if essential[i]]
    return new_blocks


def _local_search(blocks, seed=0, time_limit=30):
    """Local search to remove redundant blocks and try swaps."""
    rng = random.Random(seed)
    blocks = list(blocks)
    start_time = time.time()
    
    # First pass: remove redundant blocks
    prev_len = len(blocks) + 1
    while len(blocks) < prev_len:
        prev_len = len(blocks)
        blocks = _remove_redundant(blocks)
    
    # Try to remove blocks and repair
    for _ in range(50):
        if time.time() - start_time > time_limit:
            break
        if not blocks:
            break
        
        # Try removing a random block
        idx = rng.randrange(len(blocks))
        candidate = blocks[:idx] + blocks[idx+1:]
        
        if _is_valid_covering(candidate):
            blocks = candidate
        else:
            # Try to repair by adding a block that covers the most uncovered triples
            # Find uncovered triples
            covered = np.zeros(_NUM_TRIPLES, dtype=np.uint8)
            for block in candidate:
                for combo in itertools.combinations(block, 3):
                    covered[_TRIPLE_TO_IDX[combo]] = 1
            
            uncovered_indices = np.nonzero(1 - covered)[0]
            if len(uncovered_indices) == 0:
                blocks = candidate
                continue
            
            # Greedily add one block to cover most uncovered triples
            best_block = None
            best_count = 0
            # Try some candidate blocks
            candidates_to_try = []
            # Random 7-subsets
            for _ in range(100):
                cand = tuple(sorted(rng.sample(range(24), 7)))
                candidates_to_try.append(cand)
            # Also try blocks from the original set
            candidates_to_try.extend(blocks)
            
            for cand in candidates_to_try:
                count = 0
                for combo in itertools.combinations(cand, 3):
                    t_idx = _TRIPLE_TO_IDX[combo]
                    if not covered[t_idx]:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_block = cand
            
            if best_block is not None and best_count > 0:
                new_candidate = candidate + [best_block]
                if _is_valid_covering(new_candidate) and len(new_candidate) < len(blocks):
                    blocks = new_candidate
    
    # Final redundant removal
    blocks = _remove_redundant(blocks)
    
    return blocks


def _simulated_annealing(blocks, seed=0, time_limit=60):
    """Simulated annealing to minimize block count."""
    rng = random.Random(seed)
    blocks = list(blocks)
    start_time = time.time()
    
    # Remove redundant first
    blocks = _remove_redundant(blocks)
    
    best_blocks = list(blocks)
    best_count = len(blocks)
    
    temperature = 5.0
    cooling = 0.999
    
    while time.time() - start_time < time_limit:
        if temperature < 0.1:
            break
        
        # Generate a perturbation
        perturbation_type = rng.random()
        
        if perturbation_type < 0.5 and len(blocks) > 1:
            # Remove a random block
            idx = rng.randrange(len(blocks))
            candidate = blocks[:idx] + blocks[idx+1:]
        elif perturbation_type < 0.8:
            # Swap a block with a random 7-subset
            idx = rng.randrange(len(blocks))
            new_block = tuple(sorted(rng.sample(range(24), 7)))
            candidate = blocks[:idx] + [new_block] + blocks[idx+1:]
        else:
            # Add a random block
            new_block = tuple(sorted(rng.sample(range(24), 7)))
            candidate = blocks + [new_block]
        
        # Evaluate
        if not _is_valid_covering(candidate):
            # Invalid - reject
            continue
        
        delta = len(candidate) - len(blocks)
        
        if delta <= 0:
            # Improvement or same
            blocks = candidate
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = list(blocks)
        else:
            # Worse - accept with probability exp(-delta/temperature)
            if rng.random() < np.exp(-delta / temperature):
                blocks = candidate
        
        temperature *= cooling
    
    # Final cleanup
    best_blocks = _remove_redundant(best_blocks)
    return best_blocks


def construct(v, k, t):
    """Construct a covering design C(v,k,t)."""
    if v != 24 or k != 7 or t != 3:
        # Fallback for other parameters
        return _greedy_covering(v, k, t, seed=0, time_limit=10)
    
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    best_blocks = None
    best_count = float('inf')
    
    # Phase 1: Multiple greedy constructions with different seeds
    for seed in range(20):
        if time.time() > deadline - 60:
            break
        blocks = _greedy_covering(24, 7, 3, seed=seed, time_limit=3)
        if blocks and _is_valid_covering(blocks):
            blocks = _remove_redundant(blocks)
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks
    
    if best_blocks is None:
        # Emergency: just do a simple greedy
        best_blocks = _greedy_covering(24, 7, 3, seed=0, time_limit=5)
        if not best_blocks or not _is_valid_covering(best_blocks):
            # Should not happen, but just in case
            return []
    
    # Phase 2: Local search on the best solution
    if time.time() < deadline - 30:
        ls_blocks = _local_search(best_blocks, seed=0, time_limit=20)
        if _is_valid_covering(ls_blocks) and len(ls_blocks) < best_count:
            best_blocks = ls_blocks
            best_count = len(ls_blocks)
    
    # Phase 3: Simulated annealing
    if time.time() < deadline - 15:
        sa_blocks = _simulated_annealing(best_blocks, seed=0, time_limit=25)
        if _is_valid_covering(sa_blocks) and len(sa_blocks) < best_count:
            best_blocks = sa_blocks
            best_count = len(sa_blocks)
    
    # Final check
    if not _is_valid_covering(best_blocks):
        # Fallback
        best_blocks = _greedy_covering(24, 7, 3, seed=0, time_limit=5)
    
    # Convert to lists
    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
