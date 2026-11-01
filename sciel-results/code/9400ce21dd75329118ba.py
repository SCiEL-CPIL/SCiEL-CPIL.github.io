import itertools
import time
import random
from collections import defaultdict

# Precomputation for v=24, k=7, t=3
V = 24
K = 7
T = 3

def _precompute():
    triples = list(itertools.combinations(range(V), T))
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    
    all_blocks = list(itertools.combinations(range(V), K))
    block_triples = []
    for b in all_blocks:
        b_set = set(b)
        ts = []
        for t in itertools.combinations(b, T):
            ts.append(triple_to_idx[t])
        block_triples.append(ts)
        
    block_to_idx = {b: i for i, b in enumerate(all_blocks)}
    
    # Map triple index -> list of block indices covering it
    triple_to_blocks = defaultdict(list)
    for b_idx, ts in enumerate(block_triples):
        for t_idx in ts:
            triple_to_blocks[t_idx].append(b_idx)

    return triple_to_idx, all_blocks, block_triples, block_to_idx, triple_to_blocks

triple_to_idx, all_blocks, block_triples, block_to_idx, triple_to_blocks = _precompute()
N_TRIPLES = len(triple_to_idx)
N_BLOCKS = len(all_blocks)

def _greedy_cover_indices(seed=0):
    """Greedy covering returning block indices."""
    rng = random.Random(seed)
    uncovered = [1] * N_TRIPLES
    blocks_idx = []
    
    while any(uncovered):
        # Pick a random uncovered triple
        unc_idx = None
        while unc_idx is None:
            t_idx = rng.randrange(N_TRIPLES)
            if uncovered[t_idx]:
                unc_idx = t_idx
        
        # Get candidate blocks covering this triple
        candidates = triple_to_blocks[unc_idx]
        
        best_block = None
        best_gain = -1
        
        for b_idx in candidates:
            gain = 0
            for t_idx in block_triples[b_idx]:
                if uncovered[t_idx]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block = b_idx
        
        if best_block is not None:
            blocks_idx.append(best_block)
            for t_idx in block_triples[best_block]:
                if uncovered[t_idx]:
                    uncovered[t_idx] = 0
    
    return blocks_idx

def _remove_redundant(blocks_idx, cover_count, rng):
    """Remove redundant blocks. Modifies blocks_idx and cover_count in place? 
       Better to return new ones to be safe, or modify carefully."""
    # We modify in place for speed, but must be careful with indices.
    n = len(blocks_idx)
    if n == 0:
        return blocks_idx, cover_count
    
    changed = True
    while changed:
        changed = False
        order = list(range(len(blocks_idx)))
        rng.shuffle(order)
        
        for idx in order:
            b_idx = blocks_idx[idx]
            redundant = True
            for t_idx in block_triples[b_idx]:
                if cover_count[t_idx] <= 1:
                    redundant = False
                    break
            
            if redundant:
                # Remove block
                del blocks_idx[idx]
                for t_idx in block_triples[b_idx]:
                    cover_count[t_idx] -= 1
                changed = True
                break
    
    return blocks_idx, cover_count

def _local_search(blocks_idx, cover_count, deadline, seed=0):
    """Local search to minimize block count."""
    rng = random.Random(seed)
    blocks_idx, cover_count = _remove_redundant(blocks_idx, cover_count, rng)
    
    best_blocks_idx = list(blocks_idx)
    best_count = len(blocks_idx)
    
    stagnant_count = 0
    max_stagnant = 500
    
    while time.time() < deadline:
        if len(blocks_idx) < best_count:
            best_count = len(blocks_idx)
            best_blocks_idx = list(blocks_idx)
            stagnant_count = 0
            if best_count == 69:
                break
        
        if len(blocks_idx) == 0:
            break
            
        i = rng.randrange(len(blocks_idx))
        old_b_idx = blocks_idx[i]
        new_b_idx = rng.randrange(N_BLOCKS)
        
        if new_b_idx == old_b_idx:
            continue
            
        # Check validity of swap
        valid = True
        for t_idx in block_triples[old_b_idx]:
            cover_count[t_idx] -= 1
            if cover_count[t_idx] < 0:
                valid = False
                break
        
        if not valid:
            for t_idx in block_triples[old_b_idx]:
                cover_count[t_idx] += 1
            continue
        
        for t_idx in block_triples[new_b_idx]:
            cover_count[t_idx] += 1
            
        blocks_idx[i] = new_b_idx
        
        # Remove redundants
        blocks_idx, cover_count = _remove_redundant(blocks_idx, cover_count, rng)
        
        if len(blocks_idx) < len(best_blocks_idx):
            stagnant_count = 0
        else:
            stagnant_count += 1
            if stagnant_count > max_stagnant:
                # Revert
                blocks_idx = list(best_blocks_idx)
                cover_count = [0] * N_TRIPLES
                for b_idx in blocks_idx:
                    for t_idx in block_triples[b_idx]:
                        cover_count[t_idx] += 1
                break

    return best_blocks_idx, best_count

def construct(v, k, t):
    deadline = time.time() + 110
    
    if v != 24 or k != 7 or t != 3:
        # Fallback
        def _generic_greedy(v, k, t, seed=0):
            rng = random.Random(seed)
            all_tsets = [frozenset(c) for c in itertools.combinations(range(v), t)]
            uncovered = set(all_tsets)
            blocks = []
            while uncovered:
                start_tset = rng.choice(list(uncovered))
                block = list(start_tset)
                while len(block) < k:
                    best_point = None
                    best_gain = -1
                    block_set = set(block)
                    for x in range(v):
                        if x in block_set:
                            continue
                        gain = 0
                        for rest in itertools.combinations(block, t - 1):
                            new_tset = frozenset(rest) | {x}
                            if new_tset in uncovered:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_point = x
                    block.append(best_point)
                block.sort()
                blocks.append(block)
                for c in itertools.combinations(block, t):
                    uncovered.discard(frozenset(c))
            return blocks
        return _generic_greedy(v, k, t, seed=0)

    best_blocks_idx = None
    best_count = float('inf')
    
    for seed in range(30):
        if time.time() > deadline:
            break
        
        greedy_idx = _greedy_cover_indices(seed=seed)
        
        cover_count = [0] * N_TRIPLES
        for b_idx in greedy_idx:
            for t_idx in block_triples[b_idx]:
                cover_count[t_idx] += 1
        
        ls_idx, ls_count = _local_search(greedy_idx, cover_count, deadline, seed=seed)
        
        if ls_count < best_count:
            best_count = ls_count
            best_blocks_idx = ls_idx
            
        if best_count == 69:
            break
            
    if best_blocks_idx is None:
        greedy_idx = _greedy_cover_indices(seed=0)
        best_blocks_idx = greedy_idx
        best_count = len(greedy_idx)

    return [list(all_blocks[b_idx]) for b_idx in best_blocks_idx]
