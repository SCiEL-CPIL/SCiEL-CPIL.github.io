"""Bitset-accelerated greedy covering with vectorized block selection and local search."""

import time
import random
import itertools
import numpy as np

# Precompute for v=24, k=7, t=3
_C24_3 = 2024
_NK = 245040  # C(24,7)
_C7_3 = 35    # C(7,3)

_COMBO_INDEX = {}
_idx = 0
for a in range(24):
    for b in range(a + 1, 24):
        for c in range(b + 1, 24):
            _COMBO_INDEX[(a, b, c)] = _idx
            _idx += 1
assert _idx == _C24_3

_IDX_TO_COMBO = [None] * _C24_3
for combo, idx in _COMBO_INDEX.items():
    _IDX_TO_COMBO[idx] = combo
_IDX_TO_COMBO = tuple(_IDX_TO_COMBO)

# Precompute all 7-subsets and their triple coverage
_ALL_BLOCKS = []
_BLOCK_TRIPLES = np.zeros((NK, _C7_3), dtype=np.int32)
_BLOCK_TO_IDX = {}

for _i, _blk in enumerate(itertools.combinations(range(24), 7)):
    _ALL_BLOCKS.append(_blk)
    _BLOCK_TO_IDX[_blk] = _i
    for _j, _tri in enumerate(itertools.combinations(_blk, 3)):
        _BLOCK_TRIPLES[_i, _j] = _COMBO_INDEX[_tri]

_ALL_BLOCKS = tuple(_ALL_BLOCKS)


def _fast_greedy(seed, deadline):
    """Vectorized whole-block greedy set cover."""
    rng = random.Random(seed)
    uncovered = np.ones(_C24_3, dtype=bool)
    blocks = []
    
    while uncovered.any() and time.time() < deadline:
        # Compute gain for each block: number of currently uncovered triples it covers
        gains = np.sum(uncovered[_BLOCK_TRIPLES], axis=1)
        max_gain = int(np.max(gains))
        if max_gain == 0:
            break
        
        candidates = np.where(gains == max_gain)[0]
        chosen = int(candidates[rng.randint(0, len(candidates) - 1)])
        blocks.append(_ALL_BLOCKS[chosen])
        uncovered[_BLOCK_TRIPLES[chosen]] = False
    
    return blocks


def _local_search(blocks, deadline):
    """Remove redundant blocks and try pair merges."""
    if not blocks:
        return blocks
    
    block_indices = [_BLOCK_TO_IDX[b] for b in blocks]
    n_triples = _C24_3
    cov_count = np.zeros(n_triples, dtype=np.int32)
    
    for bi in block_indices:
        cov_count[_BLOCK_TRIPLES[bi]] += 1
    
    # Phase 1: Remove redundant blocks (those with no unique triples)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        new_block_indices = []
        for bi in block_indices:
            tris = _BLOCK_TRIPLES[bi]
            # If no triple in this block has coverage count 1, it's redundant
            if not np.any(cov_count[tris] == 1):
                cov_count[tris] -= 1
                changed = True
            else:
                new_block_indices.append(bi)
        
        if changed:
            block_indices = new_block_indices
    
    # Phase 2: Try to merge pairs of blocks
    rng = random.Random(999)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(block_indices)
        rng.shuffle(block_indices)
        
        for i in range(n):
            if time.time() >= deadline:
                break
            for j in range(i + 1, n):
                if time.time() >= deadline:
                    break
                
                bi, bj = block_indices[i], block_indices[j]
                tris_i = _BLOCK_TRIPLES[bi]
                tris_j = _BLOCK_TRIPLES[bj]
                
                # Find triples only covered by block i or block j
                only_i = tris_i[cov_count[tris_i] == 1]
                only_j = tris_j[cov_count[tris_j] == 1]
                
                # Points that must be in the merged block
                must_points = set()
                for ti in only_i:
                    must_points.update(_IDX_TO_COMBO[ti])
                for ti in only_j:
                    must_points.update(_IDX_TO_COMBO[ti])
                
                if len(must_points) > 7:
                    continue
                
                # Try to find a single block that covers all must_points
                # and all critical triples from both blocks
                found = False
                for candidate_idx in range(NK):
                    cand_tris = _BLOCK_TRIPLES[candidate_idx]
                    cand_set = set(_ALL_BLOCKS[candidate_idx])
                    
                    if not must_points.issubset(cand_set):
                        continue
                    
                    # Check if all critical triples from i and j are covered
                    all_ok = True
                    for ti in only_i:
                        if ti not in cand_tris:
                            all_ok = False
                            break
                    if all_ok:
                        for ti in only_j:
                            if ti not in cand_tris:
                                all_ok = False
                                break
                    
                    if not all_ok:
                        continue
                    
                    # Check that removing i and j doesn't uncover any other critical triples
                    all_ok2 = True
                    for ti in tris_i:
                        if cov_count[ti] == 1 and ti not in cand_tris:
                            all_ok2 = False
                            break
                    if all_ok2:
                        for ti in tris_j:
                            if cov_count[ti] == 1 and ti not in cand_tris:
                                all_ok2 = False
                                break
                    
                    if not all_ok2:
                        continue
                    
                    # Perform the merge
                    cov_count[tris_i] -= 1
                    cov_count[tris_j] -= 1
                    cov_count[cand_tris] += 1
                    
                    block_indices.pop(j)
                    block_indices.pop(i)
                    block_indices.append(candidate_idx)
                    
                    improved = True
                    found = True
                    break
                
                if found:
                    break
            
            if improved:
                break
    
    return [_ALL_BLOCKS[bi] for bi in block_indices]


def construct(v, k, t):
    start = time.time()
    deadline = start + 115.0
    
    if v == 24 and k == 7 and t == 3:
        best_blocks = None
        best_count = float('inf')
        
        seeds = [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000, 6000,
                 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000]
        
        for seed in seeds:
            if time.time() > deadline - 25:
                break
            
            blocks = _fast_greedy(seed, deadline - 20)
            
            if len(blocks) < best_count:
                refined = _local_search(blocks, deadline - 5)
                if len(refined) < best_count:
                    best_blocks = refined
                    best_count = len(refined)
                    
                    if best_count <= 76:
                        break
        
        if best_blocks is not None:
            return [list(b) for b in best_blocks]
    
    # Fallback
    return _generic_greedy(v, k, t, start, deadline)


def _generic_greedy(v, k, t, start, deadline):
    """Fallback for non-24-point cases."""
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    while uncovered and time.time() < deadline - 5:
        block = list(min(uncovered))
        while len(block) < k:
            def gain(x):
                return sum(1 for rest in itertools.combinations(block, t - 1)
                           if tuple(sorted(rest + (x,))) in uncovered)
            block.append(max((x for x in range(v) if x not in block), key=gain))
        block.sort()
        blocks.append(block)
        uncovered.difference_update(itertools.combinations(block, t))
    return blocks
