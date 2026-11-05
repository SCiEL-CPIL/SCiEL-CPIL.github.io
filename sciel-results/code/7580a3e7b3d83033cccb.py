"""Covering design C(v,k,t) solver with greedy + local search (1-for-1 swaps and pruning)."""
import time
import itertools
import numpy as np
import random

def construct(v, k, t):
    deadline = time.time() + 140  # safety margin within 150s
    
    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_index = {s: i for i, s in enumerate(all_tsubs)}
    
    # Greedy construction
    uncovered = np.ones(n_tsubs, dtype=np.int32)
    n_uncovered = n_tsubs
    blocks = []
    
    while n_uncovered > 0:
        if time.time() > deadline:
            break
        
        # Find first uncovered t-subset
        unc_idx = np.where(uncovered == 1)[0]
        if len(unc_idx) == 0:
            break
        anchor = list(all_tsubs[unc_idx[0]])
        block = list(anchor)
        
        while len(block) < k:
            if time.time() > deadline:
                break
            block_set = set(block)
            b = len(block)
            
            best_x = None
            best_gain = -1
            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                if b >= t - 1:
                    for comb in itertools.combinations(block, t - 1):
                        s = tuple(sorted(comb + (x,)))
                        idx = tsub_index.get(s)
                        if idx is not None and uncovered[idx] == 1:
                            gain += 1
                else:
                    # Can't cover new t-subs yet, just pick arbitrarily
                    gain = 0
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                best_x = next(x for x in range(v) if x not in block_set)
            block.append(best_x)
        
        if len(block) < k:
            break
        
        block.sort()
        blocks.append(block)
        
        for s in itertools.combinations(block, t):
            idx = tsub_index[s]
            if uncovered[idx] == 1:
                uncovered[idx] = 0
                n_uncovered -= 1
    
    if n_uncovered > 0:
        # Fallback: add blocks to cover remaining
        while n_uncovered > 0:
            unc_idx = np.where(uncovered == 1)[0]
            if len(unc_idx) == 0:
                break
            anchor = list(all_tsubs[unc_idx[0]])
            # Complete to k-subset
            remaining = [x for x in range(v) if x not in set(anchor)]
            random.shuffle(remaining)
            block = anchor + remaining[:k - t]
            block.sort()
            blocks.append(block)
            for s in itertools.combinations(block, t):
                idx = tsub_index[s]
                if uncovered[idx] == 1:
                    uncovered[idx] = 0
                    n_uncovered -= 1
    
    # Local search phase
    if time.time() > deadline:
        return blocks
    
    n_blocks = len(blocks)
    
    # Precompute: for each t-subset, which blocks cover it
    # coverage[i] = set of block indices that cover t-subset i
    # Use a list of lists for efficiency
    coverage = [[] for _ in range(n_tsubs)]
    for bi, block in enumerate(blocks):
        for s in itertools.combinations(block, t):
            idx = tsub_index[s]
            coverage[idx].append(bi)
    
    # Helper: can we remove block bi?
    # A block can be removed if for every t-subset it covers, len(coverage[idx]) > 1
    def can_remove(bi, blocks_list, coverage_list):
        for s in itertools.combinations(blocks_list[bi], t):
            idx = tsub_index[s]
            if len(coverage_list[idx]) <= 1:
                return False
        return True
    
    # Try to remove redundant blocks
    changed = True
    while changed and time.time() < deadline - 5:
        changed = False
        for bi in range(len(blocks)):
            if can_remove(bi, blocks, coverage):
                # Remove block bi
                for s in itertools.combinations(blocks[bi], t):
                    idx = tsub_index[s]
                    coverage[idx].remove(bi)
                blocks.pop(bi)
                # Reindex coverage
                for i in range(n_tsubs):
                    cov = coverage[i]
                    new_cov = []
                    for j in cov:
                        if j < bi:
                            new_cov.append(j)
                        else:
                            new_cov.append(j - 1)
                    coverage[i] = new_cov
                changed = True
                break
    
    # Local search: 1-for-1 swaps
    # Try replacing block i with a new random block
    if time.time() > deadline - 5:
        return blocks
    
    rng = random.Random(42)
    
    # Precompute all k-subsets? C(19,7) = 50388, manageable
    all_ksubs = list(itertools.combinations(range(v), k))
    
    # For each existing block, precompute which t-subsets it covers
    def get_tsubs_of_block(block):
        return [tsub_index[s] for s in itertools.combinations(block, t)]
    
    block_tsubs = [get_tsubs_of_block(b) for b in blocks]
    
    # Try swaps
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        n_blocks = len(blocks)
        for bi in range(n_blocks):
            if time.time() > deadline - 5:
                break
            # Try to find a replacement block
            # The replacement must cover all t-subsets in block bi that are ONLY covered by block bi
            exclusive = set()
            for idx in block_tsubs[bi]:
                if len(coverage[idx]) == 1:
                    exclusive.add(idx)
            
            if not exclusive:
                # Can remove, but we already tried that
                continue
            
            # Try random candidate blocks
            candidates = rng.sample(range(len(all_ksubs)), min(200, len(all_ksubs)))
            found = False
            for ci in candidates:
                cand = all_ksubs[ci]
                # Check if cand covers all exclusive t-subsets
                ok = True
                cand_set = set(cand)
                for idx in exclusive:
                    s = all_tsubs[idx]
                    if not all(p in cand_set for p in s):
                        ok = False
                        break
                if not ok:
                    continue
                
                # Also check: after replacing bi with cand, do we still cover everything?
                # For t-subsets in cand that weren't covered before, that's fine (adds coverage)
                # For t-subsets in bi that are covered by others, that's fine
                # The only risk: t-subsets in bi that are NOT in cand and NOT covered by others
                # But we already checked exclusive t-subsets are in cand
                # What about t-subsets in bi that are covered by others? They're still covered.
                # What about t-subsets in bi that are covered by bi AND others? Still covered by others.
                # So the only requirement is: all exclusive t-subsets of bi are in cand.
                
                # Replace
                old_block = blocks[bi]
                old_tsubs = block_tsubs[bi]
                new_tsubs = get_tsubs_of_block(cand)
                
                # Update coverage
                for idx in old_tsubs:
                    coverage[idx].remove(bi)
                for idx in new_tsubs:
                    coverage[idx].append(bi)
                
                blocks[bi] = list(cand)
                block_tsubs[bi] = new_tsubs
                improved = True
                found = True
                break
            
            if found:
                break
    
    return blocks
