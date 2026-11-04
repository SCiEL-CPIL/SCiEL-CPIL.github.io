"""Covering C(19,8,5): randomized minimal cover via iterative removal of non-essential blocks."""
import numpy as np
import itertools
import random

def construct(v, k, t):
    # Precompute all t-subsets and map to indices
    t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(t_subsets)
    t_index = {s: i for i, s in enumerate(t_subsets)}
    
    # Precompute all k-blocks
    k_blocks = list(itertools.combinations(range(v), k))
    n_k = len(k_blocks)
    
    # For each block, compute the list of t-subset indices it covers
    block_t_list = []
    for block in k_blocks:
        covered = [t_index[ts] for ts in itertools.combinations(block, t)]
        block_t_list.append(covered)
    
    # Coverage counts: how many remaining blocks cover each t-subset
    coverage = np.zeros(n_t, dtype=np.int32)
    for covered in block_t_list:
        for idx in covered:
            coverage[idx] += 1
    
    # Active flags
    active = np.ones(n_k, dtype=bool)
    
    # For each t-subset, maintain list of blocks that cover it
    # blocks_for_s[s] = list of block indices that cover t-subset s
    blocks_for_s = [[] for _ in range(n_t)]
    for i in range(n_k):
        for idx in block_t_list[i]:
            blocks_for_s[idx].append(i)
    
    # Convert to arrays for efficiency
    for s in range(n_t):
        blocks_for_s[s] = np.array(blocks_for_s[s], dtype=np.int32)
    
    # Randomized removal: repeatedly pick a random active block and remove it
    # if it is non-essential (all its t-subsets have coverage >= 2)
    # To make this efficient, we use a queue of candidates.
    
    # Initial: all blocks are candidates
    # We'll use a list as a stack/queue
    candidates = list(range(n_k))
    random.shuffle(candidates)
    
    removed = set()
    
    while candidates:
        i = candidates.pop()
        if i in removed or not active[i]:
            continue
        
        # Check if block i is removable: all its t-subsets have coverage >= 2
        covered = block_t_list[i]
        removable = True
        for idx in covered:
            if coverage[idx] < 2:
                removable = False
                break
        
        if removable:
            # Remove block i
            active[i] = False
            removed.add(i)
            
            # Update coverage and add newly non-essential blocks to candidates
            for idx in covered:
                coverage[idx] -= 1
                if coverage[idx] == 1:
                    # The remaining block covering this t-subset must be kept
                    # But other blocks that also covered this t-subset may now be removable
                    # Add all blocks covering this t-subset (except i) to candidates
                    for j in blocks_for_s[idx]:
                        if j != i and j not in removed and active[j]:
                            candidates.append(j)
    
    # Collect remaining blocks
    result = [list(k_blocks[i]) for i in range(n_k) if active[i]]
    
    # Verify coverage
    if n_t > 0:
        if np.any(coverage < 1):
            # Fallback: greedy add blocks to cover uncovered t-subsets
            uncovered = set()
            for idx in range(n_t):
                if coverage[idx] < 1:
                    uncovered.add(t_subsets[idx])
            
            while uncovered:
                best_block = None
                best_gain = -1
                for i in range(n_k):
                    covered = block_t_list[i]
                    gain = sum(1 for idx in covered if t_subsets[idx] in uncovered)
                    if gain > best_gain:
                        best_gain = gain
                        best_block = i
                if best_block is None or best_gain <= 0:
                    break
                result.append(list(k_blocks[best_block]))
                for idx in block_t_list[best_block]:
                    s = t_subsets[idx]
                    if s in uncovered:
                        uncovered.discard(s)
                    coverage[idx] += 1
            result.sort()
    
    return result
