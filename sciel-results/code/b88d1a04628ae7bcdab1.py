import time
import itertools
import numpy as np
import random

def _classic_superperm(n):
    """Classic recursive superpermutation construction."""
    if n == 1:
        return [1]
    word = _classic_superperm(n - 1)
    m = n
    perms, seen = [], set()
    # Extract (n-1)-permutations in order of appearance
    for i in range(len(word) - m + 2):
        window = tuple(word[i:i + m - 1])
        if len(set(window)) == m - 1 and window not in seen:
            seen.add(window)
            perms.append(window)
            
    new = []
    for p in perms:
        block = list(p) + [m] + list(p)
        # Find max overlap with end of new
        overlap = 0
        max_ov = min(len(new), len(block))
        for o in range(max_ov, 0, -1):
            if new[len(new) - o:] == block[:o]:
                overlap = o
                break
        new += block[overlap:]
    return new

def _build_level_perms(n):
    """
    Build the list of (n-1)-length permutations in the order they appear
    in the classic superpermutation of level n-1.
    """
    if n == 1:
        return [[1]]
    word = _classic_superperm(n - 1)
    perms, seen = [], set()
    m = n - 1
    for i in range(len(word) - m + 1):
        window = tuple(word[i:i + m])
        if len(set(window)) == m and window not in seen:
            seen.add(window)
            perms.append(window)
    return [list(p) for p in perms]

def _compute_overlap_matrix(blocks):
    """
    Compute the overlap matrix for a list of blocks.
    ov[i][j] is the maximum overlap of suffix of blocks[i] and prefix of blocks[j].
    """
    n_blocks = len(blocks)
    block_len = len(blocks[0])
    # Overlap can be at most block_len - 1 (since blocks are distinct, full overlap impossible unless identical)
    # Actually, if blocks are distinct permutations, max overlap is block_len - 1?
    # Block length is 2*n - 1.
    # Max overlap is block_len - 1.
    
    ov = np.zeros((n_blocks, n_blocks), dtype=np.int8)
    
    # We can speed this up by checking only up to max possible overlap.
    # But 720x720 = 518,400 pairs.
    # Each check is O(L). L=13.
    # Total ops ~ 6.7M. This is fast in NumPy if vectorized, or even in Python if optimized.
    
    for i in range(n_blocks):
        b_i = blocks[i]
        for j in range(n_blocks):
            if i == j:
                continue
            b_j = blocks[j]
            # Max overlap is min(len(b_i), len(b_j)) - 1 = 12
            # We check from 12 down to 0
            # b_i[-k:] == b_j[:k]
            for k in range(12, 0, -1):
                if b_i[-k:] == b_j[:k]:
                    ov[i][j] = k
                    break
    return ov

def _optimize_order_sa(blocks, ov, time_limit=100.0):
    """
    Simulated Annealing to optimize the order of blocks.
    """
    start_time = time.time()
    n_blocks = len(blocks)
    
    # Initial order: 0, 1, ..., N-1
    order = np.arange(n_blocks)
    
    # Calculate initial cost (total length)
    # Length = len(block[0]) + sum(len(block[k]) - ov[order[k-1]][order[k]])
    # Since len is constant L, minimize cost = L*N - sum(ov)
    # Maximize sum(ov)
    
    def get_overlap_sum(ord):
        # ord is a numpy array of indices
        # sum ov[ord[k-1], ord[k]] for k=1..N-1
        # We can use np.sum(ov[np.ix_(ord[:-1], ord[1:])])
        return np.sum(ov[np.ix_(ord[:-1], ord[1:])])
    
    best_order = order.copy()
    best_score = get_overlap_sum(best_order)
    current_order = order.copy()
    current_score = best_score
    
    T = 10.0
    T_min = 0.01
    alpha = 0.999
    
    while T > T_min and (time.time() - start_time) < time_limit:
        # Random move: 2-opt (reverse segment)
        i = np.random.randint(0, n_blocks - 1)
        j = np.random.randint(i + 1, n_blocks)
        
        # Calculate delta score
        # Old edges: (i-1, i), (i, i+1), ..., (j-1, j), (j, j+1)
        # New edges: (i-1, j), (j, j-1), ..., (i+1, i), (i, j+1)
        
        old_score = 0
        new_score = 0
        
        # Helper to get edge weight
        # Note: if i=0, no left edge. If j=N-1, no right edge.
        
        # Left boundary
        if i > 0:
            old_score += ov[current_order[i-1], current_order[i]]
            new_score += ov[current_order[i-1], current_order[j]]
        
        # Right boundary
        if j < n_blocks - 1:
            old_score += ov[current_order[j], current_order[j+1]]
            new_score += ov[current_order[i], current_order[j+1]]
            
        # Internal edges
        # Old: sum_{k=i}^{j-1} ov[current_order[k], current_order[k+1]]
        # New: sum_{k=i}^{j-1} ov[current_order[k+1], current_order[k]]
        
        # Extract segment
        seg = current_order[i:j+1]
        
        # Old internal
        old_internal = np.sum(ov[np.ix_(seg[:-1], seg[1:])])
        # New internal (reversed)
        # ov[seg[k+1], seg[k]]
        new_internal = np.sum(ov[np.ix_(seg[1:], seg[:-1])])
        
        old_score += old_internal
        new_score += new_internal
        
        delta = new_score - old_score
        
        if delta > 0 or np.random.rand() < np.exp(delta / T):
            # Accept
            current_order[i:j+1] = seg[::-1]
            current_score += delta
            
            if current_score > best_score:
                best_score = current_score
                best_order = current_order.copy()
                
        T *= alpha
        
    return best_order

def construct(n):
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        # Classic length 19
        word = _classic_superperm(n)
        return "".join(map(str, word))
    if n == 5:
        word = _classic_superperm(n)
        return "".join(map(str, word))
    if n == 6:
        word = _classic_superperm(n)
        return "".join(map(str, word))
        
    if n == 7:
        perm_list = _build_level_perms(n)
        m = n
        n_perms = len(perm_list)
        
        # Get classic word for fallback
        classic_word = _classic_superperm(n)
        classic_len = len(classic_word)
        
        # Build blocks
        blocks = [list(p) + [m] + list(p) for p in perm_list]
        
        # Compute overlap matrix
        ov = _compute_overlap_matrix(blocks)
        
        # Optimize
        time_left = deadline - (time.time() - start_time)
        if time_left > 10.0:
            best_order = _optimize_order_sa(blocks, ov, time_limit=time_left - 5.0)
            
            # Build word from best_order
            word = list(blocks[best_order[0]])
            for k in range(1, len(best_order)):
                b = blocks[best_order[k]]
                # Find overlap with end of word
                # We can use the precomputed ov matrix for the overlap between blocks,
                # but we need the overlap between the *current word suffix* and the *block prefix*.
                # Since the word is built by concatenating blocks with max overlap,
                # the suffix of the word is exactly the suffix of the previous block.
                # So the overlap is ov[prev_block_idx, curr_block_idx].
                prev_idx = best_order[k-1]
                curr_idx = best_order[k]
                ov_val = ov[prev_idx, curr_idx]
                word += b[ov_val:]
                
            if len(word) < classic_len:
                return "".join(map(str, word))
        
        return "".join(map(str, classic_word))
        
    # For n > 7, just use classic
    word = _classic_superperm(n)
    return "".join(map(str, word))
