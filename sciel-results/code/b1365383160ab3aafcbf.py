"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import numpy as np


def _word_to_idx(word, q, n):
    """Convert a word (tuple or list) to an integer index in 0..q^n-1."""
    idx = 0
    for d in word:
        idx = idx * q + d
    return idx


def _idx_to_word(idx, q, n):
    """Convert an integer index to a word (list)."""
    word = []
    for _ in range(n):
        word.append(idx % q)
        idx //= q
    word.reverse()
    return word


def _tetracode_solution(q, n, radius):
    """Direct sum construction: 9 * q^(n-4) words."""
    if q != 3 or n < 4:
        return None
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]


def _precompute_neighbors(q, n, radius):
    """
    Precompute neighbor lists and reverse neighbor lists.
    neighbors[i]: list of indices covered by word i.
    reverse_neighbors[j]: list of indices that cover word j.
    """
    total = q ** n
    neighbors = [[] for _ in range(total)]
    reverse_neighbors = [[] for _ in range(total)]
    
    for i in range(total):
        word = _idx_to_word(i, q, n)
        covered = [i]
        
        if radius >= 1:
            for pos in range(n):
                orig_val = word[pos]
                for new_val in range(q):
                    if new_val == orig_val:
                        continue
                    new_word = word[:]
                    new_word[pos] = new_val
                    j = _word_to_idx(new_word, q, n)
                    covered.append(j)
        
        neighbors[i] = covered
        for j in covered:
            reverse_neighbors[j].append(i)
            
    return neighbors, reverse_neighbors


def _greedy_cover_efficient(q, n, radius, deadline, start_time):
    """
    Greedy set cover with O(1) updates per covered point.
    """
    total = q ** n
    neighbors, reverse_neighbors = _precompute_neighbors(q, n, radius)
    
    covered = np.zeros(total, dtype=bool)
    codewords = []
    
    uncovered_count = np.array([len(nbrs) for nbrs in neighbors], dtype=np.int64)
    
    while time.time() < deadline:
        best_idx = int(np.argmax(uncovered_count))
        best_count = int(uncovered_count[best_idx])
        
        if best_count == 0:
            break
            
        codewords.append(_idx_to_word(best_idx, q, n))
        
        for j in neighbors[best_idx]:
            if not covered[j]:
                covered[j] = True
                for i in reverse_neighbors[j]:
                    uncovered_count[i] -= 1
                    
        if np.all(covered):
            break
            
    if np.all(covered):
        return codewords
    return None


def _prune_codewords(codewords, q, n, radius, deadline, start_time):
    """
    Remove redundant codewords.
    """
    total = q ** n
    neighbors, _ = _precompute_neighbors(q, n, radius)
    
    cw_indices = [_word_to_idx(cw, q, n) for cw in codewords]
    
    cover_count = np.zeros(total, dtype=np.int64)
    for idx in cw_indices:
        for j in neighbors[idx]:
            cover_count[j] += 1
            
    keep = [True] * len(codewords)
    
    for _ in range(5):
        if time.time() > deadline:
            break
        changed = False
        for i, idx in enumerate(cw_indices):
            if not keep[i]:
                continue
            min_count = np.min(cover_count[neighbors[idx]])
            if min_count > 1:
                keep[i] = False
                changed = True
                for j in neighbors[idx]:
                    cover_count[j] -= 1
        if not changed:
            break
            
    return [codewords[i] for i in range(len(codewords)) if keep[i]]


def _local_search(codewords, q, n, radius, deadline, start_time):
    """
    Local search: try to remove or swap codewords to reduce size while maintaining coverage.
    Uses bitset operations for speed.
    """
    total = q ** n
    neighbors, reverse_neighbors = _precompute_neighbors(q, n, radius)
    
    cw_indices = [_word_to_idx(cw, q, n) for cw in codewords]
    num_cw = len(cw_indices)
    
    # cover_count[j] = number of codewords covering word j
    cover_count = np.zeros(total, dtype=np.int64)
    for idx in cw_indices:
        for j in neighbors[idx]:
            cover_count[j] += 1
    
    # Try removals
    for _ in range(10):
        if time.time() > deadline:
            break
        changed = False
        for i in range(num_cw - 1, -1, -1):
            if time.time() > deadline:
                break
            idx = cw_indices[i]
            # Check if all neighbors are covered by at least one other codeword
            min_count = np.min(cover_count[neighbors[idx]])
            if min_count > 1:
                # Remove this codeword
                for j in neighbors[idx]:
                    cover_count[j] -= 1
                cw_indices.pop(i)
                num_cw -= 1
                changed = True
        if not changed:
            break
    
    # Try swaps: remove one, add one that covers more uncovered
    for _ in range(5):
        if time.time() > deadline:
            break
        changed = False
        for i in range(num_cw):
            if time.time() > deadline:
                break
            idx = cw_indices[i]
            # Temporarily remove
            for j in neighbors[idx]:
                cover_count[j] -= 1
            
            # Find best replacement
            best_new_idx = -1
            best_gain = 0
            # Only consider candidates that cover some currently uncovered (count==0) words
            # Actually, we need to find a word that, when added, restores coverage
            # of all words that had count==0 after removal.
            
            # Words that became uncovered
            uncovered_after_removal = []
            for j in neighbors[idx]:
                if cover_count[j] == 0:
                    uncovered_after_removal.append(j)
            
            if not uncovered_after_removal:
                # No uncovered words, so we can just remove
                cw_indices.pop(i)
                num_cw -= 1
                changed = True
                continue
            
            # Find a candidate that covers all uncovered_after_removal
            # This is expensive, so limit to candidates near the removed word
            # For now, just try random candidates
            for _ in range(50):
                candidate = random.randint(0, total - 1)
                if candidate == idx:
                    continue
                # Check if candidate covers all uncovered_after_removal
                covers_all = True
                for j in uncovered_after_removal:
                    if j not in neighbors[candidate]:
                        covers_all = False
                        break
                if covers_all:
                    # Compute gain: how many additional words does it cover?
                    gain = 0
                    for j in neighbors[candidate]:
                        if cover_count[j] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_new_idx = candidate
            
            if best_new_idx >= 0:
                # Perform swap
                cw_indices[i] = best_new_idx
                for j in neighbors[best_new_idx]:
                    cover_count[j] += 1
                changed = True
            else:
                # Restore
                for j in neighbors[idx]:
                    cover_count[j] += 1
        
        if not changed:
            break
    
    return [_idx_to_word(idx, q, n) for idx in cw_indices]


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0
    
    tc = _tetracode_solution(q, n, radius)
    if tc is not None:
        best = tc
        best_len = len(tc)
    else:
        best = []
        best_len = float('inf')
    
    try:
        greedy_result = _greedy_cover_efficient(q, n, radius, deadline - 30.0, start_time)
        if greedy_result is not None:
            if len(greedy_result) < best_len:
                best = greedy_result
                best_len = len(best)
                
                if time.time() < deadline - 15.0:
                    pruned = _prune_codewords(best, q, n, radius, deadline - 10.0, start_time)
                    if pruned and len(pruned) < best_len:
                        total = q ** n
                        covered = np.zeros(total, dtype=bool)
                        for cw in pruned:
                            idx = _word_to_idx(cw, q, n)
                            word = _idx_to_word(idx, q, n)
                            covered[idx] = True
                            for pos in range(n):
                                orig_val = word[pos]
                                for new_val in range(q):
                                    if new_val == orig_val:
                                        continue
                                    new_word = word[:]
                                    new_word[pos] = new_val
                                    j = _word_to_idx(new_word, q, n)
                                    covered[j] = True
                            
                        if np.all(covered):
                            best = pruned
                            best_len = len(best)
                
                # Local search
                if time.time() < deadline - 5.0:
                    local_result = _local_search(best, q, n, radius, deadline - 2.0, start_time)
                    if local_result and len(local_result) < best_len:
                        total = q ** n
                        covered = np.zeros(total, dtype=bool)
                        for cw in local_result:
                            idx = _word_to_idx(cw, q, n)
                            word = _idx_to_word(idx, q, n)
                            covered[idx] = True
                            for pos in range(n):
                                orig_val = word[pos]
                                for new_val in range(q):
                                    if new_val == orig_val:
                                        continue
                                    new_word = word[:]
                                    new_word[pos] = new_val
                                    j = _word_to_idx(new_word, q, n)
                                    covered[j] = True
                            
                        if np.all(covered):
                            best = local_result
                            best_len = len(best)
    except Exception as e:
        pass
        
    return best
# EVOLVE-BLOCK-END
