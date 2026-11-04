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


def _build_neighbor_masks(q, n, radius):
    """
    Precompute neighbor masks for all words in the space.
    Returns a numpy array of shape (total, total) with dtype=bool,
    where neighbor_masks[i] is a boolean vector indicating which words
    are within Hamming distance <= radius from word i.
    """
    total = q ** n
    masks = np.zeros((total, total), dtype=bool)
    
    for idx in range(total):
        word = _idx_to_word(idx, q, n)
        masks[idx, idx] = True
        if radius >= 1:
            for pos in range(n):
                orig_val = word[pos]
                for new_val in range(q):
                    if new_val == orig_val:
                        continue
                    new_word = word[:]
                    new_word[pos] = new_val
                    new_idx = _word_to_idx(new_word, q, n)
                    masks[idx, new_idx] = True
    return masks


def _tetracode_solution(q, n, radius):
    """Direct sum construction: 9 * q^(n-4) words."""
    if q != 3 or n < 4:
        return None
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]


def _verify_coverage(codewords, q, n, radius, neighbor_masks):
    """Verify that all words are covered by the given codewords."""
    total = q ** n
    covered = np.zeros(total, dtype=bool)
    for cw in codewords:
        idx = _word_to_idx(cw, q, n)
        covered |= neighbor_masks[idx]
    return bool(np.all(covered))


def _greedy_cover(q, n, radius, deadline, start_time, neighbor_masks=None):
    """
    Greedy set cover with efficient bitset operations.
    Returns a list of codewords (lists of ints) or None if timeout.
    """
    total = q ** n
    
    if neighbor_masks is None:
        neighbor_masks = _build_neighbor_masks(q, n, radius)
    
    covered = np.zeros(total, dtype=bool)
    codewords = []
    
    # uncovered_count[i] = number of currently uncovered words that codeword i covers
    uncovered_count = np.array([np.sum(~covered & nm) for nm in neighbor_masks], dtype=np.int64)
    
    while time.time() < deadline:
        # Find the candidate with max uncovered_count
        best_idx = int(np.argmax(uncovered_count))
        best_count = int(uncovered_count[best_idx])
        
        if best_count == 0:
            break
        
        # Add best_idx as a codeword
        codewords.append(_idx_to_word(best_idx, q, n))
        
        # Update covered
        covered |= neighbor_masks[best_idx]
        
        # Check if all covered
        if np.all(covered):
            break
        
        # Update uncovered_count: recount uncovered words each candidate covers
        # Optimization: only update candidates that could be affected
        # (those that cover any of the newly covered words)
        newly_covered = np.where(~covered & neighbor_masks[best_idx])[0]
        if len(newly_covered) > 0:
            # For each candidate, subtract the number of newly covered words it covers
            # We can vectorize this: for each newly covered word j, find all candidates i
            # where neighbor_masks[i, j] is True, and decrement uncovered_count[i]
            # This is equivalent to: uncovered_count -= neighbor_masks[:, newly_covered].sum(axis=1)
            # But that's expensive. Instead, just recompute for candidates that matter.
            # Actually, the simplest correct approach: recompute all.
            # But that's O(total^2) per step. Let's use a smarter update.
            
            # For each newly covered word, decrement count for all candidates covering it
            # neighbor_masks[:, newly_covered] has shape (total, len(newly_covered))
            # Sum over axis=1 gives how many newly covered words each candidate covers
            decrement = neighbor_masks[:, newly_covered].sum(axis=1).astype(np.int64)
            uncovered_count -= decrement
    
    if np.all(covered):
        return codewords
    
    return None


def _greedy_prune(codewords, q, n, radius, deadline, start_time, neighbor_masks=None):
    """Remove redundant codewords."""
    total = q ** n
    
    if neighbor_masks is None:
        neighbor_masks = _build_neighbor_masks(q, n, radius)
    
    # For each word, count how many codewords cover it
    word_cover_count = np.zeros(total, dtype=np.int64)
    cover_idx_list = []
    for cw in codewords:
        idx = _word_to_idx(cw, q, n)
        cover_idx_list.append(idx)
        word_cover_count += neighbor_masks[idx].astype(np.int64)
    
    keep = [True] * len(codewords)
    for i in range(len(codewords)):
        if not keep[i]:
            continue
        idx = cover_idx_list[i]
        # Codeword i is removable if all words it covers are covered by at least one other codeword
        covered_by_others = word_cover_count[neighbor_masks[idx]]
        if np.all(covered_by_others > 1):
            keep[i] = False
            # Update word_cover_count
            word_cover_count[neighbor_masks[idx]] -= 1
    
    return [codewords[i] for i in range(len(codewords)) if keep[i]]


def _simulated_annealing_refine(codewords, q, n, radius, deadline, start_time, neighbor_masks=None):
    """
    Simulated annealing to escape local minima in the code size.
    Tries swaps (remove one, add one) and removals, accepting worse solutions
    with probability based on a cooling schedule.
    """
    total = q ** n
    
    if neighbor_masks is None:
        neighbor_masks = _build_neighbor_masks(q, n, radius)
    
    # Work with indices for efficiency
    cw_indices = [_word_to_idx(cw, q, n) for cw in codewords]
    current_size = len(cw_indices)
    best_indices = cw_indices[:]
    best_size = current_size
    
    # Temperature schedule
    initial_temp = 1.0
    final_temp = 0.001
    num_iterations = 5000  # Adjust based on time
    cooling_rate = (final_temp / initial_temp) ** (1.0 / max(1, num_iterations - 1))
    
    temp = initial_temp
    rng = random.Random(42)
    
    for iter_num in range(num_iterations):
        if time.time() >= deadline:
            break
        
        # Choose a random move
        move_type = rng.random()
        
        if move_type < 0.7 and current_size > 0:
            # Try removal: remove a random codeword
            if current_size <= 1:
                continue
            remove_idx = rng.randrange(current_size)
            new_indices = cw_indices[:remove_idx] + cw_indices[remove_idx+1:]
            new_size = current_size - 1
            
            # Check if still covers everything
            covered = np.zeros(total, dtype=bool)
            for idx in new_indices:
                covered |= neighbor_masks[idx]
            
            if np.all(covered):
                # Accept (improvement)
                cw_indices = new_indices
                current_size = new_size
                if current_size < best_size:
                    best_indices = cw_indices[:]
                    best_size = current_size
            else:
                # Uncovered count
                uncovered = int(np.sum(~covered))
                # Accept with probability exp(-uncovered / temp)
                if uncovered > 0 and temp > 0:
                    prob = np.exp(-uncovered / max(temp, 1e-10))
                    if rng.random() < prob:
                        cw_indices = new_indices
                        current_size = new_size
                        if current_size < best_size:
                            best_indices = cw_indices[:]
                            best_size = current_size
        elif move_type < 0.9:
            # Try swap: remove one, add one
            if current_size == 0:
                continue
            remove_idx = rng.randrange(current_size)
            add_idx = rng.randrange(total)
            
            # Avoid adding the same one we're removing
            if add_idx == cw_indices[remove_idx]:
                continue
            
            new_indices = cw_indices[:remove_idx] + cw_indices[remove_idx+1:] + [add_idx]
            new_size = current_size  # Same size
            
            covered = np.zeros(total, dtype=bool)
            for idx in new_indices:
                covered |= neighbor_masks[idx]
            
            if np.all(covered):
                # Accept (same size, but might enable future removals)
                cw_indices = new_indices
                current_size = new_size
            else:
                uncovered = int(np.sum(~covered))
                if uncovered > 0 and temp > 0:
                    prob = np.exp(-uncovered / max(temp, 1e-10))
                    if rng.random() < prob:
                        cw_indices = new_indices
                        current_size = new_size
        else:
            # Try addition (to escape a bad state)
            add_idx = rng.randrange(total)
            if add_idx in cw_indices:
                continue
            new_indices = cw_indices + [add_idx]
            new_size = current_size + 1
            
            covered = np.zeros(total, dtype=bool)
            for idx in new_indices:
                covered |= neighbor_masks[idx]
            
            # Addition always covers more, so it's never worse in coverage
            # But it increases size, so accept with low probability
            if temp > 0:
                prob = np.exp(-1.0 / max(temp, 1e-10))
                if rng.random() < prob:
                    cw_indices = new_indices
                    current_size = new_size
        
        # Cool down
        temp *= cooling_rate
    
    # Convert back to codewords
    return [_idx_to_word(idx, q, n) for idx in best_indices]


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Precompute neighbor masks once
    neighbor_masks = _build_neighbor_masks(q, n, radius)
    
    # Try tetracode first as a baseline
    tc = _tetracode_solution(q, n, radius)
    if tc is not None:
        best = tc
        best_len = len(tc)
    else:
        best = []
        best_len = float('inf')
    
    # Try greedy cover
    try:
        greedy_result = _greedy_cover(q, n, radius, deadline, start_time, neighbor_masks)
        if greedy_result is not None:
            # Verify coverage
            if _verify_coverage(greedy_result, q, n, radius, neighbor_masks) and len(greedy_result) < best_len:
                best = greedy_result
                best_len = len(best)
                # Try to prune
                if time.time() < deadline - 10.0:
                    pruned = _greedy_prune(best, q, n, radius, deadline, start_time, neighbor_masks)
                    if pruned:
                        if _verify_coverage(pruned, q, n, radius, neighbor_masks) and len(pruned) < best_len:
                            best = pruned
                            best_len = len(best)
    except Exception:
        pass
    
    # Try simulated annealing refinement if we have time
    if best and time.time() < deadline - 20.0:
        try:
            sa_result = _simulated_annealing_refine(best, q, n, radius, deadline, start_time, neighbor_masks)
            if sa_result:
                # Verify and prune
                sa_pruned = _greedy_prune(sa_result, q, n, radius, deadline, start_time, neighbor_masks)
                if sa_pruned and _verify_coverage(sa_pruned, q, n, radius, neighbor_masks) and len(sa_pruned) < best_len:
                    best = sa_pruned
                    best_len = len(best)
        except Exception:
            pass
    
    # Final fallback: if best is empty, use tetracode
    if not best:
        if tc is not None:
            best = tc
        else:
            # Return an empty list (will be invalid, but better than crashing)
            best = []
    
    return best
# EVOLVE-BLOCK-END
