"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np


def _ball_indices(word, q, n, radius):
    """Return the flat indices of all words within Hamming distance `radius` of `word`."""
    # Precompute all words within radius via itertools
    idxs = []
    # radius 1: word itself + 1 position * (q-1) changes
    if radius == 1:
        idxs.append(int(np.ravel_multi_index(tuple(word), (q,)*n)))
        for pos in range(n):
            for v in range(q):
                if v == word[pos]:
                    continue
                w2 = list(word)
                w2[pos] = v
                idxs.append(int(np.ravel_multi_index(tuple(w2), (q,)*n)))
    return idxs


def _tetracode(q, n, radius):
    """Direct product: tetracode on first 4 coords, all words on remaining n-4 coords."""
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]


def _greedy_cover_fast(q, n, radius, deadline):
    """
    Greedy set cover: at each step pick the codeword whose ball covers the most
    currently-uncovered words. Use numpy bitsets for speed.
    """
    total = q ** n
    # Precompute balls as sorted lists of flat indices
    # For n=6, q=3: 729 words, each ball has 1 + 6*2 = 13 indices
    all_words = [list(w) for w in itertools.product(range(q), repeat=n)]
    balls = [_ball_indices(w, q, n, radius) for w in all_words]
    
    # uncovered: boolean array
    uncovered = np.ones(total, dtype=np.uint8)
    uncovered_count = total
    chosen = []
    
    # Precompute ball sets as frozensets for fast intersection, or use numpy
    # For 729 words and 13-element balls, use a different approach:
    # For each candidate, compute how many of its ball positions are still uncovered
    
    # Use a list of sets for uncovered tracking
    uncovered_set = set(range(total))
    
    while uncovered_set and time.time() < deadline:
        best_cand = -1
        best_count = 0
        for i, ball in enumerate(balls):
            # count how many in ball are still uncovered
            cnt = sum(1 for idx in ball if idx in uncovered_set)
            if cnt > best_count:
                best_count = cnt
                best_cand = i
        if best_cand < 0:
            break
        chosen.append(all_words[best_cand])
        # Remove covered
        for idx in balls[best_cand]:
            uncovered_set.discard(idx)
    
    return chosen


def _remove_redundant(codewords, q, n, radius):
    """Try to remove each codeword; if the covering property still holds, keep it removed."""
    total = q ** n
    # Build coverage: for each word, which codewords cover it
    # For small spaces, just check coverage directly
    chosen = list(codewords)
    changed = True
    while changed:
        changed = False
        for i in range(len(chosen)):
            # Try removing chosen[i]
            remaining = chosen[:i] + chosen[i+1:]
            # Check if all words are still covered
            if not remaining:
                # Can't remove if it's the only one (unless n=0)
                continue
            # Quick check: build union of balls
            covered = set()
            for cw in remaining:
                for idx in _ball_indices(cw, q, n, radius):
                    covered.add(idx)
            if len(covered) == total:
                chosen = remaining
                changed = True
                break
    return chosen


def _construct_k3_6_1(q, n, radius, deadline):
    """
    Specialized construction for q=3, n=6, radius=1.
    Start with the 81-word direct product, then greedily remove redundant words
    using a smarter ordering and swap-based local search.
    """
    # Start with tetracode product (81 words)
    codewords = _tetracode(q, n, radius)
    
    total = q ** n
    all_words = list(itertools.product(range(q), repeat=n))
    all_words_set = set(all_words)
    
    # Precompute balls for all 729 possible words as numpy arrays
    balls_cache = {}
    for w in all_words:
        balls_cache[w] = np.array(_ball_indices(list(w), q, n, radius), dtype=np.int32)
    
    # Build coverage counts using numpy
    cover_count = np.zeros(total, dtype=np.int32)
    for cw in codewords:
        idxs = balls_cache[tuple(cw)]
        cover_count[idxs] += 1
    
    # Helper: count how many words in a ball have cover_count == 1 (uniquely covered)
    def _unique_count(cw_tuple):
        idxs = balls_cache[cw_tuple]
        return np.sum(cover_count[idxs] == 1)
    
    # Phase 1: Greedy removal with smart ordering
    # Sort codewords by number of uniquely-covered words (ascending)
    # Remove those with 0 unique words first
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Compute unique counts for all current codewords
        unique_counts = []
        for i, cw in enumerate(codewords):
            cw_t = tuple(cw)
            uc = _unique_count(cw_t)
            unique_counts.append((uc, i))
        
        # Sort by unique count ascending, then by index
        unique_counts.sort()
        
        # Try removing in order of fewest unique words
        for uc, i in unique_counts:
            if time.time() > deadline:
                return codewords
            if uc > 0:
                break  # No more removable codewords in this pass
            cw = tuple(codewords[i])
            idxs = balls_cache[cw]
            # Verify all have cover_count > 1
            if np.all(cover_count[idxs] > 1):
                cover_count[idxs] -= 1
                codewords.pop(i)
                changed = True
                break  # Restart with updated counts
    
    # Phase 2: Swap-based local search
    # Try replacing one codeword with another to enable further reductions
    # or to find a smaller equivalent code
    best_size = len(codewords)
    
    # Precompute all possible words not currently in codewords
    current_set = set(tuple(cw) for cw in codewords)
    candidate_words = [w for w in all_words if w not in current_set]
    
    # For each codeword, try swapping with candidates
    # Check if swap maintains coverage: 
    # - Words only covered by old codeword must be covered by new codeword
    # - Words covered by new codeword but not old are fine (they were covered before)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_set = set(tuple(cw) for cw in codewords)
        
        # Recompute cover_count
        cover_count[:] = 0
        for cw in codewords:
            idxs = balls_cache[tuple(cw)]
            cover_count[idxs] += 1
        
        for i in range(len(codewords)):
            if time.time() > deadline:
                return codewords
            
            old_cw = tuple(codewords[i])
            old_idxs = balls_cache[old_cw]
            
            # Find words uniquely covered by old_cw
            unique_mask = cover_count[old_idxs] == 1
            unique_words = old_idxs[unique_mask]
            
            if len(unique_words) == 0:
                # Can remove instead of swap
                cover_count[old_idxs] -= 1
                codewords.pop(i)
                improved = True
                break
            
            # Try swapping with candidates that cover all unique_words
            # For efficiency, only try a subset of candidates
            # Score candidates by how many of their ball words are already well-covered
            best_swap = None
            best_swap_score = -1
            
            # Sample candidates or use all if feasible
            # For 729 words, trying all candidates for each swap is expensive
            # Use a heuristic: try candidates that are close to old_cw
            for cand in candidate_words:
                if time.time() > deadline:
                    return codewords
                
                cand_idxs = balls_cache[cand]
                
                # Check if cand covers all unique_words
                # unique_words are flat indices, cand_idxs are flat indices
                # Check intersection
                if not np.isin(unique_words, cand_idxs).all():
                    continue
                
                # Score: prefer candidates that cover words with low cover_count
                # (i.e., words that are barely covered)
                # After swap, cover_count changes:
                # - old_idxs: cover_count -= 1
                # - cand_idxs: cover_count += 1
                # We want to avoid creating new unique words
                # Estimate: count how many words in cand_idxs would become unique
                # after the swap (i.e., cover_count[cand_idxs] == 1 and not in old_idxs)
                
                # Words in cand_idxs that are not in old_idxs
                new_words_mask = ~np.isin(cand_idxs, old_idxs)
                new_words = cand_idxs[new_words_mask]
                
                # These words will have cover_count increased by 1
                # They become "more covered" which is good
                # Words in old_idxs not in cand_idxs will have cover_count decreased by 1
                # These might become unique (bad)
                lost_words_mask = ~np.isin(old_idxs, cand_idxs)
                lost_words = old_idxs[lost_words_mask]
                
                # For lost_words, their new cover_count is cover_count[lost_words] - 1
                # They become unique if cover_count[lost_words] - 1 == 1, i.e., cover_count[lost_words] == 2
                new_unique_count = np.sum(cover_count[lost_words] == 2)
                
                # Prefer swaps that create fewer new unique words
                score = -new_unique_count
                if score > best_swap_score:
                    best_swap_score = score
                    best_swap = cand
            
            if best_swap is not None:
                # Perform the swap
                old_idxs = balls_cache[old_cw]
                new_idxs = balls_cache[best_swap]
                
                cover_count[old_idxs] -= 1
                cover_count[new_idxs] += 1
                
                codewords[i] = list(best_swap)
                improved = True
                break
    
    return codewords


def construct(q, n, radius):
    start_time = time.time()
    # Reserve 10s safety margin
    deadline = start_time + 110.0
    
    if q == 3 and n == 6 and radius == 1:
        result = _construct_k3_6_1(q, n, radius, deadline)
        # Final verification and cleanup
        if result is None:
            return _tetracode(q, n, radius)
        return result
    else:
        # Fallback: direct product construction
        return _tetracode(q, n, radius)
# EVOLVE-BLOCK-END
