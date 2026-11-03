"""Construct a ternary covering code K_3(6,1) using greedy + local search."""
import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius.
    Returns a list of codewords (each a list of n integers in 0..q-1).
    """
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    # For the specific case of q=3, n=6, radius=1, use a specialized approach
    # General fallback for other parameters
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback: direct sum construction (may not be optimal but valid)
        # For small n, brute force or simple greedy
        return _generic_construct(q, n, radius, deadline)
    
    # Specialized for K_3(6,1)
    # Total space: 3^6 = 729 words
    # Each codeword covers 1 + 6*2 = 13 words (itself + 12 neighbors at distance 1)
    # Lower bound: ceil(729/13) = 57, but best known is 73
    
    # Precompute all words
    all_words = np.array(list(itertools.product(range(3), repeat=6)), dtype=np.int32)
    num_words = 729
    
    # Precompute neighborhoods: for each word index, which words are within distance 1
    # Actually, we want: for each candidate codeword position, which space words it covers
    # Since codewords are chosen from the same space, a codeword at position w covers all words within dist 1 of w
    
    # Precompute the coverage mask: cover_mask[i] = set of word indices covered by codeword at word i
    # This is a 729x729 boolean matrix, but we can be smarter
    
    # For each word, compute which other words are at distance <= 1
    # Distance 1 means: same in 5 positions, different in 1 position (or same in all 6 for distance 0)
    
    # Precompute neighbor lists
    # For word w, neighbors are w itself plus words that differ in exactly one position
    # For position j, the two other values give 2 neighbors per position, 6 positions => 12 neighbors + self = 13
    
    # Build a mapping from word tuple to index
    word_to_idx = {}
    for i, w in enumerate(all_words):
        word_to_idx[tuple(w)] = i
    
    # Precompute coverage: cover_set[i] = list of word indices covered by placing codeword at word i
    cover_set = []
    for i in range(num_words):
        w = all_words[i]
        covered = {i}
        for pos in range(6):
            for val in range(3):
                if val == w[pos]:
                    continue
                new_w = list(w)
                new_w[pos] = val
                idx = word_to_idx[tuple(new_w)]
                covered.add(idx)
        cover_set.append(covered)
    
    # Greedy set cover: repeatedly pick the codeword that covers the most uncovered words
    uncovered = set(range(num_words))
    selected = []
    
    while uncovered and (time.time() - start_time) < deadline:
        best_idx = -1
        best_count = -1
        for i in range(num_words):
            if i in selected_set if 'selected_set' in dir() else False:
                pass
            count = len(uncovered & cover_set[i])
            if count > best_count:
                best_count = count
                best_idx = i
        if best_idx == -1 or best_count <= 0:
            break
        selected.append(best_idx)
        uncovered -= cover_set[best_idx]
    
    # Convert to codewords
    codewords = [all_words[i].tolist() for i in selected]
    
    # Local search: try to remove codewords if coverage is maintained
    # Try to swap or remove
    selected_set = set(selected)
    
    # Phase 1: Try removing each codeword
    changed = True
    while changed and (time.time() - start_time) < deadline - 10:
        changed = False
        for i in range(len(selected) - 1, -1, -1):
            if (time.time() - start_time) > deadline - 5:
                break
            # Check if removing selected[i] still covers everything
            # Compute coverage of remaining codewords
            remaining = selected[:i] + selected[i+1:]
            if not remaining:
                continue
            # Check if all words are covered
            covered = set()
            for idx in remaining:
                covered |= cover_set[idx]
            if covered == set(range(num_words)):
                selected = remaining
                selected_set = set(selected)
                changed = True
                break
    
    # Phase 2: Try swapping - replace one codeword with another to enable further removals
    # This is more complex; do a few rounds of random restarts if time permits
    
    if (time.time() - start_time) < deadline - 20:
        # Try a few random restarts with different greedy orders
        best_solution = selected[:]
        for restart in range(3):
            if (time.time() - start_time) > deadline - 15:
                break
            # Random perturbation of greedy
            random.seed(42 + restart)
            uncovered = set(range(num_words))
            sel = []
            # Add some random initial picks
            for _ in range(5):
                if not uncovered:
                    break
                idx = random.choice(list(uncovered))
                sel.append(idx)
                uncovered -= cover_set[idx]
            
            while uncovered and (time.time() - start_time) < deadline - 10:
                best_idx = -1
                best_count = -1
                for i in range(num_words):
                    if i in sel:
                        continue
                    count = len(uncovered & cover_set[i])
                    if count > best_count:
                        best_count = count
                        best_idx = i
                if best_idx == -1 or best_count <= 0:
                    break
                sel.append(best_idx)
                uncovered -= cover_set[best_idx]
            
            # Try to remove
            changed = True
            while changed and (time.time() - start_time) < deadline - 10:
                changed = False
                for i in range(len(sel) - 1, -1, -1):
                    if (time.time() - start_time) > deadline - 8:
                        break
                    remaining = sel[:i] + sel[i+1:]
                    if not remaining:
                        continue
                    covered = set()
                    for idx in remaining:
                        covered |= cover_set[idx]
                    if covered == set(range(num_words)):
                        sel = remaining
                        changed = True
                        break
            
            if len(sel) < len(best_solution):
                best_solution = sel[:]
    
    codewords = [all_words[i].tolist() for i in best_solution]
    
    # Verify coverage
    final_covered = set()
    for idx in best_solution:
        final_covered |= cover_set[idx]
    assert final_covered == set(range(num_words)), "Coverage failed"
    
    # Ensure distinctness
    assert len(set(tuple(c) for c in codewords)) == len(codewords)
    
    return codewords


def _generic_construct(q, n, radius, deadline):
    """Generic fallback construction."""
    start_time = time.time()
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int32)
    num_words = q ** n
    
    # Build coverage mask
    cover_mask = np.zeros((num_words, num_words), dtype=np.bool_)
    for i in range(num_words):
        for j in range(num_words):
            dist = np.sum(all_words[i] != all_words[j])
            if dist <= radius:
                cover_mask[i, j] = True
    
    uncovered_mask = np.ones(num_words, dtype=np.bool_)
    selected = []
    selected_set = set()
    
    while uncovered_mask.any() and (time.time() - start_time) < deadline:
        scores = (cover_mask & uncovered_mask[None, :]).sum(axis=1)
        for s in selected_set:
            scores[s] = -1
        best_idx = int(np.argmax(scores))
        best_count = int(scores[best_idx])
        if best_count <= 0:
            break
        selected.append(best_idx)
        selected_set.add(best_idx)
        uncovered_mask &= ~cover_mask[best_idx]
    
    return [all_words[i].tolist() for i in selected]
