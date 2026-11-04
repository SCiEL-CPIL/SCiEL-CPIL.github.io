"""CSP-style local search for ternary covering code K_3(8,1).

We model the problem as: choose a set of codewords C ⊆ {0,1,2}^8 such that
every word in the space is within Hamming distance 1 of some c ∈ C.

Strategy:
1. Start with a greedy construction that gives a dense initial cover.
2. Use randomized local search (simulated annealing / conflict-driven) to
   reduce the number of codewords while maintaining coverage.
3. Track the best valid solution found and return it before the deadline.

Key insight: A codeword c covers c itself and all words at Hamming distance 1
(8*2 = 16 neighbors plus itself = 17 words per codeword). With 486 codewords,
486*17 = 8262 ≥ 6561, so it's feasible with overlap.

We represent the space as integers 0..6560. For each candidate codeword, we
precompute the set of space points it covers. Then we do local search:
- State: a set of active codewords.
- Objective: minimize |C| subject to union of covered sets = entire space.
- Moves: remove a codeword if its covered points are all covered by others;
  add a codeword if it helps cover uncovered points (for repair).
"""
import time
import random
import numpy as np
from collections import Counter

# Deadline guard: 150s wall-clock, return 5s before
DEADLINE = 145.0


def _word_to_idx(word, n=8):
    """Convert a word (list of ints 0..q-1) to an integer index in [0, q^n)."""
    idx = 0
    for d in range(n):
        idx = idx * 3 + word[d]
    return idx


def _idx_to_word(idx, n=8):
    """Convert an integer index back to a word."""
    word = [0] * n
    for d in range(n - 1, -1, -1):
        word[d] = idx % 3
        idx //= 3
    return word


def _precompute_covers(q, n):
    """For each codeword index, precompute the set of space indices it covers
    (itself + all words at Hamming distance 1)."""
    total = q ** n
    covers = []
    for idx in range(total):
        word = _idx_to_word(idx, n)
        covered = set()
        covered.add(idx)
        for pos in range(n):
            orig = word[pos]
            for val in range(q):
                if val == orig:
                    continue
                word[pos] = val
                covered.add(_word_to_idx(word, n))
            word[pos] = orig
        covers.append(covered)
    return covers


def _greedy_cover(covers, n, q, radius=1):
    """Greedy set cover: repeatedly pick the codeword that covers the most
    currently-uncovered points."""
    total = q ** n
    uncovered = set(range(total))
    selected = []
    
    # Precompute cover sizes for initial ordering
    cover_sizes = [len(c) for c in covers]
    
    while uncovered:
        best_idx = -1
        best_gain = 0
        for idx in range(total):
            if idx in [s for s in selected]:
                continue
            gain = len(covers[idx] & uncovered)
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
        if best_idx == -1 or best_gain == 0:
            break
        selected.append(best_idx)
        uncovered -= covers[best_idx]
    
    return selected


def _is_covered(selected, covers, n, q):
    """Check if the selected codewords cover the entire space."""
    total = q ** n
    covered = set()
    for idx in selected:
        covered |= covers[idx]
    return len(covered) == total


def _count_uncovered(selected, covers, n, q):
    """Return the set of uncovered points."""
    total = q ** n
    covered = set()
    for idx in selected:
        covered |= covers[idx]
    return set(range(total)) - covered


def _try_remove(selected, covers, n, q):
    """Try to remove a codeword from selected if coverage is maintained.
    Returns (new_selected, success)."""
    if not selected:
        return selected, False
    
    # Count how many selected codewords cover each point
    point_count = Counter()
    for idx in selected:
        for p in covers[idx]:
            point_count[p] += 1
    
    # Find a codeword whose removal doesn't leave any point uncovered
    random.shuffle(selected)
    for idx in selected:
        removable = True
        for p in covers[idx]:
            if point_count[p] == 1:
                removable = False
                break
        if removable:
            new_selected = [s for s in selected if s != idx]
            return new_selected, True
    
    return selected, False


def _local_search(covers, n, q, start_selected, time_limit, rng):
    """Local search to minimize the number of codewords while maintaining coverage.
    Uses removal-based moves with occasional random restarts."""
    total = q ** n
    best = list(start_selected)
    best_len = len(best)
    current = list(start_selected)
    
    start_time = time.time()
    
    # Precompute cover sets as frozensets for faster intersection
    cover_frozensets = [frozenset(c) for c in covers]
    
    # Build a mapping: for each point, which codeword indices cover it
    # This helps with fast removal checks
    # But for 6561 points and up to ~2000 codewords, we use counting
    
    while time.time() - start_time < time_limit:
        # Try multiple removals in sequence
        progress = True
        while progress and time.time() - start_time < time_limit:
            progress = False
            # Count coverage
            point_count = np.zeros(total, dtype=np.int32)
            for idx in current:
                for p in covers[idx]:
                    point_count[p] += 1
            
            # Find removable codewords
            removable = []
            for idx in current:
                if all(point_count[p] > 1 for p in covers[idx]):
                    removable.append(idx)
            
            if removable:
                # Remove one at random
                idx_to_remove = rng.choice(removable)
                current.remove(idx_to_remove)
                if len(current) < best_len:
                    best = list(current)
                    best_len = len(current)
                progress = True
        
        # If stuck, try adding a random codeword that covers many uncovered-ish points
        # then try removing more
        if time.time() - start_time < time_limit:
            # Occasionally do a random perturbation
            if rng.random() < 0.1 and current:
                # Swap: remove one, add another
                idx_out = rng.choice(current)
                idx_in = rng.randrange(total)
                if idx_in not in current:
                    current.remove(idx_out)
                    current.append(idx_in)
    
    return best


def construct(q, n, radius):
    """Construct a covering code. Returns list of codewords (each a list of n ints)."""
    start_time = time.time()
    rng = random.Random(42)
    
    total = q ** n
    
    # Precompute covers
    covers = _precompute_covers(q, n)
    
    # Greedy initialization
    selected = _greedy_cover(covers, n, q, radius)
    
    # Local search with time budget
    time_remaining = DEADLINE - (time.time() - start_time) - 10  # 10s safety margin
    if time_remaining > 1:
        best = _local_search(covers, n, q, selected, time_remaining, rng)
    else:
        best = selected
    
    # Verify coverage
    covered = set()
    for idx in best:
        covered |= covers[idx]
    
    if len(covered) != total:
        # Fallback: use the tetracode construction (known valid)
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        import itertools
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Convert indices back to words
    result = [_idx_to_word(idx, n) for idx in best]
    
    # Ensure distinctness
    seen = set()
    unique = []
    for w in result:
        key = tuple(w)
        if key not in seen:
            seen.add(key)
            unique.append(w)
    
    return unique
