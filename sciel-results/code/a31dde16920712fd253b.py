"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct(n):
    # Deadline guard: evaluator kills after 150s; leave safety margin
    deadline = time.time() + 130.0

    # For small n, use the classic recursive construction (optimal for n<=4)
    if n <= 4:
        return _classic_superperm(n)

    # Strategy: "symbol rotation" constructive algorithm.
    # Build a superpermutation by maintaining an ordered list of (m-1)-permutations.
    # At step m, for each (m-1)-permutation p in the current order, emit block p,m,p
    # with maximal overlap. The order of permutations is derived by cyclically
    # rotating the symbol labels of the previous step's permutation sequence.
    #
    # Key insight: the classic construction already uses maximal overlap between
    # consecutive blocks. The "rotation" approach tries different orderings of the
    # (m-1)-permutations at each step to see if a different global ordering yields
    # a shorter total word (due to different overlap patterns at boundaries).

    word = _rotation_superperm(n, deadline)
    current_len = len(word)

    # Try to shorten via local search: find and remove redundant characters
    word = _try_shorten(word, n, deadline)

    return word


def _classic_superperm(n):
    """Classic recursive construction of a superpermutation of length 1!+2!+...+n!"""
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find maximum overlap
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return "".join(map(str, word))


def _rotation_superperm(n, deadline):
    """
    Symbol-rotation constructive algorithm.
    
    At each step m (from 2 to n), we have an ordered list of (m-1)-permutations.
    We generate the m-superpermutation by emitting blocks p,m,p for each p in order,
    with maximal overlap between consecutive blocks.
    
    The "rotation" idea: instead of just using the order from the previous word,
    we also try cyclically rotating the symbol labels (i.e., applying a cyclic
    permutation to the symbols in each (m-1)-perm) to generate a different
    ordering, and pick whichever gives the shorter result.
    """
    # Start with the base
    word = [1]
    
    for m in range(2, n + 1):
        if time.time() > deadline - 10.0:
            # Not enough time left; fall back to classic for remaining steps
            word = _complete_classic_from_word(word, m, n)
            break
        
        # Extract the ordered list of (m-1)-permutations from current word
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # Try the standard order first
        best_word = _emit_blocks(word, perms, m)
        best_len = len(best_word)
        
        # Try cyclic rotations of symbol labels
        # A cyclic rotation by k means: replace each symbol s with ((s-1+k) % (m-1)) + 1
        # This generates a different ordering of the (m-1)-permutations
        for k in range(1, m - 1):
            if time.time() > deadline - 5.0:
                break
            rotated_perms = []
            for p in perms:
                rotated = tuple(((x - 1 + k) % (m - 1)) + 1 for x in p)
                rotated_perms.append(rotated)
            candidate = _emit_blocks(word, rotated_perms, m)
            if len(candidate) < best_len:
                best_len = len(candidate)
                best_word = candidate
        
        word = best_word
    
    return "".join(map(str, word))


def _emit_blocks(base_word, perms, m):
    """
    Given a base word and an ordered list of (m-1)-permutations,
    emit blocks p,m,p with maximal overlap, starting from the base word.
    """
    new = list(base_word)
    for p in perms:
        block = list(p) + [m] + list(p)
        # Find maximum overlap between end of new and start of block
        max_overlap = min(len(new), len(block))
        overlap = 0
        for o in range(max_overlap, -1, -1):
            if o == 0 or new[len(new) - o:] == block[:o]:
                overlap = o
                break
        new += block[overlap:]
    return new


def _complete_classic_from_word(word, start_m, n):
    """Complete the classic construction from an intermediate word."""
    word = list(word)
    for m in range(start_m, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return "".join(map(str, word))


def _try_shorten(word, n, deadline):
    """Try to shorten the superpermutation by removing redundant characters."""
    word = list(word)
    length = len(word)
    
    # Build a set of all permutations needed
    target_perms = set(itertools.permutations(range(1, n + 1)))
    
    # Track which permutations are covered
    def get_covered_perms(w):
        covered = set()
        for i in range(len(w) - n + 1):
            p = tuple(w[i:i+n])
            if len(set(p)) == n:
                covered.add(p)
        return covered
    
    # This is too slow for n=7. Let's use a different approach.
    # Instead, try specific known improvements.
    
    # For n=5, the improvement from 153 to 152 is well-known.
    # The pattern involves removing one character from a specific position.
    
    # Let's try a greedy approach: try removing each character and check
    # if all permutations are still covered. Use a bitmap for efficiency.
    
    if n > 6:
        # Too slow for n=7 with full verification
        return "".join(map(str, word))
    
    # Use a faster verification method
    # Represent each permutation as an integer for fast lookup
    perm_to_id = {}
    id_to_perm = []
    for i, p in enumerate(itertools.permutations(range(1, n + 1))):
        perm_to_id[p] = i
        id_to_perm.append(p)
    num_perms = len(id_to_perm)
    
    def perm_to_int(p):
        # Encode permutation as integer using factorial number system
        result = 0
        for i, val in enumerate(p):
            result = result * n + val
        return result
    
    # This encoding is not unique for all tuples, but for permutations of 1..n it is
    # Actually, let's just use the tuple directly in a set.
    
    # For n=5, num_perms = 120, which is manageable.
    # For n=6, num_perms = 720, still manageable.
    
    if n <= 6:
        covered = get_covered_perms(word)
        
        # Try removing each position
        improved = False
        for pos in range(length - 1, -1, -1):
            if time.time() > deadline:
                break
            # Try removing position pos
            test_word = word[:pos] + word[pos+1:]
            new_covered = get_covered_perms(test_word)
            if new_covered == target_perms:
                word = test_word
                covered = new_covered
                improved = True
                # Continue trying to remove more
                # Reset position since word changed
                pos = len(word) - 1
        
        if improved:
            return "".join(map(str, word))
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
