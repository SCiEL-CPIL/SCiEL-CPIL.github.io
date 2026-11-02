"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import defaultdict


def construct(q, n, radius):
    """Construct a covering code. For q=3, n=7, radius=1, find a small ternary covering code."""
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback: direct sum construction
        return _direct_sum(q, n, radius)
    
    return _solve_ternary_7_1(deadline)


def _direct_sum(q, n, radius):
    """Direct sum construction as fallback."""
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    # General fallback: all-zero plus greedy
    return _greedy_construct(q, n, radius)


def _greedy_construct(q, n, radius):
    """Simple greedy construction as a last resort."""
    # Start with all-zero codeword, greedily add codewords covering most uncovered words
    all_words = list(itertools.product(range(q), repeat=n))
    covered = [False] * len(all_words)
    codewords = []
    
    for _ in range(1000):  # limit iterations
        best_word = None
        best_count = 0
        for w in all_words:
            if covered[w]:
                continue
            # Count how many uncovered words this would cover
            count = 0
            for d in range(radius + 1):
                for pos in itertools.combinations(range(n), d):
                    for vals in itertools.product(range(q), repeat=d):
                        new_word = list(w)
                        for i, v in zip(pos, vals):
                            new_word[i] = v
                        idx = tuple(new_word)
                        if not covered[idx]:
                            count += 1
            if count > best_count:
                best_count = count
                best_word = w
        if best_word is None or best_count == 0:
            break
        codewords.append(list(best_word))
        # Mark covered
        for d in range(radius + 1):
            for pos in itertools.combinations(range(n), d):
                for vals in itertools.product(range(q), repeat=d):
                    new_word = list(best_word)
                    for i, v in zip(pos, vals):
                        new_word[i] = v
                    covered[tuple(new_word)] = True
    return codewords


def _solve_ternary_7_1(deadline):
    """Solve the ternary (7,1) covering code problem."""
    q, n, radius = 3, 7, 1
    total_words = q ** n  # 2187
    
    # Generate tetracode direct sum: 243 codewords
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    codewords = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=3):
            codewords.append(h + list(tail))
    
    # Build coverage: for each word, which codewords cover it
    # Use efficient representation
    # Each codeword covers itself + 6 neighbors (7*2=14 words total per codeword)
    
    def neighbors(word):
        """Generate all words within Hamming distance 1."""
        yield tuple(word)
        for i in range(n):
            for v in range(q):
                if v != word[i]:
                    w = list(word)
                    w[i] = v
                    yield tuple(w)
    
    # Map word to index
    word_to_idx = {}
    for w in codewords:
        word_to_idx[tuple(w)] = len(word_to_idx)
    
    # Build coverage counts: for each word in space, count how many codewords cover it
    # We'll track coverage as a dict: word -> set of codeword indices covering it
    # But that's expensive. Instead, use counting.
    
    # Precompute: for each codeword, the set of words it covers
    cw_covers = []
    for cw in codewords:
        covered_set = set()
        for nb in neighbors(cw):
            covered_set.add(nb)
        cw_covers.append(covered_set)
    
    # coverage_count[word] = number of codewords covering word
    coverage_count = defaultdict(int)
    for i, covered_set in enumerate(cw_covers):
        for w in covered_set:
            coverage_count[w] += 1
    
    # Also track: for each codeword, how many words are ONLY covered by it
    # A codeword is redundant if no word is exclusively covered by it
    
    # More efficient: for each word, track the list of codewords that cover it
    # But memory... 2187 words, each covered by up to ~14*243/2187 ≈ 1.5 codewords on average
    # Actually each word is covered by at most 1 + 7*2 = 15 potential codewords (itself + neighbors)
    # But we only have 243 codewords, so at most 243 cover any word, typically few.
    
    # Let's use: for each word, a list of codeword indices covering it
    word_coverers = defaultdict(list)
    for i, covered_set in enumerate(cw_covers):
        for w in covered_set:
            word_coverers[w].append(i)
    
    # Active codewords
    active = [True] * len(codewords)
    
    # Iteratively remove redundant codewords
    removed_any = True
    while removed_any and time.time() < deadline:
        removed_any = False
        for i in range(len(codewords)):
            if not active[i]:
                continue
            # Check if codeword i is redundant
            # It's redundant if for every word w in cw_covers[i], there exists another active codeword j != i that also covers w
            is_redundant = True
            for w in cw_covers[i]:
                # Find if any other active codeword covers w
                found = False
                for j in word_coverers[w]:
                    if j != i and active[j]:
                        found = True
                        break
                if not found:
                    is_redundant = False
                    break
            if is_redundant:
                active[i] = False
                removed_any = True
    
    # Get result
    result = [codewords[i] for i in range(len(codewords)) if active[i]]
    
    # Verify coverage
    if _verify_coverage(result, q, n, radius):
        return result
    
    # If verification failed (shouldn't happen), return all
    return codewords


def _verify_coverage(codewords, q, n, radius):
    """Verify that all words are within distance radius of some codeword."""
    covered = set()
    for cw in codewords:
        for d in range(radius + 1):
            if d == 0:
                covered.add(tuple(cw))
            else:
                for pos in itertools.combinations(range(n), d):
                    for vals in itertools.product(range(q), repeat=d):
                        w = list(cw)
                        for i, v in zip(pos, vals):
                            w[i] = v
                        covered.add(tuple(w))
    return len(covered) == q ** n
# EVOLVE-BLOCK-END
