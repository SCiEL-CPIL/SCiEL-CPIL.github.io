"""Improved ternary covering code K_3(8,1) via greedy + local search."""
import time
import random
import itertools

# Deadline guard: evaluator kills at 150s wall-clock; we target finishing by ~135s.
DEADLINE = 135.0


def _hamming_dist(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)


def _covers(c, word, radius):
    return _hamming_dist(c, word) <= radius


def _all_neighbors(c, q, n, radius):
    """Return all words within Hamming distance <= radius of c."""
    neighbors = {tuple(c)}
    # radius 1: add all single-position changes
    if radius >= 1:
        for i in range(n):
            for v in range(q):
                if v == c[i]:
                    continue
                tmp = list(c)
                tmp[i] = v
                neighbors.add(tuple(tmp))
    return neighbors


def _check_covering(code, q, n, radius):
    """Check if code covers the whole space. Uses neighbor sets for efficiency."""
    covered = set()
    for c in code:
        covered |= _all_neighbors(c, q, n, radius)
    return len(covered) == q ** n


def construct(q, n, radius):
    """
    Construct a covering code for {0..q-1}^n with given radius.
    Uses greedy farthest-point + local search (add/remove/swap).
    """
    start_time = time.time()
    total_space = q ** n

    # --- Greedy construction: farthest-point sampling ---
    # Start with a random seed, then iteratively add the word farthest from current code.
    # For n=8, q=3: space is 6561, manageable.
    
    # Precompute all words as tuples for speed
    all_words = [list(w) for w in itertools.product(range(q), repeat=n)]
    all_word_tuples = [tuple(w) for w in all_words]
    
    # For efficiency, precompute neighbor sets for all words (radius 1)
    # Actually, for greedy we need distance to nearest center, which is expensive.
    # Instead, use a different approach: greedy by coverage gain.
    
    # Greedy: start empty, at each step add the word that covers the most uncovered words.
    # This is O(|code| * |space| * n) per step, which for |space|=6561 and |code|~500 is ~3.2M * 8 * 500 = too slow.
    
    # Better: use the direct-sum construction as initial code, then local search.
    
    # Initial code: direct sum of tetracode
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    code = [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
    
    # Verify initial code
    if not _check_covering(code, q, n, radius):
        # Fallback: brute force greedy
        code = []
        for w in all_words:
            code.append(list(w))
        # This is too large; use greedy instead
        code = []
        uncovered = set(all_word_tuples)
        for w in all_words:
            if time.time() - start_time > DEADLINE - 30:
                break
            new_cover = set()
            for i in range(n):
                for v in range(q):
                    tmp = list(w)
                    tmp[i] = v
                    t = tuple(tmp)
                    if t in uncovered:
                        new_cover.add(t)
            if w in uncovered:
                new_cover.add(tuple(w))
            if len(new_cover) > 0:
                code.append(list(w))
                uncovered -= new_cover
                if not uncovered:
                    break
    
    best_code = list(code)
    best_size = len(code)
    
    # Local search: try to remove a codeword if coverage is maintained
    # Then try to swap: remove one, add another that helps more
    
    # Precompute coverage: for each codeword, which words it uniquely covers
    # For efficiency, compute for each word in space, which codewords cover it
    
    # Given time constraints, do a simple local search:
    # 1. Try removing each codeword; if still covered, remove it.
    # 2. Repeat until no more removals.
    
    def try_remove(code_list):
        """Try to remove redundant codewords."""
        removed = True
        while removed:
            removed = False
            for i in range(len(code_list)):
                if time.time() - start_time > DEADLINE - 10:
                    return code_list
                candidate = code_list[:i] + code_list[i+1:]
                if _check_covering(candidate, q, n, radius):
                    code_list = candidate
                    removed = True
                    break
        return code_list
    
    # Apply removal pass
    code = try_remove(code)
    if len(code) < best_size:
        best_code = list(code)
        best_size = len(code)
    
    # Try multiple random restarts with different initial codes
    rng = random.Random(42)
    
    for restart in range(20):
        if time.time() - start_time > DEADLINE - 20:
            break
        
        # Generate a random initial code of size ~best_size
        target_size = best_size
        random_code = [list(rng.choice(all_words)) for _ in range(target_size)]
        # Deduplicate
        seen = set()
        dedup_code = []
        for c in random_code:
            t = tuple(c)
            if t not in seen:
                seen.add(t)
                dedup_code.append(c)
        random_code = dedup_code
        
        # Check if it covers; if not, try to fix by adding words
        if not _check_covering(random_code, q, n, radius):
            # Add uncovered words
            covered = set()
            for c in random_code:
                covered |= _all_neighbors(c, q, n, radius)
            uncovered = [w for w in all_word_tuples if w not in covered]
            for w in uncovered:
                random_code.append(list(w))
        
        # Now do removal pass
        random_code = try_remove(random_code)
        
        if len(random_code) < best_size:
            best_code = list(random_code)
            best_size = len(random_code)
    
    # Final removal pass on best code
    best_code = try_remove(best_code)
    
    return best_code
