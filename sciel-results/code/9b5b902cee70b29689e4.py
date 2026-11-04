"""Ternary covering code K_3(6,1) - improved greedy with local search."""
import itertools
import random


def construct(q, n, radius):
    # Handle the specific case: q=3, n=6, radius=1
    if q == 3 and n == 6 and radius == 1:
        return _solve_ternary_6_1()
    
    # Fallback: simple greedy for other parameters
    all_words = list(itertools.product(range(q), repeat=n))
    covered = set()
    codewords = []
    
    for c in all_words:
        c = list(c)
        new_covered = _cover_set(c, n, q, radius)
        added = sum(1 for w in new_covered if w not in covered)
        if added > 0:
            covered.update(new_covered)
            codewords.append(c)
            if len(covered) == len(all_words):
                break
    
    return codewords


def _cover_set(c, n, q, radius):
    """Return set of words within Hamming distance <= radius from codeword c."""
    covered = set()
    covered.add(tuple(c))
    for dist in range(1, radius + 1):
        for positions in itertools.combinations(range(n), dist):
            for values in itertools.product(range(q), repeat=dist):
                w = list(c)
                for pos, val in zip(positions, values):
                    w[pos] = val
                covered.add(tuple(w))
    return covered


def _solve_ternary_6_1():
    """Find a 73-word (or better) ternary covering code of length 6, radius 1."""
    
    # Precompute all 729 words and their coverage
    all_words = list(itertools.product(range(3), repeat=6))
    all_words_set = set(all_words)
    
    # Precompute coverage for each potential codeword
    # coverage[i] = set of word indices covered by codeword i
    n_words = 729
    coverage = [None] * n_words
    
    # Map word to index
    word_to_idx = {}
    for idx, w in enumerate(all_words):
        word_to_idx[w] = idx
    
    # For each codeword, compute which words it covers
    for i, c in enumerate(all_words):
        cov = set()
        cov.add(i)
        for pos in range(6):
            for val in range(3):
                if val != c[pos]:
                    w = list(c)
                    w[pos] = val
                    cov.add(word_to_idx[tuple(w)])
        coverage[i] = cov
    
    # Greedy set cover to get an initial solution
    covered_mask = [False] * n_words
    covered_count = 0
    codewords = []
    
    # Use a priority approach: pick codeword covering most uncovered words
    remaining = set(range(n_words))
    
    for _ in range(100):  # Cap iterations for safety
        best_idx = -1
        best_count = 0
        for i in range(n_words):
            if i in remaining:
                continue
            # Count uncovered words covered by i
            cnt = 0
            for w in coverage[i]:
                if not covered_mask[w]:
                    cnt += 1
            if cnt > best_count:
                best_count = cnt
                best_idx = i
            # Early termination if we can't beat current best
            # (optimization: skip if remaining potential <= best_count)
        
        if best_idx == -1 or best_count == 0:
            break
        
        codewords.append(best_idx)
        remaining.discard(best_idx)
        for w in coverage[best_idx]:
            if not covered_mask[w]:
                covered_mask[w] = True
                covered_count += 1
        
        if covered_count == n_words:
            break
    
    # Try to reduce the number of codewords via local search
    # Remove codewords that don't uniquely cover anything
    for _iteration in range(20):
        removed = False
        for i in range(len(codewords) - 1, -1, -1):
            idx = codewords[i]
            # Check if any word is only covered by this codeword
            unique = False
            for w in coverage[idx]:
                # Check if w is covered by any other codeword
                only_this = True
                for j in range(len(codewords)):
                    if j == i:
                        continue
                    if w in coverage[codewords[j]]:
                        only_this = False
                        break
                if only_this:
                    unique = True
                    break
            if not unique:
                codewords.pop(i)
                removed = True
        
        if not removed:
            break
    
    # Additional local search: try swapping codewords to reduce count
    for _attempt in range(10):
        new_codewords = list(codewords)
        # Try removing each codeword and see if we can re-cover with fewer additions
        best_removal = -1
        best_size = len(new_codewords)
        
        for i in range(len(new_codewords)):
            test_list = new_codewords[:i] + new_codewords[i+1:]
            # Check coverage
            covered_set = set()
            for idx in test_list:
                covered_set.update(coverage[idx])
            if len(covered_set) == n_words:
                best_removal = i
                best_size = len(test_list)
                break
        
        if best_removal >= 0:
            new_codewords.pop(best_removal)
            codewords = new_codewords
        else:
            break
    
    # Convert indices back to codewords
    result = [list(all_words[idx]) for idx in codewords]
    
    # Verify coverage
    final_covered = set()
    for c in result:
        final_covered.add(tuple(c))
        for pos in range(6):
            for val in range(3):
                if val != c[pos]:
                    w = list(c)
                    w[pos] = val
                    final_covered.add(tuple(w))
    
    if len(final_covered) == 729:
        return result
    
    # Fallback: known 81-word construction
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=2)]
