"""Improved ternary covering code construction for K_3(6,1)."""
# EVOLVE-BLOCK-START
import itertools
import random
import time


def construct(q, n, radius):
    # Known best: 73 (van Laarhoven et al. 1989). Lower bound 71.
    # We use a greedy + local-search approach with a time budget.
    start_time = time.time()
    deadline = start_time + 115  # leave 5s safety margin
    
    if n != 6 or q != 3 or radius != 1:
        # Fallback: direct sum of tetracode
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    all_words = list(itertools.product(range(3), repeat=6))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    def hamming_dist(a, b):
        return sum(1 for x, y in zip(a, b) if x != y)
    
    def covered_set(code):
        """Return a set of indices of all words within distance 1 of any codeword."""
        s = set()
        for cw in code:
            idx = word_to_idx[tuple(cw)]
            s.add(idx)
            for pos in range(6):
                for val in range(3):
                    if cw[pos] != val:
                        lst = list(cw)
                        lst[pos] = val
                        s.add(word_to_idx[tuple(lst)])
        return s
    
    # Greedy construction
    def greedy_build():
        uncovered = set(range(729))
        code = []
        while uncovered:
            # Pick word that covers most uncovered words
            best_word = None
            best_cover = 0
            for w in all_words:
                idx = word_to_idx[w]
                # Count how many uncovered words this would cover
                cover_count = 0
                for d in range(3):  # distance 0, 1
                    if d == 0:
                        if idx in uncovered:
                            cover_count += 1
                    else:
                        for pos in range(6):
                            for val in range(3):
                                if w[pos] != val:
                                    lst = list(w)
                                    lst[pos] = val
                                    if word_to_idx[tuple(lst)] in uncovered:
                                        cover_count += 1
                if cover_count > best_cover:
                    best_cover = cover_count
                    best_word = list(w)
                if time.time() > deadline:
                    return code
            if best_word is None:
                break
            code.append(best_word)
            # Update uncovered
            idx = word_to_idx[tuple(best_word)]
            uncovered.discard(idx)
            for pos in range(6):
                for val in range(3):
                    if best_word[pos] != val:
                        lst = list(best_word)
                        lst[pos] = val
                        uncovered.discard(word_to_idx[tuple(lst)])
        return code
    
    # Start with a reasonable initial code
    code = greedy_build()
    
    if time.time() > deadline:
        return code
    
    # Local search: try removing redundant codewords, then swap
    def is_redundant(code, i):
        """Check if removing code[i] still covers all words."""
        remaining = [code[j] for j in range(len(code)) if j != i]
        cov = covered_set(remaining)
        return len(cov) == 729
    
    def try_remove_redundant(code):
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(len(code)):
                if is_redundant(code, i):
                    code.pop(i)
                    changed = True
                    break
        return code
    
    def try_swap(code):
        """Try to replace a codeword with another to allow further reductions."""
        best_code = list(code)
        best_len = len(code)
        random.shuffle(best_code)
        
        for i in range(len(best_code)):
            if time.time() > deadline:
                break
            # Try replacing best_code[i] with a random word
            for _ in range(200):
                if time.time() > deadline:
                    break
                new_w = list(random.choice(all_words))
                trial = best_code[:i] + [new_w] + best_code[i+1:]
                if len(trial) < best_len:
                    # Check if still covers everything
                    cov = covered_set(trial)
                    if len(cov) == 729:
                        best_code = trial
                        best_len = len(trial)
                        # Try to remove more
                        best_code = try_remove_redundant(best_code)
                        best_len = len(best_code)
        return best_code
    
    # Remove redundant codewords
    code = try_remove_redundant(code)
    
    # Try multiple swap iterations
    best_code = list(code)
    for _ in range(50):
        if time.time() > deadline:
            break
        cand = try_swap(best_code)
        if len(cand) < len(best_code):
            best_code = cand
    
    return best_code
# EVOLVE-BLOCK-END
