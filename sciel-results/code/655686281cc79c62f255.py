"""Football pool problem: ternary covering code K_3(7,1).

Strategy:
1. Use the known best construction for K_3(7,1) with 186 codewords (van Laarhoven et al., 1989)
   as a starting point if we can construct it.
2. Since explicitly listing all 186 codewords is tedious, we use a greedy set-cover approach
   that is fast enough for n=7, q=3, radius=1 (2187 words, each covered by 1 + 2*7 = 15 words).
3. The greedy algorithm picks the codeword that covers the most uncovered words at each step.
   With 2187 words and ~186 steps, each step scans 2187 candidates * 15 neighbors = ~32k ops,
   total ~6M ops, well within 120s.

We also try to improve upon the greedy result with a local search (swap/replacement) if time allows,
but the primary goal is to match or beat 186.

Actually, the greedy set-cover for covering codes often gets close to optimal for small instances.
For K_3(7,1), the optimal is 186. Greedy might get slightly worse (e.g., 190-200).
To do better, we can use a more sophisticated approach: start from a good initial code and do local search.

Let's use a hybrid:
- Generate a good initial code using a structured method or greedy.
- Then do local search: try replacing one codeword with another to reduce the total size while maintaining coverage.

But given time constraints, let's first implement a fast greedy and see.

Actually, a better approach for this specific small instance: use integer programming? No, too slow.

Let's use a greedy algorithm with a twist: process words in a smart order, and use bitsets for speed.

We represent the set of all 2187 words as a bitset (or a boolean array).
For each candidate codeword, we precompute which words it covers (itself + 14 neighbors).
Greedy: repeatedly pick the codeword that covers the most currently-uncovered words.

We can speed this up by using numpy or by using Python sets, but 2187 is small.

Let's implement greedy with efficient counting.
"""

import itertools
import numpy as np


def construct(q, n, radius):
    # Only handle the specific case q=3, n=7, radius=1 efficiently.
    # For other parameters, fall back to a simple method.
    if q == 3 and n == 7 and radius == 1:
        return _construct_3_7_1()
    
    # Fallback: use the direct sum method from the seed if possible, else greedy.
    # For generality, do a greedy set cover.
    return _greedy_cover(q, n, radius)


def _construct_3_7_1():
    """Construct a covering code for q=3, n=7, radius=1.
    
    We use a greedy set-cover approach followed by a minimization step.
    The universe is all 3^7 = 2187 words.
    Each codeword covers itself and its 14 neighbors (Hamming distance 1).
    We pick codewords greedily to maximize coverage of uncovered words,
    then remove any redundant codewords to minimize the set size.
    """
    # Generate all words
    all_words = list(itertools.product(range(3), repeat=7))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    N = len(all_words)  # 2187
    
    # Precompute for each codeword, the list of word indices it covers
    # A word covers itself and all words at Hamming distance 1.
    covers = []
    for w in all_words:
        covered = set()
        covered.add(word_to_idx[w])
        for i in range(7):
            for d in range(1, 3):
                new_w = list(w)
                new_w[i] = (w[i] + d) % 3
                covered.add(word_to_idx[tuple(new_w)])
        covers.append(covered)
    
    # Greedy set cover
    uncovered = set(range(N))
    code_indices = []
    while uncovered:
        best = None
        best_count = -1
        for i in range(N):
            # Count how many uncovered words this codeword covers
            count = len(covers[i] & uncovered)
            if count > best_count:
                best_count = count
                best = i
        # Add the best codeword index
        code_indices.append(best)
        uncovered -= covers[best]
        
    # Minimization phase: remove redundant codewords
    # cover_count[i] is the number of codewords in code_indices that cover word i
    cover_count = np.zeros(N, dtype=int)
    for c_idx in code_indices:
        for w_idx in covers[c_idx]:
            cover_count[w_idx] += 1
            
    changed = True
    while changed:
        changed = False
        for i in range(len(code_indices)):
            c_idx = code_indices[i]
            # Check if removing c_idx leaves any word uncovered
            # A word is left uncovered if its count becomes 0.
            # Since we only remove one codeword at a time, a word becomes uncovered
            # if its current count is 1 and it is covered by c_idx.
            is_essential = False
            for w_idx in covers[c_idx]:
                if cover_count[w_idx] == 1:
                    is_essential = True
                    break
            
            if not is_essential:
                # Remove c_idx
                for w_idx in covers[c_idx]:
                    cover_count[w_idx] -= 1
                code_indices.pop(i)
                changed = True
                break # Restart the loop to handle index shifts
        
    # Convert indices back to codewords
    code = [list(all_words[idx]) for idx in code_indices]
    return code


def _greedy_cover(q, n, radius):
    """General greedy set cover for covering codes."""
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    N = len(all_words)
    
    # Precompute covers for each word
    covers = []
    for w in all_words:
        covered = set()
        covered.add(word_to_idx[w])
        # Generate all words within Hamming distance radius
        # For small radius, enumerate all changes
        for r in range(1, radius + 1):
            for positions in itertools.combinations(range(n), r):
                for deltas in itertools.product(range(1, q), repeat=r):
                    new_w = list(w)
                    for pos, d in zip(positions, deltas):
                        new_w[pos] = (w[pos] + d) % q
                    covered.add(word_to_idx[tuple(new_w)])
        covers.append(covered)
    
    uncovered = set(range(N))
    code = []
    while uncovered:
        best = None
        best_count = -1
        for i in range(N):
            count = len(covers[i] & uncovered)
            if count > best_count:
                best_count = count
                best = i
        code.append(list(all_words[best]))
        uncovered -= covers[best]
    
    return code
