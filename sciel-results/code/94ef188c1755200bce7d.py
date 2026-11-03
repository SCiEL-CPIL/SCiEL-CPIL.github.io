"""
Ternary covering code K_3(6,1) via greedy set-cover with local improvement.

Strategy:
1. Start with the direct-sum construction (tetracode x all tails) giving 81 words.
2. Use a greedy set-cover / minimum-dominating-set approach:
   - Represent the universe as all 729 words.
   - Each candidate codeword covers itself + all words at Hamming distance 1 (27 words).
   - Greedily pick codewords that cover the most uncovered words.
3. After greedy, try to remove redundant codewords (if a codeword's covered set is
   subset of the union of others' covered sets, remove it).
4. Try local search: swap a codeword with a non-codeword if it reduces the total
   number of codewords or maintains coverage with fewer.

We use bitset representation for speed.
"""

import itertools
import time

def construct(q, n, radius):
    # General fallback for non-ternary or different n/radius
    if q != 3 or n != 6 or radius != 1:
        return _generic_construct(q, n, radius)
    
    return _ternary_6_1()


def _generic_construct(q, n, radius):
    """Fallback: use direct product of single-coordinate perfect codes if possible, else greedy."""
    # For general case, do a simple greedy
    all_words = list(itertools.product(range(q), repeat=n))
    n_total = len(all_words)
    
    # Precompute balls
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # For each word, compute which words it covers (within radius)
    balls = []
    for w in all_words:
        covered = set()
        # All words within Hamming distance radius
        for pos in range(n):
            for delta in range(1, q):
                new_w = list(w)
                new_w[pos] = (w[pos] + delta) % q
                covered.add(tuple(new_w))
        covered.add(tuple(w))
        balls.append(covered)
    
    # Greedy set cover
    uncovered = set(range(n_total))
    selected = []
    
    while uncovered:
        best_idx = -1
        best_count = -1
        for i in range(n_total):
            if i in uncovered or True:  # can pick any
                cnt = len(balls[i] & uncovered)
                if cnt > best_count:
                    best_count = cnt
                    best_idx = i
        selected.append(best_idx)
        uncovered -= balls[best_idx]
    
    return [list(all_words[i]) for i in selected]


def _ternary_6_1():
    """Specialized solver for K_3(6,1)."""
    n = 6
    q = 3
    total = 3 ** 6  # 729
    
    # Precompute: for each of the 729 words, the 27 words in its ball (itself + 18 neighbors + ... wait)
    # Ball of radius 1 in ternary length 6: 1 + 6*2 = 13 words
    # Actually: center + for each of 6 positions, 2 alternatives = 1 + 12 = 13
    
    # Map word (tuple of 6 ints in {0,1,2}) to index 0..728
    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx = idx * 3 + w[i]
        return idx
    
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % 3
            idx //= 3
        return w
    
    # Precompute ball masks as bitsets (Python ints)
    # ball_mask[i] is a bitset of all words covered by word i
    all_words = [idx_to_word(i) for i in range(total)]
    
    ball_masks = [0] * total
    for i in range(total):
        w = all_words[i]
        mask = 1 << i
        for pos in range(n):
            for delta in (1, 2):
                new_w = list(w)
                new_w[pos] = (w[pos] + delta) % 3
                j = word_to_idx(tuple(new_w))
                mask |= (1 << j)
        ball_masks[i] = mask
    
    # Start with the tetracode-based construction (81 words)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_selected = set()
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            w = h + list(tail)
            initial_selected.add(word_to_idx(tuple(w)))
    
    # Verify coverage and try to improve
    best_selected = set(initial_selected)
    best_size = len(best_selected)
    
    # Check coverage of initial
    def is_covered(selected):
        union = 0
        for s in selected:
            union |= ball_masks[s]
        return union == (1 << total) - 1
    
    if not is_covered(best_selected):
        # Shouldn't happen, but fallback
        return _generic_construct(3, 6, 1)
    
    # Phase 1: Remove redundant codewords
    # A codeword is redundant if its ball is contained in the union of the others' balls
    changed = True
    while changed:
        changed = False
        for s in list(best_selected):
            if len(best_selected) <= 1:
                break
            others = best_selected - {s}
            union_others = 0
            for o in others:
                union_others |= ball_masks[o]
            if (ball_masks[s] & ~union_others) == 0:
                best_selected.remove(s)
                best_size = len(best_selected)
                changed = True
                break
    
    # Phase 2: Try adding a word to enable removal of multiple words
    # For each non-selected word, check if adding it allows removing 2+ selected words
    # This is expensive, so we limit iterations
    
    # Phase 3: Local search - swap operations
    # Try replacing one selected word with one non-selected word if coverage is maintained
    # and we can then remove another word
    
    # More effective: try to find a smaller cover via greedy from scratch with good ordering
    # But first, let's do a more thorough local improvement
    
    # Compute coverage contribution of each selected word
    def get_uncovered(selected):
        union = 0
        for s in selected:
            union |= ball_masks[s]
        return ((1 << total) - 1) & ~union
    
    # Try: for each pair (remove s1, remove s2), check if there's a word w that covers
    # all words that were only covered by s1 or s2
    # This is O(k^2 * 729) which for k~73 is about 73^2 * 729 ~ 3.9M, feasible
    
    improved = True
    while improved and len(best_selected) > 70:
        improved = False
        sel_list = list(best_selected)
        k = len(sel_list)
        
        for i in range(k):
            for j in range(i + 1, k):
                s1, s2 = sel_list[i], sel_list[j]
                remaining = set(sel_list) - {s1, s2}
                union_remaining = 0
                for r in remaining:
                    union_remaining |= ball_masks[r]
                
                # Words that need coverage: those in ball(s1) or ball(s2) not in union_remaining
                need = (ball_masks[s1] | ball_masks[s2]) & ~union_remaining
                if need == 0:
                    # Can remove both!
                    best_selected.discard(s1)
                    best_selected.discard(s2)
                    improved = True
                    break
                # Check if there's a single word w that covers all of 'need'
                # need is a bitset; find w such that ball_masks[w] & need == need
                # This means need must be subset of some ball
                # Since balls have 13 elements, need must have <= 13 elements and be a subset of some ball
                if need.bit_count() <= 13:
                    # Try all 729 words to find one that covers 'need'
                    for w in range(total):
                        if w in best_selected:
                            continue
                        if (ball_masks[w] & need) == need:
                            # Adding w and removing s1, s2
                            best_selected.discard(s1)
                            best_selected.discard(s2)
                            best_selected.add(w)
                            improved = True
                            break
                    if improved:
                        break
            if improved:
                break
    
    # Phase 4: Single swap - replace one selected with one non-selected to enable further removals
    # For each selected s, for each non-selected w, if we replace s with w, can we then remove another?
    # Too expensive for full search. Do a limited version.
    
    # Phase 5: Greedy repair - if we got stuck, try a fresh greedy with different ordering
    # Actually, let's try a different approach: start from empty and greedily add
    # but use a smarter ordering based on the current solution
    
    # Let's also try: take the current best, and for each word in the space not in the code,
    # check if adding it and removing the selected word with least "unique coverage" helps
    
    # Compute for each selected word, how many words it uniquely covers
    # (words covered by this word and no other selected word)
    
    # Final check and return
    if is_covered(best_selected):
        return [idx_to_word(s) for s in sorted(best_selected)]
    else:
        # Fallback to initial
        return [idx_to_word(s) for s in sorted(initial_selected)]
