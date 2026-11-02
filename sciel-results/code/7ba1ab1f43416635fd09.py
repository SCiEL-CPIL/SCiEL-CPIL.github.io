"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import heapq
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = 150.0 - 5.0  # safety margin
    
    # For the specific case q=3, n=7, radius=1
    # Total words: 3^7 = 2187
    # Each codeword covers 1 + 7*(3-1) = 15 words (itself + 14 neighbors)
    # Lower bound: ceil(2187/15) = 146, but actual lower bound is 156
    # Best known: 186
    
    # Strategy: Greedy maximum coverage with lazy evaluation, then local search
    # with swap and removal to reduce code size.
    
    total_words = q ** n
    if total_words > 100000:
        # Fallback for larger problems: direct sum construction
        if n >= 4 and q == 3 and radius == 1:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Generic fallback: cover each word individually (valid but large)
            return [list(word) for word in itertools.product(range(q), repeat=n)]
    
    # Represent words as integers for efficiency
    def to_int(word):
        val = 0
        for i, v in enumerate(word):
            val = val * q + v
        return val
    
    def to_word(val):
        word = []
        for i in range(n):
            word.append(val % q)
            val //= q
        return word
    
    # Precompute neighbors: for each word, its ball of radius 1
    # For radius 1: the word itself + all words differing in exactly one position
    # Number of neighbors per word: 1 + n*(q-1)
    
    # We'll use a greedy set cover approach:
    # 1. Start with all words uncovered
    # 2. Repeatedly pick the codeword that covers the most uncovered words
    # 3. Then do local search: try removing codewords, try swapping
    
    # For efficiency, we'll track coverage counts
    # covered_count[w] = number of selected codewords whose ball contains w
    
    # Step 1: Greedy construction
    # For each potential codeword c, its ball B(c) = {w : dist(c,w) <= radius}
    # For radius=1, q=3, n=7: |B(c)| = 1 + 7*2 = 15
    
    # Precompute balls for all possible codewords? 3^7 = 2187 balls of size 15.
    # That's manageable.
    
    all_words = list(range(total_words))
    
    # Precompute ball membership: ball[c] = set of words covered by codeword c
    # Actually, let's use a different representation for speed.
    # For each word w, precompute which codewords cover it.
    # cover[w] = list of codewords c such that w is in B(c)
    
    # For radius 1: w is in B(c) iff dist(c,w) <= 1
    # dist(c,w) = 0 if c==w, 1 if they differ in exactly one position
    
    # Let's precompute for each word w, the set of codewords that cover it
    # A codeword c covers w if c == w or c differs from w in exactly one position
    
    # For a given word w, the codewords that cover it are:
    # - w itself
    # - for each position i, for each value v != w[i], the word w with position i changed to v
    
    # So cover[w] has size 1 + n*(q-1)
    
    cover = {}
    for w in all_words:
        word = to_word(w)
        cands = [w]
        for i in range(n):
            for v in range(q):
                if v != word[i]:
                    new_word = word[:]
                    new_word[i] = v
                    cands.append(to_int(new_word))
        cover[w] = cands
    
    # Now greedy set cover:
    # uncovered = set of all words
    # For each codeword c, gain(c) = number of uncovered words in B(c)
    # Pick c with max gain, add to solution, update uncovered
    
    # To make this efficient, we'll track for each codeword how many uncovered words it covers.
    # Initially, gain(c) = |B(c)| = 1 + n*(q-1) for all c.
    
    # Actually, let's think about it differently.
    # uncovered_count = total_words
    # For each codeword c, gain(c) = number of words w in B(c) that are still uncovered.
    
    # When we add codeword c to the solution, for each w in B(c), we decrement its cover count.
    # When cover_count[w] becomes 0... wait, no. We need to track which words are covered.
    
    # Let me use a cleaner approach:
    # covered[w] = boolean, whether w is covered by at least one selected codeword
    # For each candidate codeword c, gain(c) = sum of (not covered[w] for w in B(c))
    
    # To update efficiently: when we select codeword c, for each w in B(c):
    #   if not covered[w]: covered[w] = True, and for each other codeword c' that also covers w, gain(c') -= 1
    
    # This is O(|B(c)| * |cover[w]|) per selection, which is 15 * 15 = 225 per step.
    # With ~150-200 steps, that's ~45000 operations. Very fast.
    
    # Let's implement this.
    
    covered = [False] * total_words
    selected = []
    
    # gain[c] = number of currently uncovered words in B(c)
    gain = [1 + n * (q - 1)] * total_words  # initial gain for all codewords
    
    # For each word w, cover[w] is the list of codewords that cover w.
    # When w becomes covered, we need to decrement gain[c] for all c in cover[w].
    
    # Use a max-heap for lazy evaluation of gains.
    # heap entries: (-gain[c], c, version[c])
    # version[c] tracks how many times gain[c] has been updated.
    
    version = [0] * total_words
    heap = [(-gain[c], c, 0) for c in all_words]
    heapq.heapify(heap)
    
    uncovered_count = total_words
    
    while uncovered_count > 0:
        # Get the codeword with max gain
        neg_gain, c, ver = heapq.heappop(heap)
        if ver != version[c]:
            continue  # stale entry
        if -neg_gain <= 0:
            break  # no more gain
        
        # Select codeword c
        selected.append(c)
        
        # Update: for each word w in B(c), if not covered, mark as covered and update gains
        # B(c) = cover[c]... wait, no. cover[w] is the list of codewords covering w.
        # We need B(c) = {w : c in cover[w]}
        # Let me precompute B(c) for each c.
        
        # Actually, let me just compute B(c) on the fly or precompute it.
        # For efficiency, let's precompute balls[c] = list of words covered by codeword c.
        
        # Hmm, I didn't precompute that. Let me do it now.
        # Actually, for radius 1, B(c) is easy to compute.
        
        # Let me restructure: precompute balls for all codewords.
        # balls[c] = list of words in B(c)
        
        # This is getting complex. Let me simplify by just computing gains directly.
        
        # Alternative simpler approach: since the problem is small (2187 words),
        # let's just do a straightforward greedy without the heap optimization.
        
        break  # Let me restart with a cleaner implementation
    
    # Let me restart with a cleaner approach.
    # For 2187 words, even O(n^2) is fine.
    
    # Precompute balls
    balls = []
    for c in all_words:
        word_c = to_word(c)
        ball = [c]
        for i in range(n):
            for v in range(q):
                if v != word_c[i]:
                    new_word = word_c[:]
                    new_word[i] = v
                    ball.append(to_int(new_word))
        balls.append(ball)
    
    # Greedy set cover
    covered = [False] * total_words
    selected = []
    uncovered_count = total_words
    
    while uncovered_count > 0:
        best_c = -1
        best_gain = -1
        for c in all_words:
            g = 0
            for w in balls[c]:
                if not covered[w]:
                    g += 1
            if g > best_gain:
                best_gain = g
                best_c = c
            if time.time() > deadline:
                break
        if best_c == -1 or best_gain <= 0:
            break
        selected.append(best_c)
        for w in balls[best_c]:
            if not covered[w]:
                covered[w] = True
                uncovered_count -= 1
        if time.time() > deadline:
            break
    
    # Local search: try to remove codewords
    # A codeword c can be removed if all words in B(c) are covered by other selected codewords.
    
    # For each word w, let cover_count[w] = number of selected codewords that cover w.
    cover_count = [0] * total_words
    for c in selected:
        for w in balls[c]:
            cover_count[w] += 1
    
    # Try removing codewords
    changed = True
    while changed:
        changed = False
        for i in range(len(selected)):
            c = selected[i]
            # Check if c is redundant: all w in B(c) have cover_count[w] > 1
            redundant = True
            for w in balls[c]:
                if cover_count[w] <= 1:
                    redundant = False
                    break
            if redundant:
                # Remove c
                for w in balls[c]:
                    cover_count[w] -= 1
                selected.pop(i)
                changed = True
                break
        if time.time() > deadline:
            break
    
    # Swap local search: try replacing one codeword with another to reduce total count
    # Actually, swaps don't reduce count. We want to remove codewords.
    # Better: try adding a codeword that allows removing two existing ones.
    # Or: try 1-for-2 swaps.
    
    # For now, let's just do the removal pass and return.
    
    # Convert back to word lists
    result = [to_word(c) for c in selected]
    # Ensure distinctness
    seen = set()
    unique_result = []
    for r in result:
        key = tuple(r)
        if key not in seen:
            seen.add(key)
            unique_result.append(r)
    
    return unique_result
# EVOLVE-BLOCK-END
