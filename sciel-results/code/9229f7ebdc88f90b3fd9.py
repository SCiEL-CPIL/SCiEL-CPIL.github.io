import time
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 100  # 100 seconds limit to be safe
    
    # 1. Generate seed codewords
    # Direct sum of tetracode (length 4) and all words of length n-4
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    seed_codewords = [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
    
    # Convert to tuples for hashing
    codewords = set(tuple(c) for c in seed_codewords)
    
    # Helper to convert word to int
    def word_to_int(w):
        val = 0
        for i in range(n):
            val = val * q + w[i]
        return val

    # Helper to get neighbors (ball of radius 1)
    def get_ball(w):
        # w is a tuple
        ball = [word_to_int(w)]
        for i in range(n):
            for val in range(q):
                if val != w[i]:
                    new_w = list(w)
                    new_w[i] = val
                    ball.append(word_to_int(tuple(new_w)))
        return set(ball)

    # Precompute balls for initial codewords
    # This might be expensive if we recompute, so we maintain a coverage count or set of covered words.
    # Since we only remove, we can maintain a set of "uncovered" words? 
    # No, we need to know if removing c leaves any word uncovered.
    # A word is covered if it is in the ball of at least one codeword.
    # If we remove c, a word w in ball(c) becomes uncovered if c was the ONLY codeword covering w.
    # So we can maintain a count of how many codewords cover each word.
    
    all_words = [word_to_int(tuple(w)) for w in itertools.product(range(q), repeat=n)]
    cover_count = {w: 0 for w in all_words}
    
    # Initialize cover counts
    for c in codewords:
        for w in get_ball(c):
            cover_count[w] += 1
            
    # Greedy removal
    # We can try to remove codewords in a specific order.
    # Codewords that are "less useful" (cover more words that are also covered by others) are better candidates.
    # Or just iterate randomly.
    
    # To be efficient, we can check if min(cover_count[w] for w in ball(c)) > 1.
    # If yes, c can be removed.
    
    # Let's iterate.
    # We might need multiple passes if removing one opens up removal of another?
    # Actually, if c is removable, removing it decreases cover_count for words in ball(c).
    # This might make other codewords removable that weren't before?
    # No, removing a codeword only decreases coverage. It cannot make a previously non-removable codeword removable.
    # Wait. If c' was not removable because some word w was only covered by c' and c.
    # If we remove c, w is now only covered by c'. c' is still not removable (w is still covered by c').
    # But if w was covered by c' and c, and we remove c, w is still covered by c'.
    # The condition for c' to be removable is that ALL words in ball(c') have count > 1.
    # If we remove c, counts decrease. So if c' was not removable (some w had count 1), it remains not removable (count is still 1 or becomes 0 which is bad, but we only remove if valid).
    # Actually, if we remove c, and w was covered by c and c', count becomes 1. c' is still not removable.
    # So a single pass might not be enough if we process in the wrong order?
    # No, if c is removable, removing it is safe.
    # Does removing c make c' removable?
    # c' is removable if for all w in ball(c'), count[w] > 1.
    # Before removing c, count[w] >= 1.
    # If count[w] was 2 (covered by c' and c), after removing c, count[w] becomes 1.
    # So c' becomes LESS likely to be removable.
    # So removing codewords makes it harder for others to be removed.
    # Therefore, we should remove as many as possible in one pass, but the order doesn't matter for the set of removable codewords?
    # If c and c' are both removable, removing c doesn't prevent removing c'?
    # If c is removable, all w in ball(c) have count >= 2.
    # If c' is removable, all w in ball(c') have count >= 2.
    # If we remove c, counts in ball(c) decrease by 1.
    # If w is in ball(c) and ball(c'), count[w] was >= 2, now >= 1.
    # So c' might NOT be removable anymore if it relied on w having count 2.
    # So order matters. We want to remove "central" codewords first? Or "peripheral"?
    # Actually, we want to keep the code valid.
    # Let's just iterate until no more can be removed.
    
    removed = True
    while removed and time.time() < deadline:
        removed = False
        # Randomize order to avoid local minima? Or sort by something?
        # Let's try sorting by the number of "exclusive" words (words covered only by this codeword).
        # But we can't remove codewords with exclusive words.
        # So all candidates have 0 exclusive words.
        # Let's just iterate over a copy of the set.
        for c in list(codewords):
            if time.time() > deadline:
                break
            
            # Check if c is removable
            removable = True
            for w in get_ball(c):
                if cover_count[w] <= 1:
                    removable = False
                    break
            
            if removable:
                # Remove c
                codewords.remove(c)
                for w in get_ball(c):
                    cover_count[w] -= 1
                removed = True
                # We can break and restart the loop to re-evaluate others?
                # Or continue? If we continue, we might remove c' which was removable before but might not be now?
                # No, if c' was removable, it had count >= 2 for all its words.
                # Removing c decreases counts. So c' might become non-removable.
                # So it is safer to restart the scan or just continue and check again?
                # If we continue, and c' is checked later, we will check its current counts.
                # So it will correctly identify if it's still removable.
                # So we don't need to restart.
                
    # Convert back to list of lists
    return [list(c) for c in codewords]
