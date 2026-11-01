import itertools
import time

def construct(q, n, radius):
    """
    Constructs a covering code using a greedy set cover algorithm followed by redundancy removal.
    Optimized for q=3, n=8, radius=1.
    """
    start_time = time.time()
    timeout = 110  # seconds, leave 10s margin for safety

    # Generate all words as tuples
    all_words = list(itertools.product(range(q), repeat=n))
    N = len(all_words)
    
    # Precompute balls for all words (Hamming distance <= radius)
    # For radius=1, ball size is 1 + n*(q-1)
    balls = {}
    for w in all_words:
        ball = set()
        ball.add(w)
        if radius >= 1:
            for i in range(n):
                for val in range(q):
                    if val == w[i]:
                        continue
                    new_w = list(w)
                    new_w[i] = val
                    ball.add(tuple(new_w))
        # If radius > 1, we would need to add more, but task is R=1
        balls[w] = ball
        
    # Greedy Set Cover
    uncovered = set(all_words)
    codewords = []
    
    while uncovered:
        if time.time() - start_time > timeout:
            break
            
        best_word = None
        max_new_coverage = 0
        
        for w in all_words:
            # Calculate how many words in ball[w] are in uncovered
            count = len(balls[w] & uncovered)
            if count > max_new_coverage:
                max_new_coverage = count
                best_word = w
                
        if best_word is None or max_new_coverage == 0:
            # Should not happen if uncovered is not empty and balls are correct
            break
            
        codewords.append(best_word)
        uncovered.difference_update(balls[best_word])
        
    # Post-processing: Remove redundant codewords
    if codewords:
        cover_count = {w: 0 for w in all_words}
        for c in codewords:
            for w in balls[c]:
                cover_count[w] += 1
                
        changed = True
        while changed:
            if time.time() - start_time > timeout:
                break
            changed = False
            # Iterate over a copy since we might modify codewords
            for i, c in enumerate(list(codewords)):
                redundant = True
                for w in balls[c]:
                    if cover_count[w] <= 1:
                        redundant = False
                        break
                if redundant:
                    codewords.pop(i)
                    for w in balls[c]:
                        cover_count[w] -= 1
                    changed = True
                    
    # Convert to list of lists
    return [list(c) for c in codewords]
