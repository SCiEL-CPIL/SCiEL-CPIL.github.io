import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110
    
    total_words = q ** n
    
    def to_int(word):
        val = 0
        for i in range(n):
            val = val * q + word[i]
        return val

    def to_word(val):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = val % q
            val //= q
        return word

    # Precompute balls
    balls = []
    for w in range(total_words):
        word = to_word(w)
        ball = [w]
        for i in range(n):
            orig = word[i]
            for delta in range(1, q):
                new_val = (orig + delta) % q
                word[i] = new_val
                ball.append(to_int(word))
            word[i] = orig
        balls.append(ball)
        
    # Strategy 1: Prune the 729 product code
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = []
    for h in tetracode:
        for tail in range(3 ** (n - 4)):
            t_word = to_word(tail)
            full_word = h + t_word
            initial_code.append(to_int(full_word))
            
    code1 = set(initial_code)
    coverage = [0] * total_words
    for cw in code1:
        for v in balls[cw]:
            coverage[v] += 1
            
    current_codewords = list(code1)
    random.seed(42)
    random.shuffle(current_codewords)
    
    while time.time() < deadline:
        to_remove = []
        for cw in current_codewords:
            if time.time() > deadline:
                break
            redundant = True
            for v in balls[cw]:
                if coverage[v] <= 1:
                    redundant = False
                    break
            if redundant:
                to_remove.append(cw)
                
        if not to_remove:
            break
            
        for cw in to_remove:
            code1.remove(cw)
            current_codewords.remove(cw)
            for v in balls[cw]:
                coverage[v] -= 1
                
    best_code = list(code1)
    
    # Strategy 2: Greedy Set Cover
    # Only if we have time and the pruned code is not already very good
    if time.time() < deadline - 10:
        uncovered = [1] * total_words # 1 means uncovered
        greedy_code = []
        
        # Precompute list of balls for faster access? Already have 'balls'.
        
        # To speed up greedy, we can maintain a count of uncovered neighbors for each candidate.
        # But updating that for every added word is expensive.
        # Simple greedy: 
        # For each step, find w that maximizes sum(uncovered[v] for v in balls[w])
        
        while sum(uncovered) > 0 and time.time() < deadline:
            best_w = -1
            max_cov = 0
            
            # Optimization: We can skip words that are already covered? No, we need to cover uncovered words.
            # We only need to consider words that cover at least one uncovered word.
            
            # Simple loop over all words
            for w in range(total_words):
                # Quick check: if w's ball has no uncovered words, skip
                # But we don't know that without checking.
                cov = 0
                for v in balls[w]:
                    if uncovered[v]:
                        cov += 1
                        if cov > max_cov:
                            # Early break if we can't beat max? 
                            # Max possible is 17. If max_cov is 17, we can break inner?
                            pass
                
                if cov > max_cov:
                    max_cov = cov
                    best_w = w
                    
                    # If max_cov is 17 (perfect coverage of ball), we can break outer loop?
                    # No, we just want the best for this step.
                    # If max_cov == 17, it's the best possible.
                    if max_cov == 17:
                        break
                        
            if best_w == -1:
                break # Should not happen if uncovered > 0
                
            greedy_code.append(best_w)
            for v in balls[best_w]:
                uncovered[v] = 0
                
            if len(greedy_code) < len(best_code):
                best_code = greedy_code
                
    return [to_word(cw) for cw in best_code]
