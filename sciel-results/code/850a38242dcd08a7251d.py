import itertools

def construct(q, n, radius):
    all_words = list(itertools.product(range(q), repeat=n))
    
    # Precompute balls
    balls = []
    for w in all_words:
        ball = {w}
        if radius >= 1:
            for i in range(n):
                for val in range(q):
                    if val == w[i]:
                        continue
                    neighbor = list(w)
                    neighbor[i] = val
                    ball.add(tuple(neighbor))
        # Note: For this task radius is always 1, but keeping it general is good.
        # If radius > 1, we would need to generate more neighbors. 
        # Given the task is specifically K_3(6,1), radius=1 is guaranteed by the contract context, 
        # but the function signature takes radius. 
        # For robustness, let's implement a general radius check if needed, 
        # but for q=3, n=6, radius=1, the above is sufficient.
        # Let's assume radius=1 for the specific optimization, but handle general case simply.
        balls.append(ball)
        
    uncovered = set(all_words)
    codewords = []
    
    while uncovered:
        best_idx = -1
        max_covered = 0
        
        for i in range(len(all_words)):
            covered_count = len(balls[i] & uncovered)
            if covered_count > max_covered:
                max_covered = covered_count
                best_idx = i
        
        if best_idx == -1:
            break
            
        best_word = all_words[best_idx]
        codewords.append(best_word)
        uncovered -= balls[best_idx]
        
    return [list(c) for c in codewords]
