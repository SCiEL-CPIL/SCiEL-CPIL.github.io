<<<<<<< SEARCH
def _encode(word, q, n):
    """Encode a word as an integer in base q."""
    val = 0
    for i, digit in enumerate(word):
        val = val * q + digit
    return val


def _decode(val, q, n):
    """Decode an integer back to a word."""
    word = [0] * n
    for i in range(n - 1, -1, -1):
        word[i] = val % q
        val //= q
    return word


def _hamming_dist(w1, w2):
    """Hamming distance between two words."""
    return sum(1 for a, b in zip(w1, w2) if a != b)


def _generate_balls(q, n, radius):
    """Precompute the ball (set of words within radius) for each word."""
    total = q ** n
    balls = []
    for idx in range(total):
        center = _decode(idx, q, n)
        ball = set()
        for word in itertools.product(range(q), repeat=n):
            if _hamming_dist(center, word) <= radius:
                ball.add(_encode(word, q, n))
        balls.append(ball)
    return balls


def _greedy_cover(q, n, radius, balls, random_seed=0):
    """Greedy set cover to build a covering code."""
    total = q ** n
    uncovered = set(range(total))
    codewords = []
    
    # Precompute coverage count for each candidate
    # For each word, count how many uncovered words it covers
    while uncovered:
        best_word = -1
        best_count = -1
        for word in range(total):
            if word in codewords:
                continue
            count = len(balls[word] & uncovered)
            if count > best_count:
                best_count = count
                best_word = word
            if best_count == len(uncovered):
                break
        codewords.append(best_word)
        uncovered -= balls[best_word]
    
    return codewords


def _local_search_remove(codewords, q, n, radius, balls):
    """Try to remove codewords that are redundant."""
    total = q ** n
    covered = set()
    for cw in codewords:
        covered |= balls[cw]
    
    changed = True
    while changed:
        changed = False
        for i in range(len(codewords)):
            # Check if removing codeword[i] still covers everything
            remaining = set(codewords[:i] + codewords[i+1:])
            new_covered = set()
            for cw in remaining:
                new_covered |= balls[cw]
            if new_covered == set(range(total)):
                codewords.pop(i)
                covered = new_covered
                changed = True
                break
    return codewords


def construct(q, n, radius):
    total = q ** n
    
    # For small instances, use exact/greedy + local search
    if total <= 10000:
        balls = _generate_balls(q, n, radius)
        
        # Greedy construction
        codewords = _greedy_cover(q, n, radius, balls)
        
        # Local search to remove redundant codewords
        codewords = _local_search_remove(codewords, q, n, radius, balls)
        
        # Convert back to list form
        return [_decode(cw, q, n) for cw in codewords]
    else:
        # Fallback: direct sum with tetracode for larger n
        if n < 4:
            # For very small n, brute force
            return [list(word) for word in itertools.product(range(q), repeat=n)]
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
=======
def _encode(word, q, n):
    val = 0
    for d in word:
        val = val * q + d
    return val

def _decode(val, q, n):
    word = [0] * n
    for i in range(n - 1, -1, -1):
        word[i] = val % q
        val //= q
    return word

def _precompute_balls(q, n, radius):
    total = q ** n
    balls = [set() for _ in range(total)]
    
    # Efficient ball generation for radius 1
    if radius == 1:
        for center_idx in range(total):
            center = _decode(center_idx, q, n)
            ball = {center_idx}
            for i in range(n):
                for sym in range(q):
                    if sym != center[i]:
                        new_word = center.copy()
                        new_word[i] = sym
                        ball.add(_encode(new_word, q, n))
            balls[center_idx] = ball
    else:
        # Fallback for other radii
        for center_idx in range(total):
            center = _decode(center_idx, q, n)
            ball = set()
            for w in itertools.product(range(q), repeat=n):
                dist = sum(1 for a, b in zip(center, w) if a != b)
                if dist <= radius:
                    ball.add(_encode(w, q, n))
            balls[center_idx] = ball
            
    return balls

def construct(q, n, radius):
    total = q ** n
    
    # Fallback for very large instances (not expected for this task)
    if total > 100000:
        balls = _precompute_balls(q, n, radius)
        uncovered = set(range(total))
        codewords = []
        while uncovered:
            best_w = -1
            best_cov = -1
            for w in range(total):
                if w in codewords:
                    continue
                cov = len(balls[w] & uncovered)
                if cov > best_cov:
                    best_cov = cov
                    best_w = w
            codewords.append(best_w)
            uncovered -= balls[best_w]
        return [_decode(cw, q, n) for cw in codewords]
        
    balls = _precompute_balls(q, n, radius)
    max_ball_size = max(len(b) for b in balls)
    all_words = set(range(total))
    
    # Greedy initial solution
    uncovered_greedy = set(range(total))
    greedy_code = []
    while uncovered_greedy:
        best_w = -1
        best_cov = -1
        for w in range(total):
            if w in greedy_code:
                continue
            cov = len(balls[w] & uncovered_greedy)
            if cov > best_cov:
                best_cov = cov
                best_w = w
        greedy_code.append(best_w)
        uncovered_greedy -= balls[best_w]
        
    best_size = len(greedy_code)
    best_solution = greedy_code
    
    # Search for better solutions using Branch and Bound
    # Fix first codeword to 0
    lb = (total + max_ball_size - 1) // max_ball_size
    
    current_code = [0]
    uncovered = all_words - balls[0]
    
    def search(uncovered_set, depth, max_depth):
        nonlocal best_solution, best_size
        
        if not uncovered_set:
            if depth < best_size:
                best_size = depth
                best_solution = current_code.copy()
            return True
        
        if depth >= max_depth:
            return False
            
        if len(uncovered_set) > (max_depth - depth) * max_ball_size:
            return False
            
        candidates = []
        for w in range(total):
            if w in current_code:
                continue
            cov = len(balls[w] & uncovered_set)
            if cov > 0:
                candidates.append((cov, w))
                
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        top_k = 20
        for cov, w in candidates[:top_k]:
            current_code.append(w)
            new_uncovered = uncovered_set - balls[w]
            
            if search(new_uncovered, depth + 1, max_depth):
                current_code.pop()
                return True
            
            current_code.pop()
            
        return False

    # Try to find smaller solutions
    # Start from lb, go up to best_size - 1
    for target_size in range(lb, best_size):
        current_code = [0]
        uncovered = all_words - balls[0]
        
        if search(uncovered, 1, target_size):
            break
            
    return [_decode(cw, q, n) for cw in best_solution]
>>>>>>> REPLACE
