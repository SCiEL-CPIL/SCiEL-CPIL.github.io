import time
import random
import numpy as np
from itertools import product

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin before 150s kill
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in product(range(q), repeat=n-4)]
        else:
            return [list(t) for t in product(range(q), repeat=n)]
    
    # Precompute all words and their neighbors
    all_words = np.array(list(product(range(3), repeat=6)), dtype=np.int8)
    N = 729
    
    # Precompute neighbor sets (including self) for each word
    # For each word, its ball of radius 1 includes itself + 6*2 = 19 words
    def get_ball(word):
        ball = [word]
        for i in range(6):
            for v in range(3):
                if v != word[i]:
                    w = list(word)
                    w[i] = v
                    ball.append(tuple(w))
        return set(ball)
    
    # Build index mapping
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    
    # Precompute balls as index sets
    balls = []
    for w in all_words:
        ball_set = get_ball(tuple(w))
        balls.append([word_to_idx[x] for x in ball_set])
    
    # Greedy set cover: start with empty, add codewords that cover most uncovered words
    uncovered = set(range(N))
    codewords = []
    
    while uncovered:
        if time.time() > deadline - 5:
            break
        best_cw = None
        best_count = -1
        for cw in range(N):
            if not uncovered:
                break
            covered = len(set(balls[cw]) & uncovered)
            if covered > best_count:
                best_count = covered
                best_cw = cw
        if best_cw is None:
            break
        codewords.append(best_cw)
        uncovered -= set(balls[best_cw])
    
    # Convert to list of codeword tuples
    code = [list(all_words[cw]) for cw in codewords]
    
    # Try to remove redundant codewords
    # A codeword is redundant if its ball is fully covered by other codewords' balls
    def is_redundant(idx, code_list):
        if idx >= len(code_list):
            return False
        ball = set(balls[code_list[idx]])
        for j in range(len(code_list)):
            if j == idx:
                continue
            if ball.issubset(set(balls[code_list[j]])):
                return True
        return False
    
    # Iterative removal pass
    improved = True
    while improved and time.time() < deadline - 3:
        improved = False
        for i in range(len(code_list) if 'code_list' in dir() else len(codewords)):
            pass
    
    # Redo with proper tracking
    code_list = list(codewords)
    improved = True
    while improved and time.time() < deadline - 3:
        improved = False
        new_code_list = []
        for i, cw in enumerate(code_list):
            # Check if cw is redundant given all others in code_list
            ball = set(balls[cw])
            redundant = False
            for j, other_cw in enumerate(code_list):
                if j == i:
                    continue
                if ball.issubset(set(balls[other_cw])):
                    redundant = True
                    break
            if redundant:
                improved = True
            else:
                new_code_list.append(cw)
        code_list = new_code_list
    
    # Verify coverage
    covered = set()
    for cw in code_list:
        covered |= set(balls[cw])
    
    if len(covered) != N:
        # Fallback to greedy result if something went wrong
        code_list = codewords
    
    result = [list(all_words[cw]) for cw in code_list]
    
    # Ensure distinctness
    seen = set()
    unique_result = []
    for cw in result:
        t = tuple(cw)
        if t not in seen:
            seen.add(t)
            unique_result.append(cw)
    
    return unique_result
