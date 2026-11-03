import numpy as np
import itertools
import random
import time

def construct(q, n, radius):
    # The task is fixed to q=3, n=8, radius=1
    if q != 3 or n != 8 or radius != 1:
        # Fallback to direct sum for other parameters if needed, though task is specific
        # But for safety, we can just return the direct sum for general case
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Specific optimization for q=3, n=8, r=1
    M = 3**8
    
    # Helper to get digits
    def to_digits(x):
        digits = [0]*n
        for i in range(n):
            digits[i] = x % 3
            x //= 3
        return digits
    
    def to_int(digits):
        x = 0
        for i in range(n-1, -1, -1):
            x = x * 3 + digits[i]
        return x

    # Precompute balls for all words
    balls = [None] * M
    for i in range(M):
        digits = to_digits(i)
        ball = [i]
        for pos in range(n):
            d_plus = digits.copy()
            d_plus[pos] = (d_plus[pos] + 1) % 3
            ball.append(to_int(d_plus))
            d_minus = digits.copy()
            d_minus[pos] = (d_minus[pos] - 1) % 3
            ball.append(to_int(d_minus))
        balls[i] = ball
    
    # Greedy set cover initialization
    uncovered = set(range(M))
    code = set()
    
    # Greedy: repeatedly pick codeword covering most uncovered words
    while uncovered:
        best_c = None
        best_count = -1
        # Sample a subset to speed up if needed, but M=6561 is manageable
        for c in range(M):
            if c in code:
                continue
            covered = 0
            for p in balls[c]:
                if p in uncovered:
                    covered += 1
            if covered > best_count:
                best_count = covered
                best_c = c
            if covered == len(uncovered):
                break
        if best_c is not None:
            code.add(best_c)
            for p in balls[best_c]:
                if p in uncovered:
                    uncovered.discard(p)
    
    in_code = np.zeros(M, dtype=bool)
    for c in code:
        in_code[c] = True
        
    count = np.zeros(M, dtype=np.int32)
    for c in code:
        for p in balls[c]:
            count[p] += 1
    
    # Local Search with time limit
    start_time = time.time()
    time_limit = 110  # seconds
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        
        # Phase 1: Remove redundant codewords
        to_remove = []
        for c in list(code):
            redundant = True
            for p in balls[c]:
                if count[p] == 1:
                    redundant = False
                    break
            if redundant:
                to_remove.append(c)
                
        if to_remove:
            for c in to_remove:
                code.remove(c)
                in_code[c] = False
                for p in balls[c]:
                    count[p] -= 1
            improved = True
            continue
        
        # Phase 2: Try to swap a codeword to enable removal of others
        swapped = False
        # Randomize order to escape local minima
        code_list = list(code)
        random.shuffle(code_list)
        
        for c in code_list:
            if (time.time() - start_time) >= time_limit:
                break
            
            # Find unique points
            unique_points = []
            for p in balls[c]:
                if count[p] == 1:
                    unique_points.append(p)
            
            if not unique_points:
                code.remove(c)
                in_code[c] = False
                for p in balls[c]:
                    count[p] -= 1
                improved = True
                swapped = True
                break
                
            # Find candidates for c_new that cover all unique points
            candidates = set(balls[unique_points[0]])
            for p in unique_points[1:]:
                candidates.intersection_update(balls[p])
            
            # Filter out existing code
            candidates -= code
            
            if candidates:
                # Pick candidate that maximizes number of points with count > 1 (to enable more removals)
                best_c_new = None
                best_score = -1
                for c_new in list(candidates)[:10]:  # Limit candidates for speed
                    score = 0
                    for p in balls[c_new]:
                        if count[p] > 1:
                            score += 1
                    if score > best_score:
                        best_score = score
                        best_c_new = c_new
                
                if best_c_new is not None:
                    code.remove(c)
                    in_code[c] = False
                    for p in balls[c]:
                        count[p] -= 1
                        
                    code.add(best_c_new)
                    in_code[best_c_new] = True
                    for p in balls[best_c_new]:
                        count[p] += 1
                        
                    improved = True
                    swapped = True
                    break
                
        if swapped:
            continue
            
        # If no improvement in this pass, break
        break

    return [list(map(int, to_digits(c))) for c in code]
