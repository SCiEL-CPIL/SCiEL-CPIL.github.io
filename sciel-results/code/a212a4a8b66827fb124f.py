import itertools
import random
import time

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0
    
    N = q ** n
    
    def int_to_digits(num):
        digits = []
        for _ in range(n):
            digits.append(num % q)
            num //= q
        return digits
    
    def digits_to_int(digits):
        val = 0
        for i in range(n):
            val += digits[i] * (q ** i)
        return val

    # Precompute balls
    balls = []
    for w in range(N):
        digits = int_to_digits(w)
        ball = set([w])
        if radius >= 1:
            for i in range(n):
                orig = digits[i]
                for new_val in range(q):
                    if new_val != orig:
                        new_digits = digits[:]
                        new_digits[i] = new_val
                        ball.add(digits_to_int(new_digits))
        balls.append(ball)
        
    best_size = N
    best_code = list(range(N))
    
    def greedy_cover():
        covered = set(range(N))
        code = []
        code_set = set()
        while covered:
            best_c = -1
            best_count = -1
            candidates = []
            for c in range(N):
                if c in code_set:
                    continue
                inter_count = len(balls[c] & covered)
                if inter_count > best_count:
                    best_count = inter_count
                    best_c = c
                    candidates = [c]
                elif inter_count == best_count:
                    candidates.append(c)
            
            if not candidates:
                break
                
            c = random.choice(candidates)
            code.append(c)
            code_set.add(c)
            covered -= balls[c]
            
        return code

    def local_search_optimize(code):
        coverage_count = [0] * N
        for c in code:
            for w in balls[c]:
                coverage_count[w] += 1
        
        # Verify validity
        for w in range(N):
            if coverage_count[w] == 0:
                return code
                
        improved = True
        while improved:
            improved = False
            for i in range(len(code)):
                c = code[i]
                can_remove = True
                for w in balls[c]:
                    if coverage_count[w] == 1:
                        can_remove = False
                        break
                if can_remove:
                    for w in balls[c]:
                        coverage_count[w] -= 1
                    code.pop(i)
                    improved = True
                    break
        return code

    iterations = 0
    while time.time() < deadline:
        iterations += 1
        code = greedy_cover()
        code = local_search_optimize(code)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
            
        if best_size <= 73:
             if iterations > 100:
                 break

    return best_code
