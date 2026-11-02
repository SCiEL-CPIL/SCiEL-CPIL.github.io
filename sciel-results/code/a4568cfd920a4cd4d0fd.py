import time
import numpy as np

def construct(q, n, radius):
    if q != 3 or n != 7 or radius != 1:
        # Fallback for generic case, though task is specific
        # Simple greedy for generic
        N = q**n
        start_time = time.time()
        deadline = start_time + 115
        
        # Precompute balls for generic q, n, radius
        # This might be slow for large n, but n=7 is small
        balls = []
        for i in range(N):
            digits = []
            x = i
            for _ in range(n):
                digits.append(x % q)
                x //= q
            digits.reverse()
            
            ball = set()
            # Generate all words within radius
            # For radius 1:
            ball.add(i)
            for pos in range(n):
                orig = digits[pos]
                for val in range(q):
                    if val == orig: continue
                    new_digits = digits[:]
                    new_digits[pos] = val
                    new_val = 0
                    for d in new_digits:
                        new_val = new_val * q + d
                    ball.add(new_val)
            balls.append(list(ball))
            
        cover_count = np.zeros(N, dtype=np.int8)
        code = []
        
        while time.time() < deadline:
            best_score = -1
            best_word = -1
            for i in range(N):
                if i in code: continue
                score = sum(1 for j in balls[i] if cover_count[j] == 0)
                if score > best_score:
                    best_score = score
                    best_word = i
            
            if best_score == 0:
                break
                
            code.append(best_word)
            for j in balls[best_word]:
                cover_count[j] += 1
                
        return code

    # Specific optimized path for q=3, n=7, radius=1
    start_time = time.time()
    deadline = start_time + 115
    
    N = 2187
    
    # Precompute balls
    balls = []
    for i in range(N):
        digits = []
        x = i
        for _ in range(7):
            digits.append(x % 3)
            x //= 3
        digits.reverse()
        
        ball = [i]
        for pos in range(7):
            orig = digits[pos]
            for val in range(3):
                if val == orig: continue
                new_digits = digits[:]
                new_digits[pos] = val
                new_val = 0
                for d in new_digits:
                    new_val = new_val * 3 + d
                ball.append(new_val)
        balls.append(ball)
        
    # Greedy Construction
    cover_count = np.zeros(N, dtype=np.int8)
    code = []
    code_set = set()
    
    while True:
        if time.time() > deadline:
            break
        best_score = -1
        best_word = -1
        for i in range(N):
            if i in code_set: continue
            score = 0
            for j in balls[i]:
                if cover_count[j] == 0:
                    score += 1
            if score > best_score:
                best_score = score
                best_word = i
        
        if best_score == 0:
            break
        
        code.append(best_word)
        code_set.add(best_word)
        for j in balls[best_word]:
            cover_count[j] += 1
            
    # Local Search
    def can_remove(c):
        for j in balls[c]:
            if cover_count[j] <= 1:
                return False
        return True

    def do_remove(c):
        code_set.remove(c)
        code.remove(c)
        for j in balls[c]:
            cover_count[j] -= 1

    def do_add(c):
        code_set.add(c)
        code.append(c)
        for j in balls[c]:
            cover_count[j] += 1

    def remove_redundant():
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for c in list(code):
                if can_remove(c):
                    do_remove(c)
                    changed = True

    # Initial removal
    remove_redundant()
    
    # Swap search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for c in list(code):
            # Try swapping c with w
            for w in range(N):
                if w in code_set:
                    continue
                
                # Check validity
                valid = True
                for j in balls[c]:
                    if cover_count[j] == 1 and j not in balls[w]:
                        valid = False
                        break
                
                if valid:
                    # Apply swap
                    do_remove(c)
                    do_add(w)
                    
                    # Try to remove
                    size_before = len(code)
                    remove_redundant()
                    
                    if len(code) < size_before:
                        improved = True
                    else:
                        # Revert
                        do_remove(w)
                        do_add(c)
                        
                        # If we reverted, we might want to break inner loop?
                        # No, other w might work.
                        
    return code
