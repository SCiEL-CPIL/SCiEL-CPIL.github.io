import math
import time

def construct(n):
    if n <= 1:
        return "1"
    
    target = math.factorial(n)
    seen = set()
    # Start with 1..n-1
    word = list(range(1, n))
    
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin
    
    while len(seen) < target:
        best_k = -1
        best_next_count = -1
        best_immediate = False
        
        current_suffix = word[-(n-1):] if len(word) >= n-1 else word
        
        for k in range(1, n + 1):
            new_window = tuple(current_suffix + [k])
            if len(set(new_window)) != n:
                continue
                
            is_new = new_window not in seen
            
            # Lookahead: how many next moves are good?
            next_count = 0
            new_suffix = new_window[1:]
            for j in range(1, n + 1):
                next_window = tuple(new_suffix + [j])
                if len(set(next_window)) == n and next_window not in seen:
                    next_count += 1
            
            if is_new:
                # Immediate gain
                if next_count > best_next_count or (next_count == best_next_count and (best_k == -1 or k < best_k)):
                    best_next_count = next_count
                    best_k = k
                    best_immediate = True
            else:
                # No immediate gain, but keep track in case no immediate gain is possible
                if not best_immediate:
                    if next_count > best_next_count or (next_count == best_next_count and (best_k == -1 or k < best_k)):
                        best_next_count = next_count
                        best_k = k
        
        if best_k == -1:
            # No distinct window possible. Must repeat a symbol.
            best_k = 1
        
        word.append(best_k)
        
        new_window = tuple(word[-n:])
        if len(set(new_window)) == n:
            seen.add(new_window)
            
        if time.time() > deadline:
            break
    
    return "".join(map(str, word))
