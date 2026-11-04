import time
import random
from itertools import permutations
from heapq import heappush, heappop
import sys

def construct(n):
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # Base case
    if n == 1:
        return "1"
    
    if n == 2:
        return "121"
    
    if n == 3:
        # Known shortest: 123121321
        return "123121321"
    
    # For n >= 4, use the recursive construction with optimization
    # The standard recursive construction gives length sum_{k=1}^n k!
    # For n=7, this is 5913. The known best is 5905.
    # We'll use a greedy approach to try to find a shorter one.
    
    # For n=7, let's try a different approach:
    # Use the fact that we can build up from smaller superpermutations
    
    # Strategy: Build S_1, S_2, ..., S_n using greedy TSP at each step
    # but with more sophisticated overlap calculation
    
    word = [1]
    
    for k in range(2, n + 1):
        time_remaining = deadline - time.time()
        
        # Extract unique (k-1)-permutations from current word
        perms = []
        seen = set()
        for i in range(len(word) - k + 2):
            window = tuple(word[i:i + k - 1])
            if len(set(window)) == k - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        num_perms = len(perms)
        
        if num_perms == 0:
            return "".join(map(str, word))
        
        # Determine if we should use fast mode
        use_fast_mode = (time_remaining < 15.0) or (num_perms > 200)
        
        if use_fast_mode:
            best_order = list(range(num_perms))
        else:
            # Greedy TSP with multiple random restarts
            best_order = None
            best_len = float('inf')
            
            # Try a few random starts
            if num_perms <= 100:
                num_starts = min(num_perms, 50)
            else:
                num_starts = 20
                
            starts = random.sample(range(num_perms), num_starts)
            
            for start_idx in starts:
                if time.time() > deadline - 5.0:
                    break
                
                visited = [False] * num_perms
                order = []
                current = start_idx
                visited[current] = True
                order.append(current)
                
                for _ in range(num_perms - 1):
                    if time.time() > deadline - 5.0:
                        break
                    best_next = -1
                    best_overlap = -1
                    p_curr = perms[current]
                    
                    for next_idx in range(num_perms):
                        if visited[next_idx]:
                            continue
                        p_next = perms[next_idx]
                        
                        # Calculate overlap
                        max_possible = min(len(p_curr), len(p_next))
                        overlap = 0
                        for o in range(min(k-2, max_possible), -1, -1):
                            if o == 0:
                                overlap = 0
                                break
                            if p_curr[-o:] == p_next[:o]:
                                overlap = o
                                break
                                
                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_next = next_idx
                            
                    if best_next == -1:
                        break
                    visited[best_next] = True
                    order.append(best_next)
                    current = best_next
                    
                if len(order) < num_perms:
                    continue
                
                # Calculate total length
                total_len = 2*(k-1) + 1
                
                for i in range(1, len(order)):
                    p_prev = perms[order[i-1]]
                    p_curr = perms[order[i]]
                    max_possible = min(len(p_prev), len(p_curr))
                    overlap = 0
                    for o in range(min(k-2, max_possible), -1, -1):
                        if o == 0:
                            overlap = 0
                            break
                        if p_prev[-o:] == p_curr[:o]:
                            overlap = o
                            break
                    total_len += (2*(k-1) + 1) - overlap
                
                if total_len < best_len:
                    best_len = total_len
                    best_order = order
                    
        if best_order is None:
            best_order = list(range(num_perms))
            
        # Construct new word
        new_word = []
        for idx, p_idx in enumerate(best_order):
            p = perms[p_idx]
            block = list(p) + [k] + list(p)
            
            if idx == 0:
                new_word = block
            else:
                # Find maximum overlap between end of new_word and start of block
                max_overlap = min(len(new_word), len(block))
                overlap = 0
                for o in range(min(k, max_overlap), -1, -1):
                    if o == 0:
                        overlap = 0
                        break
                    if new_word[-o:] == block[:o]:
                        overlap = o
                        break
                new_word += block[overlap:]
                
        word = new_word
        
        # Early exit if we're out of time
        if time.time() > deadline - 5.0:
            break
    
    # Verify we have a valid superpermutation
    # If not, fall back to a known valid construction
    def is_valid_superperm(word, n):
        if len(word) < n:
            return False
        perms_found = set()
        for i in range(len(word) - n + 1):
            window = tuple(word[i:i + n])
            if len(set(window)) == n:
                perms_found.add(window)
        return len(perms_found) == int(__import__('math').factorial(n))
    
    if not is_valid_superperm(word, n):
        # Fallback: use the standard recursive construction
        word = [1]
        for k in range(2, n + 1):
            perms = []
            seen = set()
            for i in range(len(word) - k + 2):
                window = tuple(word[i:i + k - 1])
                if len(set(window)) == k - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            
            new_word = []
            for idx, p in enumerate(perms):
                block = list(p) + [k] + list(p)
                if idx == 0:
                    new_word = block
                else:
                    max_overlap = min(len(new_word), len(block))
                    overlap = 0
                    for o in range(min(k, max_overlap), -1, -1):
                        if o == 0:
                            overlap = 0
                            break
                        if new_word[-o:] == block[:o]:
                            overlap = o
                            break
                    new_word += block[overlap:]
            word = new_word
    
    return "".join(map(str, word))
