import itertools
import random
import time

def construct(n):
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
        
    all_perms = list(itertools.permutations(range(1, n + 1)))
    perm_set = set(all_perms)
    total_perms = len(all_perms)
    
    def validate(word):
        if len(word) < n:
            return False
        seen = set()
        for i in range(len(word) - n + 1):
            w = tuple(word[i:i+n])
            if w in perm_set:
                seen.add(w)
        return len(seen) == total_perms

    def standard_superperm(n):
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                overlap = 0
                for o in range(min(len(new), len(block)), -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return word

    seed_word = standard_superperm(n)
    if not validate(seed_word):
        # Fallback if seed is invalid (should not happen)
        return "".join(map(str, seed_word))
        
    best_word = seed_word
    best_len = len(seed_word)
    
    start_time = time.time()
    deadline = start_time + 145
    
    def greedy_search(start_perm, time_limit):
        word = list(start_perm)
        visited = {tuple(start_perm)}
        
        while len(visited) < total_perms:
            if time.time() > time_limit:
                return None
            
            last = word[-(n-1):]
            
            # Case 1: Last n-1 is a permutation
            if len(set(last)) == n - 1:
                # Find the unique symbol that completes the permutation
                missing = [x for x in range(1, n+1) if x not in last]
                c_next = missing[0]
                new_window = tuple(last + [c_next])
                
                if new_window not in visited:
                    word.append(c_next)
                    visited.add(new_window)
                else:
                    # Stuck in permutation cycle. Must break.
                    # Pick a random other symbol
                    other_symbols = [x for x in range(1, n+1) if x != c_next]
                    c_break = random.choice(other_symbols)
                    word.append(c_break)
                    # The new window is non-perm. We don't add to visited.
                    
            # Case 2: Last n-1 is NOT a permutation
            else:
                # Look for any symbol c such that last + c is a permutation and unvisited
                found = False
                # Shuffle to break ties randomly
                symbols = list(range(1, n+1))
                random.shuffle(symbols)
                for c in symbols:
                    new_window = tuple(last + [c])
                    if len(set(new_window)) == n:
                        if new_window not in visited:
                            word.append(c)
                            visited.add(new_window)
                            found = True
                            break
                            
                if not found:
                    # No unvisited perm reachable in 1 step.
                    # Pick a random symbol to move forward
                    c = random.choice(symbols)
                    word.append(c)
                    
        return word

    attempts = 0
    while time.time() < deadline:
        attempts += 1
        start_perm = random.sample(range(1, n + 1), n)
        
        # Allocate time based on remaining
        remaining = deadline - time.time()
        if remaining < 2:
            break
            
        inner_limit = time.time() + min(5, remaining) # Max 5s per attempt
        
        result = greedy_search(start_perm, inner_limit)
        
        if result:
            if validate(result):
                if len(result) < best_len:
                    best_len = len(result)
                    best_word = result
                    # Early stop if we hit theoretical lower bound or known best
                    if best_len <= 5905:
                        break

    return "".join(map(str, best_word))
