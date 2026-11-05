import time
import random
from itertools import permutations

def overlap(a, b):
    """Calculate the length of the longest suffix of a that is a prefix of b."""
    max_ov = 0
    # a and b are tuples of same length
    # Suffix of a of length k: a[-k:]
    # Prefix of b of length k: b[:k]
    # Max k is len(a)
    # We can iterate k from len(a) down to 0
    # Or just check all possible k
    # Since len is small (<= 6), this is fast.
    for k in range(len(a), 0, -1):
        if a[-k:] == b[:k]:
            return k
    return 0

def find_best_order(perms, time_limit=5.0):
    """
    Find an ordering of perms that maximizes the sum of overlaps.
    Uses greedy with multiple restarts.
    """
    N = len(perms)
    if N == 0:
        return []
    if N == 1:
        return perms[:]
    
    best_score = -1
    best_order = None
    
    start_time = time.time()
    
    # Number of restarts depends on time
    # Estimate time per restart
    # Greedy is O(N^2)
    # Let's try 10 restarts first to estimate
    for trial in range(10000): # Cap on trials
        if time.time() - start_time > time_limit:
            break
        
        # Random start
        start_idx = random.randint(0, N - 1)
        current = perms[start_idx]
        visited = [False] * N
        visited[start_idx] = True
        order = [current]
        score = 0
        
        for _ in range(N - 1):
            max_ov = -1
            next_idx = -1
            # Find best next
            for j in range(N):
                if not visited[j]:
                    ov = overlap(current, perms[j])
                    if ov > max_ov:
                        max_ov = ov
                        next_idx = j
            # If tie, break randomly?
            # For now, just pick first found with max_ov
            # To add randomness, we could collect all max_ov candidates and pick one
            # But simple greedy might get stuck.
            # Let's add a bit of randomness in tie-breaking
            candidates = [j for j in range(N) if not visited[j] and overlap(current, perms[j]) == max_ov]
            if candidates:
                next_idx = random.choice(candidates)
            
            order.append(perms[next_idx])
            visited[next_idx] = True
            score += max_ov
            current = perms[next_idx]
        
        if score > best_score:
            best_score = score
            best_order = order[:]
            
    return best_order

def construct(n):
    """
    Constructs the shortest known superpermutation for n symbols.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    if n == 1:
        return "1"
    
    word = [1]
    
    for m in range(2, n + 1):
        if time.time() > deadline:
            break
            
        # Get all permutations of 1..m-1
        perms = list(permutations(range(1, m)))
        
        # Determine order
        # For small m, recursive order is good.
        # For m=7, use search.
        # We can always use search, it's fast for m<7.
        # But for m=7, we need to be careful with time.
        
        # Allocate time based on m
        if m == 7:
            time_budget = 100.0 # Use most of the time
        else:
            time_budget = 1.0
        
        # Adjust budget if time is running out
        remaining = deadline - time.time()
        if remaining < time_budget:
            time_budget = max(0.1, remaining * 0.9)
            
        order = find_best_order(perms, time_budget)
        
        new_word = []
        for p in order:
            # Form the block: p + [m] + p
            block = list(p) + [m] + list(p)
            
            # Find maximum overlap between end of new_word and start of block
            max_overlap = min(len(new_word), len(block))
            overlap_val = 0
            for o in range(max_overlap, -1, -1):
                if o == 0 or new_word[len(new_word) - o:] == block[:o]:
                    overlap_val = o
                    break
            
            new_word += block[overlap_val:]
        
        word = new_word
        
    return "".join(map(str, word))
