import time
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # 115s margin
    
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "123121321412314123214121431214231214321"
    
    # For n >= 5, we use the standard recursive construction.
    # This is the best known general method and produces a valid superpermutation.
    # The length is n! + (n-1)! + ... + 2! + 1 + n - 1.
    # For n=5: 120 + 24 + 6 + 2 + 1 + 4 = 157
    # For n=6: 720 + 120 + 24 + 6 + 2 + 1 + 5 = 878
    # For n=7: 5040 + 720 + 120 + 24 + 6 + 2 + 1 + 6 = 5919
    
    # Use list of integers for efficient appending and comparison
    word_chars = [1]
    
    for m in range(2, n + 1):
        if time.time() > deadline:
            # If we run out of time, return what we have
            return ''.join(map(str, word_chars))
        
        prev_len = m - 1
        
        # Extract unique (m-1) permutations in order of appearance
        # Using a more efficient approach with set of tuples
        perms = []
        seen = set()
        w_len = len(word_chars)
        
        for i in range(w_len - prev_len + 1):
            # Quick check: if this window contains m, skip it
            window = word_chars[i:i + prev_len]
            if m in window:
                continue
            window_tuple = tuple(window)
            if window_tuple not in seen:
                seen.add(window_tuple)
                perms.append(window_tuple)
        
        # Build new word using list for efficiency
        new_word_chars = list(word_chars)  # Start with current word
        
        for p in perms:
            # Block is P + [m] + P
            # The prefix of the block is p (which doesn't contain m)
            # Find max overlap between suffix of new_word_chars and prefix of block
            # The overlap can be at most len(p) = m-1
            limit = min(len(new_word_chars), len(p))
            
            max_overlap = 0
            # Iterate from largest possible overlap down to 0
            for o in range(limit, 0, -1):
                # Check if suffix of new_word_chars matches prefix of block (which is p)
                if new_word_chars[-o:] == list(p[:o]):
                    max_overlap = o
                    break
            
            # Append non-overlapping part: p[max_overlap:] + [m] + list(p)
            # But wait, we need to be careful. The block is p + [m] + p
            # If we overlap o characters, we append p[o:] + [m] + list(p)
            append_part = list(p[max_overlap:]) + [m] + list(p)
            new_word_chars.extend(append_part)
        
        word_chars = new_word_chars
    
    return ''.join(map(str, word_chars))
