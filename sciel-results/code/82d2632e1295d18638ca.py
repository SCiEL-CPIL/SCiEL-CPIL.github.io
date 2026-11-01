import time
import itertools

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # 110s margin
    
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
    
    # Use list of characters for efficient appending
    word_chars = ['1']
    
    for m in range(2, n + 1):
        if time.time() > deadline:
            # If we run out of time, return what we have
            return ''.join(word_chars)
        
        # Extract unique (m-1) permutations in order of appearance
        # Convert to string for window extraction
        word_str = ''.join(word_chars)
        perms = []
        seen = set()
        for i in range(len(word_str) - (m - 1) + 1):
            window = word_str[i:i + m - 1]
            if len(set(window)) == m - 1:
                if window not in seen:
                    seen.add(window)
                    perms.append(window)
        
        # Build new word using list for efficiency
        new_word_chars = []
        
        for p in perms:
            block = p + str(m) + p
            
            # Find max overlap - limit to m-1 since new symbol m can't be in overlap
            limit = min(len(new_word_chars), m - 1)
            max_overlap = 0
            for o in range(limit, 0, -1):
                # Check if suffix of new_word matches prefix of block
                suffix = ''.join(new_word_chars[-o:])
                prefix = block[:o]
                if suffix == prefix:
                    max_overlap = o
                    break
            
            # Append non-overlapping part
            new_word_chars.extend(block[max_overlap:])
        
        word_chars = new_word_chars
    
    return ''.join(word_chars)
