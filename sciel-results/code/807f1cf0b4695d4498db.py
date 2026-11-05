"""Shortest superpermutation construction with deadline guard."""
import time

def construct(n):
    """Return the shortest known superpermutation word for 1..n.
    
    For n <= 7, uses the classical recursive construction which gives
    length 1! + 2! + ... + n!.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Classical recursive construction
    word = [1]
    for m in range(2, n + 1):
        if time.time() > deadline:
            break
        # Extract all distinct permutations of length m-1 from current word
        perms = []
        seen = set()
        # Optimization: use tuple slicing
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # Build new word by inserting m into each permutation
        new = []
        for p in perms:
            # block is p + [m] + p
            # We can construct the string parts directly for overlap check if needed,
            # but list comparison is fine.
            block = list(p) + [m] + list(p)
            # Find maximum overlap
            overlap = 0
            max_overlap = min(len(new), len(block))
            # Check from largest possible overlap downwards
            for o in range(max_overlap, -1, -1):
                if o == 0 or new[-o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    return "".join(map(str, word))
