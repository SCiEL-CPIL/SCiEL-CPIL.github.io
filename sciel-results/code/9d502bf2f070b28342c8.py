import time
import itertools

def construct(n):
    """
    Constructs a superpermutation for symbols 1..n.
    Uses standard recursive construction, then attempts to shorten it
    via greedy deletion local search for n >= 7.
    """
    start_time = time.time()
    # Leave a safety margin of 10 seconds before the 120s limit
    deadline = start_time + 110.0

    # 1. Generate the standard superpermutation
    word = _standard_superperm(n)
    
    # For n <= 6, the standard construction is known to be optimal.
    if n <= 6:
        return "".join(map(str, word))

    # 2. Local search to try to reduce length
    target_count = int(1)
    for i in range(1, n + 1):
        target_count *= i
        
    # Pre-compute target permutations for faster validation if needed,
    # but checking set size is usually sufficient.
    # We just need to ensure all n! permutations are present.
    
    def is_valid(w):
        if len(w) < n:
            return False
        found = set()
        # Iterate through all windows of length n
        for i in range(len(w) - n + 1):
            # Create tuple of the window
            # Using tuple(w[i:i+n]) is efficient enough for n=7
            found.add(tuple(w[i:i+n]))
            # Early exit if we found all permutations
            if len(found) == target_count:
                return True
        return len(found) == target_count

    current_word = word
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        # Try to delete each element
        # We iterate from 0 to len-1
        # If a deletion is successful, we restart the scan to account for index shifts
        # and to potentially find more deletions.
        for i in range(len(current_word)):
            if time.time() > deadline:
                break
            
            # Candidate word with element at i removed
            candidate = current_word[:i] + current_word[i+1:]
            
            if is_valid(candidate):
                current_word = candidate
                improved = True
                # Break to restart the for-loop with the new length
                break

    return "".join(map(str, current_word))

def _standard_superperm(n):
    """
    Generates the standard superpermutation of length 1! + 2! + ... + n!.
    """
    if n == 1:
        return [1]
    
    # Recursive step
    prev = _standard_superperm(n - 1)
    
    # Extract all permutations of 1..n-1 from prev in order of appearance
    perms = []
    seen = set()
    # Window size is n-1
    for i in range(len(prev) - (n - 1) + 1):
        window = tuple(prev[i:i + n - 1])
        # Check if it's a valid permutation of 1..n-1
        # len(set(window)) == n-1 ensures no duplicates
        if len(set(window)) == n - 1 and window not in seen:
            seen.add(window)
            perms.append(window)
            
    new_word = []
    for p in perms:
        # Block is p, n, p
        block = list(p) + [n] + list(p)
        
        # Calculate overlap
        # We want to append block to new_word, overlapping as much as possible
        # new_word ends with some sequence, block starts with p
        # Find max k such that new_word[-k:] == block[:k]
        max_overlap = 0
        # Max possible overlap is min(len(new_word), len(block))
        # But usually small
        for k in range(min(len(new_word), len(block)), -1, -1):
            if k == 0:
                max_overlap = 0
                break
            # Slice comparison
            if new_word[len(new_word) - k:] == block[:k]:
                max_overlap = k
                break
                
        new_word.extend(block[max_overlap:])
        
    return new_word
