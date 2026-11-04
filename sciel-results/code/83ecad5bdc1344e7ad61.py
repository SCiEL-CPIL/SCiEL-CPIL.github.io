import time

def construct(n):
    """
    Construct the shortest known superpermutation for n symbols.
    For n <= 5, the classic recursive construction is optimal.
    For n >= 6, it is not optimal, but it is the standard baseline.
    This function implements the classic recursive construction.
    """
    # Deadline guard
    start_time = time.time()
    deadline = 110.0  # 10 seconds margin before 120s limit
    
    # Base case
    if n == 1:
        return "1"
    if n == 2:
        return "121"
        
    # Classic recursive construction
    # S_1 = "1"
    # S_n is constructed from S_{n-1} by replacing each (n-1)-permutation P with P n P
    # and overlapping as much as possible.
    
    word = [1]
    
    for m in range(2, n + 1):

        # Extract all unique (m-1)-permutations from the current word in order of appearance
        perms = []
        seen = set()
        # The word length is L. Windows of size m-1.
        # Note: The word contains all (m-1)! permutations of 1..m-1.
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            # A window is a permutation if it has distinct elements
            if len(set(window)) == m - 1:
                if window not in seen:
                    seen.add(window)
                    perms.append(window)
        
        # Construct the new word
        new_word = []
        for p in perms:
            # Block is p + [m] + p
            block = list(p) + [m] + list(p)
            
            # Find maximum overlap with the end of new_word
            # Theoretically, the overlap is m-2 for this construction.
            # Check this specific case first for speed.
            target_overlap = m - 2
            overlap = 0
            if len(new_word) >= target_overlap and len(block) >= target_overlap:
                if new_word[-target_overlap:] == block[:target_overlap]:
                    overlap = target_overlap
                else:
                    # Fallback to general search
                    max_overlap = min(len(new_word), len(block))
                    for o in range(max_overlap, -1, -1):
                        if o == 0:
                            break
                        if new_word[len(new_word) - o:] == block[:o]:
                            overlap = o
                            break
            else:
                # Handle small cases
                max_overlap = min(len(new_word), len(block))
                for o in range(max_overlap, -1, -1):
                    if o == 0:
                        break
                    if new_word[len(new_word) - o:] == block[:o]:
                        overlap = o
                        break
            
            # Append the non-overlapping part
            new_word.extend(block[overlap:])
        
        word = new_word
        
    return "".join(map(str, word))
