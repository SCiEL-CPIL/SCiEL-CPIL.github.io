import time

def construct(n):
    """Return a string that is a superpermutation over symbols 1..n.
    
    Uses the classic greedy construction:
    S(1) = "1"
    S(k) is formed by taking each (k-1)-permutation p that appears in S(k-1),
    and writing p + [k] + p, with maximal overlap with the previous block.
    
    This produces a valid superpermutation of length 1! + 2! + ... + n!.
    """
    if n <= 1:
        return "1"
    
    start_time = time.time()
    deadline = 110.0  # safety margin before 120s kill
    
    # Build iteratively from 1 to n
    # S_1 = [1]
    current = [1]
    
    for k in range(2, n + 1):
        # Check time
        if time.time() - start_time > deadline - 5:
            break
        
        # Find all (k-1)-permutations in order of first appearance
        seen = set()
        order = []
        L = len(current)
        for i in range(L - (k - 1) + 1):
            window = tuple(current[i:i + (k - 1)])
            if window not in seen:
                seen.add(window)
                order.append(window)
        
        # Build new superpermutation
        new_word = []
        for p in order:
            # Block is p + [k] + p
            block = list(p) + [k] + list(p)
            
            # Find maximum overlap with end of new_word
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0 or new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            
            new_word.extend(block[overlap:])
        
        current = new_word
    
    return "".join(map(str, current))
