import itertools
import time

def construct(n):
    if n == 1:
        return "1"
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Classic recursive superpermutation construction:
    # S(1) = "1"
    # S(m) = for each permutation p of S(m-1)'s windows: p + [m] + p, maximally overlapped
    # This produces length 1! + 2! + ... + n!
    word = [1]
    for m in range(2, n + 1):
        if time.time() > deadline:
            break
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find maximum overlap between end of new and start of block
            overlap = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    # Convert to string
    result = "".join(map(str, word))
    
    # Optional: try a quick local improvement pass if time permits
    if time.time() < deadline - 5:
        # Check if we can find any shorter valid superpermutation by trying
        # to remove redundant middle sections. This is a heuristic polish.
        # For n=7, the classic construction gives 5913; best known is 5905.
        # A full optimization is intractable, but we can try small local moves.
        # However, given the time budget, we'll just return the classic result.
        pass
    
    return result
