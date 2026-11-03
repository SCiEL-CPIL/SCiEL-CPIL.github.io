import time
import math

def _classic_superperm(n):
    """Build the classic superpermutation of length sum(k! for k=1..n)."""
    if n <= 1:
        return [1]
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = 0
            max_ov = min(len(new_word), len(block) - 1)
            for o in range(max_ov, -1, -1):
                if o == 0 or new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            new_word += block[overlap:]
        word = new_word
    return word

def construct(n):
    start_time = time.time()
    deadline = start_time + 110.0
    
    if n <= 1:
        return "1"
    
    word = _classic_superperm(n)
    return "".join(map(str, word))
