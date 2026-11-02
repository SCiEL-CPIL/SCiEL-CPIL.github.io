import time
import random
import itertools
from collections import Counter

def construct(n):
    if n <= 1:
        return "1"
    
    def build_standard_superperm(n):
        word = [1]
        for m in range(2, n + 1):
            seen = set()
            perms = []
            for i in range(len(word) - (m - 1) + 1):
                w = tuple(word[i:i + m - 1])
                if w not in seen:
                    seen.add(w)
                    perms.append(w)
            
            new_word = []
            for p in perms:
                block = list(p) + [m] + list(p)
                max_ovl = 0
                min_len = min(len(new_word), len(block))
                for o in range(min_len, -1, -1):
                    if new_word[len(new_word) - o:] == block[:o]:
                        max_ovl = o
                        break
                new_word.extend(block[max_ovl:])
            word = new_word
        return word

    word = build_standard_superperm(n)
    
    all_perms = set(itertools.permutations(range(1, n + 1)))
    
    def get_windows(w):
        return [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    
    def is_valid(w):
        if len(w) < n:
            return False
        windows = set(get_windows(w))
        return all_perms.issubset(windows)
        
    if not is_valid(word):
        return "".join(map(str, word))
        
    start_time = time.time()
    max_time = 100
    
    L = len(word)
    
    while time.time() - start_time < max_time:
        progress = False
        # Try a batch of random deletions
        for _ in range(1000):
            if time.time() - start_time > max_time:
                break
            if len(word) <= n:
                break
                
            i = random.randint(1, len(word) - 2)
            new_word = word[:i] + word[i+1:]
            
            if is_valid(new_word):
                word = new_word
                L = len(word)
                progress = True
                
        if not progress:
            # If no progress in a batch, maybe we are stuck.
            # We can try a different strategy or break.
            # Given the small gap (8 chars), we might need more iterations.
            # Let's continue until time runs out.
            pass
            
    return "".join(map(str, word))
