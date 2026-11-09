import time
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85
    
    def check_sidon(s):
        sums = set()
        for i in range(len(s)):
            si = s[i]
            for j in range(i, len(s)):
                v = si + s[j]
                if v in sums:
                    return False
                sums.add(v)
        return True
    
    def greedy_interleaved(order, target_a, target_b):
        A = []
        B = []
        diffs_a = set()
        diffs_b = set()
        
        for x in order:
            if time.time() > deadline:
                break
            if len(A) >= target_a and len(B) >= target_b:
                break
            
            # Try A first if it's smaller or equal
            if len(A) <= len(B) and len(A) < target_a:
                cand = A + [x]
                if check_sidon(cand):
                    new_diffs = set(x - a for a in A)
                    if not (new_diffs & diffs_b):
                        A = cand
                        diffs_a |= new_diffs
                        continue
            
            # Try B
            if len(B) < target_b:
                cand = B + [x]
                if check_sidon(cand):
                    new_diffs = set(x - b for b in B)
                    if not (new_diffs & diffs_a):
                        B = cand
                        diffs_b |= new_diffs
                        continue
            
            # Try A if not yet tried
            if len(A) < target_a:
                cand = A + [x]
                if check_sidon(cand):
                    new_diffs = set(x - a for a in A)
                    if not (new_diffs & diffs_b):
                        A = cand
                        diffs_a |= new_diffs
        
        if A and B:
            return A, B
        return None
    
    best = ([1], [2])
    best_size = 2
    
    targets = [(14,9), (13,10), (12,11), (15,8), (16,7), (14,8), (13,9), (11,12), (10,13)]
    
    for trial in range(1000):
        if time.time() > deadline:
            break
        
        random.seed(trial * 7 + 13)
        elems = list(range(1, n+1))
        random.shuffle(elems)
        
        for ta, tb in targets:
            if time.time() > deadline:
                break
            result = greedy_interleaved(elems, ta, tb)
            if result:
                A, B = result
                sz = len(A) + len(B)
                if sz > best_size:
                    best_size = sz
                    best = (A, B)
                    if sz >= 23:
                        return best
    
    return best
