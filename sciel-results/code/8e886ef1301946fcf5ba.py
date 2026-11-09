import time
import random

N = 256
DEADLINE = 55.0

def construct():
    start = time.time()
    random.seed(123)
    
    def greedy_build():
        S = []
        sums = set()
        for x in range(1, N + 1):
            added = [x + a for a in S] + [2 * x]
            if len(set(added)) == len(added) and not sums.intersection(added):
                S.append(x)
                sums.update(added)
        return S
    
    def get_sums(S):
        s = set()
        for i, a in enumerate(S):
            for j in range(i, len(S)):
                s.add(a + S[j])
        return s
    
    def is_sidon(S):
        s = set()
        for i, a in enumerate(S):
            for j in range(i, len(S)):
                v = a + S[j]
                if v in s:
                    return False
                s.add(v)
        return True
    
    def is_maximal(S, ss):
        Sset = set(S)
        for x in range(1, N + 1):
            if x in Sset:
                continue
            blocked = False
            for a in S:
                if x + a in ss:
                    blocked = True
                    break
            if not blocked and 2 * x in ss:
                blocked = True
            if not blocked:
                return False
        return True
    
    best_S = greedy_build()
    best_len = len(best_S)
    
    for _ in range(500):
        if time.time() - start > DEADLINE:
            break
        
        S = list(best_S)
        ss = get_sums(S)
        
        # Try to remove elements
        changed = True
        while changed and time.time() - start < DEADLINE:
            changed = False
            for i in range(len(S)):
                if time.time() - start > DEADLINE:
                    break
                trial = S[:i] + S[i+1:]
                if not trial:
                    continue
                if is_sidon(trial):
                    ss_t = get_sums(trial)
                    if is_maximal(trial, ss_t):
                        S = trial
                        ss = ss_t
                        changed = True
                        break
        
        if len(S) < best_len:
            best_S = list(S)
            best_len = len(S)
    
    return best_S
