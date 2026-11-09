import numpy as np
import random

def construct():
    N = 256
    best_S = set()
    best_score = -1
    
    def creates_4ap(x, in_S):
        for d in range(1, (N - x) // 3 + 1):
            if in_S[x + d] and in_S[x + 2 * d] and in_S[x + 3 * d]:
                return True
        for d in range(1, min(x, (N - x) // 2) + 1):
            if in_S[x - d] and in_S[x + d] and in_S[x + 2 * d]:
                return True
        for d in range(1, min(x // 2, N - x) + 1):
            if in_S[x - 2 * d] and in_S[x - d] and in_S[x + d]:
                return True
        for d in range(1, x // 3 + 1):
            if in_S[x - 3 * d] and in_S[x - 2 * d] and in_S[x - d]:
                return True
        return False

    def calc_gain(x, in_S):
        gain = 0
        for d in range(1, (N - x) // 2 + 1):
            if in_S[x + d] and in_S[x + 2 * d]:
                gain += 1
        for d in range(1, min(x, N - x) + 1):
            if in_S[x - d] and in_S[x + d]:
                gain += 1
        for d in range(1, x // 2 + 1):
            if in_S[x - 2 * d] and in_S[x - d]:
                gain += 1
        return gain

    def count_3aps(S):
        in_S = [False] * (N + 1)
        for x in S:
            in_S[x] = True
        count = 0
        for a in range(1, N + 1):
            if not in_S[a]:
                continue
            for d in range(1, (N - a) // 2 + 1):
                if in_S[a + d] and in_S[a + 2 * d]:
                    count += 1
        return count

    for _ in range(100):
        S = set()
        in_S = [False] * (N + 1)
        candidates = set(range(1, N + 1))
        
        while candidates:
            best_gain = -1
            best_candidates = []
            
            for x in candidates:
                if creates_4ap(x, in_S):
                    continue
                
                gain = calc_gain(x, in_S)
                
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [x]
                elif gain == best_gain:
                    best_candidates.append(x)
            
            if not best_candidates:
                break
            
            best_x = random.choice(best_candidates)
            
            S.add(best_x)
            in_S[best_x] = True
            candidates.remove(best_x)
            
        score = count_3aps(S)
        if score > best_score:
            best_score = score
            best_S = S
            
    return sorted(best_S)
