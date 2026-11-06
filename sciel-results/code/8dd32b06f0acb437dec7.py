import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin for 150s limit
    
    best_set = [1, 2, 5, 11]
    
    def compute_sums(S):
        sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                sums.add(S[i] + S[j])
        return sums

    def can_add(x, S, used_sums):
        # x is not in S
        for y in S:
            if x + y in used_sums:
                return False
        if 2 * x in used_sums:
            return False
        return True

    def greedy_solve(randomize=False, lookahead=False):
        S = [1, 2, 5, 11]
        used_sums = compute_sums(S)
        candidates = [x for x in range(12, 285)]
        
        while candidates:
            valid_candidates = []
            for x in candidates:
                if can_add(x, S, used_sums):
                    valid_candidates.append(x)
            
            if not valid_candidates:
                break
            
            if lookahead:
                best_x = None
                best_score = -1
                # To speed up, we can sample if valid_candidates is large?
                # 280 is small enough to check all.
                for x in valid_candidates:
                    temp_S = S + [x]
                    temp_used = set(used_sums)
                    for y in S:
                        temp_used.add(x + y)
                    temp_used.add(2 * x)
                    
                    score = 0
                    for z in candidates:
                        if z == x:
                            continue
                        if can_add(z, temp_S, temp_used):
                            score += 1
                    if score > best_score:
                        best_score = score
                        best_x = x
                x = best_x
            elif randomize:
                x = random.choice(valid_candidates)
            else:
                x = min(valid_candidates)
                
            S.append(x)
            for y in S[:-1]:
                used_sums.add(x + y)
            used_sums.add(2 * x)
            
            # Remove x from candidates
            # List removal is O(N), N=280, acceptable.
            candidates.remove(x)
            
            if time.time() > deadline:
                break
        
        return S

    # 1. Fast randomized runs
    # These are very fast, so we can do many.
    while time.time() < deadline - 20.0: # Reserve 20s for lookahead
        res = greedy_solve(randomize=True, lookahead=False)
        if len(res) > len(best_set):
            best_set = res

    # 2. Lookahead runs
    # These are slower, so do fewer.
    while time.time() < deadline:
        res = greedy_solve(randomize=False, lookahead=True)
        if len(res) > len(best_set):
            best_set = res

    return best_set
