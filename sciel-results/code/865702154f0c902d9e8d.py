import time
import random

def construct():
    start_time = time.time()
    deadline = 110  # Safety margin for 120s limit

    target_diffs = set(range(1, 31))

    def get_diffs(S):
        return set(y - x for i, x in enumerate(S) for y in S[i+1:])

    def is_sidon(S):
        if not S:
            return True
        diffs = get_diffs(S)
        return len(diffs) == len(S) * (len(S) - 1) // 2

    def covers_all(S):
        return get_diffs(S).issuperset(target_diffs)

    def greedy_construct(randomize=False):
        S = [0]
        covered = set()
        
        max_steps = 100
        for _ in range(max_steps):
            if time.time() - start_time > deadline:
                return None
            if covered.issuperset(target_diffs):
                return S
            
            missing = target_diffs - covered
            if not missing:
                return S

            if randomize:
                d_miss = random.choice(list(missing))
            else:
                d_miss = min(missing)
            
            candidates = set()
            for y in S:
                c1 = y + d_miss
                c2 = y - d_miss
                if c1 >= 0 and c1 not in S:
                    candidates.add(c1)
                if c2 >= 0 and c2 not in S:
                    candidates.add(c2)
            
            valid_candidates = []
            for c in candidates:
                new_S = sorted(S + [c])
                if is_sidon(new_S):
                    valid_candidates.append(c)
            
            if not valid_candidates:
                # Try to cover any missing diff
                found = False
                for d in missing:
                    cands_d = set()
                    for y in S:
                        c1 = y + d
                        c2 = y - d
                        if c1 >= 0 and c1 not in S:
                            cands_d.add(c1)
                        if c2 >= 0 and c2 not in S:
                            cands_d.add(c2)
                    
                    for c in cands_d:
                        new_S = sorted(S + [c])
                        if is_sidon(new_S):
                            new_cov = get_diffs(new_S)
                            if d in new_cov:
                                S = new_S
                                covered = new_cov
                                found = True
                                break
                    if found:
                        break
                
                if not found:
                    return None
                continue

            if randomize:
                c = random.choice(valid_candidates)
            else:
                c = min(valid_candidates)
            
            S.append(c)
            S.sort()
            covered = get_diffs(S)
        
        if covers_all(S):
            return S
        return None

    def local_search(S):
        S = sorted(S)
        improved = True
        while improved:
            if time.time() - start_time > deadline:
                break
            improved = False
            for i in range(len(S) - 1, -1, -1):
                x = S[i]
                
                # Check removal
                S_rem = S[:i] + S[i+1:]
                if covers_all(S_rem) and is_sidon(S_rem):
                    S = S_rem
                    improved = True
                    break
                
                # Check replacement
                if x == 0:
                    continue
                
                for y in range(x - 1, -1, -1):
                    if y in S:
                        continue
                    new_S = S[:i] + [y] + S[i+1:]
                    new_S.sort()
                    if is_sidon(new_S) and covers_all(new_S):
                        S = new_S
                        improved = True
                        break
            
        return S

    best_set = None
    best_max = float('inf')

    # 1. Deterministic Greedy
    S = greedy_construct(randomize=False)
    if S:
        S = local_search(S)
        max_elem = max(S)
        if max_elem < best_max:
            best_max = max_elem
            best_set = S[:]

    # 2. Randomized Greedy Restarts
    for restart in range(200):
        if time.time() - start_time > deadline:
            break
        
        S = greedy_construct(randomize=True)
        if S:
            S = local_search(S)
            max_elem = max(S)
            if max_elem < best_max:
                best_max = max_elem
                best_set = S[:]
        
        if best_max < 80:
            break

    if best_set:
        return best_set
    else:
        # Fallback
        return [0, 1, 3, 7, 12, 20, 31, 45, 62, 82, 105, 131]

