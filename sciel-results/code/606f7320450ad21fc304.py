import time
import random

N = 256

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s margin
    
    def get_sums(S):
        sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                sums.add(S[i] + S[j])
        return sums

    def greedy_maximal(order=None):
        if order is None:
            order = list(range(1, N + 1))
        selected = []
        sums = set()
        for x in order:
            # Check if x can be added
            ok = True
            for a in selected:
                if (x + a) in sums:
                    ok = False
                    break
            if ok and (2 * x) in sums:
                ok = False
            
            if ok:
                selected.append(x)
                sums.update([x + a for a in selected[:-1]] + [2 * x])
        return selected

    # Initialize
    S = greedy_maximal()
    best_S = S.copy()
    best_size = len(S)
    
    while time.time() < deadline:
        if len(S) < 2:
            break
            
        found_swap = False
        attempts = 0
        # Use more attempts if we have lots of time
        max_attempts = 50 if time.time() < deadline - 10.0 else 10
        
        while attempts < max_attempts and time.time() < deadline:
            attempts += 1
            # Pick random distinct a, b
            idx_a, idx_b = random.sample(range(len(S)), 2)
            a, b = S[idx_a], S[idx_b]
            
            S_rem = S.copy()
            S_rem.remove(a)
            S_rem.remove(b)
            
            sums_rem = get_sums(S_rem)
            
            # Try candidates c
            candidates = [c for c in range(1, N + 1) if c not in S_rem]
            random.shuffle(candidates)
            
            for c in candidates:
                # Check Sidon for S_rem U {c}
                if 2 * c in sums_rem:
                    continue
                    
                new_sums = set()
                valid_sidon = True
                for x in S_rem:
                    s = c + x
                    if s in sums_rem or s in new_sums:
                        valid_sidon = False
                        break
                    new_sums.add(s)
                
                if not valid_sidon:
                    continue
                
                # S_new is Sidon. Check Maximal.
                S_new = S_rem.copy()
                S_new.append(c)
                S_new_set = set(S_new)
                
                sums_new = sums_rem.union(new_sums)
                sums_new.add(2 * c)
                
                is_max = True
                for y in range(1, N + 1):
                    if y in S_new_set:
                        continue
                    blocked = False
                    if 2 * y in sums_new:
                        blocked = True
                    else:
                        for x in S_new:
                            if (y + x) in sums_new:
                                blocked = True
                                break
                    if not blocked:
                        is_max = False
                        break
                
                if is_max:
                    S = S_new
                    found_swap = True
                    break
            
            if found_swap:
                break
        
        if len(S) < best_size:
            best_size = len(S)
            best_S = S.copy()
            
        if not found_swap:
            # Restart with new random greedy
            order = list(range(1, N + 1))
            random.shuffle(order)
            S = greedy_maximal(order)
            
    return best_S
