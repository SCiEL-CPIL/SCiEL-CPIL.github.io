import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    # 1. Generate smooth numbers (2^a * 3^b * 5^c) <= 128
    smooths = set()
    for a in range(8):
        for b in range(6):
            for c in range(4):
                val = (2 ** a) * (3 ** b) * (5 ** c)
                if val <= 128:
                    smooths.add(val)
    smooth_list = sorted(smooths)
    smooth_set = set(smooth_list)
    
    def is_antichain(terms):
        for i in range(len(terms)):
            for j in range(i + 1, len(terms)):
                a, b = terms[i], terms[j]
                if a % b == 0 or b % a == 0:
                    return False
        return True

    results = []
    
    # 2. For each target 1..128, find min-term antichain representation
    for target in range(1, 129):
        best_terms = None
        best_count = 31
        
        # Try increasing k
        for k in range(1, 7): # Limit search depth to 6 for performance
            if time.time() > deadline:
                break
            
            found = False
            if k == 1:
                if target in smooth_set:
                    best_terms = [target]
                    best_count = 1
                    found = True
            elif k == 2:
                for i in range(len(smooth_list)):
                    a = smooth_list[i]
                    if a >= target: break
                    b = target - a
                    if b in smooth_set and b > a:
                        if is_antichain([a, b]):
                            best_terms = [a, b]
                            best_count = 2
                            found = True
                            break
            elif k == 3:
                for i in range(len(smooth_list)):
                    a = smooth_list[i]
                    if a * 3 >= target: break # Simple pruning
                    for j in range(i + 1, len(smooth_list)):
                        b = smooth_list[j]
                        if a + b >= target: break
                        c = target - a - b
                        if c in smooth_set and c > b:
                            if is_antichain([a, b, c]):
                                best_terms = [a, b, c]
                                best_count = 3
                                found = True
                                break
                    if found: break
            else:
                # For k >= 4, use a simple backtracking search
                def dfs(rem, start_idx, current_terms):
                    if rem == 0:
                        if is_antichain(current_terms):
                            return list(current_terms)
                        return None
                    if len(current_terms) >= k:
                        return None
                    for idx in range(start_idx, len(smooth_list)):
                        val = smooth_list[idx]
                        if val > rem:
                            break
                        # Pruning: if val divides any existing or vice versa, skip
                        valid_div = True
                        for t in current_terms:
                            if t % val == 0 or val % t == 0:
                                valid_div = False
                                break
                        if not valid_div:
                            continue
                        current_terms.append(val)
                        res = dfs(rem - val, idx + 1, current_terms)
                        if res:
                            return res
                        current_terms.pop()
                    return None

                res = dfs(target, 0, [])
                if res:
                    best_terms = res
                    best_count = len(res)
                    found = True
            
            if found:
                break
        
        if best_terms is None:
            # Fallback: Random search for a valid antichain
            for _ in range(50):
                if time.time() > deadline: break
                terms = []
                rem = target
                candidates = [s for s in smooth_list if s < rem]
                if not candidates: continue
                valid = True
                for _ in range(30):
                    if rem == 0: break
                    if not candidates:
                        valid = False
                        break
                    c = random.choice(candidates)
                    if c in terms: continue
                    div_ok = True
                    for t in terms:
                        if t % c == 0 or c % t == 0:
                            div_ok = False
                            break
                    if not div_ok: continue
                    terms.append(c)
                    rem -= c
                    candidates = [s for s in candidates if s not in terms and s < rem]
                if valid and rem == 0 and terms:
                    best_terms = terms
                    best_count = len(terms)
                    break
        
        if best_terms is None:
            # Last resort
            if target in smooth_set:
                best_terms = [target]
            else:
                best_terms = []
        
        results.append(best_terms)

    return results
