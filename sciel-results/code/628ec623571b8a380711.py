# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    
    # Generate smooth numbers up to 128 with their exponent vectors
    smooth = []
    smooth_exp = {}  # value -> (a, b, c) where value = 2^a * 3^b * 5^c
    for x in range(1, 129):
        y = x
        a = b = c = 0
        while y % 2 == 0:
            y //= 2
            a += 1
        while y % 3 == 0:
            y //= 3
            b += 1
        while y % 5 == 0:
            y //= 5
            c += 1
        if y == 1:
            smooth.append(x)
            smooth_exp[x] = (a, b, c)
    
    smooth_set = set(smooth)
    n_smooth = len(smooth)
    
    # Precompute divisibility: divs[i][j] = True if smooth[i] divides smooth[j]
    # Using exponent vectors: smooth[i] divides smooth[j] iff exp_i <= exp_j component-wise
    divs = [[False] * n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        ei = smooth_exp[smooth[i]]
        for j in range(i + 1, n_smooth):
            ej = smooth_exp[smooth[j]]
            if ei[0] <= ej[0] and ei[1] <= ej[1] and ei[2] <= ej[2]:
                divs[i][j] = True
    
    def is_antichain(terms):
        """Check if terms form an antichain using exponent vectors."""
        exps = [smooth_exp[t] for t in terms]
        for i in range(len(exps)):
            for j in range(i + 1, len(exps)):
                ei, ej = exps[i], exps[j]
                # Check if ei <= ej or ej <= ei (one divides the other)
                if (ei[0] <= ej[0] and ei[1] <= ej[1] and ei[2] <= ej[2]) or \
                   (ej[0] <= ei[0] and ej[1] <= ei[1] and ej[2] <= ei[2]):
                    return False
        return True
    
    def find_rep(n):
        # k=1
        if n in smooth_set:
            return [n]
        
        # k=2
        for i in range(n_smooth):
            a = smooth[i]
            if a >= n:
                break
            b = n - a
            if b <= a:
                continue
            if b in smooth_set:
                # Check antichain: a does not divide b and b does not divide a
                # Since a < b, only need to check if a divides b
                ei = smooth_exp[a]
                ej = smooth_exp[b]
                if not (ei[0] <= ej[0] and ei[1] <= ej[1] and ei[2] <= ej[2]):
                    return [a, b]
        
        # k=3
        for i in range(n_smooth):
            a = smooth[i]
            if a >= n:
                break
            for j in range(i + 1, n_smooth):
                b = smooth[j]
                if a + b >= n:
                    break
                c = n - a - b
                if c <= b:
                    continue
                if c in smooth_set:
                    # a < b < c, check antichain
                    if is_antichain([a, b, c]):
                        return [a, b, c]
        
        # k=4
        for i in range(n_smooth):
            a = smooth[i]
            if a >= n:
                break
            for j in range(i + 1, n_smooth):
                b = smooth[j]
                if a + b >= n:
                    break
                for k in range(j + 1, n_smooth):
                    c = smooth[k]
                    if a + b + c >= n:
                        break
                    d = n - a - b - c
                    if d <= c:
                        continue
                    if d in smooth_set:
                        if is_antichain([a, b, c, d]):
                            return [a, b, c, d]
        
        # k=5
        for i in range(n_smooth):
            a = smooth[i]
            if a >= n:
                break
            for j in range(i + 1, n_smooth):
                b = smooth[j]
                if a + b >= n:
                    break
                for k in range(j + 1, n_smooth):
                    c = smooth[k]
                    if a + b + c >= n:
                        break
                    for l in range(k + 1, n_smooth):
                        d = smooth[l]
                        if a + b + c + d >= n:
                            break
                        e = n - a - b - c - d
                        if e <= d:
                            continue
                        if e in smooth_set:
                            if is_antichain([a, b, c, d, e]):
                                return [a, b, c, d, e]
        
        # k=6: use combinations for speed
        pool = [x for x in smooth if x < n]
        if len(pool) >= 6:
            for combo in combinations(pool, 6):
                if sum(combo) == n and is_antichain(combo):
                    return list(combo)
        
        # k=7
        if len(pool) >= 7:
            for combo in combinations(pool, 7):
                if sum(combo) == n and is_antichain(combo):
                    return list(combo)
        
        # Generic backtracking for k > 7
        def backtrack(target, start_idx, current_sum, current_list, current_indices):
            if current_sum == target:
                return current_list[:]
            if current_sum > target:
                return None
            
            for idx in range(start_idx, n_smooth):
                term = smooth[idx]
                if current_sum + term > target:
                    break
                
                # Check antichain property with new term using exponent vectors
                valid = True
                for existing_idx in current_indices:
                    e1 = smooth_exp[smooth[existing_idx]]
                    e2 = smooth_exp[term]
                    # Check if either divides the other
                    if (e1[0] <= e2[0] and e1[1] <= e2[1] and e1[2] <= e2[2]) or \
                       (e2[0] <= e1[0] and e2[1] <= e1[1] and e2[2] <= e1[2]):
                        valid = False
                        break
                
                if valid:
                    current_list.append(term)
                    current_indices.append(idx)
                    res = backtrack(target, idx + 1, current_sum + term, current_list, current_indices)
                    if res:
                        return res
                    current_list.pop()
                    current_indices.pop()
            return None

        res = backtrack(n, 0, 0, [], [])
        if res:
            return res
        
        # Absolute fallback: should not happen
        return [1]

    answers = []
    for n in range(1, 129):
        if time.time() > deadline - 5:
            # If time is tight, break but we should have all answers by now
            pass
        
        found = find_rep(n)
        if found is None:
            # This should not happen, but handle it
            found = [n] if n in smooth_set else [1]
        
        answers.append(found)
    
    return answers
# EVOLVE-BLOCK-END
