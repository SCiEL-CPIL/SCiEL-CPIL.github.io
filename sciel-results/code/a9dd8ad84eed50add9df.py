import time
from itertools import combinations

def construct():
    start_time = time.time()
    timeout = 80.0  # Safety margin below 90s limit
    
    # Generate all 5-smooth numbers up to 128
    smooth_numbers = []
    for x in range(1, 129):
        y = x
        for p in (2, 3, 5):
            while y % p == 0:
                y //= p
        if y == 1:
            smooth_numbers.append(x)
    
    smooth_set = set(smooth_numbers)
    n_smooth = len(smooth_numbers)
    
    # Map value -> index
    val_to_idx = {v: i for i, v in enumerate(smooth_numbers)}
    
    # Precompute divisibility relation: divs[i] = set of indices j where smooth_numbers[i] divides smooth_numbers[j] or vice versa
    # For antichain: no two elements should divide each other
    incompatible = [set() for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            a, b = smooth_numbers[i], smooth_numbers[j]
            if b % a == 0 or a % b == 0:
                incompatible[i].add(j)
                incompatible[j].add(i)
    
    # For each target n, find the minimum number of terms
    answers = []
    
    # Precompute for each n the best (minimum term count) representation
    # We'll do this efficiently by checking k=1,2,3,... in order
    
    # For k=1: n must be smooth
    # For k=2: find a,b smooth, a+b=n, a!=b, a not dividing b and b not dividing a
    # For k=3: find a,b,c smooth, a+b+c=n, all distinct, pairwise incomparable
    # etc.
    
    # Let's precompute all valid antichains of small size up to some limit
    # and use them to quickly find representations
    
    # Strategy: for each target n, try k=1,2,3,4,5,... until found
    # Use efficient search for small k
    
    # Precompute: for each pair (i,j) that form an antichain, their sum
    # pair_sum_to_indices: sum -> list of (i,j) pairs that are antichains
    pair_antichains = {}  # sum -> list of (idx_i, idx_j)
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            if j not in incompatible[i]:
                s = smooth_numbers[i] + smooth_numbers[j]
                if s <= 128:
                    if s not in pair_antichains:
                        pair_antichains[s] = []
                    pair_antichains[s].append((i, j))
    
    # Precompute triples that form antichains
    # triple_antichains: sum -> list of (i,j,k)
    triple_antichains = {}
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            if j in incompatible[i]:
                continue
            for k in range(j + 1, n_smooth):
                if k in incompatible[i] or k in incompatible[j]:
                    continue
                s = smooth_numbers[i] + smooth_numbers[j] + smooth_numbers[k]
                if s <= 128:
                    if s not in triple_antichains:
                        triple_antichains[s] = []
                    triple_antichains[s].append((i, j, k))
    
    # Precompute quadruples
    quad_antichains = {}
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            if j in incompatible[i]:
                continue
            for k in range(j + 1, n_smooth):
                if k in incompatible[i] or k in incompatible[j]:
                    continue
                for l in range(k + 1, n_smooth):
                    if l in incompatible[i] or l in incompatible[j] or l in incompatible[k]:
                        continue
                    s = smooth_numbers[i] + smooth_numbers[j] + smooth_numbers[k] + smooth_numbers[l]
                    if s <= 128:
                        if s not in quad_antichains:
                            quad_antichains[s] = []
                        quad_antichains[s].append((i, j, k, l))
    
    # Precompute quintuples
    quint_antichains = {}
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            if j in incompatible[i]:
                continue
            for k in range(j + 1, n_smooth):
                if k in incompatible[i] or k in incompatible[j]:
                    continue
                for l in range(k + 1, n_smooth):
                    if l in incompatible[i] or l in incompatible[j] or l in incompatible[k]:
                        continue
                    for m in range(l + 1, n_smooth):
                        if m in incompatible[i] or m in incompatible[j] or m in incompatible[k] or m in incompatible[l]:
                            continue
                        s = smooth_numbers[i] + smooth_numbers[j] + smooth_numbers[k] + smooth_numbers[l] + smooth_numbers[m]
                        if s <= 128:
                            if s not in quint_antichains:
                                quint_antichains[s] = []
                            quint_antichains[s].append((i, j, k, l, m))
    
    # Now for each target, find minimum term count
    for n in range(1, 129):
        if time.time() - start_time > timeout:
            # Fill remaining with best effort
            while len(answers) < 128:
                m = len(answers) + 1
                if m in smooth_set:
                    answers.append([m])
                else:
                    answers.append([1, m - 1] if (m - 1) in smooth_set else [m])
            break
        
        found = None
        
        # k=1
        if n in smooth_set:
            found = [n]
        # k=2
        elif n in pair_antichains:
            i, j = pair_antichains[n][0]
            found = [smooth_numbers[i], smooth_numbers[j]]
        # k=3
        elif n in triple_antichains:
            i, j, k = triple_antichains[n][0]
            found = [smooth_numbers[i], smooth_numbers[j], smooth_numbers[k]]
        # k=4
        elif n in quad_antichains:
            i, j, k, l = quad_antichains[n][0]
            found = [smooth_numbers[i], smooth_numbers[j], smooth_numbers[k], smooth_numbers[l]]
        # k=5
        elif n in quint_antichains:
            i, j, k, l, m = quint_antichains[n][0]
            found = [smooth_numbers[i], smooth_numbers[j], smooth_numbers[k], smooth_numbers[l], smooth_numbers[m]]
        else:
            # Fallback: search for k=6 or more
            candidates = [x for x in smooth_numbers if x <= n]
            found = None
            for k in range(6, len(candidates) + 1):
                if time.time() - start_time > timeout:
                    break
                for terms in combinations(candidates, k):
                    if sum(terms) != n:
                        continue
                    # Check antichain
                    is_antichain = True
                    for idx in range(len(terms)):
                        for jdx in range(idx + 1, len(terms)):
                            if val_to_idx[terms[jdx]] in incompatible[val_to_idx[terms[idx]]]:
                                is_antichain = False
                                break
                        if not is_antichain:
                            break
                    if is_antichain:
                        found = list(terms)
                        break
                if found is not None:
                    break
            
            if found is None:
                # Absolute fallback
                found = [n] if n in smooth_set else [1, n - 1]
        
        answers.append(found)
    
    # If we broke out early, fill remaining
    while len(answers) < 128:
        n = len(answers) + 1
        if n in smooth_set:
            answers.append([n])
        elif n in pair_antichains:
            i, j = pair_antichains[n][0]
            answers.append([smooth_numbers[i], smooth_numbers[j]])
        elif n in triple_antichains:
            i, j, k = triple_antichains[n][0]
            answers.append([smooth_numbers[i], smooth_numbers[j], smooth_numbers[k]])
        elif n in quad_antichains:
            i, j, k, l = quad_antichains[n][0]
            answers.append([smooth_numbers[i], smooth_numbers[j], smooth_numbers[k], smooth_numbers[l]])
        elif n in quint_antichains:
            i, j, k, l, m = quint_antichains[n][0]
            answers.append([smooth_numbers[i], smooth_numbers[j], smooth_numbers[k], smooth_numbers[l], smooth_numbers[m]])
        else:
            answers.append([n] if n in smooth_set else [1, n - 1])
    
    return answers
