"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    start_time = time.time()
    timeout = 85.0  # Safety margin below 90s limit
    
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
    
    # Precompute divisibility: div[i][j] = True if smooth_numbers[i] divides smooth_numbers[j]
    div_matrix = [[False] * n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i != j:
                a, b = smooth_numbers[i], smooth_numbers[j]
                if b % a == 0 or a % b == 0:
                    div_matrix[i][j] = True
    
    # Map value -> index in smooth_numbers
    val_to_idx = {v: i for i, v in enumerate(smooth_numbers)}
    
    # Precompute all valid antichains of size 1-4 (small enough for brute force)
    # For each size k, store list of tuples that form antichains
    valid_antichains = {1: [], 2: [], 3: [], 4: []}
    
    # Size 1: all smooth numbers
    for v in smooth_numbers:
        valid_antichains[1].append((v,))
    
    # Size 2: all pairs where neither divides the other
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            if not div_matrix[i][j]:
                valid_antichains[2].append((smooth_numbers[i], smooth_numbers[j]))
    
    # Size 3: all triples forming antichains
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            if div_matrix[i][j]:
                continue
            for k in range(j + 1, n_smooth):
                if not div_matrix[i][k] and not div_matrix[j][k]:
                    valid_antichains[3].append((smooth_numbers[i], smooth_numbers[j], smooth_numbers[k]))
    
    # Size 4: all quadruples forming antichains
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            if div_matrix[i][j]:
                continue
            for k in range(j + 1, n_smooth):
                if div_matrix[i][k] or div_matrix[j][k]:
                    continue
                for l in range(k + 1, n_smooth):
                    if not div_matrix[i][l] and not div_matrix[j][l] and not div_matrix[k][l]:
                        valid_antichains[4].append((smooth_numbers[i], smooth_numbers[j], smooth_numbers[k], smooth_numbers[l]))
    
    # For faster lookup, create a dict mapping sum -> list of antichains for each size
    sum_to_antichains = {1: {}, 2: {}, 3: {}, 4: {}}
    for k in range(1, 5):
        for ach in valid_antichains[k]:
            s = sum(ach)
            if s <= 128:
                if s not in sum_to_antichains[k]:
                    sum_to_antichains[k][s] = []
                sum_to_antichains[k][s].append(ach)
    
    answers = []
    
    for n in range(1, 129):
        if time.time() - start_time > timeout:
            break
        
        found = None
        
        # Try increasing number of terms
        for k in range(1, 5):
            # Check if there's a precomputed antichain of size k summing to n
            if k in sum_to_antichains and n in sum_to_antichains[k]:
                # Pick the first one
                found = list(sum_to_antichains[k][n][0])
                break
        
        if found is None:
            # Fallback: use combinations for k >= 5 or if precomputation missed something
            candidates = [x for x in smooth_numbers if x <= n]
            cand_set = set(candidates)
            
            for k in range(1, min(8, len(candidates) + 1)):  # Limit to 8 terms max
                if time.time() - start_time > timeout:
                    break
                
                if k == 1:
                    if n in smooth_set:
                        found = [n]
                        break
                    continue
                
                if k == 2:
                    for a in candidates:
                        b = n - a
                        if b <= a:
                            continue
                        if b in cand_set:
                            if b % a != 0 and a % b != 0:
                                found = [a, b]
                                break
                    if found is not None:
                        break
                    continue
                
                if k == 3:
                    # Efficient 3-term search
                    for i, a in enumerate(candidates):
                        if a >= n:
                            break
                        for j in range(i + 1, len(candidates)):
                            b = candidates[j]
                            if a + b >= n:
                                break
                            c = n - a - b
                            if c <= b:
                                continue
                            if c in cand_set:
                                # Check antichain property
                                ia, ib, ic = val_to_idx[a], val_to_idx[b], val_to_idx[c]
                                if not div_matrix[ia][ib] and not div_matrix[ia][ic] and not div_matrix[ib][ic]:
                                    found = [a, b, c]
                                    break
                        if found is not None:
                            break
                    if found is not None:
                        break
                    continue
                
                if k == 4:
                    # Efficient 4-term search
                    for i, a in enumerate(candidates):
                        if a >= n:
                            break
                        for j in range(i + 1, len(candidates)):
                            b = candidates[j]
                            if a + b >= n:
                                break
                            for k2 in range(j + 1, len(candidates)):
                                c = candidates[k2]
                                if a + b + c >= n:
                                    break
                                d = n - a - b - c
                                if d <= c:
                                    continue
                                if d in cand_set:
                                    # Check antichain property
                                    ia, ib, ic, id_ = val_to_idx[a], val_to_idx[b], val_to_idx[c], val_to_idx[d]
                                    if not div_matrix[ia][ib] and not div_matrix[ia][ic] and not div_matrix[ia][id_] and \
                                       not div_matrix[ib][ic] and not div_matrix[ib][id_] and not div_matrix[ic][id_]:
                                        found = [a, b, c, d]
                                        break
                                if found is not None:
                                    break
                            if found is not None:
                                break
                        if found is not None:
                            break
                    if found is not None:
                        break
                    continue
                
                # For k >= 5, use combinations (should rarely be needed)
                for terms in combinations(candidates, k):
                    if sum(terms) != n:
                        continue
                    
                    is_antichain = True
                    for ii in range(len(terms)):
                        for jj in range(ii + 1, len(terms)):
                            if div_matrix[val_to_idx[terms[ii]]][val_to_idx[terms[jj]]]:
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
            if n in smooth_set:
                found = [n]
            else:
                found = [n]  # Should not happen
        
        answers.append(found)
    
    # If we timed out, fill remaining
    while len(answers) < 128:
        n = len(answers) + 1
        found = None
        for k in range(1, 5):
            if k in sum_to_antichains and n in sum_to_antichains[k]:
                found = list(sum_to_antichains[k][n][0])
                break
        if found is None:
            found = [n] if n in smooth_set else [1, n - 1]
        answers.append(found)
    
    return answers
# EVOLVE-BLOCK-END
