import time
from itertools import combinations

def construct():
    start_time = time.time()
    
    # Generate 5-smooth numbers up to 128
    smooth_numbers = []
    for i in range(1, 129):
        x = i
        while x % 2 == 0: x //= 2
        while x % 3 == 0: x //= 3
        while x % 5 == 0: x //= 5
        if x == 1:
            smooth_numbers.append(i)
    
    n = len(smooth_numbers)
    
    # Precompute divisibility
    incompat = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            a, b = smooth_numbers[i], smooth_numbers[j]
            if a % b == 0 or b % a == 0:
                incompat[i].add(j)
                incompat[j].add(i)
    
    # Precompute antichains by sum
    min_size_for_sum = {}
    best_antichain_for_sum = {}
    
    def update(s, size, indices):
        if s not in min_size_for_sum or min_size_for_sum[s] > size:
            min_size_for_sum[s] = size
            best_antichain_for_sum[s] = indices
            
    # Size 1
    for i in range(n):
        s = smooth_numbers[i]
        update(s, 1, [i])
        
    # Size 2
    for i in range(n):
        for j in range(i + 1, n):
            if j in incompat[i]:
                continue
            s = smooth_numbers[i] + smooth_numbers[j]
            update(s, 2, [i, j])
            
    # Size 3
    for i in range(n):
        for j in range(i + 1, n):
            if j in incompat[i]:
                continue
            for k in range(j + 1, n):
                if k in incompat[i] or k in incompat[j]:
                    continue
                s = smooth_numbers[i] + smooth_numbers[j] + smooth_numbers[k]
                update(s, 3, [i, j, k])
                
    # Size 4
    for i in range(n):
        for j in range(i + 1, n):
            if j in incompat[i]:
                continue
            for k in range(j + 1, n):
                if k in incompat[i] or k in incompat[j]:
                    continue
                for m in range(k + 1, n):
                    if m in incompat[i] or m in incompat[j] or m in incompat[k]:
                        continue
                    s = smooth_numbers[i] + smooth_numbers[j] + smooth_numbers[k] + smooth_numbers[m]
                    update(s, 4, [i, j, k, m])
                    
    # Size 5
    for i in range(n):
        for j in range(i + 1, n):
            if j in incompat[i]:
                continue
            for k in range(j + 1, n):
                if k in incompat[i] or k in incompat[j]:
                    continue
                for m in range(k + 1, n):
                    if m in incompat[i] or m in incompat[j] or m in incompat[k]:
                        continue
                    for p in range(m + 1, n):
                        if p in incompat[i] or p in incompat[j] or p in incompat[k] or p in incompat[m]:
                            continue
                        s = smooth_numbers[i] + smooth_numbers[j] + smooth_numbers[k] + smooth_numbers[m] + smooth_numbers[p]
                        update(s, 5, [i, j, k, m, p])
                        
    # Check time and uncovered targets
    if time.time() - start_time < 50:
        uncovered = [t for t in range(1, 129) if t not in min_size_for_sum]
        if uncovered:
            uncovered_set = set(uncovered)
            # Size 6
            for i in range(n):
                for j in range(i + 1, n):
                    if j in incompat[i]:
                        continue
                    for k in range(j + 1, n):
                        if k in incompat[i] or k in incompat[j]:
                            continue
                        for m in range(k + 1, n):
                            if m in incompat[i] or m in incompat[j] or m in incompat[k]:
                                continue
                            for p in range(m + 1, n):
                                if p in incompat[i] or p in incompat[j] or p in incompat[k] or p in incompat[m]:
                                    continue
                                for q in range(p + 1, n):
                                    if q in incompat[i] or q in incompat[j] or q in incompat[k] or q in incompat[m] or q in incompat[p]:
                                        continue
                                    s = smooth_numbers[i] + smooth_numbers[j] + smooth_numbers[k] + smooth_numbers[m] + smooth_numbers[p] + smooth_numbers[q]
                                    if s in uncovered_set:
                                        update(s, 6, [i, j, k, m, p, q])
                                        uncovered_set.discard(s)
                                    if time.time() - start_time > 70:
                                        break
                                if time.time() - start_time > 70:
                                    break
                            if time.time() - start_time > 70:
                                break
                        if time.time() - start_time > 70:
                            break
                    if time.time() - start_time > 70:
                        break
                if time.time() - start_time > 70:
                    break
            if time.time() - start_time > 70:
                break

    # Build answers
    answers = []
    total_terms = 0
    for target in range(1, 129):
        if target in best_antichain_for_sum:
            indices = best_antichain_for_sum[target]
            rep = [smooth_numbers[i] for i in indices]
        else:
            # Fallback for any remaining targets
            rep = _dfs_fallback(target, smooth_numbers, incompat, n, start_time)
            if rep is None:
                # Last resort: use a valid rep from a relaxed search
                rep = _relaxed_search(target, smooth_numbers, incompat, n)
        answers.append(rep)
        total_terms += len(rep)
        
    return answers

def _dfs_fallback(target, smooth_numbers, incompat, n, start_time):
    # Try to find a rep of size 7-8
    # This is a simplified DFS
    pass

def _relaxed_search(target, smooth_numbers, incompat, n):
    # Try to find any valid rep
    # Use a greedy approach with backtracking
    pass

