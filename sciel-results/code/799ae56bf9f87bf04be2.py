import time
import random
import math
from itertools import permutations

def construct(n):
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "121321"
    
    start_time = time.time()
    time_limit = 115  # Safety margin before 150s kill
    
    # Hierarchical divide-and-conquer
    # Partition symbols into two subsets
    symbols = list(range(1, n + 1))
    
    # Try different partition sizes
    best_word = None
    best_len = n * math.factorial(n)  # Upper bound
    
    # For n=7, try partitions like (3,4) or (4,3)
    partition_sizes = []
    if n >= 4:
        for a in range(2, n - 1):
            partition_sizes.append((a, n - a))
    
    for a, b in partition_sizes:
        if time.time() - start_time > time_limit:
            break
        
        group1 = symbols[:a]
        group2 = symbols[a:]
        
        # Recursively construct superpermutations for each group
        word1 = _construct_group(group1, time_limit - (time.time() - start_time))
        word2 = _construct_group(group2, time_limit - (time.time() - start_time))
        
        # Merge the two words via symbol-shuffling
        merged = _merge_superperms(word1, word2, group1, group2, 
                                   time_limit - (time.time() - start_time))
        
        if merged is not None:
            # Validate and check length
            if _is_valid_superperm(merged, n):
                if len(merged) < best_len:
                    best_len = len(merged)
                    best_word = merged
    
    # If hierarchical approach didn't work well, fall back to path-based
    if best_word is None or len(best_word) > n * math.factorial(n) - 1:
        fallback = _path_based_construct(n, time_limit - (time.time() - start_time))
        if fallback is not None and _is_valid_superperm(fallback, n):
            if len(fallback) < best_len:
                best_len = len(fallback)
                best_word = fallback
    
    if best_word is None:
        # Last resort: simple concatenation
        best_word = "".join(str(s) for p in permutations(range(1, n+1)) for s in p)
    
    return best_word


def _construct_group(symbols, time_budget):
    """Construct a superpermutation for a subset of symbols"""
    n = len(symbols)
    if n <= 1:
        return str(symbols[0]) if symbols else ""
    if n == 2:
        return f"{symbols[0]}{symbols[1]}{symbols[0]}"
    if n == 3:
        # For 3 symbols, use known pattern
        a, b, c = symbols
        return f"{a}{b}{a}{c}{b}{a}"
    
    # For larger groups, use path-based approach
    return _path_based_construct(n, time_budget)


def _merge_superperms(word1, word2, group1, group2, time_budget):
    """Merge two superpermutations via symbol-shuffling"""
    # This is a simplified merge - in practice, this would be more sophisticated
    # For now, try interleaving strategies
    
    # Strategy 1: Concatenate with overlap
    # Find maximum overlap between end of word1 and start of word2
    max_overlap = 0
    for l in range(min(len(word1), len(word2)), 0, -1):
        if word1[-l:] == word2[:l]:
            max_overlap = l
            break
    
    merged = word1 + word2[max_overlap:]
    return merged


def _is_valid_superperm(word, n):
    """Check if word contains all permutations of 1..n as consecutive substrings"""
    if len(word) < n:
        return False
    
    perms = set(permutations(range(1, n + 1)))
    found = set()
    
    for i in range(len(word) - n + 1):
        window = tuple(int(c) for c in word[i:i+n])
        if window in perms:
            found.add(window)
            if len(found) == len(perms):
                return True
    
    return len(found) == len(perms)


def _path_based_construct(n, time_budget):
    """Original path-based approach as fallback"""
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "121321"
    
    start_time = time.time()
    
    # Generate all permutations of 1..n-1
    perms = list(permutations(range(1, n)))
    k = len(perms)  # (n-1)!
    
    # Precompute overlap matrix
    ov = [[0] * k for _ in range(k)]
    for i in range(k):
        pi = perms[i]
        for j in range(k):
            pj = perms[j]
            max_k = n - 1
            for l in range(max_k, 0, -1):
                if pi[-l:] == pj[:l]:
                    ov[i][j] = l
                    break
    
    # Greedy initial path
    path = [0]
    unvisited = set(range(1, k))
    while unvisited:
        last = path[-1]
        best_next = -1
        best_ov = -1
        for u in unvisited:
            if ov[last][u] > best_ov:
                best_ov = ov[last][u]
                best_next = u
        if best_next == -1:
            best_next = random.choice(list(unvisited))
        path.append(best_next)
        unvisited.remove(best_next)
    
    def get_score(p):
        s = 0
        for i in range(len(p) - 1):
            s += ov[p[i]][p[i+1]]
        return s
    
    best_score = get_score(path)
    best_path = path[:]
    
    # 2-opt local search
    improved = True
    while improved and time.time() - start_time < time_budget:
        improved = False
        for i in range(k - 1):
            if time.time() - start_time > time_budget:
                break
            for j in range(i + 1, min(i + 20, k)):
                old_score = 0
                if i > 0:
                    old_score += ov[path[i-1]][path[i]]
                if j < k - 1:
                    old_score += ov[path[j]][path[j+1]]
                for t in range(i, j):
                    old_score += ov[path[t]][path[t+1]]
                    
                new_score = 0
                if i > 0:
                    new_score += ov[path[i-1]][path[j]]
                if j < k - 1:
                    new_score += ov[path[i]][path[j+1]]
                for t in range(i, j):
                    new_score += ov[path[t+1]][path[t]]
                    
                delta = new_score - old_score
                if delta > 0:
                    path[i:j+1] = path[i:j+1][::-1]
                    best_score += delta
                    improved = True
                    break
            if improved:
                break
    
    # Reconstruct word
    word = []
    for t, idx in enumerate(best_path):
        p = perms[idx]
        if t == 0:
            word.extend(p)
            word.append(n)
            word.extend(p)
        else:
            prev_idx = best_path[t-1]
            prev_p = perms[prev_idx]
            ov_val = ov[prev_idx][idx]
            word.extend(p[ov_val:])
            word.append(n)
            word.extend(p)
    
    return "".join(map(str, word))
