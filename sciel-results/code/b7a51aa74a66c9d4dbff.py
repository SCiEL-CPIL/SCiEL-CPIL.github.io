import time
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110
    
    total_words = q ** n
    
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx
    
    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word
    
    # Precompute coverage
    coverage = []
    for idx in range(total_words):
        word = idx_to_word(idx)
        cov = [idx]
        for pos in range(n):
            for val in range(q):
                if val != word[pos]:
                    new_word = word[:]
                    new_word[pos] = val
                    cov.append(word_to_idx(new_word))
        coverage.append(cov)
    
    # Greedy set cover
    uncovered = [True] * total_words
    uncovered_count = total_words
    codewords = []
    used = [False] * total_words
    
    while uncovered_count > 0 and time.time() < deadline - 30:
        best_cw = -1
        best_count = 0
        for idx in range(total_words):
            if used[idx]:
                continue
            count = 0
            for w in coverage[idx]:
                if uncovered[w]:
                    count += 1
            if count > best_count:
                best_count = count
                best_cw = idx
        if best_cw == -1 or best_count == 0:
            break
        used[best_cw] = True
        codewords.append(best_cw)
        for w in coverage[best_cw]:
            if uncovered[w]:
                uncovered[w] = False
                uncovered_count -= 1
    
    # Build coverage count
    cov_count = [0] * total_words
    cw_set = set(codewords)
    for cw in codewords:
        for w in coverage[cw]:
            cov_count[w] += 1
    
    # Redundancy removal
    def try_remove_redundant():
        removed = True
        while removed and time.time() < deadline - 10:
            removed = False
            for cw in list(cw_set):
                redundant = True
                for w in coverage[cw]:
                    if cov_count[w] == 1:
                        redundant = False
                        break
                if redundant:
                    for w in coverage[cw]:
                        cov_count[w] -= 1
                    cw_set.discard(cw)
                    removed = True
    
    try_remove_redundant()
    
    # Local search
    if time.time() < deadline - 10:
        for _ in range(200):
            if time.time() > deadline - 10:
                break
            cw_list = list(cw_set)
            if not cw_list:
                break
            cw = random.choice(cw_list)
            word = idx_to_word(cw)
            new_word = word[:]
            pos = random.randint(0, n - 1)
            new_val = random.randint(0, q - 1)
            if new_val == word[pos]:
                continue
            new_word[pos] = new_val
            new_cw = word_to_idx(new_word)
            if new_cw in cw_set:
                continue
            # Check if cw can be removed
            can_remove = True
            for w in coverage[cw]:
                if cov_count[w] == 1:
                    can_remove = False
                    break
            if not can_remove:
                continue
            # Remove cw
            for w in coverage[cw]:
                cov_count[w] -= 1
            cw_set.discard(cw)
            # Add new_cw
            cw_set.add(new_cw)
            for w in coverage[new_cw]:
                cov_count[w] += 1
            # Try to remove a redundant codeword
            for cw2 in list(cw_set):
                if cw2 == new_cw:
                    continue
                redundant = True
                for w in coverage[cw2]:
                    if cov_count[w] == 1:
                        redundant = False
                        break
                if redundant:
                    for w in coverage[cw2]:
                        cov_count[w] -= 1
                    cw_set.discard(cw2)
                    break
    
    result = [idx_to_word(cw) for cw in cw_set]
    return result
