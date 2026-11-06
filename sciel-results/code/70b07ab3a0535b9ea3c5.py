import time
from itertools import combinations

_START = time.time()
DEADLINE = 85  # Leave 5s margin

def construct():
    t0 = time.time()
    
    # 1. Generate all 2^a * 3^b * 5^c smooth numbers <= 128
    smooth = []
    smooth_set = set()
    for a in range(0, 8):
        p2 = 2 ** a
        if p2 > 128: break
        for b in range(0, 6):
            p3 = 3 ** b
            if p2 * p3 > 128: break
            for c in range(0, 5):
                p5 = 5 ** c
                v = p2 * p3 * p5
                if v > 128: break
                if v not in smooth_set:
                    smooth_set.add(v)
                    smooth.append(v)
    smooth.sort()
    n_smooth = len(smooth)
    
    # Map value to index for quick lookup
    val_to_idx = {v: i for i, v in enumerate(smooth)}
    is_smooth_arr = [False] * 129
    for v in smooth:
        is_smooth_arr[v] = True
        
    # Precompute divisibility relationship
    # div[i][j] = True if smooth[i] divides smooth[j]
    div = [[False] * n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        vi = smooth[i]
        for j in range(n_smooth):
            if i != j and smooth[j] % vi == 0:
                div[i][j] = True

    # best_count[s] = minimum number of terms
    # best_terms[s] = the actual terms
    best_count = [1000] * 129
    best_terms = [None] * 129
    
    # Helper to check if a tuple of indices forms an antichain
    def is_antichain_indices(indices):
        k = len(indices)
        for i in range(k):
            for j in range(i + 1, k):
                if div[indices[i]][indices[j]] or div[indices[j]][indices[i]]:
                    return False
        return True

    # Precompute all valid antichains of size 1 to 5
    # Size 1
    for i in range(n_smooth):
        s = smooth[i]
        if s <= 128 and 1 < best_count[s]:
            best_count[s] = 1
            best_terms[s] = [smooth[i]]

    # Size 2
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            if div[i][j] or div[j][i]:
                continue
            s = smooth[i] + smooth[j]
            if s > 128:
                continue
            if 2 < best_count[s]:
                best_count[s] = 2
                best_terms[s] = [smooth[i], smooth[j]]

    # Size 3
    for i in range(n_smooth):
        vi = smooth[i]
        for j in range(i + 1, n_smooth):
            vj = smooth[j]
            if div[i][j] or div[j][i]:
                continue
            for k in range(j + 1, n_smooth):
                vk = smooth[k]
                if div[i][k] or div[k][i] or div[j][k] or div[k][j]:
                    continue
                s = vi + vj + vk
                if s > 128:
                    continue
                if 3 < best_count[s]:
                    best_count[s] = 3
                    best_terms[s] = [vi, vj, vk]

    # Size 4
    for i in range(n_smooth):
        vi = smooth[i]
        for j in range(i + 1, n_smooth):
            vj = smooth[j]
            if div[i][j] or div[j][i]:
                continue
            for k in range(j + 1, n_smooth):
                vk = smooth[k]
                if div[i][k] or div[k][i] or div[j][k] or div[k][j]:
                    continue
                for l in range(k + 1, n_smooth):
                    vl = smooth[l]
                    if div[i][l] or div[l][i] or div[j][l] or div[l][j] or div[k][l] or div[l][k]:
                        continue
                    s = vi + vj + vk + vl
                    if s > 128:
                        continue
                    if 4 < best_count[s]:
                        best_count[s] = 4
                        best_terms[s] = [vi, vj, vk, vl]

    # Size 5
    for i in range(n_smooth):
        vi = smooth[i]
        for j in range(i + 1, n_smooth):
            vj = smooth[j]
            if div[i][j] or div[j][i]:
                continue
            for k in range(j + 1, n_smooth):
                vk = smooth[k]
                if div[i][k] or div[k][i] or div[j][k] or div[k][j]:
                    continue
                for l in range(k + 1, n_smooth):
                    vl = smooth[l]
                    if div[i][l] or div[l][i] or div[j][l] or div[l][j] or div[k][l] or div[l][k]:
                        continue
                    for m in range(l + 1, n_smooth):
                        vm = smooth[m]
                        if (div[i][m] or div[m][i] or div[j][m] or div[m][j] or 
                            div[k][m] or div[m][k] or div[l][m] or div[m][l]):
                            continue
                        s = vi + vj + vk + vl + vm
                        if s > 128:
                            continue
                        if 5 < best_count[s]:
                            best_count[s] = 5
                            best_terms[s] = [vi, vj, vk, vl, vm]

    # Fallback backtracking for any n not covered (should be rare)
    def backtrack(rem, min_idx, current_list, current_set):
        if rem == 0:
            return current_list
        if time.time() - t0 > DEADLINE:
            return None
        for idx in range(min_idx, n_smooth):
            s = smooth[idx]
            if s > rem:
                break
            valid = True
            for x in current_set:
                if s % x == 0 or x % s == 0:
                    valid = False
                    break
            if not valid:
                continue
            current_list.append(s)
            current_set.add(s)
            res = backtrack(rem - s, idx + 1, current_list, current_set)
            if res is not None:
                return res
            current_list.pop()
            current_set.remove(s)
        return None
    
    answers = []
    for n in range(1, 129):
        if best_terms[n] is not None:
            answers.append(best_terms[n])
        else:
            res = backtrack(n, 0, [], set())
            if res is not None:
                answers.append(res)
            else:
                # Last resort: find any valid representation
                # This should not happen for n <= 128
                res = backtrack(n, 0, [], set())
                if res is not None:
                    answers.append(res)
                else:
                    answers.append([1, 2, 3, 5, 7, 9, 11, 13, 15, 17])
    
    return answers
