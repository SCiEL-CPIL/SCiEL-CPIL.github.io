# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 80.0

    # Generate smooth numbers up to 128 with exponent vectors
    smooth = []
    smooth_exp = {}
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
    smooth_idx = {x: i for i, x in enumerate(smooth)}

    # Precompute incompatibility
    incompat = [[False] * n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        ei = smooth_exp[smooth[i]]
        for j in range(i + 1, n_smooth):
            ej = smooth_exp[smooth[j]]
            if (ei[0] <= ej[0] and ei[1] <= ej[1] and ei[2] <= ej[2]) or \
               (ej[0] <= ei[0] and ej[1] <= ei[1] and ej[2] <= ei[2]):
                incompat[i][j] = True
                incompat[j][i] = True

    # Precompute all valid k-term antichains for small k (k=2,3,4)
    # For each target sum, store the representation with fewest terms
    best_rep = {}  # target -> list of terms

    # k=1
    for x in smooth:
        if x <= 128:
            best_rep[x] = [x]

    # k=2: precompute all valid pairs
    for i in range(n_smooth):
        a = smooth[i]
        for j in range(i + 1, n_smooth):
            b = smooth[j]
            s = a + b
            if s <= 128:
                if not incompat[i][j]:
                    if s not in best_rep or len(best_rep[s]) > 2:
                        best_rep[s] = [a, b]

    # k=3: precompute all valid triples
    for i in range(n_smooth):
        a = smooth[i]
        for j in range(i + 1, n_smooth):
            b = smooth[j]
            if incompat[i][j]:
                continue
            s2 = a + b
            for k in range(j + 1, n_smooth):
                c = smooth[k]
                if incompat[i][k] or incompat[j][k]:
                    continue
                s = s2 + c
                if s <= 128:
                    if s not in best_rep or len(best_rep[s]) > 3:
                        best_rep[s] = [a, b, c]

    # k=4: precompute all valid quadruples
    for i in range(n_smooth):
        a = smooth[i]
        for j in range(i + 1, n_smooth):
            b = smooth[j]
            if incompat[i][j]:
                continue
            for k in range(j + 1, n_smooth):
                c = smooth[k]
                if incompat[i][k] or incompat[j][k]:
                    continue
                s3 = a + b + c
                for m in range(k + 1, n_smooth):
                    d = smooth[m]
                    if incompat[i][m] or incompat[j][m] or incompat[k][m]:
                        continue
                    s = s3 + d
                    if s <= 128:
                        if s not in best_rep or len(best_rep[s]) > 4:
                            best_rep[s] = [a, b, c, d]

    # For remaining targets, use backtracking
    memo = {}

    def backtrack(target, start_idx, current_sum, current_terms, current_indices, max_terms, min_terms):
        if current_sum == target:
            if len(current_terms) >= min_terms:
                return current_terms[:]
            return None
        if current_sum > target:
            return None
        if len(current_terms) >= max_terms:
            return None

        remaining = target - current_sum
        if remaining <= 0:
            return None

        # Pruning: check if we can still reach target with remaining terms
        # Minimum sum of (max_terms - len(current_terms)) smallest available terms
        # This is expensive, so only do it when close to the limit
        if len(current_terms) > max_terms - 3:
            # Check if enough small terms remain
            available = [smooth[idx] for idx in range(start_idx, n_smooth) if smooth[idx] <= remaining]
            if len(available) < (max_terms - len(current_terms)):
                return None
            # Sum of smallest needed terms
            available.sort()
            needed = max_terms - len(current_terms)
            if sum(available[:needed]) > remaining:
                return None

        for idx in range(start_idx, n_smooth):
            term = smooth[idx]
            if term > remaining:
                break

            # Check compatibility with all existing terms
            valid = True
            for existing_idx in current_indices:
                if incompat[idx][existing_idx]:
                    valid = False
                    break

            if valid:
                current_terms.append(term)
                current_indices.append(idx)
                res = backtrack(target, idx + 1, current_sum + term, current_terms, current_indices, max_terms, min_terms)
                if res:
                    return res
                current_terms.pop()
                current_indices.pop()
        return None

    def find_rep(n):
        if n in memo:
            return memo[n]
        if n in best_rep:
            memo[n] = best_rep[n]
            return best_rep[n]

        # Try k=5,6,7 with backtracking
        for k in range(5, 8):
            res = backtrack(n, 0, 0, [], [], k, k)
            if res:
                memo[n] = res
                return res

        # For larger k, use randomized search
        import random
        rng = random.Random(n)
        pool = [x for x in smooth if x < n]
        pool_set = set(pool)

        for k in range(8, min(31, len(pool) + 1)):
            for attempt in range(200):
                if time.time() > deadline - 5:
                    break
                terms = rng.sample(pool, k)
                if sum(terms) == n:
                    # Check antichain
                    valid = True
                    indices = [smooth_idx[t] for t in terms]
                    for i in range(len(indices)):
                        for j in range(i + 1, len(indices)):
                            if incompat[indices[i]][indices[j]]:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        memo[n] = sorted(terms)
                        return sorted(terms)
            if time.time() > deadline - 5:
                break

        # Fallback: exhaustive backtracking without depth limit
        res = backtrack(n, 0, 0, [], [], 30, 1)
        if res:
            memo[n] = res
            return res

        # Absolute fallback
        memo[n] = [1]
        return [1]

    answers = []
    for n in range(1, 129):
        if time.time() > deadline - 5:
            for m in range(n, 129):
                if m in memo:
                    answers.append(memo[m])
                elif m in best_rep:
                    answers.append(best_rep[m])
                else:
                    found = find_rep(m)
                    answers.append(found)
            break

        found = find_rep(n)
        if found is None:
            found = [n] if n in smooth_set else [1]
        answers.append(found)

    return answers
# EVOLVE-BLOCK-END
