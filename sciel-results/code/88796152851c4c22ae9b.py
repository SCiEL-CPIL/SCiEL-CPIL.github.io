import math
import time

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s margin

    # Precompute weights
    weights = [1.0 / (512 - a) for a in range(512)]

    # Precompute coprime matrix for fast lookup
    coprime = [[False] * 512 for _ in range(512)]
    for a in range(1, 512):
        for b in range(a + 1, 512):
            coprime[a][b] = (math.gcd(a, b) == 1)
            coprime[b][a] = coprime[a][b]
    for a in range(1, 512):
        coprime[a][a] = True  # self-coprime (not used but safe)

    # Greedy initialization: pick largest weights first
    chosen = []
    for a in range(511, 0, -1):
        if all(coprime[a][b] for b in chosen):
            chosen.append(a)

    best_set = chosen[:]
    best_score = sum(weights[a] for a in chosen)

    def greedy_fill(remaining_set, max_iter=512):
        """Fill remaining_set with best available coprime elements."""
        s = list(remaining_set)
        s_set = set(s)
        for b in range(511, 0, -1):
            if b in s_set:
                continue
            if all(coprime[b][c] for c in s):
                s.append(b)
                s_set.add(b)
        return s

    # Local search: try removing one element and re-greedy-filling
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i, a in enumerate(best_set):
            # Try removing a
            remaining = [x for j, x in enumerate(best_set) if j != i]
            new_set = greedy_fill(remaining)

            new_score = sum(weights[b] for b in new_set)
            if new_score > best_score:
                best_score = new_score
                best_set = new_set
                improved = True
                break

    # Second pass: try swapping one element with another
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i, a in enumerate(best_set):
            # Try replacing a with each candidate b not in set
            remaining = [x for j, x in enumerate(best_set) if j != i]
            rem_set = set(remaining)
            best_candidate = None
            best_candidate_score = best_score
            for b in range(511, 0, -1):
                if b in rem_set or b == a:
                    continue
                if all(coprime[b][c] for c in remaining):
                    new_score = sum(weights[x] for x in remaining) + weights[b]
                    if new_score > best_candidate_score:
                        best_candidate_score = new_score
                        best_candidate = b

            if best_candidate is not None:
                best_set = [best_candidate if x == a else x for x in best_set]
                best_score = best_candidate_score
                improved = True
                break

    # Third pass: 2-swap local search (remove two, add two)
    if time.time() < deadline:
        n = len(best_set)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(n):
                for j in range(i + 1, n):
                    remaining = [x for k, x in enumerate(best_set) if k != i and k != j]
                    rem_set = set(remaining)
                    best_pair = None
                    best_pair_score = best_score
                    # Try adding pairs
                    for b in range(511, 0, -1):
                        if b in rem_set:
                            continue
                        if not all(coprime[b][c] for c in remaining):
                            continue
                        for c in range(b - 1, 0, -1):
                            if c in rem_set or c == b:
                                continue
                            if not all(coprime[c][d] for d in remaining) or not coprime[b][c]:
                                continue
                            new_score = sum(weights[x] for x in remaining) + weights[b] + weights[c]
                            if new_score > best_pair_score:
                                best_pair_score = new_score
                                best_pair = (b, c)
                    if best_pair is not None:
                        b, c = best_pair
                        new_set = remaining + [b, c]
                        best_set = new_set
                        best_score = best_pair_score
                        improved = True
                        break
                if improved:
                    break

    return best_set
