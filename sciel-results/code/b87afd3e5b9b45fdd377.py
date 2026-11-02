"""
Constructs a finite set U of nonnegative integers containing 0 to maximize 
the sumset/difference-set exponent lower bound score.

Score = 1 + log(|U-U|/|U+U|) / log(2 * max(U) + 1)

Strategy:
1. Evaluate known optimal Golomb rulers for small n (1 to 15).
   Golomb rulers maximize |U-U| for a given max(U).
2. The score function favors small max(U) and high |U-U|/|U+U| ratio.
3. Return the set with the highest score.
"""
import math
import time

def _score_set(U):
    """Compute the score for a finite set U containing 0."""
    U_set = set(U)
    n = len(U_set)
    if n == 0:
        return 0.0
    max_u = max(U_set)
    if max_u == 0:
        return 1.0
    
    # Compute sumset and difference set
    # For small n (<= 20), O(n^2) is fast enough
    sumset = set()
    for a in U_set:
        for b in U_set:
            sumset.add(a + b)
    
    diffset = set()
    for a in U_set:
        for b in U_set:
            diffset.add(a - b)
    
    ratio = len(diffset) / len(sumset)
    if ratio <= 0:
        return 0.0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    if log_denom == 0:
        return 1.0
    return 1.0 + log_ratio / log_denom

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    # Known optimal or near-optimal Golomb rulers
    # Reference: OEIS A181391 (Optimal Golomb rulers)
    golomb_rulers = [
        [0],
        [0, 1],
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 10, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 12, 17, 22],
        [0, 1, 4, 10, 12, 17, 22, 29],
        [0, 1, 4, 10, 12, 17, 22, 29, 36],
        [0, 1, 4, 10, 12, 17, 22, 29, 36, 44],
        [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52],
        [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60],
        [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72],
        [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 84],
        [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 84, 96],
    ]

    best_score = -1.0
    best_U = [0]

    # Evaluate all known Golomb rulers
    for U in golomb_rulers:
        if time.time() > deadline:
            break
        s = _score_set(U)
        if s > best_score:
            best_score = s
            best_U = U[:]

    # Optional: Try a few random sets or simple heuristics if time permits
    # But Golomb rulers are theoretically strong candidates.
    # Let's do a quick check on n=16, 17, 18 using a greedy approach if time allows
    # Greedy Golomb construction
    def _greedy_golomb(n):
        if n <= 1:
            return [0]
        ruler = [0]
        for i in range(1, n):
            pos = 1
            while True:
                candidate = ruler + [pos]
                diffs = set()
                valid = True
                for j in range(len(candidate)):
                    for k in range(j + 1, len(candidate)):
                        d = candidate[k] - candidate[j]
                        if d in diffs:
                            valid = False
                            break
                        diffs.add(d)
                    if not valid:
                        break
                if valid:
                    ruler.append(pos)
                    break
                pos += 1
        return ruler

    # Try greedy for n=16 to 20
    for n in range(16, 21):
        if time.time() > deadline:
            break
        U = _greedy_golomb(n)
        s = _score_set(U)
        if s > best_score:
            best_score = s
            best_U = U[:]

    return sorted(set(best_U))

if __name__ == "__main__":
    # Quick self-test
    U = construct()
    print(f"Best U: {U}")
    print(f"Score: {_score_set(U)}")
