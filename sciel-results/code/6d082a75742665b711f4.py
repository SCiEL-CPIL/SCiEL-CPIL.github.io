import random
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s out of 90s wall-clock

    N = 180

    # Precompute all forbidden triples: 1/a = 1/b + 1/c with a,b,c distinct
    # Parametrization: a = k*x*y, b = k*x*(x+y), c = k*y*(x+y), x<y, gcd(x,y)=1
    triples = []
    for k in range(1, N + 1):
        for x in range(1, N + 1):
            for y in range(x + 1, N + 1):
                a = k * x * y
                if a > N:
                    break
                b = k * x * (x + y)
                c = k * y * (x + y)
                if b > N or c > N:
                    continue
                if a == b or a == c or b == c:
                    continue
                if gcd(x, y) != 1:
                    continue
                triples.append((a, b, c))

    # Build conflict graph: two elements conflict if they appear together in some triple
    # For each element, store the set of elements it conflicts with
    conflict = {i: set() for i in range(1, N + 1)}
    for (a, b, c) in triples:
        conflict[a].add(b)
        conflict[a].add(c)
        conflict[b].add(a)
        conflict[b].add(c)
        conflict[c].add(a)
        conflict[c].add(b)

    def is_valid(S):
        """Check if set S has no forbidden triple."""
        S_set = set(S)
        for (a, b, c) in triples:
            if a in S_set and b in S_set and c in S_set:
                return False
        return True

    def count_conflicts(S, n):
        """Count how many elements in S conflict with n."""
        return len(conflict[n] & S)

    def greedy_add(S, candidates, order):
        """Add elements from candidates in given order if safe."""
        S = set(S)
        for n in order:
            if n in S:
                continue
            # Check if adding n creates any forbidden triple
            safe = True
            for t in triples:
                if n in t:
                    others = [x for x in t if x != n]
                    if others[0] in S and others[1] in S:
                        safe = False
                        break
            if safe:
                S.add(n)
        return S

    def local_search(S, max_iters=2000):
        """Try 1-swap moves: remove one element, add another."""
        S = set(S)
        best_size = len(S)
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            # Try removing each element and see if we can add more
            for rem in list(S):
                if time.time() > deadline:
                    break
                S_trial = S - {rem}
                # Try to add back elements not in S_trial
                for add in range(1, N + 1):
                    if add in S_trial or add == rem:
                        continue
                    # Check if adding add is safe
                    safe = True
                    for t in triples:
                        if add in t:
                            others = [x for x in t if x != add]
                            if others[0] in S_trial and others[1] in S_trial:
                                safe = False
                                break
                    if safe:
                        S_trial.add(add)
                        break  # Only add one per removal attempt
                if len(S_trial) > len(S):
                    S = S_trial
                    improved = True
                    break
        return S

    # Strategy 1: Start with upper half, greedily add lower half in random order
    best_S = set()
    best_size = 0

    # Multiple random restarts with different strategies
    for trial in range(30):
        if time.time() > deadline - 5:
            break

        # Start with upper half
        S = set(range(91, 181))

        # Random order for lower half
        candidates = list(range(1, 91))
        random.shuffle(candidates)

        # Greedy add
        S = greedy_add(S, candidates, candidates)

        # Local search
        if time.time() < deadline - 5:
            S = local_search(S, max_iters=500)

        if len(S) > best_size:
            best_size = len(S)
            best_S = S

    # Final verification
    if not is_valid(best_S):
        # Fallback: just return upper half which is always valid
        best_S = set(range(91, 181))

    return list(best_S)
