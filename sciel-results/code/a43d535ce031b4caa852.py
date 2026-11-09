"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    DEADLINE = 115  # safety margin before 120s evaluator limit
    N = 284
    PRESCRIBED = {1, 2, 5, 11}

    def compute_sums(S):
        s = set()
        for i, a in enumerate(S):
            for b in S[i:]:
                s.add(a + b)
        return s

    def can_add(S, sums, x):
        if x in S:
            return False
        for y in S:
            if x + y in sums:
                return False
        if 2 * x in sums:
            return False
        return True

    def greedy_construct(rng, max_iter=50):
        """Try multiple random orderings of candidates."""
        base = list(PRESCRIBED)
        base.sort()
        best = None
        best_size = 0

        for trial in range(max_iter):
            if time.time() - start > DEADLINE:
                break
            S = list(base)
            sums = compute_sums(S)
            candidates = [x for x in range(1, N + 1) if x not in PRESCRIBED]
            rng.shuffle(candidates)

            for x in candidates:
                if time.time() - start > DEADLINE:
                    break
                if can_add(S, sums, x):
                    S.append(x)
                    for y in S[:-1]:
                        sums.add(x + y)
                    sums.add(2 * x)

            S.sort()
            if len(S) > best_size:
                best_size = len(S)
                best = list(S)

        return best if best else base

    def local_search(S, rng):
        """Try to improve size via swap operations."""
        S = list(S)
        S.sort()
        sums = compute_sums(S)
        improved = True
        iterations = 0

        while improved and time.time() - start < DEADLINE:
            improved = False
            iterations += 1

            # Try to add a new element
            candidates = [x for x in range(1, N + 1) if x not in set(S)]
            rng.shuffle(candidates)
            for x in candidates:
                if time.time() - start > DEADLINE:
                    return S
                if can_add(S, sums, x):
                    S.append(x)
                    S.sort()
                    for y in S[:-1]:
                        sums.add(x + y)
                    sums.add(2 * x)
                    improved = True
                    break

            # Try swap: remove one element, add another to maintain or increase size
            if not improved:
                for i in range(len(S)):
                    if time.time() - start > DEADLINE:
                        return S
                    removed = S[i]
                    S_without = S[:i] + S[i+1:]
                    sums_without = compute_sums(S_without)
                    
                    # Find candidates to add
                    candidates = [x for x in range(1, N + 1) if x not in set(S_without)]
                    rng.shuffle(candidates)
                    for x in candidates:
                        if time.time() - start > DEADLINE:
                            S = S_without + [removed]
                            return S
                        if can_add(S_without, sums_without, x):
                            S = S_without + [x]
                            S.sort()
                            sums = compute_sums(S)
                            improved = True
                            break
                    if improved:
                        break

        return S

    # Initial greedy
    rng = random.Random(42)
    S = greedy_construct(rng, max_iter=20)
    
    # Local search to improve
    S = local_search(S, rng)
    
    # More greedy attempts with different seeds, keep best
    best_S = S
    best_size = len(S)
    
    for seed in range(100):
        if time.time() - start > DEADLINE:
            break
        rng2 = random.Random(seed)
        S2 = greedy_construct(rng2, max_iter=5)
        if len(S2) > best_size:
            S2 = local_search(S2, rng2)
            if len(S2) > best_size:
                best_S = S2
                best_size = len(S2)
    
    return best_S
# EVOLVE-BLOCK-END
