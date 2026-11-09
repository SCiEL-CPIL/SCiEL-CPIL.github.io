"""Combined: fast evaluation + multi-start local search with structured inits and greedy seed."""
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill

    def evaluate(S):
        """Count distinct d>0 such that a, a+d, a+2d all in S for some a.
        Uses sorted list and set membership for speed."""
        S_set = set(S)
        valid_diffs = set()
        n = len(S)
        for i in range(n):
            a = S[i]
            for j in range(i + 1, n):
                b = S[j]
                d = b - a
                c = 2 * b - a
                if c in S_set:
                    valid_diffs.add(d)
        return len(valid_diffs)

    def local_search(S, max_iters=3000):
        """Hill climbing: try random single-element replacements, accept if score doesn't decrease."""
        S = S[:]
        current_score = evaluate(S)
        S_set = set(S)

        for _ in range(max_iters):
            if time.time() >= deadline:
                break
            idx = random.randint(0, 31)
            old_val = S[idx]

            new_val = random.randint(1, 512)
            if new_val in S_set:
                continue

            # Try the swap
            S[idx] = new_val
            S_set.remove(old_val)
            S_set.add(new_val)
            new_score = evaluate(S)

            if new_score >= current_score:
                current_score = new_score
            else:
                # Revert
                S[idx] = old_val
                S_set.remove(new_val)
                S_set.add(old_val)

        return S, current_score

    def greedy_construct():
        """Greedy: start from a seed and add elements that maximize new distinct differences."""
        S = []
        S_set = set()
        current_diffs = set()

        # Start with a small seed that has some structure
        seed = [1, 2, 3, 16, 17, 18, 32, 33, 34]
        for x in seed:
            if x not in S_set:
                S.append(x)
                S_set.add(x)

        # Update current diffs
        def update_diffs(S_set):
            diffs = set()
            lst = sorted(S_set)
            n = len(lst)
            for i in range(n):
                a = lst[i]
                for j in range(i + 1, n):
                    b = lst[j]
                    d = b - a
                    c = 2 * b - a
                    if c in S_set:
                        diffs.add(d)
            return diffs

        current_diffs = update_diffs(S_set)

        while len(S) < 32:
            best_val = None
            best_gain = -1
            # Sample candidates for speed
            candidates = random.sample(range(1, 513), min(150, 512 - len(S_set)))
            for v in candidates:
                if v in S_set:
                    continue
                new_S_set = S_set | {v}
                new_diffs = update_diffs(new_S_set)
                gain = len(new_diffs) - len(current_diffs)
                if gain > best_gain:
                    best_gain = gain
                    best_val = v

            if best_val is not None:
                S.append(best_val)
                S_set.add(best_val)
                current_diffs = update_diffs(S_set)
            else:
                # Fallback: pick a random value
                best_val = random.randint(1, 512)
                while best_val in S_set:
                    best_val = random.randint(1, 512)
                S.append(best_val)
                S_set.add(best_val)
                current_diffs = update_diffs(S_set)

        return S, len(current_diffs)

    # Generate diverse initial candidates
    candidates = []

    # 1. Consecutive integers
    candidates.append(list(range(1, 33)))

    # 2. Arithmetic progressions with various steps
    for step in [2, 3, 4, 5, 7, 10, 15, 20]:
        start = 1
        end = start + 31 * step
        if end <= 512:
            candidates.append([start + i * step for i in range(32)])

    # 3. Random sets
    random.seed(42)
    for _ in range(20):
        candidates.append(random.sample(range(1, 513), 32))

    # 4. Structured: mix of small and large values
    for _ in range(10):
        small = random.sample(range(1, 100), 16)
        large = random.sample(range(400, 513), 16)
        candidates.append(small + large)

    # 5. Quadratic-like sets
    for a in [1, 2, 3, 5]:
        vals = [a * i * i + 1 for i in range(32)]
        if max(vals) <= 512:
            candidates.append(vals)

    # 6. Grid-based construction (4x8)
    M = 64
    S = set()
    for i in range(4):
        for j in range(8):
            val = i * M + j + 1
            if 1 <= val <= 512:
                S.add(val)
    candidates.append(list(S))

    # 7. Greedy constructions
    for _ in range(5):
        gS, gScore = greedy_construct()
        candidates.append(gS)

    # Evaluate all candidates
    best_S = None
    best_score = 0
    for cand in candidates:
        if len(set(cand)) != 32:
            continue
        score = evaluate(cand)
        if score > best_score:
            best_score = score
            best_S = cand[:]

    # Multi-start local search
    num_starts = 15
    for s in range(num_starts):
        if time.time() >= deadline:
            break

        # Create a perturbed version of best_S
        if s == 0:
            start_S = best_S[:]
        else:
            start_S = best_S[:]
            # Perturb 5-15 random elements
            num_perturb = random.randint(5, 15)
            indices = random.sample(range(32), num_perturb)
            S_set = set(start_S)
            for idx in indices:
                new_val = random.randint(1, 512)
                while new_val in S_set:
                    new_val = random.randint(1, 512)
                S_set.remove(start_S[idx])
                S_set.add(new_val)
                start_S[idx] = new_val

        # Run local search
        result_S, result_score = local_search(start_S, max_iters=3000)

        if result_score > best_score:
            best_score = result_score
            best_S = result_S[:]

    return best_S
