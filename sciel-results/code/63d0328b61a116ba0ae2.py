import time
import random
from itertools import combinations

def construct():
    start = time.time()
    deadline = start + 130.0  # safety margin under the 150 s wall-clock kill

    def elapsed():
        return time.time() - start

    def is_sidon(S):
        diffs = set()
        n = len(S)
        for i in range(n):
            for j in range(i + 1, n):
                d = S[j] - S[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True

    def covers_1_to_30(S):
        diffs = set()
        n = len(S)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(S[j] - S[i])
        return all(d in diffs for d in range(1, 31))

    def greedy_construct(bound, rng):
        """Greedy construction: pick smallest valid x that maximizes new coverage."""
        S = [0]
        used = set()
        need = set(range(1, 31))
        for _ in range(32):
            best_x = None
            best_score = -1
            for x in range(S[-1] + 1, bound + 1):
                ok = True
                score = 0
                for s in S:
                    d = x - s
                    if d in used:
                        ok = False
                        break
                    if d in need:
                        score += 1
                if ok and score > best_score:
                    best_score = score
                    best_x = x
                    if best_score >= len(need) * 2:
                        break
            if best_x is None:
                break
            S.append(best_x)
            for s in S[:-1]:
                d = S[-1] - s
                used.add(d)
            need -= used
            if not need:
                break
        return S

    def local_search(S, bound, rng):
        """Iterative improvement: try swapping or removing elements to reduce max."""
        S = sorted(S)
        for _ in range(500):
            if elapsed() > deadline - 5.0:
                break
            improved = False
            # Try removing elements (if still valid)
            for i in range(len(S) - 1, 0, -1):
                trial = S[:i] + S[i + 1:]
                if trial and is_sidon(trial) and covers_1_to_30(trial):
                    S = trial
                    improved = True
                    break
            if improved:
                continue
            # Try replacing an element with a smaller one
            for i in range(1, len(S)):
                if S[i] <= 1:
                    continue
                for new_x in range(1, S[i]):
                    if new_x in S:
                        continue
                    trial = S[:i] + [new_x] + S[i + 1:]
                    if is_sidon(trial) and covers_1_to_30(trial):
                        S = trial
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        return S

    def simulated_annealing(S, bound, rng, steps=2000):
        """SA with random perturbations to escape local minima."""
        S = sorted(S)
        temp = 1.0
        cooling = 0.999
        for step in range(steps):
            if elapsed() > deadline - 5.0:
                break
            # Random perturbation: swap or change one element
            if not S:
                break
            i = rng.randint(0, len(S) - 1)
            if i == 0:
                continue  # keep 0
            old_val = S[i]
            new_val = old_val + rng.randint(-20, 20)
            if new_val <= 0 or new_val > bound:
                continue
            if new_val in S:
                continue
            S[i] = new_val
            S = sorted(S)
            if not is_sidon(S) or not covers_1_to_30(S):
                S[i] = old_val
                S = sorted(S)
            temp *= cooling
        return S

    # Strategy: multi-start greedy + local search
    best_S = None
    best_max = float('inf')
    rng = random.Random(42)

    # Try several bounds and restarts
    bounds = [100, 150, 200, 250, 300, 400, 500]
    for bound in bounds:
        if elapsed() > deadline - 30.0:
            break
        for trial in range(20):
            if elapsed() > deadline - 10.0:
                break
            S = greedy_construct(bound, rng)
            if not S or not is_sidon(S) or not covers_1_to_30(S):
                continue
            S = local_search(S, bound, rng)
            if not S or not is_sidon(S) or not covers_1_to_30(S):
                continue
            mx = max(S)
            if mx < best_max:
                best_max = mx
                best_S = list(S)

    # Fallback: known valid Sidon set (squares don't work, use a verified one)
    if best_S is None or not is_sidon(best_S) or not covers_1_to_30(best_S):
        # Use a greedy result with generous bound as fallback
        best_S = greedy_construct(1000, rng)
        if not best_S or not is_sidon(best_S) or not covers_1_to_30(best_S):
            # Absolute last resort: expand greedily
            best_S = [0]
            used = set()
            need = set(range(1, 31))
            for _ in range(32):
                found = False
                for x in range(best_S[-1] + 1, best_S[-1] + 500):
                    ok = True
                    for s in best_S:
                        d = x - s
                        if d in used:
                            ok = False
                            break
                    if ok:
                        best_S.append(x)
                        for s in best_S[:-1]:
                            used.add(x - s)
                        need -= used
                        found = True
                        break
                if not found:
                    break
                if not need:
                    break

    # Final validation
    if not is_sidon(best_S) or not covers_1_to_30(best_S) or 0 not in best_S:
        # Emergency fallback
        best_S = [0]
        used = set()
        need = set(range(1, 31))
        for _ in range(32):
            found = False
            for x in range(best_S[-1] + 1, best_S[-1] + 1000):
                ok = True
                for s in best_S:
                    d = x - s
                    if d in used:
                        ok = False
                        break
                if ok:
                    best_S.append(x)
                    for s in best_S[:-1]:
                        used.add(x - s)
                    need -= used
                    found = True
                    break
            if not found:
                break
            if not need:
                break

    return best_S
