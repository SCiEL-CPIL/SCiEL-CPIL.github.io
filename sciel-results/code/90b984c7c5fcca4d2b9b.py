import time
import random
from itertools import combinations

def construct():
    """
    Find a Sidon set containing 0 with all differences 1..30, minimizing the largest element.
    Uses constraint-driven greedy construction with backtracking and local refinement.
    """
    start_time = time.time()
    DEADLINE = 145.0  # Leave 5s margin before 150s kill

    def get_diffs(S):
        """Get set of all positive differences."""
        S = sorted(S)
        diffs = set()
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                d = S[j] - S[i]
                diffs.add(d)
        return diffs

    def is_sidon(S):
        """Check if S is a Sidon set (all positive differences distinct)."""
        S = sorted(S)
        diffs = set()
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                d = S[j] - S[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True

    def covers_all(S):
        """Check if S covers all differences 1..30."""
        diffs = get_diffs(S)
        return all(d in diffs for d in range(1, 31))

    def missing_diffs(S):
        """Return set of missing differences in 1..30."""
        diffs = get_diffs(S)
        return set(range(1, 31)) - diffs

    def score_set(S):
        """Score: (missing_count, max_element). Lower is better."""
        S = sorted(S)
        diffs = get_diffs(S)
        missing = 30 - len(diffs & set(range(1, 31)))
        return (missing, max(S))

    # --- Greedy construction with max-coverage and smallest-x tie-break ---
    def greedy_construct(max_size=32, max_val=1000000):
        """
        Build a Sidon set greedily: at each step, try candidate elements
        (in increasing order) and pick the one that covers the most
        currently-missing differences while maintaining Sidon property.
        """
        S = [0]
        target = set(range(1, 31))
        covered = set()

        for step in range(max_size - 1):
            if time.time() - start_time > DEADLINE - 10:
                break
            if covered == target:
                break

            best_x = None
            best_new_cover = 0
            best_S = None

            # Try candidates: small values first, then larger
            # Focus on values that could create useful differences
            existing = set(S)
            current_diffs = get_diffs(S)

            # Generate candidate values
            candidates = set()
            for s in S:
                for d in target - covered:
                    candidates.add(s + d)
                    if s - d > 0:
                        candidates.add(s - d)
            # Also try some random small values
            for _ in range(200):
                candidates.add(random.randint(1, max(50, 2 * max(S))))

            candidates = [c for c in candidates if c > 0 and c not in existing and c <= max_val]
            candidates.sort()

            # Limit candidates for speed
            if len(candidates) > 500:
                candidates = candidates[:500]

            for x in candidates:
                trial = sorted(S + [x])
                if not is_sidon(trial):
                    continue
                trial_diffs = get_diffs(trial)
                new_cover = len(trial_diffs & target) - len(covered)
                if new_cover > best_new_cover:
                    best_new_cover = new_cover
                    best_x = x
                    best_S = trial
                    if best_new_cover >= len(target - covered):
                        break  # Can't do better

            if best_x is None:
                # No single element helps; try adding a pair
                found = False
                for x in candidates[:200]:
                    trial1 = sorted(S + [x])
                    if not is_sidon(trial1):
                        continue
                    for y in candidates[200:400]:
                        if y <= x:
                            continue
                        trial2 = sorted(S + [x, y])
                        if not is_sidon(trial2):
                            continue
                        trial2_diffs = get_diffs(trial2)
                        new_cover = len(trial2_diffs & target) - len(covered)
                        if new_cover > 0:
                            S = trial2
                            covered = trial2_diffs & target
                            found = True
                            break
                    if found:
                        break
                if not found:
                    # Give up on this greedy path
                    break
            else:
                S = best_S
                covered = get_diffs(S) & target

        return S

    # --- Local refinement: try to reduce diameter ---
    def local_refine(S, time_limit=20.0):
        """
        Try to reduce max(S) by replacing elements with smaller values.
        """
        S = list(S)
        S.sort()
        best = S[:]
        best_score = score_set(best)
        t0 = time.time()

        while time.time() - t0 < time_limit:
            improved = False
            # Try replacing each non-zero element with a smaller value
            for i in range(len(S)):
                if S[i] == 0:
                    continue
                old_val = S[i]
                # Try values from 1 up to old_val-1
                # Focus on values that might cover missing diffs
                missing = missing_diffs(S)
                if not missing:
                    break

                for new_val in range(1, old_val):
                    if new_val in S:
                        continue
                    trial = S[:]
                    trial[i] = new_val
                    trial.sort()
                    if len(set(trial)) < len(trial):
                        continue
                    if not is_sidon(trial):
                        continue
                    new_score = score_set(trial)
                    if new_score < best_score:
                        S = trial
                        best = S[:]
                        best_score = new_score
                        improved = True
                        break

            if not improved:
                # Try removing an element if we still cover all diffs
                if best_score[0] == 0:
                    for i in range(len(S)):
                        if S[i] == 0:
                            continue
                        trial = S[:]
                        trial.pop(i)
                        if covers_all(trial) and is_sidon(trial):
                            new_score = score_set(trial)
                            if new_score < best_score:
                                S = trial
                                best = S[:]
                                best_score = new_score
                                improved = True
                                break
                if not improved:
                    break

        return best

    # --- Main: run greedy with different random seeds, keep best ---
    best_S = None
    best_score = (30, 10**9)

    # First, try a deterministic greedy
    S0 = greedy_construct()
    if S0 and len(S0) > 0:
        s0 = score_set(S0)
        if s0 < best_score:
            best_score = s0
            best_S = S0

    # Try multiple random restarts with different candidate orderings
    for restart in range(50):
        if time.time() - start_time > DEADLINE - 30:
            break

        random.seed(restart * 12345 + 678)
        S = greedy_construct()
        if S and len(S) > 0:
            s = score_set(S)
            if s < best_score:
                best_score = s
                best_S = S

        # Local refinement on the best so far
        if time.time() - start_time < DEADLINE - 10:
            refined = local_refine(best_S if best_S else S, time_limit=min(15, DEADLINE - 10 - (time.time() - start_time)))
            if refined:
                rs = score_set(refined)
                if rs < best_score:
                    best_score = rs
                    best_S = refined

        if time.time() - start_time > DEADLINE - 5:
            break

    # Final refinement
    if best_S and time.time() - start_time < DEADLINE - 2:
        final = local_refine(best_S, time_limit=max(1, DEADLINE - 2 - (time.time() - start_time)))
        if final:
            fs = score_set(final)
            if fs < best_score:
                best_score = fs
                best_S = final

    # Ensure validity
    if best_S is None or not covers_all(best_S) or not is_sidon(best_S):
        # Fallback: simple construction
        best_S = [0]
        covered = set()
        for d in range(1, 31):
            if d in covered:
                continue
            # Add element to create difference d
            for x in range(1, 1000000):
                if x in best_S:
                    continue
                trial = sorted(best_S + [x])
                if is_sidon(trial):
                    trial_diffs = get_diffs(trial)
                    if d in trial_diffs:
                        best_S = trial
                        covered = trial_diffs & set(range(1, 31))
                        break
            else:
                # Force add
                y = 2 * max(best_S) + d + 1
                best_S = sorted(best_S + [y, y + d])
                covered = get_diffs(best_S) & set(range(1, 31))

    return best_S
