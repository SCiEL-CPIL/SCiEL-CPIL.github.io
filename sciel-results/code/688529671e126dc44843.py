import time
import random
from itertools import combinations


def is_sidon(subset):
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True


def largest_sidon_subset_size(S):
    S = sorted(S)
    n = len(S)
    # Try from largest k down
    for k in range(n, 0, -1):
        for combo in combinations(S, k):
            if is_sidon(combo):
                return k
    return 1


def count_sum_collisions(S):
    """Count how many pair sums are duplicated (higher = more collisions = harder for Sidon)"""
    sums = {}
    n = len(S)
    for i in range(n):
        for j in range(i, n):
            s = S[i] + S[j]
            sums[s] = sums.get(s, 0) + 1
    # Total collisions = sum of (count-1) for each sum that appears more than once
    collisions = 0
    for v in sums.values():
        if v > 1:
            collisions += v - 1
    return collisions


def _generate_cluster_candidates():
    """Generate structured candidate sets with high additive collision density."""
    candidates = []
    # Two tight clusters of 6
    for base1 in range(0, 60, 4):
        for base2 in range(base1 + 12, 129, 4):
            c1 = [base1 + i for i in range(6)]
            c2 = [base2 + i for i in range(6)]
            if max(c2) <= 128 and len(set(c1 + c2)) == 12:
                candidates.append(c1 + c2)
    # Three clusters of 4
    for b1 in range(0, 40, 4):
        for b2 in range(b1 + 8, 70, 4):
            for b3 in range(b2 + 8, 129, 4):
                c1 = [b1 + i for i in range(4)]
                c2 = [b2 + i for i in range(4)]
                c3 = [b3 + i for i in range(4)]
                if max(c3) <= 128 and len(set(c1 + c2 + c3)) == 12:
                    candidates.append(c1 + c2 + c3)
    # Single tight cluster of 12
    for base in range(0, 117, 3):
        s = [base + i for i in range(12)]
        if max(s) <= 128:
            candidates.append(s)
    # Mixed: 8 tight + 4 spread
    for base in range(0, 80, 5):
        tight = [base + i for i in range(8)]
        spread = [base + 10 + i * 3 for i in range(4)]
        s = tight + spread
        if max(s) <= 128 and len(set(s)) == 12:
            candidates.append(s)
    return candidates


def _greedy_collision_local_search(S, deadline):
    """Improve S by repeatedly replacing one element with the candidate that
    maximizes total sum collisions, then verify with exact evaluator."""
    S = sorted(S)
    best_val = largest_sidon_subset_size(S)
    best_set = S[:]

    # Collision-maximizing local moves
    for iteration in range(300):
        if time.time() > deadline:
            break
        improved = False
        current_coll = count_sum_collisions(best_set)

        for i in range(12):
            if time.time() > deadline:
                break
            old = best_set[i]
            best_new = None
            best_new_coll = current_coll

            # Try a sample of replacement values
            candidates = random.sample(range(0, 129), min(60, 129 - 11))
            for c in candidates:
                if c == old or c in best_set:
                    continue
                trial = best_set[:]
                trial[i] = c
                coll = count_sum_collisions(trial)
                if coll > best_new_coll:
                    best_new_coll = coll
                    best_new = c

            if best_new is not None:
                new_set = best_set[:]
                new_set[i] = best_new
                val = largest_sidon_subset_size(new_set)
                if val < best_val:
                    best_val = val
                    best_set = new_set
                    improved = True
                    break  # restart iteration with new best

        if not improved:
            # Try a random jump to escape local optimum
            if random.random() < 0.3:
                i = random.randint(0, 11)
                new_val = random.randint(0, 128)
                if new_val != best_set[i] and new_val not in best_set:
                    trial = best_set[:]
                    trial[i] = new_val
                    val = largest_sidon_subset_size(trial)
                    if val < best_val:
                        best_val = val
                        best_set = trial

    return best_set, best_val


def construct():
    start_time = time.time()
    deadline = start_time + 55

    best_set = None
    best_val = 99

    # Phase 1: Evaluate structured cluster candidates
    random.seed(12345)
    cluster_candidates = _generate_cluster_candidates()
    # Shuffle and evaluate a subset
    random.shuffle(cluster_candidates)
    for cand in cluster_candidates:
        if time.time() > deadline:
            break
        if len(set(cand)) != 12:
            continue
        val = largest_sidon_subset_size(cand)
        if val < best_val:
            best_val = val
            best_set = cand[:]

    # Phase 2: Greedy collision-maximizing construction from random seeds
    for restart in range(50):
        if time.time() > deadline:
            break

        current = set()
        current.add(random.randint(0, 128))

        for step in range(11):
            if time.time() > deadline:
                break
            best_candidate = None
            best_collisions = -1
            candidates = random.sample(range(0, 129), min(40, 129 - len(current)))
            for c in candidates:
                if c in current:
                    continue
                trial = list(current) + [c]
                coll = count_sum_collisions(trial)
                if coll > best_collisions:
                    best_collisions = coll
                    best_candidate = c
            if best_candidate is not None:
                current.add(best_candidate)

        if len(current) == 12:
            S = sorted(current)
            val = largest_sidon_subset_size(S)
            if val < best_val:
                best_val = val
                best_set = S[:]

    # Phase 3: Collision-maximizing local search from best found
    if best_set is not None and time.time() < deadline - 5:
        improved_set, improved_val = _greedy_collision_local_search(best_set, deadline)
        if improved_val < best_val:
            best_val = improved_val
            best_set = improved_set

    # Fallback
    if best_set is None:
        best_set = list(range(12))
        best_val = largest_sidon_subset_size(best_set)

    return best_set
