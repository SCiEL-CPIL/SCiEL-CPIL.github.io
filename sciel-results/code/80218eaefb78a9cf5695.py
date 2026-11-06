"""Odd covering system with moduli dividing 51975; minimize uncovered residues."""
import time
import numpy as np

BASE = 51975
DEADLINE = 115.0  # safety margin under the 120 s wall-clock limit (evaluator kills at 150s)


def _divisors():
    """Generate all odd divisors of 51975 = 3^3 * 5^2 * 7 * 11, excluding 1."""
    divs = []
    for a in range(4):
        for b in range(3):
            for c in range(2):
                for d in range(2):
                    m = (3 ** a) * (5 ** b) * (7 ** c) * (11 ** d)
                    if m > 1:
                        divs.append(m)
    return sorted(divs)


DIVS = _divisors()
N = len(DIVS)


def _greedy(order, t0, deadline):
    """
    Run greedy algorithm with given order of indices.
    Returns (score, chosen_list, covered_array)
    """
    covered = np.zeros(BASE, dtype=np.uint8)
    chosen = [0] * N

    for i in order:
        m = DIVS[i]
        best_r = 0
        best_g = -1

        # Find residue that covers most uncovered points
        for r in range(m):
            g = np.sum(covered[r::m] == 0)
            if g > best_g:
                best_g = g
                best_r = r

        chosen[i] = best_r
        covered[best_r::m] = 1

        if (time.time() - t0) > deadline:
            break

    return BASE - int(np.sum(covered)), chosen, covered


def _local_search(covered, chosen, t0, deadline, max_rounds=50):
    """
    Local search: try changing residue of each modulus to improve coverage.
    Returns (score, chosen, covered)
    """
    score = BASE - int(np.sum(covered))
    improved = True
    round_num = 0

    while improved and round_num < max_rounds:
        improved = False
        round_num += 1

        for i in range(N):
            if (time.time() - t0) > deadline:
                return score, chosen, covered.copy()

            m = DIVS[i]
            current_r = chosen[i]

            # Compute what the covered array looks like without this class
            # We need to find the best alternative residue
            # First, count how many points are covered ONLY by this class
            # and how many are covered by other classes too

            # Simpler approach: try each residue and compute the change
            # Remove current contribution
            temp_covered = covered.copy()
            # We need to know which points were covered by this class AND no other
            # This is complex. Instead, let's use a different strategy:
            # For each candidate residue r, compute the marginal gain if we switch

            best_r = current_r
            best_gain = 0

            # To efficiently evaluate, we can precompute for each point how many classes cover it
            # But that's expensive to maintain. Let's use a simpler heuristic:
            # For each candidate r, the gain = (newly covered) - (lost coverage)
            # Newly covered: points in r::m that are currently uncovered
            # Lost coverage: points in current_r::m that are covered ONLY by this class

            # Count uncovered points in each candidate residue class
            # and count points covered only by this class in current residue

            # First, let's identify points covered only by this class
            # A point x is covered only by class i if covered[x]==1 and 
            # no other class j covers x.

            # This is expensive. Let's use a faster approximation:
            # Just try a few candidate residues

            candidates = [current_r]
            # Add some random candidates
            if m > 1:
                for _ in range(min(20, m)):
                    candidates.append(int(np.random.randint(0, m)))

            for r in candidates:
                if r == current_r:
                    continue

                # Compute gain of switching from current_r to r
                # Points gained: those in r::m that are currently 0
                # Points lost: those in current_r::m that are currently 1 AND not covered by others

                # Approximation: assume all points in current_r::m that are 1 are lost
                # This is conservative (overestimates loss)

                gain = np.sum(covered[r::m] == 0) - np.sum(covered[current_r::m] == 1)

                if gain > best_gain:
                    best_gain = gain
                    best_r = r

            if best_r != current_r and best_gain > 0:
                # Apply the change
                # First, we need to properly update covered
                # Remove old contribution, then add new contribution
                # But we need to know if removing old contribution uncovers points

                # Let's do it properly:
                # 1. Mark points in current_r::m as potentially uncovered
                # 2. Check if they are still covered by other classes
                # 3. Then mark points in best_r::m as covered

                # This is complex. Let's use a simpler approach:
                # Rebuild the covered array from scratch is too slow.

                # Alternative: maintain a count array
                # For now, let's just accept the approximation and update
                chosen[i] = best_r
                # Update covered: this is where it gets tricky
                # We'll need to recompute covered from scratch periodically

                improved = True

        # Periodically recompute covered to keep it accurate
        if round_num % 5 == 0:
            covered = np.zeros(BASE, dtype=np.uint8)
            for i in range(N):
                m = DIVS[i]
                r = chosen[i]
                covered[r::m] = 1
            score = BASE - int(np.sum(covered))

    return score, chosen, covered


def _refine(covered, chosen, t0, deadline):
    """
    More careful local search with proper coverage counting.
    """
    score = BASE - int(np.sum(covered))

    # Maintain a count of how many classes cover each point
    cover_count = np.zeros(BASE, dtype=np.int32)
    for i in range(N):
        m = DIVS[i]
        r = chosen[i]
        cover_count[r::m] += 1

    improved = True
    round_num = 0
    max_rounds = 100

    while improved and round_num < max_rounds:
        improved = False
        round_num += 1

        for i in range(N):
            if (time.time() - t0) > deadline:
                break

            m = DIVS[i]
            current_r = chosen[i]

            best_r = current_r
            best_score = score

            # Try a subset of candidate residues
            # Include current and some random ones
            candidates = set()
            candidates.add(current_r)
            num_candidates = min(30, m)
            for _ in range(num_candidates):
                candidates.add(int(np.random.randint(0, m)))

            for r in candidates:
                if r == current_r:
                    continue

                # Compute the change in score if we switch from current_r to r
                # Points that become newly covered: in r::m, cover_count == 0
                # Points that become uncovered: in current_r::m, cover_count == 1

                newly_covered = np.sum(cover_count[r::m] == 0)
                becomes_uncovered = np.sum(cover_count[current_r::m] == 1)

                delta = newly_covered - becomes_uncovered
                if delta > 0:
                    candidate_score = score - delta
                    if candidate_score < best_score:
                        best_score = candidate_score
                        best_r = r

            if best_r != current_r:
                # Apply the change
                cover_count[current_r::m] -= 1
                cover_count[best_r::m] += 1
                chosen[i] = best_r
                score = best_score
                improved = True

        if (time.time() - t0) > deadline:
            break

    # Rebuild covered from cover_count
    covered = (cover_count > 0).astype(np.uint8)
    score = BASE - int(np.sum(covered))

    return score, chosen, covered


def construct():
    t0 = time.time()

    # Strategy 1: Descending moduli (large moduli first) - establishes sparse baseline
    order_desc = sorted(range(N), key=lambda i: -DIVS[i])
    score_desc, chosen_desc, covered_desc = _greedy(order_desc, t0, DEADLINE - 60)

    # Strategy 2: Ascending moduli (small moduli first)
    if (time.time() - t0) < DEADLINE - 50:
        order_asc = sorted(range(N), key=lambda i: DIVS[i])
        score_asc, chosen_asc, covered_asc = _greedy(order_asc, t0, DEADLINE - 40)

        # Pick the better of the two
        if score_desc <= score_asc:
            best_score = score_desc
            best_chosen = chosen_desc
            best_covered = covered_desc
        else:
            best_score = score_asc
            best_chosen = chosen_asc
            best_covered = covered_asc
    else:
        best_score = score_desc
        best_chosen = chosen_desc
        best_covered = covered_desc

    # Local search to refine
    if (time.time() - t0) < DEADLINE - 10:
        best_score, best_chosen, best_covered = _refine(
            best_covered, best_chosen, t0, DEADLINE - 5
        )

    # Construct the final list of classes
    final_classes = []
    for i in range(N):
        m = DIVS[i]
        r = best_chosen[i]
        if r is None or r >= m:
            r = 0
        final_classes.append([m, r])

    return final_classes
