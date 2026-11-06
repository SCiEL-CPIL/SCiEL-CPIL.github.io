import math
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    N = 256
    small = list(range(2, 129))
    large = list(range(129, 257))

    # Precompute conflict graph for small numbers
    small_conflicts = {s: set() for s in small}
    for i in range(len(small)):
        s1 = small[i]
        for j in range(i + 1, len(small)):
            s2 = small[j]
            if math.lcm(s1, s2) <= N:
                small_conflicts[s1].add(s2)
                small_conflicts[s2].add(s1)

    # For each small number, precompute which large numbers it conflicts with
    # A large number y conflicts with small s if lcm(s,y) <= 256
    # Since y > 128, lcm(s,y) <= 256 means y divides some multiple of s that's <= 256
    # More directly: lcm(s,y) = s*y/gcd(s,y) <= 256
    small_large_conflicts = {s: set() for s in small}
    for s in small:
        for y in large:
            if math.lcm(s, y) <= N:
                small_large_conflicts[s].add(y)

    # Precompute: for each subset of small (represented as frozenset), which large numbers are excluded
    # Actually, a large number y is excluded if it conflicts with ANY selected small number
    # So excluded_large(S) = union of small_large_conflicts[s] for s in S

    # For a given small set S (independent set), the score is:
    # sum(1/s for s in S) + sum(1/y for y in large if y not in excluded_large(S))

    def score_for_small_set(S):
        """Compute exact reciprocal sum for a given small set S."""
        small_sum = sum(Fraction(1, s) for s in S)
        excluded = set()
        for s in S:
            excluded |= small_large_conflicts[s]
        large_sum = sum(Fraction(1, y) for y in large if y not in excluded)
        return small_sum + large_sum

    def is_independent(S):
        """Check if S is an independent set in the small conflict graph."""
        S_set = set(S)
        for s in S:
            if small_conflicts[s] & S_set:
                return False
        return True

    # Branch and bound
    # Sort small numbers by 1/s descending (prefer small numbers with high weight)
    # But we need to be careful: a small number might exclude many large numbers

    # Compute for each small number: its weight minus the loss in large sum
    # If we add s to an empty set, we gain 1/s but lose sum(1/y for y in small_large_conflicts[s])
    # This gives a heuristic for ordering

    # Let's use a recursive branch and bound
    # State: (remaining_small, current_small_set, upper_bound)
    # Upper bound: current_score + sum of 1/s for all remaining + sum of 1/y for all large not excluded by current

    best_score = Fraction(0, 1)
    best_set = set()

    # Precompute total large sum
    total_large_sum = sum(Fraction(1, y) for y in large)

    # Order small numbers: try those with high "net benefit" first
    # Net benefit of s (when added to empty): 1/s - sum(1/y for y in small_large_conflicts[s])
    net_benefit = {}
    for s in small:
        loss = sum(Fraction(1, y) for y in small_large_conflicts[s])
        net_benefit[s] = Fraction(1, s) - loss

    # Sort by net benefit descending
    small_sorted = sorted(small, key=lambda s: net_benefit[s], reverse=True)

    # For branch and bound, we need a good upper bound
    # Upper bound for a partial solution: current_score + sum(1/s for s in remaining that could be added)
    # But adding s might exclude large numbers, so a tighter bound:
    # For each remaining s, the max additional contribution is max(0, 1/s - loss_if_added)
    # But loss_if_added depends on what's already selected (already excluded large numbers)

    # Simpler upper bound: current_score + sum(1/s for s in remaining) + (total_large_sum - already_excluded_large_sum)
    # This is valid because we can't gain more than all remaining small + all non-excluded large

    def branch_and_bound(remaining, current_set, current_score, current_excluded):
        nonlocal best_score, best_set
        if time.time() > deadline:
            return

        if not remaining:
            # Evaluate this solution
            total = current_score + sum(Fraction(1, y) for y in large if y not in current_excluded)
            if total > best_score:
                best_score = total
                best_set = set(current_set)
            return

        # Upper bound
        # Best case: add all remaining small numbers (ignoring conflicts) and all non-excluded large
        ub_small = sum(Fraction(1, s) for s in remaining)
        ub_large = sum(Fraction(1, y) for y in large if y not in current_excluded)
        upper_bound = current_score + ub_small + ub_large

        if upper_bound <= best_score:
            return

        # Try adding the first remaining element
        s = remaining[0]
        rest = remaining[1:]

        # Option 1: Include s
        # Check if s conflicts with any in current_set
        if not (small_conflicts[s] & current_set):
            new_excluded = current_excluded | small_large_conflicts[s]
            new_score = current_score + Fraction(1, s)
            # Filter rest to only those not conflicting with s
            rest_filtered = [x for x in rest if x not in small_conflicts[s]]
            branch_and_bound(rest_filtered, current_set | {s}, new_score, new_excluded)

        # Option 2: Exclude s
        branch_and_bound(rest, current_set, current_score, current_excluded)

    # Start with empty set
    branch_and_bound(small_sorted, set(), Fraction(0, 1), set())

    if time.time() > deadline:
        if best_set:
            return list(best_set)
        # Fallback: just all large numbers
        return list(large)

    # Now do local search to improve
    # Try to swap elements in the small set to improve the score
    current_small_set = set(best_set)
    current_score = best_score

    # Multi-swap local search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try adding each small number not in current set
        for s in small:
            if time.time() > deadline:
                break
            if s in current_small_set:
                continue
            # Check if s can be added (no conflict with current small set)
            if small_conflicts[s] & current_small_set:
                continue
            # Compute new score
            new_excluded = set()
            for x in current_small_set | {s}:
                new_excluded |= small_large_conflicts[x]
            new_score = sum(Fraction(1, x) for x in current_small_set | {s})
            new_score += sum(Fraction(1, y) for y in large if y not in new_excluded)
            if new_score > current_score:
                current_small_set.add(s)
                current_score = new_score
                improved = True
                break

        # Try removing each element and replacing with a better one
        for s in list(current_small_set):
            if time.time() > deadline:
                break
            # Try removing s
            set_without_s = current_small_set - {s}
            excluded_without_s = set()
            for x in set_without_s:
                excluded_without_s |= small_large_conflicts[x]
            score_without_s = sum(Fraction(1, x) for x in set_without_s)
            score_without_s += sum(Fraction(1, y) for y in large if y not in excluded_without_s)

            # Try adding each other small number
            for t in small:
                if time.time() > deadline:
                    break
                if t in set_without_s:
                    continue
                if small_conflicts[t] & set_without_s:
                    continue
                new_set = set_without_s | {t}
                new_excluded = set()
                for x in new_set:
                    new_excluded |= small_large_conflicts[x]
                new_score = sum(Fraction(1, x) for x in new_set)
                new_score += sum(Fraction(1, y) for y in large if y not in new_excluded)
                if new_score > current_score:
                    current_small_set = new_set
                    current_score = new_score
                    improved = True
                    break
            if improved:
                break

    # Build final set
    final_small = current_small_set
    final_excluded = set()
    for s in final_small:
        final_excluded |= small_large_conflicts[s]
    final_large = [y for y in large if y not in final_excluded]

    return list(final_small) + final_large

if __name__ == "__main__":
    result = construct()
    print(f"Size: {len(result)}, Sum: {sum(1/x for x in result)}")
    valid = True
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if math.lcm(result[i], result[j]) <= 256:
                valid = False
                break
        if not valid:
            break
    print(f"Valid: {valid}")
