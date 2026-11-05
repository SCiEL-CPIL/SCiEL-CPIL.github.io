"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill

    def is_valid(S, target=200):
        """Check if no subset of S sums to target using bitset DP."""
        if not S:
            return True
        bits = 1  # bit i set means sum i is achievable
        for x in S:
            if x <= target:
                bits |= (bits << x)
        return (bits >> target) & 1 == 0

    # Start with known valid set: {66, 67, ..., 100}, size 35
    S = set(range(66, 101))
    best = S.copy()
    best_size = len(S)

    # Precompute conflict info: for each candidate x in [1,65],
    # which subsets of S would create a sum of 200 if x is added?
    # We use the bitset check directly for efficiency.

    # Phase 1: Greedy add - try each candidate from [1,65], add if valid
    candidates = list(range(1, 66))
    # Sort by "safety": smaller elements are riskier (more combinations to hit 200)
    # Try larger candidates first (less likely to create sum 200 with large S elements)
    candidates.sort(reverse=True)

    for x in candidates:
        if time.time() > deadline:
            break
        new_S = S | {x}
        if is_valid(new_S):
            S = new_S
            if len(S) > best_size:
                best = S.copy()
                best_size = len(S)

    # Phase 2: Local search - try swapping one element for two
    # This allows us to potentially increase cardinality
    improved = True
    while improved and time.time() < deadline:
        improved = False
        outside = set(range(1, 101)) - S
        if not outside:
            break
        outside_list = list(outside)
        # Try removing each element and adding two from outside
        for remove in list(S):
            if time.time() > deadline:
                break
            S_without = S - {remove}
            # Try pairs from outside
            for i, add1 in enumerate(outside_list):
                if time.time() > deadline:
                    break
                for add2 in outside_list[i+1:]:
                    if add1 == remove or add2 == remove:
                        continue
                    new_S = S_without | {add1, add2}
                    if len(new_S) > len(S):
                        if is_valid(new_S):
                            S = new_S
                            if len(S) > best_size:
                                best = S.copy()
                                best_size = len(S)
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break

    # Phase 3: Final greedy pass - try adding any remaining outside elements
    outside = set(range(1, 101)) - S
    for x in sorted(outside, reverse=True):
        if time.time() > deadline:
            break
        new_S = S | {x}
        if is_valid(new_S):
            S = new_S
            if len(S) > best_size:
                best = S.copy()
                best_size = len(S)

    result = list(best)
    assert len(result) > 0
    assert is_valid(set(result))
    return result
# EVOLVE-BLOCK-END
