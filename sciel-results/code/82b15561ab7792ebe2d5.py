"""Rewritten program for large interval sets avoiding a prescribed subset sum."""
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Return a nonempty subset of [1,100] with no subset summing to 200,
    maximizing cardinality.
    """
    deadline = time.time() + 55.0  # safety margin before 60s (evaluator kills at 90s, but task says 60s)
    target = 200
    max_val = 100

    # Greedy baseline: iterate from largest to smallest, include if it doesn't create a subset sum of 200
    # Use bitset for reachable sums for efficiency
    # reachable bitset: bit i set means sum i is reachable
    reachable = 1  # bit 0 set, sum 0 reachable
    selected = []

    for x in range(max_val, 0, -1):
        # Check if adding x creates a subset sum of 200
        # A subset sum of 200 exists with x if there's a subset sum of (200 - x) in current reachable
        if (200 - x) >= 0 and (reachable & (1 << (200 - x))):
            continue
        # Also need to check: if x itself is 200 (not possible here since x <= 100)
        # Add x: new sums are old sums + x
        new_reachable = reachable | (reachable << x)
        # Mask to only keep sums <= 200
        mask = (1 << (target + 1)) - 1
        new_reachable &= mask
        if (new_reachable & (1 << target)) == 0:
            selected.append(x)
            reachable = new_reachable
        # else: skip (shouldn't happen given the check above, but safe)

    best = len(selected)
    best_set = set(selected)

    # Try to improve: can we add any excluded numbers?
    # The greedy from largest to smallest is actually optimal for this type of problem?
    # Let's verify: the greedy algorithm that processes in decreasing order and includes
    # an element if it doesn't create a target sum is known to be optimal for subset sum avoidance.
    # But let's try some local search to be safe.

    # Local search: try swapping or adding
    excluded = [x for x in range(1, max_val + 1) if x not in best_set]
    
    # Try adding one excluded element at a time
    for x in excluded:
        if (200 - x) >= 0 and (reachable & (1 << (200 - x))):
            continue
        new_reachable = reachable | (reachable << x)
        new_reachable &= (1 << (target + 1)) - 1
        if (new_reachable & (1 << target)) == 0:
            # We can add x!
            best_set.add(x)
            reachable = new_reachable
            best += 1
            # Remove from excluded
            if x in excluded:
                excluded.remove(x)
    
    # Try more additions after the first round
    excluded = [x for x in range(1, max_val + 1) if x not in best_set]
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for x in list(excluded):
            if (200 - x) >= 0 and (reachable & (1 << (200 - x))):
                continue
            new_reachable = reachable | (reachable << x)
            new_reachable &= (1 << (target + 1)) - 1
            if (new_reachable & (1 << target)) == 0:
                best_set.add(x)
                reachable = new_reachable
                best += 1
                excluded.remove(x)
                changed = True

    return list(best_set)
# EVOLVE-BLOCK-END
