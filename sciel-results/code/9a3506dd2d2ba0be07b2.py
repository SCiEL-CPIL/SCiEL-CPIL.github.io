"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Return a nonempty subset of [1,100] with no subset summing to 200, maximizing cardinality.
    Uses a greedy approach in ascending order with dynamic programming reachability,
    plus a deadline guard.
    """
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    # Greedy: try to include numbers in ascending order.
    # A number x can be included if no previously reachable sum s satisfies s + x == 200,
    # i.e., (200 - x) is not in the reachable set.
    # Also, we must ensure that adding x doesn't create 200 via any combination.
    # Since we track all reachable sums up to 199, we just check if (200 - x) is reachable.
    
    selected = []
    reachable = {0}
    
    for x in range(1, 101):
        if time.time() > DEADLINE:
            break
        # Check if adding x would create a sum of 200
        # This happens if (200 - x) is in reachable
        if (200 - x) in reachable:
            continue
        # Also need to check: for any s in reachable, s + x != 200
        # This is equivalent to (200 - x) not in reachable, which we already checked.
        # But we also need to make sure that x itself doesn't cause issues with future additions.
        # The greedy check is: if (200 - x) is not in reachable, we can safely add x.
        # When we add x, new reachable sums are {s + x for s in reachable if s + x <= 199}
        # We need to ensure 200 is not among these new sums.
        # s + x = 200 => s = 200 - x, so if (200 - x) not in reachable, then 200 is not created.
        
        selected.append(x)
        # Update reachable set
        new_sums = {s + x for s in reachable if s + x <= 199}
        reachable |= new_sums

    return selected
# EVOLVE-BLOCK-END
