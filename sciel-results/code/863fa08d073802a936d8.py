"""Rewritten program: exact subset-sum avoidance via bitset reachability."""
# EVOLVE-BLOCK-START
import time

def construct():
    # Deadline guard: keep a safety margin before the 90s wall-clock kill.
    start = time.time()
    DEADLINE = start + 85.0

    # Greedy construction: iterate elements in descending order.
    # Maintain reachable sums as a Python int bitset (bit s set iff sum s reachable).
    # We only care about sums up to 200, so mask to 201 bits.
    MASK = (1 << 201) - 1  # bits 0..200

    reachable = 1  # bit 0 set: sum 0 reachable
    selected = []

    for x in range(100, 0, -1):
        if time.time() > DEADLINE:
            break
        # If we add x, new reachable sums are {s + x : s in reachable, s + x <= 200}
        # In bitset terms: shift left by x, then mask to keep only bits <= 200.
        new_reachable = (reachable << x) & MASK
        # Check if 200 would become reachable
        if not (new_reachable & (1 << 200)):
            selected.append(x)
            reachable |= new_reachable

    # If we ran out of time or finished, return what we have.
    # Ensure nonempty: if somehow empty (shouldn't happen), return [100].
    if not selected:
        selected = [100]

    return selected
# EVOLVE-BLOCK-END
