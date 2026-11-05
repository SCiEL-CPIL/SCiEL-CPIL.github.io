"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    target = 200
    # Use a bitset (as a Python int) for reachable sums up to target
    # bit i set means sum i is reachable
    
    def try_order(order):
        """Greedy: add elements in given order if it doesn't create sum=target."""
        reachable = 1  # bit 0 set, sum 0 reachable
        selected = []
        for x in order:
            # new sums = (reachable << x) & mask, where mask keeps bits <= target
            new_sums = (reachable << x) & ((1 << (target + 1)) - 1)
            if not (new_sums & (1 << target)):
                # Adding x doesn't create sum=target
                selected.append(x)
                reachable |= new_sums
        return selected
    
    # Try multiple orderings and pick the best (largest) valid set
    candidates = []
    
    # Order 1: descending (original)
    candidates.append(try_order(range(100, 0, -1)))
    
    # Order 2: ascending
    candidates.append(try_order(range(1, 101)))
    
    # Order 3: large first, then small
    candidates.append(try_order(list(range(100, 50, -1)) + list(range(50, 0, -1))))
    
    # Order 4: small first, then large
    candidates.append(try_order(list(range(1, 51)) + list(range(100, 50, -1))))
    
    # Order 5: alternating from ends
    order5 = []
    lo, hi = 1, 100
    while lo <= hi:
        order5.append(hi)
        if lo != hi:
            order5.append(lo)
        lo += 1
        hi -= 1
    candidates.append(try_order(order5))
    
    # Order 6: middle out
    order6 = list(range(50, 101)) + list(range(49, 0, -1))
    candidates.append(try_order(order6))
    
    # Pick the largest valid candidate
    best = max(candidates, key=len)
    
    # Verify best is valid (defensive check)
    # Quick check: no subset sums to 200
    # (The greedy already ensures this by construction, but let's be safe)
    
    return best
# EVOLVE-BLOCK-END
