"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 85.0
    
    def try_construct(start_val=1, end_val=257):
        values, sums = [], set()
        for x in range(start_val, end_val):
            if time.time() > deadline:
                return values
            tails, total = [x], x
            for a in reversed(values):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                values.append(x)
                sums.update(tails)
        return values
    
    # First pass: standard greedy from 1 to 256
    result = try_construct(1, 257)
    
    # If we have time, try a second pass with different starting offset
    # to see if we can get a longer sequence
    if time.time() < deadline - 10.0:
        # Try starting from a higher value to see if we get more elements
        # This is a heuristic: sometimes skipping small values allows more total
        for offset in range(1, 10):
            if time.time() > deadline - 5.0:
                break
            candidate = try_construct(1 + offset, 257)
            if len(candidate) > len(result):
                result = candidate
    
    return result
# EVOLVE-BLOCK-END
