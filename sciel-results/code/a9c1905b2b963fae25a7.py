"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = 85.0  # 90s wall-clock limit, leave 5s margin
    
    # Use the subset DP approach over divisors of a locally chosen common denominator.
    target = 2520
    reached = {0: []}
    
    for d in range(5, target + 1):
        if time.time() > deadline:
            break
        if target % d != 0:
            continue
        w = target // d
        for total in list(reached.keys()):
            if total + w <= target and (total + w) not in reached:
                reached[total + w] = reached[total] + [d]
        if target in reached:
            return reached[target]
    
    raise RuntimeError("no baseline representation")
# EVOLVE-BLOCK-END
