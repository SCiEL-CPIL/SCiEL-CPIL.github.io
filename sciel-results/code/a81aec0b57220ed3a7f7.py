"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    # Try to find a 7-term solution using a simple search
    # We know 5,6,7,8,9,10 sum to 2131/2520. Remainder 389/2520.
    # We need 1/d7 = 389/22520? No.
    
    # Let's try a specific known 7-term solution if available.
    # One such solution is:
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/252 (Invalid)
    
    # Fallback to baseline DP which is guaranteed to work
    target, reached = 2520, {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
        if time.time() > deadline:
            break
    raise RuntimeError("no baseline representation")
# EVOLVE-BLOCK-END
