"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    # Set a safe deadline well below the 60s limit
    deadline = start_time + 50.0

    # Precompute all C(16,4) subsets as 16-bit masks
    all_subsets = []
    for combo in combinations(range(16), 4):
        mask = 0
        for i in combo:
            mask |= (1 << i)
        all_subsets.append(mask)

    n_subsets = len(all_subsets)

    def greedy(order):
        chosen = []
        # Precompute pair data for faster checking? 
        # We store (a, b, a&b, a^b) ? 
        # Condition: (a&s)==ab and (b&s)==ab
        # This is equivalent to: s contains ab, and s is disjoint from (a^b) \ ab? 
        # No, a^b is symmetric difference. 
        # a\s and b\s must be empty in the parts outside ab.
        # Let core = a & b.
        # Let diff = (a | b) ^ core.
        # s is bad if (s & core) == core and (s & diff) == 0.
        pair_data = [] 
        for idx in order:
            if time.time() > deadline:
                break
            s = all_subsets[idx]
            ok = True
            for core, diff in pair_data:
                if (s & core) == core and (s & diff) == 0:
                    ok = False
                    break
            if ok:
                chosen.append(s)
                # Update pair data
                # New pairs are (s, old) for all old in chosen[:-1]
                # We need to calculate core and diff for each new pair
                for old in chosen[:-1]:
                    core = s & old
                    diff = (s | old) ^ core
                    pair_data.append((core, diff))
        return chosen

    # Lexicographic order
    best = greedy(list(range(n_subsets)))
    best_size = len(best)

    # Random restarts
    rng = random.Random(42)
    while time.time() < deadline - 5.0:
        order = list(range(n_subsets))
        rng.shuffle(order)
        result = greedy(order)
        if len(result) > best_size:
            best = result
            best_size = len(result)

    return [sorted(i + 1 for i in range(16) if (mask >> i) & 1) for mask in best]
# EVOLVE-BLOCK-END
