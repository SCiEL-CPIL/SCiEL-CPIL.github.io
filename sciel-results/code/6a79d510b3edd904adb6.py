"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random

def construct():
    best = []
    best_size = 0
    
    def greedy_greedy(start_seed=None):
        if start_seed is not None:
            random.seed(start_seed)
        chosen = []
        counts = {}
        # Try adding in random order
        candidates = list(range(1, 1001))
        random.shuffle(candidates)
        for x in candidates:
            new_sums = [x + y for y in chosen] + [2 * x]
            trial = dict(counts)
            ok = True
            for s in new_sums:
                trial[s] = trial.get(s, 0) + 1
                if trial[s] > 2:
                    ok = False
                    break
            if ok:
                chosen.append(x)
                counts = trial
        return chosen
    
    # Try multiple random seeds
    for seed in range(50):
        result = greedy_greedy(seed)
        if len(result) > best_size:
            best_size = len(result)
            best = result
    
    # Also try the original greedy as a baseline
    chosen = []
    counts = {}
    for x in range(1, 1001):
        new_sums = [x + y for y in chosen] + [2 * x]
        trial = dict(counts)
        ok = True
        for s in new_sums:
            trial[s] = trial.get(s, 0) + 1
            if trial[s] > 2:
                ok = False
                break
        if ok:
            chosen.append(x)
            counts = trial
    if len(chosen) > best_size:
        best_size = len(chosen)
        best = chosen
    
    return best
# EVOLVE-BLOCK-END
