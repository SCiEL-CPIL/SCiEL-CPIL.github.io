"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin (evaluator kills at 150s)
    
    chosen, counts = [], {}
    for x in range(1, 1001):
        if time.time() > deadline:
            break
        new_sums = [x + y for y in chosen] + [2 * x]
        trial = dict(counts)
        valid = True
        for s in new_sums:
            trial[s] = trial.get(s, 0) + 1
            if trial[s] > 2:
                valid = False
                break
        if valid:
            chosen.append(x)
            counts = trial
    return chosen
# EVOLVE-BLOCK-END
