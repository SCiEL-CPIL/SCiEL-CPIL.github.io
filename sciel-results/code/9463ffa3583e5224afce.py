import time
import random
import numpy as np

N = 256
DEADLINE = 85.0

def _evaluate(x):
    best = 1
    for s in range(1, 129):
        for r in range(s):
            sub = x[r::s]
            if sub.size < 2:
                continue
            ps = np.empty(sub.size + 1)
            ps[0] = 0
            np.cumsum(sub, out=ps[1:])
            running_min = np.minimum.accumulate(ps)
            max_diff = np.max(ps - running_min)
            running_max = np.maximum.accumulate(ps)
            min_diff = np.min(ps - running_max)
            m = int(max(max_diff, -min_diff))
            if m > best:
                best = m
    return best

def construct():
    t0 = time.time()
    rng = random.Random(777)
    
    best_x = None
    best_score = float('inf')
    
    # Try multiple random starts
    while time.time() - t0 < DEADLINE - 3:
        x = np.array([rng.choice((-1, 1)) for _ in range(N)])
        score = _evaluate(x)
        if score < best_score:
            best_score = score
            best_x = x.copy()
        
        # Local search from this start
        improved = True
        while improved and time.time() - t0 < DEADLINE - 1:
            improved = False
            for i in range(N):
                if time.time() - t0 > DEADLINE - 0.5:
                    break
                x[i] = -x[i]
                new_score = _evaluate(x)
                if new_score < best_score:
                    best_x = x.copy()
                    best_score = new_score
                    improved = True
                else:
                    x[i] = -x[i]
            if best_score == 1:
                break
        if best_score == 1:
            break
    
    return best_x.tolist()
