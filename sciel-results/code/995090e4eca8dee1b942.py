import time
import random

def construct():
    deadline = time.time() + 85
    
    def init_reps(A):
        reps = [0] * 257
        for a in A:
            for b in A:
                if a <= b:
                    s = a + b
                    if 128 <= s <= 256:
                        reps[s] += 1
        return reps
    
    def get_peak(reps):
        m = 0
        for t in range(128, 257):
            if reps[t] == 0:
                return float('inf')
            if reps[t] > m:
                m = reps[t]
        return m
    
    A = set(range(64, 129))
    reps = init_reps(A)
    best_peak = get_peak(reps)
    best_A = set(A)
    
    if time.time() > deadline:
        return sorted(best_A)
    
    random.seed(42)
    
    for step in range(200000):
        if time.time() > deadline - 1:
            break
        
        move_type = random.random()
        
        if move_type < 0.5 and len(A) > 1:
            x = random.choice(list(A))
            A.discard(x)
            s = 2*x
            if 128 <= s <= 256:
                reps[s] -= 1
            for y in A:
                if x < y:
                    s = x + y
                    if 128 <= s <= 256:
                        reps[s] -= 1
        else:
            x = random.randint(0, 128)
            if x in A:
                continue
            A.add(x)
            s = 2*x
            if 128 <= s <= 256:
                reps[s] += 1
            for y in A:
                if x < y:
                    s = x + y
                    if 128 <= s <= 256:
                        reps[s] += 1
        
        peak = get_peak(reps)
        
        if peak < best_peak:
            best_peak = peak
            best_A = set(A)
        else:
            if move_type < 0.5:
                A.add(x)
                s = 2*x
                if 128 <= s <= 256:
                    reps[s] += 1
                for y in A:
                    if x < y:
                        s = x + y
                        if 128 <= s <= 256:
                            reps[s] += 1
            else:
                A.discard(x)
                s = 2*x
                if 128 <= s <= 256:
                    reps[s] -= 1
                for y in A:
                    if x < y:
                        s = x + y
                        if 128 <= s <= 256:
                            reps[s] -= 1
    
    return sorted(best_A)
