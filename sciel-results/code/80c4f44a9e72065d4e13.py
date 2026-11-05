import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin
    
    N = 129
    TARGET_LO = 128
    TARGET_HI = 256
    
    def compute_mults(A_bool):
        mult = np.zeros(257, dtype=np.int32)
        A_list = np.where(A_bool)[0]
        for i in range(len(A_list)):
            a = A_list[i]
            for j in range(i, len(A_list)):
                b = A_list[j]
                s = a + b
                if s <= 256:
                    mult[s] += 1
        return mult
    
    def get_peak(mult):
        return int(mult[TARGET_LO:TARGET_HI+1].max())
    
    def is_valid(mult):
        return bool(np.all(mult[TARGET_LO:TARGET_HI+1] > 0))
    
    random.seed(42)
    
    # Greedy initial construction
    A_bool = np.zeros(N, dtype=bool)
    
    # Start with a moderate random base
    base_size = 42
    indices = random.sample(range(N), base_size)
    A_bool[indices] = True
    
    mult = compute_mults(A_bool)
    
    # Ensure validity by adding elements
    for t in range(TARGET_LO, TARGET_HI + 1):
        if mult[t] > 0:
            continue
        added = False
        for x in range(N):
            if A_bool[x]:
                continue
            # Check if adding x covers target t
            covers = False
            if 2 * x == t:
                covers = True
            elif t - x >= 0 and t - x < N and t - x <= x and A_bool[t - x]:
                covers = True
            if covers:
                A_bool[x] = True
                A_list = np.where(A_bool)[0]
                for a in A_list:
                    if a < x:
                        s = a + x
                        if s <= 256:
                            mult[s] += 1
                s = 2 * x
                if s <= 256:
                    mult[s] += 1
                added = True
                break
        if not added:
            x = random.randint(0, 128)
            if not A_bool[x]:
                A_bool[x] = True
                A_list = np.where(A_bool)[0]
                for a in A_list:
                    if a < x:
                        s = a + x
                        if s <= 256:
                            mult[s] += 1
                s = 2 * x
                if s <= 256:
                    mult[s] += 1
    
    current_peak = get_peak(mult)
    best_peak = current_peak
    best_A = A_bool.copy()
    
    # Simulated annealing local search
    temp = 3.0
    cooling = 0.9995
    
    def do_add(x):
        nonlocal mult, A_bool
        A_list = np.where(A_bool)[0]
        for a in A_list:
            if a < x:
                s = a + x
                if s <= 256:
                    mult[s] += 1
        s = 2 * x
        if s <= 256:
            mult[s] += 1
        A_bool[x] = True
    
    def do_remove(x):
        nonlocal mult, A_bool
        A_list = np.where(A_bool)[0]
        for a in A_list:
            if a < x:
                s = a + x
                if s <= 256:
                    mult[s] -= 1
        s = 2 * x
        if s <= 256:
            mult[s] -= 1
        A_bool[x] = False
    
    while time.time() < deadline:
        r = random.random()
        
        if r < 0.35:
            # Add element
            candidates = np.where(~A_bool)[0]
            if len(candidates) == 0:
                continue
            x = random.choice(candidates.tolist())
            old_mult = mult.copy()
            do_add(x)
            new_peak = get_peak(mult)
            if new_peak <= current_peak or random.random() < np.exp(-(new_peak - current_peak) / temp):
                current_peak = new_peak
                if current_peak < best_peak:
                    best_peak = current_peak
                    best_A = A_bool.copy()
            else:
                mult[:] = old_mult
                A_bool[x] = False
        
        elif r < 0.70:
            # Remove element
            candidates = np.where(A_bool)[0]
            if len(candidates) == 0:
                continue
            x = random.choice(candidates.tolist())
            old_mult = mult.copy()
            do_remove(x)
            if is_valid(mult):
                new_peak = get_peak(mult)
                if new_peak <= current_peak or random.random() < np.exp(-(new_peak - current_peak) / temp):
                    current_peak = new_peak
                    if current_peak < best_peak:
                        best_peak = current_peak
                        best_A = A_bool.copy()
                else:
                    mult[:] = old_mult
                    A_bool[x] = True
                    A_list = np.where(A_bool)[0]
                    for a in A_list:
                        if a < x:
                            s = a + x
                            if s <= 256:
                                mult[s] += 1
                    s = 2 * x
                    if s <= 256:
                        mult[s] += 1
            else:
                mult[:] = old_mult
                A_bool[x] = True
                A_list = np.where(A_bool)[0]
                for a in A_list:
                    if a < x:
                        s = a + x
                        if s <= 256:
                            mult[s] += 1
                s = 2 * x
                if s <= 256:
                    mult[s] += 1
        
        else:
            # Swap
            add_cands = np.where(~A_bool)[0]
            rem_cands = np.where(A_bool)[0]
            if len(add_cands) == 0 or len(rem_cands) == 0:
                continue
            x_add = random.choice(add_cands.tolist())
            x_rem = random.choice(rem_cands.tolist())
            
            old_mult = mult.copy()
            
            # Remove x_rem
            A_list = np.where(A_bool)[0]
            for a in A_list:
                if a < x_rem:
                    s = a + x_rem
                    if s <= 256:
                        mult[s] -= 1
            s = 2 * x_rem
            if s <= 256:
                mult[s] -= 1
            A_bool[x_rem] = False
            
            # Add x_add
            A_list = np.where(A_bool)[0]
            for a in A_list:
                if a < x_add:
                    s = a + x_add
                    if s <= 256:
                        mult[s] += 1
            s = 2 * x_add
            if s <= 256:
                mult[s] += 1
            A_bool[x_add] = True
            
            if is_valid(mult):
                new_peak = get_peak(mult)
                if new_peak <= current_peak or random.random() < np.exp(-(new_peak - current_peak) / temp):
                    current_peak = new_peak
                    if current_peak < best_peak:
                        best_peak = current_peak
                        best_A = A_bool.copy()
                else:
                    mult[:] = old_mult
                    A_bool[x_add] = False
                    A_bool[x_rem] = True
            else:
                mult[:] = old_mult
                A_bool[x_add] = False
                A_bool[x_rem] = True
        
        temp *= cooling
    
    result = np.where(best_A)[0]
    return list(result)
