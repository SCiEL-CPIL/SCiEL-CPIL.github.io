import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s margin
    
    # Initial best: Seed program
    best_A = set(range(64, 129))
    best_peak, best_uncovered = evaluate(best_A)
    
    # If seed is not valid (it is, peak 33), we'd need a fallback.
    # But seed is valid.
    
    best_score = best_peak  # We want to minimize peak among valid sets
    
    # Local search
    # Try different sizes
    # We will do a few restarts for different K
    for k in range(16, 26):
        if time.time() > deadline:
            break
            
        # Number of restarts depends on time remaining
        # Let's do 5 restarts per k initially, more if fast
        for restart in range(5):
            if time.time() > deadline:
                break
                
            # Init random set of size k
            A = set(random.sample(range(129), k))
            
            # Repair to ensure validity
            A = repair(A)
            
            peak, uncovered = evaluate(A)
            
            # If already valid and better, update
            if uncovered == 0 and peak < best_score:
                best_score = peak
                best_A = set(A)
            
            # Local search to minimize peak
            no_improve_count = 0
            while time.time() < deadline and no_improve_count < 500:
                # Generate move
                move_type = random.choice(['add', 'remove', 'swap'])
                new_A = set(A)
                
                if move_type == 'add':
                    candidates = [x for x in range(129) if x not in A]
                    if candidates:
                        x = random.choice(candidates)
                        new_A.add(x)
                    else:
                        no_improve_count += 1
                        continue
                elif move_type == 'remove':
                    if len(A) > 15:  # Don't go too small
                        x = random.choice(list(A))
                        new_A.remove(x)
                    else:
                        no_improve_count += 1
                        continue
                elif move_type == 'swap':
                    if A:
                        x = random.choice(list(A))
                        candidates = [y for y in range(129) if y not in A]
                        if candidates:
                            y = random.choice(candidates)
                            new_A.remove(x)
                            new_A.add(y)
                        else:
                            no_improve_count += 1
                            continue
                    else:
                        no_improve_count += 1
                        continue
                else:
                    no_improve_count += 1
                    continue
                
                new_peak, new_uncovered = evaluate(new_A)
                
                # Cost function: Prioritize coverage, then peak
                # If uncovered > 0, it's "worse" than any valid set.
                # We can define cost as (uncovered, peak)
                old_cost = (uncovered, peak)
                new_cost = (new_uncovered, new_peak)
                
                if new_cost < old_cost:
                    A = new_A
                    peak = new_peak
                    uncovered = new_uncovered
                    no_improve_count = 0
                    
                    if uncovered == 0 and peak < best_score:
                        best_score = peak
                        best_A = set(A)
                else:
                    no_improve_count += 1
                    
    return list(best_A)

def evaluate(A_set):
    A_arr = np.zeros(129)
    for x in A_set:
        A_arr[x] = 1.0
        
    conv = np.convolve(A_arr, A_arr)
    
    peak = 0
    uncovered = 0
    
    # Target sums 128 to 256
    for s in range(128, 257):
        v = conv[s]
        # Check if s/2 is in A
        n_eq = 0
        if s % 2 == 0:
            half = s // 2
            if 0 <= half <= 128 and A_arr[half] == 1.0:
                n_eq = 1
        
        m = int((v + n_eq) / 2)
        
        if m == 0:
            uncovered += 1
        elif m > peak:
            peak = m
            
    return peak, uncovered

def repair(A_set):
    A = set(A_set)
    # Limit repair steps to avoid infinite loop
    for _ in range(100):
        _, uncovered_count = evaluate(A)
        if uncovered_count == 0:
            break
        
        # Find uncovered sums
        A_arr = np.zeros(129)
        for x in A:
            A_arr[x] = 1.0
        conv = np.convolve(A_arr, A_arr)
        
        uncovered_sums = set()
        for s in range(128, 257):
            v = conv[s]
            n_eq = 1 if (s % 2 == 0 and 0 <= s // 2 <= 128 and A_arr[s // 2] == 1.0) else 0
            m = int((v + n_eq) / 2)
            if m == 0:
                uncovered_sums.add(s)
                
        if not uncovered_sums:
            break
            
        # Greedy: pick x that covers most uncovered sums
        best_x = None
        max_covered = 0
        for x in range(129):
            if x in A:
                continue
            
            covered = 0
            for a in A:
                s = x + a
                if s in uncovered_sums:
                    covered += 1
            # Also x+x
            if 2 * x in uncovered_sums:
                covered += 1
                
            if covered > max_covered:
                max_covered = covered
                best_x = x
                
        if best_x is None or max_covered == 0:
            # Stuck, pick random
            candidates = [x for x in range(129) if x not in A]
            if candidates:
                best_x = random.choice(candidates)
            else:
                break  # Should not happen if A is not full
                
        A.add(best_x)
        
    return A
