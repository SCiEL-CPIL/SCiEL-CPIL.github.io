import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    def is_valid_fast(seq):
        """Fast validity check using prefix sums"""
        if not seq:
            return True
        L = len(seq)
        P = [0] * (L + 1)
        for i in range(L):
            P[i + 1] = P[i] + seq[i]
        
        sums = set()
        for i in range(L):
            pi = P[i]
            for j in range(i + 1, L + 1):
                s = P[j] - pi
                if s in sums:
                    return False
                sums.add(s)
        return True

    def greedy_build():
        """Greedy forward construction"""
        values = []
        sums = set()
        for x in range(1, 257):
            suffix_sums = []
            current_sum = 0
            for v in reversed(values):
                current_sum += v
                suffix_sums.append(current_sum)
            
            potential_sums = [x] + [x + s for s in suffix_sums]
            
            valid = True
            for ps in potential_sums:
                if ps in sums:
                    valid = False
                    break
            
            if valid:
                values.append(x)
                for ps in potential_sums:
                    sums.add(ps)
        
        return values

    # Initial solution from greedy
    best_seq = greedy_build()
    best_len = len(best_seq)
    
    # Simulated annealing
    current_seq = list(best_seq)
    current_set = set(current_seq)
    all_candidates = list(range(1, 257))
    
    T = 1.0  # Initial temperature
    T_min = 0.01
    cooling_rate = 0.9995
    
    while time.time() < deadline and T > T_min:
        # Generate a move
        r = random.random()
        
        if r < 0.6:  # Swap: replace one element with another
            if len(current_seq) > 0:
                x_rem = random.choice(current_seq)
                available = [x for x in all_candidates if x not in current_set]
                if available:
                    x_add = random.choice(available)
                    new_seq = [x for x in current_seq if x != x_rem]
                    new_seq.append(x_add)
                    new_seq.sort()
                    
                    if is_valid_fast(new_seq):
                        # Accept if valid; use SA for length changes
                        new_len = len(new_seq)
                        delta = new_len - len(current_seq)
                        if delta >= 0 or random.random() < math.exp(-delta / T):
                            current_seq = new_seq
                            current_set = set(current_seq)
                            if len(current_seq) > best_len:
                                best_len = len(current_seq)
                                best_seq = list(current_seq)
        
        elif r < 0.8:  # Add
            available = [x for x in all_candidates if x not in current_set]
            if available:
                x = random.choice(available)
                new_seq = current_seq + [x]
                new_seq.sort()
                if is_valid_fast(new_seq):
                    current_seq = new_seq
                    current_set.add(x)
                    if len(current_seq) > best_len:
                        best_len = len(current_seq)
                        best_seq = list(current_seq)
        
        else:  # Remove
            if current_seq:
                x = random.choice(current_seq)
                current_seq.remove(x)
                current_set.remove(x)
        
        T *= cooling_rate
    
    # Final local search with pure hill climbing
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(current_seq)):
            for x in all_candidates:
                if x not in current_set:
                    new_seq = [v for j, v in enumerate(current_seq) if j != i]
                    new_seq.append(x)
                    new_seq.sort()
                    if is_valid_fast(new_seq) and len(new_seq) >= len(current_seq):
                        current_seq = new_seq
                        current_set = set(current_seq)
                        improved = True
                        if len(current_seq) > best_len:
                            best_len = len(current_seq)
                            best_seq = list(current_seq)
                        break
    
    return sorted(best_seq)
