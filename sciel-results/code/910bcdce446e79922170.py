import time
import random

def construct():
    start_time = time.time()
    TIME_LIMIT = 55.0
    MAX_VAL = 256

    def get_consecutive_sums(seq):
        sums = set()
        n = len(seq)
        for i in range(n):
            total = 0
            for j in range(i, n):
                total += seq[j]
                sums.add(total)
        return sums

    def is_valid(seq):
        if not seq:
            return True
        if seq[0] < 1 or seq[-1] > MAX_VAL:
            return False
        for i in range(len(seq) - 1):
            if seq[i] >= seq[i + 1]:
                return False
        sums = get_consecutive_sums(seq)
        return len(sums) == len(seq) * (len(seq) + 1) // 2

    def count_collisions(seq):
        sums = set()
        n = len(seq)
        collisions = 0
        for i in range(n):
            total = 0
            for j in range(i, n):
                total += seq[j]
                if total in sums:
                    collisions += 1
                else:
                    sums.add(total)
        return collisions

    def get_sequence(a1, gaps):
        seq = [a1]
        cur = a1
        for g in gaps:
            cur += g
            seq.append(cur)
        return seq

    def simulated_annealing(L, time_limit, rng):
        if L <= 0:
            return []
        
        best_seq = None
        best_cost = float('inf')
        
        # Estimate average gap
        avg_gap = MAX_VAL // L
        
        while time.time() < time_limit - 1.0:
            # Generate initial state
            # a1 small, gaps around avg
            a1 = rng.randint(1, max(1, avg_gap // 2))
            gaps = [max(1, rng.randint(avg_gap // 2, avg_gap * 2)) for _ in range(L-1)]
            seq = get_sequence(a1, gaps)
            
            # Scale if too large
            if seq[-1] > MAX_VAL:
                scale = MAX_VAL / seq[-1]
                new_gaps = [max(1, int(g * scale)) for g in gaps]
                seq = get_sequence(a1, new_gaps)
                if seq[-1] > MAX_VAL:
                    continue
            elif seq[-1] < MAX_VAL // 2:
                # Maybe scale up to use space? Not strictly necessary, but helps distinctness
                pass
            
            cost = count_collisions(seq)
            if cost == 0:
                return seq
            
            T = 10.0
            T_min = 0.01
            cooling_rate = 0.995
            current_seq = seq
            current_cost = cost
            gaps = [seq[i+1] - seq[i] for i in range(L-1)]
            
            while time.time() < time_limit - 1.0:
                if current_cost < best_cost:
                    best_cost = current_cost
                    if best_cost == 0:
                        return current_seq
                    best_seq = current_seq[:]
                
                # Pick a random move
                idx = rng.randint(0, L-1)
                # Increase delta size occasionally for jumps
                if rng.random() < 0.1:
                    delta = rng.randint(-10, 10)
                else:
                    delta = rng.randint(-3, 3)
                
                if delta == 0: continue
                
                if idx == 0:
                    new_a1 = current_seq[0] + delta
                    if new_a1 < 1: continue
                    new_seq = [new_a1]
                    cur_val = new_a1
                    for g in gaps:
                        cur_val += g
                        new_seq.append(cur_val)
                    if new_seq[-1] > MAX_VAL: continue
                else:
                    new_gaps = gaps[:]
                    new_gaps[idx-1] += delta
                    if new_gaps[idx-1] < 1: continue
                    new_seq = get_sequence(current_seq[0], new_gaps)
                    if new_seq[-1] > MAX_VAL: continue
                    
                new_cost = count_collisions(new_seq)
                
                # Acceptance criterion
                if new_cost < current_cost:
                    current_cost = new_cost
                    current_seq = new_seq
                    gaps = [new_seq[i+1] - new_seq[i] for i in range(L-1)]
                else:
                    delta_cost = new_cost - current_cost
                    if T > 0 and rng.random() < pow(2.71828, -delta_cost / T):
                        current_cost = new_cost
                        current_seq = new_seq
                        gaps = [new_seq[i+1] - new_seq[i] for i in range(L-1)]
                
                T *= cooling_rate
                if T < T_min:
                    break
        
        if best_cost == 0:
            return best_seq
        return None

    # Fallback Greedy
    def greedy_desc(start=MAX_VAL, step=1):
        vals = []
        sums = set()
        for x in range(start, 0, -step):
            new_sums = set()
            total = x
            new_sums.add(total)
            for a in reversed(vals):
                total += a
                new_sums.add(total)
            if not sums.intersection(new_sums):
                vals.append(x)
                sums.update(new_sums)
        vals.sort()
        return vals

    fallback = greedy_desc()
    best_len = len(fallback)
    
    # Try SA for decreasing L
    # Max L is likely around 12-15
    for L in range(16, best_len, -1):
        if time.time() > TIME_LIMIT - 5.0:
            break
        
        time_budget = 4.0
        t_end = time.time() + time_budget
        rng = random.Random(42 + L)
        
        res = simulated_annealing(L, t_end, rng)
        
        if res and is_valid(res):
            return res

    return fallback
