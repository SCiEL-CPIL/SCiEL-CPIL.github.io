import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    # Try to find a valid coloring for n=34 down to 1
    # Allocate time chunks to each n
    # If n=34 takes 20s and fails, we have 90s left for n=33...
    
    # We will try n=34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1
    # But we can't spend 110s / 34 on each.
    # Let's use a dynamic time budget.
    # Try 34 with 25s.
    # Try 33 with 25s.
    # Try 32 with 25s.
    # Try 31 with 25s.
    # Try 30 with 10s.
    # Try 29 with 5s.
    # ...
    # If we run out of time, return best found.
    
    best_result = {'n': 1, 'colors': []}
    
    # Time allocation strategy
    # n=34: 30s
    # n=33: 30s
    # n=32: 20s
    # n=31: 10s
    # n=30: 5s
    # n=29: 2s
    # n=28: 1s
    # n=27: 1s
    # n=26: 1s
    # n=25: 1s
    # n=24: 1s
    # n=23: 1s
    # n=22: 1s
    # n=21: 1s
    # n=20: 1s
    # n=19: 1s
    # n=18: 1s
    # n=17: 1s
    # n=16: 1s
    # n=15: 1s
    # n=14: 1s
    # n=13: 1s
    # n=12: 1s
    # n=11: 1s
    # n=10: 1s
    # n=9: 1s
    # n=8: 1s
    # n=7: 1s
    # n=6: 1s
    # n=5: 1s
    # n=4: 1s
    # n=3: 1s
    # n=2: 1s
    # n=1: 1s
    
    # Total: 30+30+20+10+5+2+1*24 = 122s. Too much.
    # Adjust:
    # n=34: 25s
    # n=33: 25s
    # n=32: 15s
    # n=31: 10s
    # n=30: 5s
    # n=29: 2s
    # n=28: 1s
    # n=27: 1s
    # n=26: 1s
    # n=25: 1s
    # n=24: 1s
    # n=23: 1s
    # n=22: 1s
    # n=21: 1s
    # n=20: 1s
    # n=19: 1s
    # n=18: 1s
    # n=17: 1s
    # n=16: 1s
    # n=15: 1s
    # n=14: 1s
    # n=13: 1s
    # n=12: 1s
    # n=11: 1s
    # n=10: 1s
    # n=9: 1s
    # n=8: 1s
    # n=7: 1s
    # n=6: 1s
    # n=5: 1s
    # n=4: 1s
    # n=3: 1s
    # n=2: 1s
    # n=1: 1s
    
    # Total: 25+25+15+10+5+2+24*1 = 106s. Safe.
    
    time_allocs = {
        34: 25,
        33: 25,
        32: 15,
        31: 10,
        30: 5,
        29: 2,
    }
    for n in range(28, 0, -1):
        time_allocs[n] = 1
        
    for n in range(34, 0, -1):
        if time.time() > deadline:
            break
            
        t_alloc = time_allocs.get(n, 1)
        t_limit = time.time() + t_alloc
        
        result = solve_n_sa(n, t_limit)
        if result is not None:
            return result
            
    return best_result

def solve_n_sa(n, deadline):
    # Precompute triples
    triples = list(itertools.combinations(range(n), 3))
    m = len(triples)
    
    # Map triple to index
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    
    # Precompute 4-sets and their triples
    four_sets = list(itertools.combinations(range(n), 4))
    four_set_triples = []
    for s in four_sets:
        trs = [triple_to_idx[t] for t in itertools.combinations(s, 3)]
        four_set_triples.append(trs)
        
    # Precompute 5-sets and their triples
    five_sets = list(itertools.combinations(range(n), 5))
    five_set_triples = []
    for s in five_sets:
        trs = [triple_to_idx[t] for t in itertools.combinations(s, 3)]
        five_set_triples.append(trs)
        
    # For each triple, which 4-sets and 5-sets does it belong to?
    triple_to_four_sets = [[] for _ in range(m)]
    triple_to_five_sets = [[] for _ in range(m)]
    
    for i, trs in enumerate(four_set_triples):
        for t_idx in trs:
            triple_to_four_sets[t_idx].append(i)
            
    for j, trs in enumerate(five_set_triples):
        for t_idx in trs:
            triple_to_five_sets[t_idx].append(j)
            
    # Initial random coloring
    colors = np.random.randint(0, 2, m).astype(np.int8)
    
    # Initialize counts
    red_count_4 = np.zeros(len(four_sets), dtype=np.int8)
    blue_count_5 = np.zeros(len(five_sets), dtype=np.int8)
    
    for i, trs in enumerate(four_set_triples):
        red_count_4[i] = np.sum(colors[trs] == 0)
        
    for j, trs in enumerate(five_set_triples):
        blue_count_5[j] = np.sum(colors[trs] == 1)
        
    # Identify violations
    viol_4 = [i for i in range(len(four_sets)) if red_count_4[i] == 4]
    viol_5 = [j for j in range(len(five_sets)) if blue_count_5[j] == 10]
    
    if not viol_4 and not viol_5:
        return {'n': n, 'colors': colors.tolist()}
        
    # Simulated Annealing
    T = 1.0
    T_min = 0.01
    cooling = 0.9999
    
    steps = 0
    best_violations = len(viol_4) + len(viol_5)
    best_colors = colors.copy()
    
    while time.time() < deadline:
        if not viol_4 and not viol_5:
            return {'n': n, 'colors': colors.tolist()}
            
        # Pick a move
        # Bias towards picking from violations
        if viol_4 and (not viol_5 or random.random() < 0.7):
            # Pick random 4-set violation
            v4_idx = random.choice(viol_4)
            trs = four_set_triples[v4_idx]
            t_idx = random.choice(trs)
        elif viol_5:
            # Pick random 5-set violation
            v5_idx = random.choice(viol_5)
            trs = five_set_triples[v5_idx]
            t_idx = random.choice(trs)
        else:
            # No violations? Should have returned.
            break
            
        # Calculate Delta E
        # Old counts
        old_red_4 = [red_count_4[i] for i in triple_to_four_sets[t_idx]]
        old_blue_5 = [blue_count_5[j] for j in triple_to_five_sets[t_idx]]
        
        old_color = colors[t_idx]
        new_color = 1 - old_color
        
        # New counts
        # If old_color was 0 (red), red counts decrease.
        # If old_color was 1 (blue), blue counts decrease.
        
        delta_e = 0
        
        # Update 4-sets
        for i in triple_to_four_sets[t_idx]:
            if old_color == 0:
                # Red count decreases
                new_count = old_red_4[triple_to_four_sets[t_idx].index(i)] # This is slow
                # Better to iterate with index
            # Let's re-structure this loop to be faster
            pass
            
        # Faster Delta E calculation
        delta_e = 0
        f_sets = triple_to_four_sets[t_idx]
        for idx_in_list, i in enumerate(f_sets):
            old_c = red_count_4[i]
            if old_color == 0:
                new_c = old_c - 1
            else:
                new_c = old_c + 1
            # Penalty: 1 if count == 4
            old_pen = 1 if old_c == 4 else 0
            new_pen = 1 if new_c == 4 else 0
            delta_e += new_pen - old_pen
            
        f_sets_5 = triple_to_five_sets[t_idx]
        for idx_in_list, j in enumerate(f_sets_5):
            old_c = blue_count_5[j]
            if old_color == 1:
                new_c = old_c - 1
            else:
                new_c = old_c + 1
            # Penalty: 1 if count == 10
            old_pen = 1 if old_c == 10 else 0
            new_pen = 1 if new_c == 10 else 0
            delta_e += new_pen - old_pen
            
        # Acceptance
        if delta_e < 0:
            accept = True
        else:
            if delta_e == 0:
                accept = True
            else:
                # Accept with prob exp(-delta_e / T)
                # delta_e is integer
                if T > 0:
                    prob = np.exp(-delta_e / T)
                    accept = random.random() < prob
                else:
                    accept = False
                    
        if accept:
            # Apply move
            colors[t_idx] = new_color
            
            # Update counts
            for i in f_sets:
                if old_color == 0:
                    red_count_4[i] -= 1
                else:
                    red_count_4[i] += 1
                    
                # Update viol_4 list (lazy)
                if red_count_4[i] == 4:
                    if i not in viol_4: # O(N) check in list
                        viol_4.append(i)
                else:
                    # If it was 4, it's no longer 4.
                    # We don't remove from list immediately to save time.
                    # We'll clean up if list gets too big or if we pick a stale entry.
                    pass
                    
            for j in f_sets_5:
                if old_color == 1:
                    blue_count_5[j] -= 1
                else:
                    blue_count_5[j] += 1
                    
                if blue_count_5[j] == 10:
                    if j not in viol_5:
                        viol_5.append(j)
                else:
                    pass
                    
            # Track best
            current_violations = len(viol_4) + len(viol_5)
            if current_violations < best_violations:
                best_violations = current_violations
                best_colors = colors.copy()
                
            # Cool
            T *= cooling
            if T < T_min:
                T = T_min
                
        steps += 1
        if steps % 1000 == 0:
            # Clean up stale violations
            # This is expensive, so do it rarely
            if len(viol_4) > 10000:
                viol_4 = [i for i in viol_4 if red_count_4[i] == 4]
            if len(viol_5) > 10000:
                viol_5 = [j for j in viol_5 if blue_count_5[j] == 10]
                
            # Restart if stuck?
            if T < T_min * 1.1:
                # Restart with new random
                colors = np.random.randint(0, 2, m).astype(np.int8)
                for i, trs in enumerate(four_set_triples):
                    red_count_4[i] = np.sum(colors[trs] == 0)
                for j, trs in enumerate(five_set_triples):
                    blue_count_5[j] = np.sum(colors[trs] == 1)
                viol_4 = [i for i in range(len(four_sets)) if red_count_4[i] == 4]
                viol_5 = [j for j in range(len(five_sets)) if blue_count_5[j] == 10]
                T = 1.0
                best_violations = len(viol_4) + len(viol_5)
                best_colors = colors.copy()

    # Return best found
    if best_violations == 0:
        return {'n': n, 'colors': best_colors.tolist()}
    return None
