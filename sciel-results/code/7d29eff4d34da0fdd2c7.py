"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    deadline = time.time() + 80
    n = 128
    targets = list(range(128, 257))
    
    # Precompute pair lists for each target
    target_pairs = {}
    for t in targets:
        pairs = []
        a_min = max(0, t - 128)
        a_max = min(128, t // 2)
        for a in range(a_min, a_max + 1):
            b = t - a
            if b <= 128 and b >= a:
                pairs.append((a, b))
        target_pairs[t] = pairs
    
    # Greedy init: start with [64,128]
    A_set = set(range(64, 129))
    
    # Compute initial multiplicities
    mults = np.zeros(257, dtype=np.int32)
    for t in targets:
        count = 0
        for a, b in target_pairs[t]:
            if a in A_set and b in A_set:
                count += 1
        mults[t] = count
    
    peak0 = int(np.max(mults[128:257]))
    best_peak = peak0
    best_A = set(A_set)
    
    # SA with incremental updates
    T = 3.0
    T_min = 0.005
    cooling = 0.9998
    
    def energy_from_mults(mults_arr):
        uncovered = np.sum(mults_arr[128:257] == 0)
        peak = np.max(mults_arr[128:257])
        return uncovered * 10000 + peak
    
    current_energy = energy_from_mults(mults)
    
    while time.time() < deadline and T > T_min:
        move_type = random.random()
        
        if move_type < 0.35:
            # Remove
            if len(A_set) <= 1:
                T *= cooling
                continue
            a = random.choice(list(A_set))
            # Update mults: for each b in A_set where b >= a
            for b in list(A_set):
                if b < a:
                    continue
                s = a + b
                if 128 <= s <= 256:
                    mults[s] -= 1
            A_set.remove(a)
            new_energy = energy_from_mults(mults)
            delta_e = new_energy - current_energy
            if delta_e < 0 or random.random() < np.exp(-delta_e / T):
                current_energy = new_energy
                p = int(np.max(mults[128:257]))
                u = int(np.sum(mults[128:257] == 0))
                if u == 0 and p < best_peak:
                    best_peak = p
                    best_A = set(A_set)
            else:
                A_set.add(a)
                for b in list(A_set):
                    if b < a:
                        continue
                    s = a + b
                    if 128 <= s <= 256:
                        mults[s] += 1
        elif move_type < 0.7:
            # Add
            candidates = [x for x in range(129) if x not in A_set]
            if not candidates:
                T *= cooling
                continue
            a = random.choice(candidates)
            A_set.add(a)
            for b in list(A_set):
                if b < a:
                    continue
                s = a + b
                if 128 <= s <= 256:
                    mults[s] += 1
            new_energy = energy_from_mults(mults)
            delta_e = new_energy - current_energy
            if delta_e < 0 or random.random() < np.exp(-delta_e / T):
                current_energy = new_energy
                p = int(np.max(mults[128:257]))
                u = int(np.sum(mults[128:257] == 0))
                if u == 0 and p < best_peak:
                    best_peak = p
                    best_A = set(A_set)
            else:
                A_set.remove(a)
                for b in list(A_set):
                    if b < a:
                        continue
                    s = a + b
                    if 128 <= s <= 256:
                        mults[s] -= 1
        else:
            # Swap
            if len(A_set) <= 1:
                T *= cooling
                continue
            a = random.choice(list(A_set))
            candidates = [x for x in range(129) if x not in A_set]
            if not candidates:
                T *= cooling
                continue
            b = random.choice(candidates)
            
            # Remove a
            for c in list(A_set):
                if c < a:
                    continue
                s = a + c
                if 128 <= s <= 256:
                    mults[s] -= 1
            A_set.remove(a)
            # Add b
            A_set.add(b)
            for c in list(A_set):
                if c < b:
                    continue
                s = b + c
                if 128 <= s <= 256:
                    mults[s] += 1
            
            new_energy = energy_from_mults(mults)
            delta_e = new_energy - current_energy
            if delta_e < 0 or random.random() < np.exp(-delta_e / T):
                current_energy = new_energy
                p = int(np.max(mults[128:257]))
                u = int(np.sum(mults[128:257] == 0))
                if u == 0 and p < best_peak:
                    best_peak = p
                    best_A = set(A_set)
            else:
                # Undo: remove b, add a
                A_set.remove(b)
                for c in list(A_set):
                    if c < b:
                        continue
                    s = b + c
                    if 128 <= s <= 256:
                        mults[s] -= 1
                A_set.add(a)
                for c in list(A_set):
                    if c < a:
                        continue
                    s = a + c
                    if 128 <= s <= 256:
                        mults[s] += 1
        
        T *= cooling
    
    return list(best_A)
# EVOLVE-BLOCK-END
