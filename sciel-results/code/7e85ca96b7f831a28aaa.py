import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit
    
    best_score = -1
    best_set = [0]
    
    # Precomputed Golomb rulers (known good structures)
    # Golomb rulers have all pairwise differences distinct, maximizing |U-U|
    GLOMB_RULERS = [
        [0, 1, 4, 10],           # n=4
        [0, 1, 4, 10, 18],       # n=5
        [0, 1, 4, 10, 18, 23],   # n=6
        [0, 1, 4, 10, 18, 23, 29], # n=7
        [0, 1, 4, 10, 18, 23, 29, 36], # n=8
        [0, 1, 4, 10, 18, 23, 29, 36, 44], # n=9
        [0, 1, 4, 10, 18, 23, 29, 36, 44, 52], # n=10
    ]
    
    def compute_score(U):
        if not U or 0 not in U:
            return -1
        max_u = max(U)
        if max_u == 0:
            return 0
        
        sumset = set()
        diffset = set()
        
        for a in U:
            for b in U:
                sumset.add(a + b)
                diffset.add(a - b)
        
        n_sum = len(sumset)
        n_diff = len(diffset)
        
        if n_sum == 0:
            return -1
        
        base = 2 * max_u + 1
        if base <= 1:
            return 0
        
        ratio = n_diff / n_sum
        if ratio <= 1:
            return 1.0
        score = 1 + math.log(ratio) / math.log(base)
        return score

    def try_set(cand):
        nonlocal best_score, best_set
        cand = sorted(set(cand))
        if not cand or 0 not in cand:
            return
        if max(cand) == 0:
            return
        s = compute_score(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)
            return True
        return False

    # Phase 1: Try precomputed Golomb rulers
    for gr in GLOMB_RULERS:
        try_set(gr)
    
    # Also try some small hand-crafted sets
    small_sets = [
        [0, 1, 3],
        [0, 1, 4],
        [0, 2, 5],
        [0, 1, 2, 5],
        [0, 1, 3, 7],
        [0, 1, 4, 6],
        [0, 2, 3, 7],
    ]
    for ss in small_sets:
        try_set(ss)
    
    # Phase 2: Local search from best found
    current = list(best_set)
    
    no_improve = 0
    max_no_improve = 200
    
    while time.time() < deadline - 10 and no_improve < max_no_improve:
        improved = False
        max_cur = max(current)
        
        # 1. Try adding an element (wider range)
        candidates_to_add = set()
        # Nearby positions
        for i in range(1, max_cur + 50):
            if i not in current:
                candidates_to_add.add(i)
        # Some larger jumps
        for _ in range(30):
            candidates_to_add.add(random.randint(max_cur + 1, max_cur + 200))
        # Also try very small additions
        for i in range(1, 10):
            if i not in current:
                candidates_to_add.add(i)
        
        for new_el in candidates_to_add:
            if new_el in current:
                continue
            cand = sorted(current + [new_el])
            if try_set(cand):
                current = list(best_set)
                improved = True
                no_improve = 0
                break
        
        # 2. Try replacing an element (wider range)
        if len(current) >= 3:
            for i in range(len(current)):
                if current[i] == 0:
                    continue
                old_el = current[i]
                # Try replacing with a wider range
                for delta in range(-50, 51):
                    new_el = old_el + delta
                    if new_el < 1 or new_el in current:
                        continue
                    if new_el == old_el:
                        continue
                    cand = list(current)
                    cand[i] = new_el
                    cand = sorted(set(cand))
                    if 0 not in cand:
                        continue
                    if try_set(cand):
                        current = list(best_set)
                        improved = True
                        no_improve = 0
                        break
                if improved:
                    break
        
        # 3. Try removing an element
        if len(current) > 3:
            for i in range(len(current)):
                if current[i] == 0:
                    continue
                cand = [x for j, x in enumerate(current) if j != i]
                if try_set(cand):
                    current = list(best_set)
                    improved = True
                    no_improve = 0
                    break
        
        # 4. Try swapping two elements
        if len(current) > 2:
            for i in range(1, len(current)):
                for j in range(i + 1, len(current)):
                    cand = list(current)
                    cand[i], cand[j] = cand[j], cand[i]
                    cand = sorted(set(cand))
                    if 0 in cand and len(cand) == len(current):
                        if try_set(cand):
                            current = list(best_set)
                            improved = True
                            no_improve = 0
                            break
                if improved:
                    break
        
        if not improved:
            no_improve += 1
            # Random perturbation
            if random.random() < 0.5:
                # Random mutation: change one element
                if len(current) > 1:
                    idx = random.randint(1, len(current) - 1)
                    new_val = random.randint(1, max_cur + 100)
                    if new_val not in current:
                        cand = list(current)
                        cand[idx] = new_val
                        cand = sorted(set(cand))
                        if 0 in cand:
                            try_set(cand)
                            current = list(best_set)
                else:
                    # Add a random element
                    new_el = random.randint(1, max_cur + 100)
                    if new_el not in current:
                        cand = sorted(current + [new_el])
                        try_set(cand)
                        current = list(best_set)
    
    # Phase 3: Final polish - try many random restarts from best
    if time.time() < deadline - 20:
        for _ in range(50):
            if time.time() > deadline - 10:
                break
            # Start from best and make several random changes
            cand = list(best_set)
            for _ in range(3):
                if time.time() > deadline - 10:
                    break
                op = random.choice(['add', 'replace', 'remove'])
                if op == 'add' and len(cand) < 20:
                    new_el = random.randint(1, max(cand) + 100)
                    if new_el not in cand:
                        cand = sorted(cand + [new_el])
                elif op == 'replace' and len(cand) > 2:
                    idx = random.randint(1, len(cand) - 1)
                    new_el = random.randint(1, max(cand) + 50)
                    if new_el not in cand:
                        cand = list(cand)
                        cand[idx] = new_el
                        cand = sorted(set(cand))
                        if 0 not in cand:
                            cand = list(best_set)
                elif op == 'remove' and len(cand) > 3:
                    idx = random.randint(1, len(cand) - 1)
                    cand = [x for j, x in enumerate(cand) if j != idx]
                try_set(cand)
    
    return list(best_set)
