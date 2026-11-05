import time
import random
import math

def _score(s):
    """Calculate the number of distinct d > 0 such that a, a+d, a+2d are in s."""
    diffs = set()
    elements = sorted(s)
    n = len(elements)
    for i in range(n):
        a = elements[i]
        for j in range(i + 1, n):
            b = elements[j]
            d = b - a
            c = b + d
            if c in s:
                diffs.add(d)
    return len(diffs)

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    best_set = None
    best_score = 0
    
    # Generate diverse initial sets
    initial_sets = []
    
    # 1. Contiguous (Seed)
    initial_sets.append(set(range(1, 33)))
    
    # 2. Random
    for _ in range(10):
        initial_sets.append(set(random.sample(range(1, 513), 32)))
        
    # 3. Arithmetic Progressions
    for d in [2, 3, 4, 5, 8, 10, 16]:
        max_start = 512 - 31 * d
        if max_start >= 1:
            a = random.randint(1, max_start)
            initial_sets.append(set(a + i * d for i in range(32)))
            
    # 4. Two Blocks
    for _ in range(5):
        s1_start = random.randint(1, 200)
        s2_start = random.randint(300, 480)
        s = set()
        # Ensure 32 distinct elements
        # Try to make them dense in their blocks to capture small d's
        # Block 1: 16 elements
        # Block 2: 16 elements
        try:
            block1 = set(random.sample(range(s1_start, s1_start + 50), 16))
            block2 = set(random.sample(range(s2_start, s2_start + 50), 16))
            s = block1 | block2
            if len(s) == 32:
                initial_sets.append(s)
        except:
            pass
            
    # Evaluate initial sets
    for s in initial_sets:
        if len(s) != 32: continue
        sc = _score(s)
        if sc > best_score:
            best_score = sc
            best_set = s
            
    if best_set is None:
        best_set = set(range(1, 33))
        best_score = _score(best_set)
        
    # Simulated Annealing
    current_set = set(best_set)
    current_score = best_score
    
    T = 10.0
    cooling = 0.999
    min_T = 0.1
    stuck_count = 0
    max_stuck = 5000
    
    while time.time() < deadline:
        # Generate move
        elem_out = random.choice(list(current_set))
        elem_in = random.randint(1, 512)
        
        if elem_in in current_set:
            continue
            
        new_set = set(current_set)
        new_set.remove(elem_out)
        new_set.add(elem_in)
        
        new_score = _score(new_set)
        delta = new_score - current_score
        
        if delta > 0:
            current_set = new_set
            current_score = new_score
            stuck_count = 0
            if new_score > best_score:
                best_score = new_score
                best_set = new_set
        else:
            # Accept with probability exp(delta/T)
            if delta < -10:
                stuck_count += 1
            elif random.random() < math.exp(delta / T):
                current_set = new_set
                current_score = new_score
            else:
                stuck_count += 1
                
        T *= cooling
        if T < min_T:
            T = min_T
            
        if stuck_count > max_stuck:
            # Perturb
            for _ in range(5):
                eo = random.choice(list(current_set))
                ei = random.randint(1, 512)
                if ei not in current_set:
                    current_set.remove(eo)
                    current_set.add(ei)
            current_score = _score(current_set)
            stuck_count = 0
            T = max(T, 5.0)
            
    return list(best_set)
