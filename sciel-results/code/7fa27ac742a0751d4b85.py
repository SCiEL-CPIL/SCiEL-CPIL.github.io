import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _witness_counts(s):
    """For each d>0, count how many (a, a+d, a+2d) triples are all in s."""
    s_set = set(s)
    elements = sorted(s_set)
    n = len(elements)
    wc = {}
    for i in range(n):
        a = elements[i]
        for j in range(i + 1, n):
            b = elements[j]
            d = b - a
            if a + 2 * d in s_set:
                wc[d] = wc.get(d, 0) + 1
    return wc

def _score(s):
    """Number of distinct d>0 with at least one 3-term AP."""
    s_set = set(s)
    elements = sorted(s_set)
    n = len(elements)
    diffs = set()
    for i in range(n):
        a = elements[i]
        for j in range(i + 1, n):
            b = elements[j]
            d = b - a
            if a + 2 * d in s_set:
                diffs.add(d)
    return len(diffs)

def _delta_score(s_set, wc, elem_out, elem_in):
    """
    Compute change in score when removing elem_out and adding elem_in.
    Returns (delta, new_wc).
    """
    # Remove elem_out: for each other element x, check if any AP involving elem_out is broken
    # An AP (a, a+d, a+2d) involves elem_out if elem_out is one of the three.
    # For each x in s_set (x != elem_out):
    #   d1 = |x - elem_out|
    #   Check triples where elem_out is the middle: (elem_out-d1, elem_out, elem_out+d1)
    #   Check triples where elem_out is the first: (elem_out, elem_out+d1, elem_out+2*d1)
    #   Check triples where elem_out is the last: (elem_out-2*d1, elem_out-d1, elem_out)
    
    affected = {}  # d -> change in count
    
    for x in s_set:
        if x == elem_out:
            continue
        d1 = abs(x - elem_out)
        if d1 == 0:
            continue
        
        # elem_out is the middle: (elem_out - d1, elem_out, elem_out + d1)
        # x is either elem_out - d1 or elem_out + d1
        if x == elem_out - d1:
            # triple: (x, elem_out, elem_out + d1)
            if elem_out + d1 in s_set:
                affected[d1] = affected.get(d1, 0) - 1
        elif x == elem_out + d1:
            # triple: (elem_out - d1, elem_out, x)
            if elem_out - d1 in s_set:
                affected[d1] = affected.get(d1, 0) - 1
        
        # elem_out is the first: (elem_out, elem_out + d1, elem_out + 2*d1)
        # x = elem_out + d1
        if x == elem_out + d1:
            if elem_out + 2 * d1 in s_set:
                affected[d1] = affected.get(d1, 0) - 1
        
        # elem_out is the last: (elem_out - 2*d1, elem_out - d1, elem_out)
        # x = elem_out - d1
        if x == elem_out - d1:
            if elem_out - 2 * d1 in s_set:
                affected[d1] = affected.get(d1, 0) - 1
    
    # Add elem_in: for each x in s_set (x != elem_out):
    for x in s_set:
        if x == elem_out:
            continue
        d1 = abs(x - elem_in)
        if d1 == 0:
            continue
        
        # elem_in is the middle: (elem_in - d1, elem_in, elem_in + d1)
        if x == elem_in - d1:
            if elem_in + d1 in s_set:
                affected[d1] = affected.get(d1, 0) + 1
        elif x == elem_in + d1:
            if elem_in - d1 in s_set:
                affected[d1] = affected.get(d1, 0) + 1
        
        # elem_in is the first: (elem_in, elem_in + d1, elem_in + 2*d1)
        if x == elem_in + d1:
            if elem_in + 2 * d1 in s_set:
                affected[d1] = affected.get(d1, 0) + 1
        
        # elem_in is the last: (elem_in - 2*d1, elem_in - d1, elem_in)
        if x == elem_in - d1:
            if elem_in - 2 * d1 in s_set:
                affected[d1] = affected.get(d1, 0) + 1
    
    # Compute delta in score (number of d with count >= 1)
    delta = 0
    for d, chg in affected.items():
        old_count = wc.get(d, 0)
        new_count = old_count + chg
        if old_count == 0 and new_count > 0:
            delta += 1
        elif old_count > 0 and new_count == 0:
            delta -= 1
    
    # Build new wc
    new_wc = dict(wc)
    for d, chg in affected.items():
        new_count = new_wc.get(d, 0) + chg
        if new_count <= 0:
            new_wc.pop(d, None)
        else:
            new_wc[d] = new_count
    
    return delta, new_wc

def _generate_initial_set():
    """Generate a good initial set using multiple strategies."""
    strategies = []
    
    # Strategy 1: Pure arithmetic progression
    for d in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 25, 30, 40, 50, 60, 80]:
        max_start = 512 - 31 * d
        if max_start >= 1:
            a = random.randint(1, max_start)
            s = set(a + i * d for i in range(32))
            if all(1 <= x <= 512 for x in s):
                strategies.append(s)
    
    # Strategy 2: Two interleaved APs
    for _ in range(10):
        d1 = random.choice([2, 3, 4, 5, 6, 8, 10, 12, 15, 20])
        d2 = random.choice([2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30])
        s = set()
        a1 = random.randint(1, max(1, 512 - 15 * d1))
        for i in range(16):
            s.add(a1 + i * d1)
        a2 = random.randint(1, max(1, 512 - 15 * d2))
        for i in range(16):
            val = a2 + i * d2
            if val not in s:
                s.add(val)
        while len(s) < 32:
            val = random.randint(1, 512)
            if val not in s:
                s.add(val)
        if len(s) == 32:
            strategies.append(s)
    
    # Strategy 3: Random
    strategies.append(set(random.sample(range(1, 513), 32)))
    
    # Strategy 4: Dense block
    base = random.randint(1, 481)
    s = set(range(base, base + 32))
    if max(s) <= 512:
        strategies.append(s)
    
    # Strategy 5: Multi-scale structure
    for _ in range(5):
        s = set()
        num_blocks = random.randint(2, 4)
        block_size = 32 // num_blocks
        for b in range(num_blocks):
            block_base = random.randint(1, 512 - block_size * 10)
            for k in range(block_size):
                val = block_base + k * random.choice([1, 2, 3])
                if 1 <= val <= 512 and val not in s:
                    s.add(val)
        while len(s) < 32:
            val = random.randint(1, 512)
            if val not in s:
                s.add(val)
        if len(s) == 32:
            strategies.append(s)
    
    # Strategy 6: Quadratic-like structure
    for _ in range(3):
        s = set()
        a = random.randint(1, 3)
        b = random.randint(1, 10)
        c = random.randint(1, 50)
        for i in range(32):
            val = a * i * i + b * i + c
            if 1 <= val <= 512 and val not in s:
                s.add(val)
        while len(s) < 32:
            val = random.randint(1, 512)
            if val not in s:
                s.add(val)
        if len(s) == 32:
            strategies.append(s)
    
    # Strategy 7: Structured random with bias toward small differences
    for _ in range(5):
        s = set()
        center = random.randint(100, 400)
        for _ in range(32):
            offset = int(random.expovariate(1.0 / 30))
            val = center + offset
            if 1 <= val <= 512 and val not in s:
                s.add(val)
            else:
                val = center - offset
                if 1 <= val <= 512 and val not in s:
                    s.add(val)
        while len(s) < 32:
            val = random.randint(1, 512)
            if val not in s:
                s.add(val)
        if len(s) == 32:
            strategies.append(s)
    
    # Strategy 8: Greedy difference coverage
    for _ in range(5):
        s = set()
        s.add(random.randint(1, 512))
        target_diffs = list(range(1, 60))
        random.shuffle(target_diffs)
        
        for d in target_diffs:
            if len(s) >= 32:
                break
            for _ in range(20):
                a = random.randint(1, 512 - 2 * d)
                if a in s and a + d not in s and a + 2 * d not in s:
                    s.add(a + d)
                    if len(s) >= 32:
                        break
                elif a not in s and a + d in s and a + 2 * d not in s:
                    s.add(a)
                    if len(s) >= 32:
                        break
                elif a not in s and a + d not in s and a + 2 * d in s:
                    s.add(a)
                    s.add(a + d)
                    if len(s) >= 32:
                        break
        
        while len(s) < 32:
            val = random.randint(1, 512)
            if val not in s:
                s.add(val)
        if len(s) == 32:
            strategies.append(s)
    
    # Pick the best
    best = None
    best_sc = -1
    for s in strategies:
        if len(s) != 32:
            continue
        sc = _score(s)
        if sc > best_sc:
            best_sc = sc
            best = s
    
    return best if best else set(random.sample(range(1, 513), 32))

def _greedy_construct():
    """
    Greedily build a 32-element set, each time adding the element
    that maximizes the number of new differences witnessed.
    """
    s = set()
    # Start with a random element
    s.add(random.randint(1, 512))
    
    for _ in range(31):
        best_elem = None
        best_gain = -1
        
        # Sample candidates (don't try all 512 for speed)
        candidates = random.sample(range(1, 513), min(100, 512 - len(s)))
        
        for cand in candidates:
            if cand in s:
                continue
            
            # Count how many new differences this would create
            # For each x in s, d = |cand - x|
            # Check if (min, mid, max) forms an AP with all three in s ∪ {cand}
            new_diffs = 0
            for x in s:
                d = abs(cand - x)
                if d == 0:
                    continue
                # cand is middle: (cand-d, cand, cand+d)
                if cand - d in s and cand + d in s:
                    # Check if this d is already witnessed
                    # We need to know if d is already in the current wc
                    pass
                # cand is first: (cand, cand+d, cand+2d)
                if cand + d in s and cand + 2 * d in s:
                    pass
                # cand is last: (cand-2d, cand-d, cand)
                if cand - 2 * d in s and cand - d in s:
                    pass
            
            # Simpler: just count how many APs this creates
            count = 0
            for x in s:
                d = abs(cand - x)
                if d == 0:
                    continue
                # Check all three positions
                # cand as middle
                if cand - d in s and cand + d in s:
                    count += 1
                # cand as first
                if cand + d in s and cand + 2 * d in s:
                    count += 1
                # cand as last
                if cand - 2 * d in s and cand - d in s:
                    count += 1
            
            if count > best_gain:
                best_gain = count
                best_elem = cand
        
        if best_elem is not None:
            s.add(best_elem)
        else:
            # Fallback: random
            while True:
                val = random.randint(1, 512)
                if val not in s:
                    s.add(val)
                    break
    
    return s

def _sa_search(initial_set, deadline):
    """
    Simulated annealing with fast delta-based scoring.
    Returns the best set found.
    """
    current_set = set(initial_set)
    current_wc = _witness_counts(current_set)
    current_score = len(current_wc)
    
    best_set = set(current_set)
    best_score = current_score
    
    T = 30.0
    cooling = 0.997
    min_T = 0.05
    stuck_count = 0
    max_stuck = 200
    
    while time.time() < deadline - 5 and T > min_T:
        elem_out = random.choice(list(current_set))
        
        # Try multiple candidates and pick the best
        best_move = None
        best_move_delta = -999
        best_move_wc = None
        
        num_candidates = 8
        for _ in range(num_candidates):
            elem_in = random.randint(1, 512)
            if elem_in in current_set:
                continue
            
            delta, new_wc = _delta_score(current_set, current_wc, elem_out, elem_in)
            
            if delta > best_move_delta:
                best_move_delta = delta
                best_move = (elem_out, elem_in)
                best_move_wc = new_wc
        
        if best_move is None:
            stuck_count += 1
            if stuck_count > max_stuck:
                break
            continue
        
        elem_out, elem_in = best_move
        delta = best_move_delta
        new_score = current_score + delta
        
        if delta > 0:
            current_set = set(current_set)
            current_set.discard(elem_out)
            current_set.add(elem_in)
            current_wc = best_move_wc
            current_score = new_score
            stuck_count = 0
            
            if new_score > best_score:
                best_score = new_score
                best_set = set(current_set)
        else:
            # Accept with probability exp(delta/T)
            if delta < -10:
                stuck_count += 1
            elif random.random() < np.exp(delta / T):
                current_set = set(current_set)
                current_set.discard(elem_out)
                current_set.add(elem_in)
                current_wc = best_move_wc
                current_score = new_score
                stuck_count = 0
            else:
                stuck_count += 1
        
        T *= cooling
        if T < min_T:
            T = min_T
        
        if stuck_count > max_stuck:
            # Perturbation: make several random moves
            for _ in range(3):
                e_out = random.choice(list(current_set))
                e_in = random.randint(1, 512)
                if e_in not in current_set:
                    current_set = set(current_set)
                    current_set.discard(e_out)
                    current_set.add(e_in)
                    current_wc = _witness_counts(current_set)
                    current_score = len(current_wc)
            stuck_count = 0
            T = max(T, 15.0)
    
    return best_set, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill
    
    best_set = None
    best_score = 0
    
    # Phase 1: Multiple restarts with SA
    num_restarts = 12
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Generate initial set
        current_set = _generate_initial_set()
        current_score = _score(current_set)
        
        if current_score > best_score:
            best_score = current_score
            best_set = set(current_set)
        
        # Also try greedy construction
        if time.time() < deadline - 10:
            greedy_set = _greedy_construct()
            greedy_score = _score(greedy_set)
            if greedy_score > best_score:
                best_score = greedy_score
                best_set = set(greedy_set)
        
        # Simulated Annealing
        sa_best, sa_score = _sa_search(current_set, deadline)
        if sa_score > best_score:
            best_score = sa_score
            best_set = set(sa_best)
        
        # Also run SA on greedy if it was better
        if time.time() < deadline - 10:
            greedy_set2 = _greedy_construct()
            greedy_score2 = _score(greedy_set2)
            if greedy_score2 > best_score:
                best_score = greedy_score2
                best_set = set(greedy_set2)
            sa_best2, sa_score2 = _sa_search(greedy_set2, deadline)
            if sa_score2 > best_score:
                best_score = sa_score2
                best_set = set(sa_best2)
    
    # Final local search on the best solution
    if best_set is not None and time.time() < deadline - 2:
        current_set = set(best_set)
        current_wc = _witness_counts(current_set)
        current_score = len(current_wc)
        
        T3 = 1.0
        cooling3 = 0.99
        min_T3 = 0.01
        stuck3 = 0
        max_stuck3 = 100
        
        while time.time() < deadline - 0.5 and T3 > min_T3:
            elem_out = random.choice(list(current_set))
            
            best_move3 = None
            best_move3_delta = -999
            best_move3_wc = None
            
            for _ in range(10):
                elem_in = random.randint(1, 512)
                if elem_in in current_set:
                    continue
                
                delta, new_wc = _delta_score(current_set, current_wc, elem_out, elem_in)
                
                if delta > best_move3_delta:
                    best_move3_delta = delta
                    best_move3 = (elem_out, elem_in)
                    best_move3_wc = new_wc
            
            if best_move3 is None:
                stuck3 += 1
                if stuck3 > max_stuck3:
                    break
                continue
            
            elem_out, elem_in = best_move3
            delta = best_move3_delta
            new_score = current_score + delta
            
            if delta > 0:
                current_set = set(current_set)
                current_set.discard(elem_out)
                current_set.add(elem_in)
                current_wc = best_move3_wc
                current_score = new_score
                stuck3 = 0
                
                if new_score > best_score:
                    best_score = new_score
                    best_set = set(current_set)
            else:
                if random.random() < np.exp(delta / T3):
                    current_set = set(current_set)
                    current_set.discard(elem_out)
                    current_set.add(elem_in)
                    current_wc = best_move3_wc
                    current_score = new_score
                    stuck3 = 0
                else:
                    stuck3 += 1
            
            T3 *= cooling3
            if T3 < min_T3:
                T3 = min_T3
            
            if stuck3 > max_stuck3:
                break
    
    return list(best_set) if best_set else list(range(1, 33))

# EVOLVE-BLOCK-END
