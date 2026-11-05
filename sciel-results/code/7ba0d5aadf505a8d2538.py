import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _compute_diffs(s):
    """Compute the set of all d > 0 such that a, a+d, a+2d are all in s."""
    s_set = set(s)
    diffs = set()
    elements = sorted(s_set)
    n = len(elements)
    for i in range(n):
        a = elements[i]
        for j in range(i + 1, n):
            b = elements[j]
            d = b - a
            if a + 2 * d in s_set:
                diffs.add(d)
    return diffs

def _score(s):
    """Score = number of distinct d > 0 with a 3-term AP."""
    return len(_compute_diffs(s))

def _generate_initial_set():
    """Generate a good initial set using multiple strategies."""
    strategies = []
    
    # Strategy 1: Pure arithmetic progression
    for d in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 25, 30, 40, 50, 60, 80, 100]:
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
    
    # Strategy 7: Greedy difference coverage
    # Try to build a set that covers many small differences
    for _ in range(5):
        s = set()
        # Start with a random base
        s.add(random.randint(1, 512))
        target_diffs = list(range(1, 60))
        random.shuffle(target_diffs)
        
        for d in target_diffs:
            if len(s) >= 32:
                break
            # Try to add elements that create this difference
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
    
    # Strategy 8: Structured random with bias toward small differences
    for _ in range(5):
        s = set()
        # Pick a center and spread elements
        center = random.randint(100, 400)
        for _ in range(32):
            # Bias toward smaller offsets
            offset = int(random.expovariate(1.0 / 30))
            val = center + offset
            if 1 <= val <= 512 and val not in s:
                s.add(val)
            else:
                # Try the other side
                val = center - offset
                if 1 <= val <= 512 and val not in s:
                    s.add(val)
        
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

def _fast_neighbor_score(current_set, elem_out, elem_in):
    """
    Compute the change in score when removing elem_out and adding elem_in.
    Returns the new score.
    """
    new_set = set(current_set)
    new_set.discard(elem_out)
    new_set.add(elem_in)
    return _score(new_set), new_set

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    best_set = None
    best_score = 0
    
    # Phase 1: Multiple restarts with simulated annealing
    num_restarts = 8
    
    for restart in range(num_restarts):
        if time.time() > deadline - 15:
            break
        
        # Generate initial set
        current_set = _generate_initial_set()
        current_score = _score(current_set)
        
        if current_score > best_score:
            best_score = current_score
            best_set = set(current_set)
        
        # Simulated Annealing
        T = 30.0
        cooling = 0.997
        min_T = 0.05
        stuck_count = 0
        max_stuck = 300
        iterations = 0
        
        while time.time() < deadline - 10 and T > min_T:
            iterations += 1
            
            # Generate a neighbor
            elem_out = random.choice(list(current_set))
            
            # Try multiple candidates and pick the best
            best_move = None
            best_move_score = -1
            best_move_set = None
            
            num_candidates = 5
            for _ in range(num_candidates):
                elem_in = random.randint(1, 512)
                if elem_in in current_set:
                    continue
                
                new_score, new_set = _fast_neighbor_score(current_set, elem_out, elem_in)
                
                if new_score > best_move_score:
                    best_move_score = new_score
                    best_move = (elem_out, elem_in)
                    best_move_set = new_set
            
            if best_move is None:
                stuck_count += 1
                if stuck_count > max_stuck:
                    break
                continue
            
            elem_out, elem_in = best_move
            new_score = best_move_score
            new_set = best_move_set
            delta = new_score - current_score
            
            if delta > 0:
                current_set = new_set
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
                    current_set = new_set
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
                        current_score = _score(current_set)
                stuck_count = 0
                T = max(T, 15.0)
        
        # Local search refinement after SA
        if time.time() < deadline - 5:
            T2 = 2.0
            cooling2 = 0.995
            min_T2 = 0.01
            stuck2 = 0
            max_stuck2 = 200
            
            while time.time() < deadline - 2 and T2 > min_T2:
                elem_out = random.choice(list(current_set))
                
                best_move2 = None
                best_move2_score = -1
                best_move2_set = None
                
                for _ in range(5):
                    elem_in = random.randint(1, 512)
                    if elem_in in current_set:
                        continue
                    
                    new_score, new_set = _fast_neighbor_score(current_set, elem_out, elem_in)
                    
                    if new_score > best_move2_score:
                        best_move2_score = new_score
                        best_move2 = (elem_out, elem_in)
                        best_move2_set = new_set
                
                if best_move2 is None:
                    stuck2 += 1
                    if stuck2 > max_stuck2:
                        break
                    continue
                
                new_score = best_move2_score
                new_set = best_move2_set
                delta = new_score - current_score
                
                if delta > 0:
                    current_set = new_set
                    current_score = new_score
                    stuck2 = 0
                    
                    if new_score > best_score:
                        best_score = new_score
                        best_set = set(current_set)
                else:
                    if random.random() < np.exp(delta / T2):
                        current_set = new_set
                        current_score = new_score
                        stuck2 = 0
                    else:
                        stuck2 += 1
                
                T2 *= cooling2
                if T2 < min_T2:
                    T2 = min_T2
                
                if stuck2 > max_stuck2:
                    break
    
    # Final local search on the best solution
    if best_set is not None and time.time() < deadline - 1:
        current_set = set(best_set)
        current_score = _score(current_set)
        
        T3 = 1.0
        cooling3 = 0.99
        min_T3 = 0.01
        stuck3 = 0
        max_stuck3 = 100
        
        while time.time() < deadline - 0.5 and T3 > min_T3:
            elem_out = random.choice(list(current_set))
            
            best_move3 = None
            best_move3_score = -1
            best_move3_set = None
            
            for _ in range(8):
                elem_in = random.randint(1, 512)
                if elem_in in current_set:
                    continue
                
                new_score, new_set = _fast_neighbor_score(current_set, elem_out, elem_in)
                
                if new_score > best_move3_score:
                    best_move3_score = new_score
                    best_move3 = (elem_out, elem_in)
                    best_move3_set = new_set
            
            if best_move3 is None:
                stuck3 += 1
                if stuck3 > max_stuck3:
                    break
                continue
            
            new_score = best_move3_score
            new_set = best_move3_set
            delta = new_score - current_score
            
            if delta > 0:
                current_set = new_set
                current_score = new_score
                stuck3 = 0
                
                if new_score > best_score:
                    best_score = new_score
                    best_set = set(current_set)
            else:
                if random.random() < np.exp(delta / T3):
                    current_set = new_set
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
