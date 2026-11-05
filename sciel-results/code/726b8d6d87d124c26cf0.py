import time
import random
import numpy as np
from itertools import combinations

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

def _score_with_diffs(s):
    """Return (score, diffs_set) for set s."""
    diffs = _compute_diffs(s)
    return len(diffs), diffs

def _fast_neighbor_score(current_set, remove_elem, add_elem, current_diffs):
    """
    Efficiently compute the new score after removing remove_elem and adding add_elem.
    Returns (new_score, new_diffs_set).
    """
    s_set = set(current_set)
    s_set.discard(remove_elem)
    s_set.add(add_elem)
    
    if len(s_set) != 32:
        return -1, set()
    
    # Compute new diffs
    new_diffs = _compute_diffs(s_set)
    return len(new_diffs), new_diffs

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
    for _ in range(15):
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
    for _ in range(8):
        s = set()
        num_blocks = random.randint(2, 4)
        block_size = 32 // num_blocks
        for b in range(num_blocks):
            block_base = random.randint(1, 512 - block_size * 10)
            step = random.choice([1, 2, 3])
            for k in range(block_size):
                val = block_base + k * step
                if 1 <= val <= 512 and val not in s:
                    s.add(val)
        while len(s) < 32:
            val = random.randint(1, 512)
            if val not in s:
                s.add(val)
        if len(s) == 32:
            strategies.append(s)
    
    # Strategy 6: Quadratic-like structure
    for _ in range(5):
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
    
    # Strategy 7: Difference-targeted construction
    # Try to maximize small differences
    for _ in range(10):
        s = set()
        # Start with a base point
        base = random.randint(1, 50)
        s.add(base)
        # Greedily add points that create new small differences
        for target_d in range(1, 30):
            # Try to find a pair in s that can form a 3-AP with difference target_d
            found = False
            for a in list(s):
                if a + target_d <= 512 and a + 2*target_d <= 512:
                    if a + target_d not in s and a + 2*target_d not in s:
                        # Check if adding these creates new diff
                        new_s = s | {a + target_d, a + 2*target_d}
                        if len(new_s) <= 32:
                            s = new_s
                            found = True
                            break
                    elif a + target_d in s and a + 2*target_d <= 512 and a + 2*target_d not in s:
                        if len(s) < 32:
                            s.add(a + 2*target_d)
                            found = True
                            break
            if found:
                continue
            # If can't extend, try random
            if len(s) < 32:
                val = random.randint(1, 512)
                if val not in s:
                    s.add(val)
        # Fill remaining
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

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    best_set = None
    best_score = 0
    
    # Phase 1: Multiple restarts with simulated annealing
    num_restarts = 8
    
    for restart in range(num_restarts):
        if time.time() > deadline - 30:
            break
        
        # Generate initial set
        current_set = _generate_initial_set()
        current_score, current_diffs = _score_with_diffs(current_set)
        
        if current_score > best_score:
            best_score = current_score
            best_set = set(current_set)
        
        # Simulated Annealing
        T = 25.0
        cooling = 0.993
        min_T = 0.05
        stuck_count = 0
        max_stuck = 800
        
        while time.time() < deadline - 10:
            # Generate a neighbor
            elem_out = random.choice(list(current_set))
            
            # Try multiple candidates to find best move
            best_move = None
            best_move_score = -1
            best_move_diffs = None
            
            # Try 5 random candidates
            for _ in range(5):
                elem_in = random.randint(1, 512)
                if elem_in in current_set:
                    continue
                
                new_score, new_diffs = _fast_neighbor_score(current_set, elem_out, elem_in, current_diffs)
                
                if new_score > best_move_score:
                    best_move_score = new_score
                    best_move = (elem_out, elem_in, new_score, new_diffs)
            
            if best_move is None:
                stuck_count += 1
                if stuck_count > max_stuck:
                    break
                continue
            
            elem_out, elem_in, new_score, new_diffs = best_move
            delta = new_score - current_score
            
            if delta > 0:
                current_set = set(current_set)
                current_set.discard(elem_out)
                current_set.add(elem_in)
                current_score = new_score
                current_diffs = new_diffs
                stuck_count = 0
                
                if new_score > best_score:
                    best_score = new_score
                    best_set = set(current_set)
            else:
                # Accept with probability exp(delta/T)
                if delta < -8:
                    stuck_count += 1
                elif random.random() < np.exp(delta / T):
                    current_set = set(current_set)
                    current_set.discard(elem_out)
                    current_set.add(elem_in)
                    current_score = new_score
                    current_diffs = new_diffs
                    stuck_count = 0
                else:
                    stuck_count += 1
            
            T *= cooling
            if T < min_T:
                T = min_T
            
            if stuck_count > max_stuck:
                # Perturbation: random restart from current
                for _ in range(3):
                    e_out = random.choice(list(current_set))
                    e_in = random.randint(1, 512)
                    if e_in not in current_set:
                        current_set = set(current_set)
                        current_set.discard(e_out)
                        current_set.add(e_in)
                current_score, current_diffs = _score_with_diffs(current_set)
                stuck_count = 0
                T = max(T, 15.0)
    
    # Phase 2: Focused local search on best solution (if time permits)
    if best_set is not None and time.time() < deadline - 5:
        current_set = set(best_set)
        current_score, current_diffs = _score_with_diffs(current_set)
        
        T = 3.0
        cooling = 0.985
        min_T = 0.02
        stuck_count = 0
        max_stuck = 1000
        
        while time.time() < deadline - 1:
            elem_out = random.choice(list(current_set))
            
            # Try multiple candidates
            best_move = None
            best_move_score = -1
            best_move_diffs = None
            
            for _ in range(6):
                elem_in = random.randint(1, 512)
                if elem_in in current_set:
                    continue
                
                new_score, new_diffs = _fast_neighbor_score(current_set, elem_out, elem_in, current_diffs)
                
                if new_score > best_move_score:
                    best_move_score = new_score
                    best_move = (elem_out, elem_in, new_score, new_diffs)
            
            if best_move is None:
                stuck_count += 1
                if stuck_count > max_stuck:
                    break
                continue
            
            elem_out, elem_in, new_score, new_diffs = best_move
            delta = new_score - current_score
            
            if delta > 0:
                current_set = set(current_set)
                current_set.discard(elem_out)
                current_set.add(elem_in)
                current_score = new_score
                current_diffs = new_diffs
                stuck_count = 0
                
                if new_score > best_score:
                    best_score = new_score
                    best_set = set(current_set)
            else:
                if random.random() < np.exp(delta / T):
                    current_set = set(current_set)
                    current_set.discard(elem_out)
                    current_set.add(elem_in)
                    current_score = new_score
                    current_diffs = new_diffs
                    stuck_count = 0
                else:
                    stuck_count += 1
            
            T *= cooling
            if T < min_T:
                T = min_T
            
            if stuck_count > max_stuck:
                break
    
    return list(best_set) if best_set else list(range(1, 33))

# EVOLVE-BLOCK-END
