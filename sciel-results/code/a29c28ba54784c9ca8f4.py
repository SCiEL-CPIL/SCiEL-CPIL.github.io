import time
import random
import numpy as np

# EVOLVE-BLOCK-START
_MAX = 512
_N = 32
_DEADLINE = 118.0  # Safety margin before 150s kill

def _score_bitmask(mask):
    """
    Count distinct d > 0 such that a, a+d, a+2d are in the set represented by mask.
    mask is a Python integer where bit i is 1 if i is in the set.
    """
    count = 0
    for d in range(1, 256):
        if (mask & (mask << d) & (mask << (2 * d))) != 0:
            count += 1
    return count

def _greedy_init(rng):
    """
    Build a set greedily by adding the element that maximizes the score gain.
    Returns a set.
    """
    current_elements = set()
    current_mask = 0
    current_score = 0
    
    for _ in range(_N):
        best_gain = -1
        best_elem = None
        for x in range(1, _MAX + 1):
            if x in current_elements:
                continue
            new_mask = current_mask | (1 << x)
            new_score = _score_bitmask(new_mask)
            gain = new_score - current_score
            if gain > best_gain or (gain == best_gain and rng.random() < 0.5):
                best_gain = gain
                best_elem = x
        
        if best_elem is not None:
            current_elements.add(best_elem)
            current_mask |= (1 << best_elem)
            current_score += best_gain
            
    return current_elements

def _local_search(elements, rng, time_limit=2.0):
    """
    Perform local search by trying 1-swaps and a limited number of random 2-swaps.
    elements: iterable of ints
    Returns a sorted list.
    """
    current_elements = set(elements)
    current_mask = 0
    for x in current_elements:
        current_mask |= (1 << x)
    current_score = _score_bitmask(current_mask)
    
    start_time = time.time()
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        
        # Try 1-swaps
        for rem in list(current_elements):
            for add in range(1, _MAX + 1):
                if add in current_elements:
                    continue
                new_mask = (current_mask ^ (1 << rem)) | (1 << add)
                new_score = _score_bitmask(new_mask)
                if new_score > current_score:
                    current_elements.remove(rem)
                    current_elements.add(add)
                    current_mask = new_mask
                    current_score = new_score
                    improved = True
                    break
            if improved:
                break
        
        if improved or time.time() - start_time >= time_limit:
            continue
            
        # Try a few random 2-swaps if no 1-swap improvement
        for _ in range(100):
            if time.time() - start_time >= time_limit:
                break
            rem_list = list(current_elements)
            rem1, rem2 = rng.sample(rem_list, 2)
            add1 = rng.randint(1, _MAX)
            add2 = rng.randint(1, _MAX)
            if add1 in current_elements or add2 in current_elements or add1 == add2:
                continue
            
            new_mask = current_mask
            new_mask ^= (1 << rem1)
            new_mask ^= (1 << rem2)
            new_mask |= (1 << add1)
            new_mask |= (1 << add2)
            
            new_score = _score_bitmask(new_mask)
            if new_score > current_score:
                current_elements.remove(rem1)
                current_elements.remove(rem2)
                current_elements.add(add1)
                current_elements.add(add2)
                current_mask = new_mask
                current_score = new_score
                improved = True
                break
                
    return sorted(current_elements)

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    best_elements = None
    best_score = -1
    
    num_restarts = 15
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        rng = random.Random(42 + restart)
        
        # 1. Greedy Initialization
        current_elements = _greedy_init(rng)
        current_mask = 0
        for x in current_elements:
            current_mask |= (1 << x)
        current_score = _score_bitmask(current_mask)
        
        if current_score > best_score:
            best_score = current_score
            best_elements = set(current_elements)
            
        # 2. Simulated Annealing
        remaining_total = _DEADLINE - (time.time() - start_time)
        time_per_restart = remaining_total / max(1, (num_restarts - restart))
        sa_start = time.time()
        sa_deadline = sa_start + min(6.0, time_per_restart * 0.8)
        
        T = 1.0
        cooling = 0.999
        min_T = 0.01
        all_elements = list(range(1, _MAX + 1))
        
        while time.time() < sa_deadline:
            if T < min_T:
                T = min_T
                
            if rng.random() < 0.3:
                # 2-swap
                rem_list = list(current_elements)
                rem1, rem2 = rng.sample(rem_list, 2)
                add1 = rng.choice(all_elements)
                add2 = rng.choice(all_elements)
                if add1 in current_elements or add2 in current_elements or add1 == add2:
                    continue
                    
                new_mask = current_mask
                new_mask ^= (1 << rem1)
                new_mask ^= (1 << rem2)
                new_mask |= (1 << add1)
                new_mask |= (1 << add2)
                
                new_score = _score_bitmask(new_mask)
                delta = new_score - current_score
                
                if delta > 0 or rng.random() < np.exp(delta / T):
                    current_elements.remove(rem1)
                    current_elements.remove(rem2)
                    current_elements.add(add1)
                    current_elements.add(add2)
                    current_mask = new_mask
                    current_score = new_score
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_elements = set(current_elements)
            else:
                # 1-swap
                rem_list = list(current_elements)
                rem = rng.choice(rem_list)
                add = rng.choice(all_elements)
                if add in current_elements:
                    continue
                    
                new_mask = (current_mask ^ (1 << rem)) | (1 << add)
                new_score = _score_bitmask(new_mask)
                delta = new_score - current_score
                
                if delta > 0 or rng.random() < np.exp(delta / T):
                    current_elements.remove(rem)
                    current_elements.add(add)
                    current_mask = new_mask
                    current_score = new_score
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_elements = set(current_elements)
                        
            T *= cooling

        # 3. Local Search Polish
        if time.time() < deadline - 2.0:
            remaining_time = deadline - time.time()
            polish_time = min(2.0, remaining_time * 0.2)
            if polish_time > 0.1:
                polished_elements = _local_search(best_elements, rng, time_limit=polish_time)
                polished_mask = 0
                for x in polished_elements:
                    polished_mask |= (1 << x)
                polished_score = _score_bitmask(polished_mask)
                
                if polished_score > best_score:
                    best_score = polished_score
                    best_elements = set(polished_elements)

    # 4. Final intensive local search
    if best_elements is not None and time.time() < deadline - 1.0:
        remaining_time = deadline - time.time() - 1.0
        if remaining_time > 0.5:
            final_rng = random.Random(999)
            final_elements = _local_search(best_elements, final_rng, time_limit=remaining_time)
            final_mask = 0
            for x in final_elements:
                final_mask |= (1 << x)
            final_score = _score_bitmask(final_mask)
            
            if final_score > best_score:
                best_score = final_score
                best_elements = set(final_elements)

    if best_elements is None:
        best_elements = set(range(1, _N + 1))
        
    return sorted(best_elements)
# EVOLVE-BLOCK-END
