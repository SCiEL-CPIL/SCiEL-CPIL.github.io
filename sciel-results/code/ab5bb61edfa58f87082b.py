import time
import random
import numpy as np

# EVOLVE-BLOCK-START
_MAX = 512
_N = 32
_DEADLINE = 115.0  # Safety margin before 150s kill

def _score_bitmask(mask):
    """
    Count distinct d > 0 such that a, a+d, a+2d are in the set represented by mask.
    mask is a Python integer where bit i is 1 if i is in the set.
    """
    count = 0
    # Max d such that 2*d + 1 <= 512 is 255.
    # If a=1, a+2d <= 512 => 2d <= 511 => d <= 255.
    for d in range(1, 256):
        if (mask & (mask << d) & (mask << (2 * d))) != 0:
            count += 1
    return count

def _greedy_init(rng):
    """
    Build a set greedily by adding the element that maximizes the score gain.
    Returns a set to maintain type consistency for SA.
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
    Perform local search by trying all single and 2-swaps.
    Elements can be a list or set.
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
                
    return sorted(current_elements)

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    best_elements = None
    best_score = -1
    
    # Run multiple restarts
    num_restarts = 10
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        rng = random.Random(42 + restart)
        
        # 1. Greedy Initialization (returns a set now)
        current_elements = _greedy_init(rng)
        current_mask = 0
        for x in current_elements:
            current_mask |= (1 << x)
        current_score = _score_bitmask(current_mask)
        
        if current_score > best_score:
            best_score = current_score
            best_elements = list(current_elements)
            
        # 2. Simulated Annealing
        # current_elements is a set. We need to sample from it.
        time_per_restart = (_DEADLINE - (time.time() - start_time)) / (num_restarts - restart)
        sa_start = time.time()
        sa_deadline = sa_start + min(10.0, time_per_restart * 0.8) # Cap SA time per restart
        
        T = 1.0
        cooling = 0.9995
        min_T = 0.001
        
        # Precompute list of all possible elements for fast sampling
        all_elements = list(range(1, _MAX + 1))
        
        while time.time() < sa_deadline:
            if T < min_T:
                T = min_T
                
            # Decide move type: 1-swap or 2-swap
            if rng.random() < 0.3:
                # 2-swap
                # Pick 2 to remove from the set
                # Convert to list for sampling, as sets are not indexable
                rem1, rem2 = rng.sample(list(current_elements), 2)
                
                # Pick 2 to add
                while True:
                    add_indices = rng.sample(all_elements, 2)
                    add1, add2 = add_indices
                    if add1 not in current_elements and add2 not in current_elements and add1 != add2:
                        break
                        
                # Apply
                new_mask = current_mask
                new_mask ^= (1 << rem1)
                new_mask ^= (1 << rem2)
                new_mask |= (1 << add1)
                new_mask |= (1 << add2)
                
                new_score = _score_bitmask(new_mask)
                delta = new_score - current_score
                
                if delta > 0 or rng.random() < np.exp(delta / T):
                    # Accept
                    current_elements.remove(rem1)
                    current_elements.remove(rem2)
                    current_elements.add(add1)
                    current_elements.add(add2)
                    current_mask = new_mask
                    current_score = new_score
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_elements = list(current_elements)
            else:
                # 1-swap
                rem = rng.choice(list(current_elements))
                while True:
                    add = rng.choice(all_elements)
                    if add not in current_elements:
                        break
                        
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
                        best_elements = list(current_elements)
                        
            T *= cooling

        # 3. Local Search Polish
        if time.time() < deadline:
            # best_elements is a list, _local_search accepts list/set
            polished_elements = _local_search(best_elements, rng, time_limit=min(2.0, deadline - time.time()))
            polished_mask = 0
            for x in polished_elements:
                polished_mask |= (1 << x)
            polished_score = _score_bitmask(polished_mask)
            
            if polished_score > best_score:
                best_score = polished_score
                best_elements = polished_elements

    if best_elements is None:
        # Fallback
        best_elements = list(range(1, _N + 1))
        
    return sorted(best_elements)
# EVOLVE-BLOCK-END
