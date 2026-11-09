import itertools
import math
import time
import numpy as np

def _score_fast(U):
    """Fast score computation using numpy for vectorized set operations."""
    U = np.array(sorted(set(U)))
    if 0 not in U:
        return 0.0
    mx = int(U[-1])
    if mx == 0:
        return 0.0
    
    # Compute all sums and differences using broadcasting
    sums = U[:, None] + U[None, :]
    diffs = U[:, None] - U[None, :]
    
    n_plus = len(np.unique(sums))
    n_minus = len(np.unique(diffs))
    
    if n_plus == 0:
        return 0.0
    ratio = n_minus / n_plus
    if ratio <= 1:
        return 1.0
    return 1.0 + math.log(ratio) / math.log(2 * mx + 1)

def _score(U):
    """Original score function for compatibility."""
    U = list(set(U))
    if 0 not in U:
        return 0.0
    mx = max(U)
    if mx == 0:
        return 0.0
    plus = set()
    minus = set()
    for a in U:
        for b in U:
            plus.add(a + b)
            minus.add(a - b)
    n_plus = len(plus)
    n_minus = len(minus)
    if n_plus == 0:
        return 0.0
    ratio = n_minus / n_plus
    if ratio <= 1:
        return 1.0
    return 1.0 + math.log(ratio) / math.log(2 * mx + 1)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    best_score = 0.0
    best_set = [0, 1, 3]
    
    # Phase 1: Exhaustive search for small sizes
    # We search for sets containing 0.
    # Size n means n-1 other elements.
    # We limit the max value to keep the search space manageable.
    max_val_small = 30
    candidates_small = list(range(1, max_val_small + 1))
    
    for size in range(2, 7):
        if time.time() > deadline - 10:
            break
        for combo in itertools.combinations(candidates_small, size - 1):
            if time.time() > deadline - 10:
                break
            U = [0] + list(combo)
            s = _score_fast(U)
            if s > best_score:
                best_score = s
                best_set = U
    
    # Phase 2: Simulated Annealing Local Search
    # Start from the best set found.
    if time.time() < deadline - 5.0:
        current_set = set(best_set)
        current_score = best_score
        
        # Define a range for new elements to try
        # We can try to extend the set or modify it.
        # To keep max(U) small, we should be careful about adding large numbers.
        # However, adding a large number increases the denominator log(2max+1).
        # It also increases |U-U| and |U+U|.
        # Let's try a range up to 200.
        max_val_large = 200
        candidates_large = list(range(1, max_val_large + 1))
        
        # Simulated Annealing parameters
        # Start with high temperature to allow large jumps, cool down.
        T_start = 0.1
        T_end = 1e-6
        cooling_rate = 0.999
        T = T_start
        
        iterations = 0
        max_iterations = 10000 # Cap iterations, but time is the real limit
        
        while time.time() < deadline - 5.0 and iterations < max_iterations:
            iterations += 1
            
            # Generate a random move
            move_type = np.random.choice(['add', 'remove', 'swap'])
            new_set = set(current_set)
            
            if move_type == 'add':
                # Pick a random element from candidates_large not in current_set
                available = [x for x in candidates_large if x not in current_set]
                if available:
                    x = np.random.choice(available)
                    new_set.add(x)
            elif move_type == 'remove':
                if len(current_set) > 1:
                    removable = [x for x in current_set if x != 0]
                    if removable:
                        x = np.random.choice(removable)
                        new_set.discard(x)
            else: # swap
                if len(current_set) > 1:
                    removable = [x for x in current_set if x != 0]
                    available = [y for y in candidates_large if y not in current_set]
                    if removable and available:
                        x = np.random.choice(removable)
                        y = np.random.choice(available)
                        new_set.discard(x)
                        new_set.add(y)
            
            if new_set == current_set:
                continue
                
            s_new = _score_fast(list(new_set))
            
            # Acceptance criterion
            if s_new > current_score:
                current_set = new_set
                current_score = s_new
                if s_new > best_score:
                    best_score = s_new
                    best_set = list(current_set)
            else:
                # Accept worse move with probability exp(-delta / T)
                delta = current_score - s_new
                if delta < 1e-9:
                    current_set = new_set
                    current_score = s_new
                else:
                    prob = math.exp(-delta / T)
                    if np.random.random() < prob:
                        current_set = new_set
                        current_score = s_new
            
            T *= cooling_rate
            if T < T_end:
                T = T_end
            
    # Phase 3: Polish
    # Perform a strict local search to ensure we are at a local optimum.
    # Iterate until no improvement is found.
    if time.time() < deadline - 2.0:
        current_set = set(best_set)
        current_score = best_score
        max_val_polish = 200
        candidates_polish = list(range(1, max_val_polish + 1))
        
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            
            # Try all adds
            for x in candidates_polish:
                if x in current_set:
                    continue
                new_set = current_set | {x}
                s = _score_fast(list(new_set))
                if s > current_score:
                    current_set = new_set
                    current_score = s
                    improved = True
                    break # Restart loop with new set
                    
            if improved:
                continue
                
            # Try all removes
            removable = [x for x in current_set if x != 0]
            for x in removable:
                new_set = current_set - {x}
                s = _score_fast(list(new_set))
                if s > current_score:
                    current_set = new_set
                    current_score = s
                    improved = True
                    break
                    
            if improved:
                continue
                
            # Try all swaps
            # This is O(N * M) which can be slow if N is large, but sets are small.
            # We limit the number of swaps to check per iteration to avoid timeout.
            # Actually, let's just check a subset if it's too slow, but N is likely < 20.
            # N=20, M=200 -> 4000 checks. Each check ~1ms -> 4s. Might be tight.
            # Let's restrict swap candidates to those close to existing elements or random subset.
            # For simplicity and safety, we'll do a limited swap search.
            
            if time.time() < deadline - 2.0:
                 # Randomized swap polish to save time
                 for _ in range(100):
                     if time.time() > deadline - 2.0:
                         break
                     if len(current_set) <= 1:
                         break
                     removable = [x for x in current_set if x != 0]
                     if not removable:
                         break
                     x = np.random.choice(removable)
                     # Try swapping with a random candidate
                     y = np.random.choice(candidates_polish)
                     if y in current_set:
                         continue
                     new_set = (current_set - {x}) | {y}
                     s = _score_fast(list(new_set))
                     if s > current_score:
                         current_set = new_set
                         current_score = s
                         improved = True
                         break

        if current_score > best_score:
            best_score = current_score
            best_set = list(current_set)

    return best_set
