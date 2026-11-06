"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from math import gcd

def construct():
    deadline = time.time() + 55
    target_num = 5
    target_den = 31
    
    best = [8, 31, 248]
    best_max = 248
    
    # Dijkstra-style search: state = (max_den_so_far, last_den, num, den, path)
    # We want to minimize max_den_so_far.
    # Start with remainder = 5/31, last_den = 0 (no denominators used yet)
    # Priority: (max_den_so_far, num/den as float for tie-breaking, last_den)
    
    # Initial state: remainder 5/31, no denominators used
    # max_den_so_far = 0 (will be set when we pick first denominator)
    # We'll track (max_den, last_den, num, den, path_tuple)
    
    # For efficiency, we use a visited set to avoid re-exploring states
    # State key: (num, den, last_den) - but this might be too large
    # Actually, since we're doing Dijkstra on max_den, once we pop a state with
    # max_den >= best_max, we can stop.
    
    # Let's use a different approach: BFS/DFS with pruning, but ordered by potential
    # Actually, let's do a proper Dijkstra where the "distance" is the max denominator.
    
    # State: (num, den, last_den) where num/den is the remaining fraction
    # Edge: choose next denominator k > last_den, new remainder = num/den - 1/k
    # Cost: max(current_max, k)
    
    # Since we want to minimize the maximum, we can use a priority queue keyed by
    # the current maximum denominator.
    
    # visited: set of (num, den, last_den) that we've already processed with a
    # given or better max_den. Since Dijkstra processes in order of max_den,
    # once we pop a state, it's optimal.
    
    # But the state space could be large. Let's bound it:
    # - num/den must be positive
    # - k > last_den
    # - k < best_max (to improve)
    # - num/den - 1/k must be positive (or zero if it's the last term)
    
    # Actually, for the last term, num/den - 1/k = 0 means 1/k = num/den,
    # so num must be 1 and k = den.
    
    # Let me reconsider. The state is (num, den, last_den) and we track the
    # path. The priority is the max denominator in the path so far.
    
    # To keep memory manageable, I'll use a dictionary to track the best
    # (min max_den) seen for each (num, den, last_den) state.
    
    # Actually, let's just do a careful BFS with the priority queue.
    
    # Initial: remainder = 5/31, last_den = 0, max_den = 0, path = ()
    # We reduce fractions to canonical form.
    
    # Priority queue entry: (max_den, last_den, num, den, path)
    # We use last_den as secondary key to ensure deterministic ordering
    
    start_num, start_den = target_num, target_den
    g = gcd(start_num, start_den)
    start_num //= g
    start_den //= g
    
    # pq: (max_den, last_den, num, den, path)
    pq = [(0, 0, start_num, start_den, ())]
    
    # best_max_for_state: (num, den, last_den) -> min max_den seen
    # This helps prune: if we've seen this state with a lower or equal max_den, skip
    best_state = {}
    
    steps = 0
    while pq:
        steps += 1
        if steps % 10000 == 0 and time.time() > deadline:
            break
        
        max_den, last_den, num, den, path = heapq.heappop(pq)
        
        # If this state was already processed with a better (lower) max_den, skip
        state_key = (num, den, last_den)
        if state_key in best_state and best_state[state_key] <= max_den:
            continue
        best_state[state_key] = max_den
        
        # If max_den >= best_max, no point continuing (we need strictly less)
        if max_den >= best_max:
            break
        
        # Determine range for next denominator k
        # k > last_den
        # num/den - 1/k > 0 (unless this is the last term, handled below)
        # So 1/k < num/den, i.e., k > den/num
        # Also, we need at least one more term, so num/den - 1/k > 0
        # Unless num/den is exactly 1/k for some k (last term)
        
        # Check if current remainder is a unit fraction: num == 1
        if num == 1:
            # Last term: k = den
            k = den
            if k > last_den and k < best_max:
                new_max = max(max_den, k)
                if new_max < best_max:
                    best_max = new_max
                    best = list(path) + [k]
            continue
        
        # k must satisfy: k > last_den and k > den/num
        # Also k < best_max (to potentially improve)
        # And num/den - 1/k > 0, so k > den/num
        # Lower bound: k_min = max(last_den + 1, floor(den/num) + 1)
        # Upper bound: k_max = best_max - 1 (to improve)
        # Also, we need num/den - 1/k to be representable, which it always is.
        
        k_min = max(last_den + 1, den // num + 1)
        # If den % num == 0, then den/num is integer, and we need k > den/num, so k >= den/num + 1
        # If den % num != 0, then den/num is not integer, and k > den/num means k >= floor(den/num) + 1
        # So k_min = max(last_den + 1, den // num + 1) works in both cases? 
        # Let's check: if den/num = 3.5, floor = 3, k > 3.5 means k >= 4 = 3+1. Yes.
        # If den/num = 3.0, floor = 3, k > 3.0 means k >= 4 = 3+1. Yes.
        
        k_max = best_max - 1
        
        if k_min > k_max:
            continue
        
        # To avoid too many iterations, cap the range
        # But we need to explore all valid k in [k_min, k_max]
        
        for k in range(k_min, k_max + 1):
            if time.time() > deadline:
                return best
            
            # New remainder: num/den - 1/k = (num*k - den) / (den*k)
            new_num = num * k - den
            new_den = den * k
            
            if new_num <= 0:
                # This shouldn't happen since k > den/num, but guard
                continue
            
            # Reduce fraction
            g = gcd(new_num, new_den)
            new_num_r = new_num // g
            new_den_r = new_den // g
            
            new_max = max(max_den, k)
            
            if new_max >= best_max:
                continue
            
            # Check if this state was already seen with a better max_den
            state_key2 = (new_num_r, new_den_r, k)
            if state_key2 in best_state and best_state[state_key2] <= new_max:
                continue
            
            heapq.heappush(pq, (new_max, k, new_num_r, new_den_r, path + (k,)))
        
        # Periodic check
        if steps % 5000 == 0 and time.time() > deadline:
            break
    
    return best
# EVOLVE-BLOCK-END
