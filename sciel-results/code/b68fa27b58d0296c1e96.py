import time
import random
from math import gcd
from functools import lru_cache

# EVOLVE-BLOCK-START

@lru_cache(maxsize=None)
def get_lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    # 1. Initial Greedy Construction (Fast baseline)
    # The seed program uses a greedy approach. We replicate a fast version of it.
    # To speed up the check, we can maintain a set of "forbidden" triples or just check against existing pairs.
    # The check is: for new x, for all pairs (a,b) in S, lcm(a,b) != lcm(a,x) and lcm(a,b) != lcm(b,x).
    # Actually the condition is: lcm(a,b) == lcm(a,x) == lcm(b,x).
    # This is equivalent to: lcm(a,b) == lcm(a,x) AND lcm(a,b) == lcm(b,x).
    
    # Let's just use the seed's logic but vectorized or optimized? 
    # The seed is O(N^3) roughly? 
    # N=128. Pairs ~ 8000. For each x, check ~8000 pairs. 128 * 8000 = 1M operations. 
    # Python math.lcm is fast enough? 
    # Let's stick to a simple greedy first to get a good starting point.
    
    def greedy_construct(order):
        selected = []
        for x in order:
            valid = True
            # Check against all pairs in selected
            # Optimization: If we store selected as a list, iterating pairs is O(k^2) for k elements.
            # Total O(N * K^2) where K is final size.
            for i in range(len(selected)):
                if not valid: break
                a = selected[i]
                for j in range(i+1, len(selected)):
                    b = selected[j]
                    l_ab = get_lcm(a, b)
                    if l_ab == get_lcm(a, x) and l_ab == get_lcm(b, x):
                        valid = False
                        break
            if valid:
                selected.append(x)
        return selected

    # Standard greedy 1..128
    initial_set = greedy_construct(range(1, 129))
    current_size = len(initial_set)
    best_set = list(initial_set)
    
    # 2. Simulated Annealing / Local Search
    # State: A set of integers.
    # Moves:
    #   - Add a random element not in set (if valid)
    #   - Remove a random element from set
    #   - Swap a random element from set with a random element not in set (if valid)
    
    # To make moves fast, we need to quickly check validity of adding/removing.
    # Pre-compute LCMs for all pairs in [1, 128]? 128*128/2 = 8128 pairs. 
    # Storing in a dict or 2D array is feasible.
    
    max_val = 128
    lcm_table = [[0] * (max_val + 1) for _ in range(max_val + 1)]
    for i in range(1, max_val + 1):
        for j in range(i, max_val + 1):
            lcm_table[i][j] = get_lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]

    # Helper to check if a set is valid (for verification, not used in inner loop)
    def is_valid_set(s):
        s_list = list(s)
        for i in range(len(s_list)):
            for j in range(i+1, len(s_list)):
                a, b = s_list[i], s_list[j]
                l_ab = lcm_table[a][b]
                for k in range(j+1, len(s_list)):
                    c = s_list[k]
                    if l_ab == lcm_table[a][c] and l_ab == lcm_table[b][c]:
                        return False
        return True

    # Current state as a boolean array for O(1) membership and fast iteration
    # But we need to iterate over elements in the set for conflict checks.
    # Let's maintain a list `current` and a set `current_set`.
    
    current = list(initial_set)
    current_set = set(current)
    not_in_set = set(range(1, 129)) - current_set
    
    def try_add(x):
        """Returns True if x can be added to current without creating a bad triple."""
        # For any pair (a, b) in current, check if lcm(a,b) == lcm(a,x) == lcm(b,x)
        for i in range(len(current)):
            a = current[i]
            for j in range(i+1, len(current)):
                b = current[j]
                l_ab = lcm_table[a][b]
                if l_ab == lcm_table[a][x] and l_ab == lcm_table[b][x]:
                    return False
        return True

    def try_remove(x):
        """Removing always results in a valid set if current is valid."""
        return True

    def try_swap(x_in, x_out):
        """Check if replacing x_out with x_in is valid."""
        # 1. Remove x_out, check if x_in is valid with remaining
        # We can simulate by checking pairs in (current - {x_out}) against x_in
        for i in range(len(current)):
            if current[i] == x_out: continue
            a = current[i]
            for j in range(len(current)):
                if j <= i: continue
                if current[j] == x_out: continue
                b = current[j]
                l_ab = lcm_table[a][b]
                if l_ab == lcm_table[a][x_in] and l_ab == lcm_table[b][x_in]:
                    return False
        return True

    # Annealing parameters
    temperature = 1.0
    cooling_rate = 0.995
    min_temperature = 0.01
    step_count = 0
    
    while time.time() < deadline and temperature > min_temperature:
        # Random move
        move_type = random.random()
        
        if move_type < 0.4 and not_in_set:
            # Add
            x = random.choice(list(not_in_set))
            if try_add(x):
                # Accept add (increases size, always good)
                current.append(x)
                current_set.add(x)
                not_in_set.remove(x)
                # Update best
                if len(current) > current_size:
                    current_size = len(current)
                    best_set = list(current)
            else:
                # Reject
                pass
        elif move_type < 0.6 and current:
            # Remove
            x = random.choice(current)
            # Always accept removal? In SA, we might reject to keep size up, 
            # but here we want to maximize size. Removing is a "bad" move unless it enables future adds.
            # Standard SA: accept if delta < 0 or with prob exp(-delta/T).
            # Here delta = -1 (size decreases). 
            # We accept with probability exp(1/T) ? No, exp(-(-1)/T) = exp(1/T) which is > 1. 
            # So we always accept removals in this heuristic? 
            # Better: only remove if it helps escape a local optimum. 
            # Let's just accept removals with some probability to explore.
            if random.random() < 0.5:
                current.remove(x)
                current_set.remove(x)
                not_in_set.add(x)
        else:
            # Swap
            if current and not_in_set:
                x_out = random.choice(current)
                x_in = random.choice(list(not_in_set))
                if try_swap(x_in, x_out):
                    # Accept swap (size unchanged, but might enable future adds)
                    # Replace x_out with x_in
                    idx = current.index(x_out)
                    current[idx] = x_in
                    current_set.remove(x_out)
                    current_set.add(x_in)
                    not_in_set.remove(x_in)
                    not_in_set.add(x_out)
                else:
                    pass
        
        temperature *= cooling_rate
        step_count += 1
        
        # If we have reached a very high size, we might want to stop earlier or keep polishing
        # Max possible size is 128.
        if current_size == 128:
            break

    # Final check and return
    # Ensure best_set is valid (it should be, as we only accepted valid moves)
    # But just in case of bugs, verify.
    if not is_valid_set(best_set):
        # Fallback to greedy
        best_set = greedy_construct(range(1, 129))
        
    return best_set

# EVOLVE-BLOCK-END
