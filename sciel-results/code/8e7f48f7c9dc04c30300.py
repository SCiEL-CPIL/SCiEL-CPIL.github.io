import time
from fractions import Fraction

def construct():
    N = 256
    # Precompute reciprocals as Fractions for exactness in final result
    # and as floats for fast bound checking
    recips_frac = [Fraction(1, i) for i in range(1, N + 1)]
    recips_float = [1.0 / i for i in range(1, N + 1)]
    
    # Suffix sums of float reciprocals for upper bounding
    # suffix_float[i] = sum(recips_float[i:] for i in 1..N)
    # We index 1-based in logic, so let's make arrays 0-based for Python but map carefully.
    # Let's just use a list where index i corresponds to value i+1? 
    # Easier: recips_float[0] is 1/1.
    # suffix[i] = sum(recips_float[i:])
    suffix_float = [0.0] * (N + 1)
    for i in range(N - 1, -1, -1):
        suffix_float[i] = suffix_float[i + 1] + recips_float[i]

    # Global variables for B&B
    best_set = []
    best_sum_frac = Fraction(0, 1)
    best_sum_float = 0.0
    start_time = time.time()
    time_limit = 85.0  # Leave 5s margin for 90s limit
    nodes_visited = 0
    max_nodes = 10000000  # Hard cap on nodes to prevent infinite loop if time check fails? No, time check is primary.

    # Helper to check if x can be added to current_set (list)
    # We will maintain a set for O(1) lookups
    def can_add(x, current_set_list, current_set_set):
        # Check conflicts with all a in current_set_set
        for a in current_set_set:
            # Case 1: a, x, b -> b = 2x - a
            b = 2 * x - a
            if b in current_set_set:
                return False
            # Case 2: x, a, b -> b = 2a - x
            b = 2 * a - x
            if b in current_set_set:
                return False
            # Case 3: a, b, x -> b = (a+x)/2
            s = a + x
            if s % 2 == 0:
                b = s // 2
                if b in current_set_set:
                    return False
        return True

    # Initial Greedy Solution to set a good lower bound
    greedy_set = []
    greedy_set_set = set()
    for x in range(1, N + 1):
        if can_add(x, greedy_set, greedy_set_set):
            greedy_set.append(x)
            greedy_set_set.add(x)
    
    if greedy_set:
        best_set = greedy_set.copy()
        best_sum_float = sum(recips_float[x-1] for x in greedy_set)
        best_sum_frac = sum(recips_frac[x-1] for x in greedy_set)

    # Recursive B&B
    def dfs(idx, current_set_list, current_set_set, current_sum_float, current_sum_frac):
        nonlocal best_set, best_sum_frac, best_sum_float, nodes_visited
        
        # Time check
        if time.time() - start_time > time_limit:
            return

        nodes_visited += 1

        # Upper bound check
        # The maximum we can add is the sum of all remaining reciprocals
        if idx > N:
            # Leaf node
            if current_sum_frac > best_sum_frac:
                best_sum_frac = current_sum_frac
                best_sum_float = current_sum_float
                best_set = current_set_list.copy()
            return

        # Prune if current sum + max possible remaining <= best
        # Add a small epsilon to account for float precision errors in bound
        if current_sum_float + suffix_float[idx] + 1e-9 <= best_sum_float:
            return

        x = idx + 1 # 1-based index
        
        # Option 1: Include x
        if can_add(x, current_set_list, current_set_set):
            new_list = current_set_list + [x]
            new_set = current_set_set | {x}
            dfs(idx + 1, new_list, new_set, current_sum_float + recips_float[idx], current_sum_frac + recips_frac[idx])
            
            # If we've hit the time limit, stop exploring further
            if time.time() - start_time > time_limit:
                return

        # Option 2: Exclude x
        dfs(idx + 1, current_set_list, current_set_set, current_sum_float, current_sum_frac)

    # Start search from index 0 (value 1)
    dfs(0, [], set(), 0.0, Fraction(0, 1))

    return best_set

