import time
import random
import math

def construct():
    N = 1000
    max_val = 1000
    max_sum = 2 * max_val
    deadline = time.time() + 110  # Safety margin for 120s limit

    # Data structures
    S = []  # List of chosen integers
    in_S = [False] * (max_val + 1)  # in_S[i] is True if i is in S
    # counts[s] = number of pairs (a,b) in S with a<=b and a+b=s
    counts = [0] * (max_sum + 1)

    # Helper to add a number to the set and update counts
    # Returns True if successful, False if it would violate B2[2]
    def try_add(x):
        # Check 2*x
        if counts[2 * x] >= 2:
            return False
        # Check x + y for y in S
        for y in S:
            if y == x:
                continue
            s = x + y
            if counts[s] >= 2:
                return False
        # If we get here, it's valid.
        # Perform addition
        S.append(x)
        in_S[x] = True
        counts[2 * x] += 1
        for y in S:
            if y != x:
                counts[x + y] += 1
        return True

    # Helper to remove a number
    def try_remove(x):
        S.remove(x)
        in_S[x] = False
        counts[2 * x] -= 1
        for y in S:
            if y != x:
                counts[x + y] -= 1

    # Helper to check if a swap (remove a, add c) is valid
    # Does not modify state, just checks
    def is_valid_swap(a, c):
        # We need to check if any resulting count > 2
        # Affected sums are:
        # - 2*a
        # - a + y for y in S, y != a
        # - 2*c
        # - c + y for y in S, y != a
        
        deltas = {}
        
        # Remove a
        deltas[2*a] = deltas.get(2*a, 0) - 1
        for y in S:
            if y != a:
                s = a + y
                deltas[s] = deltas.get(s, 0) - 1
        
        # Add c
        deltas[2*c] = deltas.get(2*c, 0) + 1
        for y in S:
            if y != a:
                s = c + y
                deltas[s] = deltas.get(s, 0) + 1
        
        # Check all affected sums
        for s, d in deltas.items():
            new_count = counts[s] + d
            if new_count > 2:
                return False
            if new_count < 0:
                return False
        
        return True

    # 1. Probabilistic Greedy Initialization
    # Instead of just taking the first valid number, we score candidates
    # based on how many "underutilized" sums they touch.
    # This encourages a more balanced distribution of representations.
    
    # We will process numbers in increasing order, but for each number,
    # we decide whether to add it based on a score.
    # However, since the set grows, the "score" changes.
    # A simpler approach: Randomized greedy.
    # Shuffle 1..N. For each x, if valid, add with probability p.
    # Or: Sort by "potential" dynamically? That's expensive.
    
    # Let's try a hybrid: 
    # 1. Standard greedy to get a baseline.
    # 2. Then, use the local search to improve.
    
    # Actually, the "inspiration" suggests selecting x with probability proportional to underutilized sums.
    # Let's implement a single pass where we evaluate all remaining candidates and pick the best one.
    # This is O(N^2 * |S|) which might be too slow for N=1000.
    # 1000 * 1000 * 300 = 300M ops. Too slow.
    
    # Alternative: Batched probabilistic selection.
    # Let's stick to the standard greedy for initialization as it's fast and robust,
    # but maybe add some randomness to break ties or skip some to allow better packing?
    # No, standard greedy is a good seed. The local search is the key.
    
    for i in range(1, max_val + 1):
        try_add(i)

    # 2. Local Search with Simulated Annealing
    # We allow "bad" moves (conflicts) to be accepted with a probability,
    # and we track the best conflict-free solution.
    
    # To allow conflicts, we need to track the number of conflicts.
    # A conflict is a sum s with counts[s] > 2.
    # Conflict count C = sum(max(0, counts[s] - 2))
    
    # We will maintain C incrementally.
    
    # Recalculate C for current state (should be 0)
    def calc_conflicts():
        c = 0
        for s in range(2, max_sum + 1):
            if counts[s] > 2:
                c += counts[s] - 2
        return c

    current_conflicts = calc_conflicts()
    best_conflicts = current_conflicts
    best_set = list(S)
    best_size = len(S)

    T = 1.0
    cooling = 0.99995
    min_T = 0.01

    while time.time() < deadline:
        if T < min_T:
            # If temperature is too low, restart with higher T to escape local minima?
            # Or just break? Let's keep going but with very low acceptance of bad moves.
            pass

        # Generate move
        # 60% Add, 40% Swap
        if random.random() < 0.6:
            # Try Add
            x = random.randint(1, max_val)
            if in_S[x]:
                continue
            
            # Calculate delta conflicts
            # Adding x increases counts for 2x and x+y
            # If a sum goes from 2 to 3, delta_conf += 1
            # If a sum goes from 3 to 4, delta_conf += 1 (and so on)
            
            delta_conf = 0
            if counts[2 * x] >= 2:
                delta_conf += 1
            for y in S:
                if y == x:
                    continue
                s = x + y
                if counts[s] >= 2:
                    delta_conf += 1
            
            # Acceptance
            if delta_conf == 0:
                # Always accept
                try_add(x)
                current_conflicts += delta_conf
                if current_conflicts < best_conflicts:
                    best_conflicts = current_conflicts
                    if best_conflicts == 0:
                        best_set = list(S)
                        if len(S) > best_size:
                            best_size = len(S)
            else:
                # Accept with prob exp(-delta_conf / T)
                if random.random() < math.exp(-delta_conf / T):
                    try_add(x)
                    current_conflicts += delta_conf
                    if current_conflicts < best_conflicts:
                        best_conflicts = current_conflicts
                        if best_conflicts == 0:
                            best_set = list(S)
                            if len(S) > best_size:
                                best_size = len(S)
            
        else:
            # Try Swap
            if not S:
                continue
            x = random.choice(S)
            y = random.randint(1, max_val)
            if in_S[y]:
                continue
            
            # Calculate delta conflicts for swap (remove x, add y)
            # This is trickier. We need to simulate the counts change.
            # Let's use the is_valid_swap logic but extended to count deltas.
            
            # Sums affected:
            # Removed: 2x, x+z (z in S, z!=x)
            # Added: 2y, y+z (z in S, z!=x)
            
            # We can calculate the new conflict count for affected sums.
            # Let's build a dictionary of delta counts for affected sums.
            
            deltas = {}
            
            # Remove x
            deltas[2*x] = deltas.get(2*x, 0) - 1
            for z in S:
                if z != x:
                    s = x + z
                    deltas[s] = deltas.get(s, 0) - 1
            
            # Add y
            deltas[2*y] = deltas.get(2*y, 0) + 1
            for z in S:
                if z != x:
                    s = y + z
                    deltas[s] = deltas.get(s, 0) + 1
            
            # Calculate delta_conf
            delta_conf = 0
            for s, d in deltas.items():
                old_c = counts[s]
                new_c = old_c + d
                old_conf = max(0, old_c - 2)
                new_conf = max(0, new_c - 2)
                delta_conf += new_conf - old_conf
            
            # Acceptance
            if delta_conf == 0:
                # Always accept
                try_remove(x)
                try_add(y)
                current_conflicts += delta_conf
                if current_conflicts < best_conflicts:
                    best_conflicts = current_conflicts
                    if best_conflicts == 0:
                        best_set = list(S)
                        if len(S) > best_size:
                            best_size = len(S)
            else:
                # Accept with prob exp(-delta_conf / T)
                if random.random() < math.exp(-delta_conf / T):
                    try_remove(x)
                    try_add(y)
                    current_conflicts += delta_conf
                    if current_conflicts < best_conflicts:
                        best_conflicts = current_conflicts
                        if best_conflicts == 0:
                            best_set = list(S)
                            if len(S) > best_size:
                                best_size = len(S)
            
        # Cooling
        T *= cooling

    # Return the best valid set found
    return best_set
