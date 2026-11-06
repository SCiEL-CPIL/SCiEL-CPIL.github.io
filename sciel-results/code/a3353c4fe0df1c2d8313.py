import time
from math import gcd

# Precomputation
N = 128
lcm_table = [[0] * (N + 1) for _ in range(N + 1)]
for i in range(1, N + 1):
    for j in range(1, N + 1):
        lcm_table[i][j] = i * j // gcd(i, j)

# conflicts[a][b] is a set of x such that {a, b, x} is a bad triple
# Note: a, b, x must be distinct.
conflicts = [[set() for _ in range(N + 1)] for _ in range(N + 1)]

# For faster lookup: for each x, store the list of (a, b) pairs such that x is in conflicts[a][b]
# i.e., adding x is invalid iff any (a,b) in bad_pairs[x] are both in the selected set
bad_pairs = [set() for _ in range(N + 1)]
for a in range(1, N + 1):
    for b in range(a + 1, N + 1):
        L_ab = lcm_table[a][b]
        for x in range(1, N + 1):
            if x == a or x == b:
                continue
            if lcm_table[a][x] == L_ab and lcm_table[b][x] == L_ab:
                conflicts[a][b].add(x)
                conflicts[b][a].add(x)
                bad_pairs[x].add((a, b))

def is_valid_addition(x, selected_set):
    # Check if adding x to selected_set creates a bad triple
    # A bad triple involves x and two elements a, b from selected_set
    # such that x is in conflicts[a][b]
    for (a, b) in bad_pairs[x]:
        if a in selected_set and b in selected_set:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    best_size = 0
    best_set = set()
    
    # Multi-start greedy with different orderings
    # We'll run several random restarts
    # And one increasing order start
    
    orders = []
    base = list(range(1, N + 1))
    orders.append(base) # Increasing
    
    # Add some random shuffles
    import random
    random.seed(42)
    for _ in range(20):
        shuffled = base.copy()
        random.shuffle(shuffled)
        orders.append(shuffled)
        
    # Also try sorting by number of divisors (ascending) to pick "simpler" numbers first?
    # Or descending?
    def num_divisors(n):
        count = 0
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count
    
    sorted_by_divs = sorted(base, key=num_divisors)
    orders.append(sorted_by_divs)
    orders.append(sorted(base, key=num_divisors, reverse=True))
    
    for order in orders:
        if time.time() > deadline:
            break
            
        current_set = set()
        for x in order:
            if time.time() > deadline:
                break
            if is_valid_addition(x, current_set):
                current_set.add(x)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set.copy()
            
    # Local Search on the best set found
    if time.time() < deadline:
        current_set = best_set.copy()
        current_size = len(current_set)
        
        # Try to improve by swapping
        # We can try to remove an element and add another, or remove two and add two
        # Since adding is the only way to increase size, we need to find a way to add.
        # If we can't add any single element, we are stuck.
        # We can try to remove a "blocking" element.
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try all possible single adds
            added = False
            for x in range(1, N + 1):
                if x not in current_set:
                    if is_valid_addition(x, current_set):
                        current_set.add(x)
                        current_size += 1
                        added = True
                        improved = True
                        break # Restart loop to check for more adds
            
            if not added:
                # Try 1-remove, 1-add swap? No, size doesn't increase.
                # Try 2-remove, 2-add?
                # This is expensive. 
                # Instead, let's try random restarts of local search if stuck.
                # Or, just accept the local max.
                
                # Let's try a simple hill climbing step:
                # Pick a random element to remove, see if we can add 2 new ones?
                # Or just do random swaps to escape local optima?
                # Since the objective is just size, and we are in a local max for adds,
                # we might be in a global max or a bad local max.
                
                # Let's try removing one element and seeing if it allows adding more than 1.
                # This is complex. 
                
                # Alternative: Run a simulated annealing step?
                # Let's just do a few random restarts of the greedy if we have time.
                pass

    # If we have time, do a more extensive search
    if time.time() < deadline:
        # Try to find a larger set via local search with swaps
        # State: set of elements
        # Move: Remove y, Add x. Valid if new set is valid.
        # This doesn't increase size.
        # To increase size, we need "Add" moves.
        # If no "Add" moves are possible, we must "Remove" to enable "Add".
        
        # Let's try: Remove a random element, then greedily add as many as possible.
        # Compare size.
        
        for _ in range(100): # Limit iterations
            if time.time() > deadline:
                break
                
            trial_set = best_set.copy()
            if not trial_set:
                continue
                
            # Remove a random element
            rem_elem = random.choice(list(trial_set))
            trial_set.remove(rem_elem)
            
            # Greedily add back
            candidates = [x for x in range(1, N + 1) if x not in trial_set]
            random.shuffle(candidates)
            
            for x in candidates:
                if time.time() > deadline:
                    break
                if is_valid_addition(x, trial_set):
                    trial_set.add(x)
            
            if len(trial_set) > best_size:
                best_size = len(trial_set)
                best_set = trial_set.copy()

    return list(best_set)

# Ensure non-empty
if not construct():
    construct = lambda: [1]
