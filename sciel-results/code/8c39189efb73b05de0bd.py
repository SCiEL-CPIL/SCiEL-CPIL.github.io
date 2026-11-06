"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin for 150s limit

    def is_valid(arr):
        """
        Checks if arr is a Sidon set and covers all differences 1..30.
        Returns True if valid, False otherwise.
        """
        if len(arr) > 32:
            return False
        if 0 not in arr:
            return False
        if max(arr) > 10**6:
            return False

        diffs = set()
        n = len(arr)
        for i in range(n):
            for j in range(i + 1, n):
                d = arr[j] - arr[i]
                if d in diffs:
                    return False
                diffs.add(d)

        # Check coverage 1..30
        for d in range(1, 31):
            if d not in diffs:
                return False
        return True

    def get_diameter(arr):
        return max(arr) - min(arr)

    def generate_greedy_set():
        """
        Generate a valid Sidon set covering 1..30 using a greedy approach.
        Returns a sorted list.
        """
        a = [0]
        diffs = set()
        
        # We need to cover 1..30.
        # Greedy: for each missing d, find a new element x such that adding x
        # creates difference d with some existing element, and doesn't create
        # duplicate differences.
        
        # To be efficient, we prioritize creating small differences first?
        # Or just iterate d=1..30 and try to realize it.
        
        # Let's try a randomized greedy:
        # While not all 1..30 covered:
        #   Pick a random missing d.
        #   Try to add an element x such that d = x - y or d = y - x for some y in a.
        #   Specifically, try x = y + d or x = y - d.
        #   Check if adding x keeps Sidon property.
        
        # To avoid infinite loops or getting stuck, we can also just add random numbers
        # and hope for coverage, but greedy is more directed.
        
        # Let's combine:
        # 1. Start with [0].
        # 2. Loop:
        #    a. Check coverage. If complete, return.
        #    b. Pick a missing difference d (random or smallest).
        #    c. Generate candidates for x: {y + d, y - d for y in a}.
        #    d. Filter candidates: x > 0, x not in a.
        #    e. For each candidate, check if adding it maintains Sidon property.
        #    f. If valid, add it.
        #    g. If no candidate works, try a different strategy (e.g., add random x, or restart).
        
        # This might get stuck. A robust fallback is the "add smallest valid x" greedy
        # which is guaranteed to terminate but might have large diameter.
        
        # Let's implement the directed greedy first.
        
        # To ensure progress, we can shuffle the order of missing d's or candidates.
        
        for _ in range(1000): # Max iterations
            if time.time() > deadline - 5:
                break
            
            missing = [d for d in range(1, 31) if d not in diffs]
            if not missing:
                return sorted(a)
            
            # Pick a random missing difference
            d = random.choice(missing)
            
            candidates = set()
            for y in a:
                x1 = y + d
                x2 = y - d
                if x1 > 0 and x1 not in a:
                    candidates.add(x1)
                if x2 > 0 and x2 not in a:
                    candidates.add(x2)
            
            # Shuffle candidates
            candidates = list(candidates)
            random.shuffle(candidates)
            
            found = False
            for x in candidates:
                # Check if adding x is valid
                new_diffs = set()
                valid = True
                for y in a:
                    dd = abs(x - y)
                    if dd in diffs or dd in new_diffs:
                        valid = False
                        break
                    new_diffs.add(dd)
                
                if valid:
                    a.append(x)
                    diffs.update(new_diffs)
                    found = True
                    break
            
            if not found:
                # If we can't add an element to realize a specific d,
                # let's just add a random element that keeps Sidon property.
                # This might help "open up" space for future differences.
                # Or restart with a different random seed/order.
                # For simplicity, let's try adding the smallest valid x > max(a).
                x = max(a) + 1
                while True:
                    valid = True
                    new_diffs = set()
                    for y in a:
                        dd = x - y
                        if dd in diffs or dd in new_diffs:
                            valid = False
                            break
                        new_diffs.add(dd)
                    if valid:
                        a.append(x)
                        diffs.update(new_diffs)
                        break
                    x += 1
                
                # If we've added too many elements without covering all, restart?
                if len(a) > 20:
                    # Restart with fresh random state
                    a = [0]
                    diffs = set()
                    continue
        
        # Fallback: pure greedy smallest x
        if not all(d in diffs for d in range(1, 31)):
            a = [0]
            diffs = set()
            x = 1
            while not all(d in diffs for d in range(1, 31)):
                valid = True
                new_diffs = set()
                for y in a:
                    dd = x - y
                    if dd in diffs or dd in new_diffs:
                        valid = False
                        break
                    new_diffs.add(dd)
                if valid:
                    a.append(x)
                    diffs.update(new_diffs)
                x += 1
        
        return sorted(a)

    def hill_climb(initial_set):
        """
        Perform simulated annealing to minimize the diameter.
        """
        current = sorted(initial_set)
        if not is_valid(current):
            # Should not happen if generator is good, but just in case
            return current

        best_diam = get_diameter(current)
        best_set = current[:]
        
        T = 50.0 # Temperature
        T_min = 0.1
        cooling = 0.9995
        
        while time.time() < deadline - 10:
            if T < T_min:
                T = T_min
            
            # Generate neighbor
            neighbor = current[:]
            move_type = random.choice(['swap', 'move', 'add_remove'])
            
            if move_type == 'swap':
                # Swap two elements? Or replace one?
                # Replacing one is more common for size preservation.
                idxs = [i for i, v in enumerate(neighbor) if v != 0]
                if idxs:
                    idx = random.choice(idxs)
                    old_val = neighbor[idx]
                    max_val = max(neighbor)
                    # Try values in a range around old_val or within [1, max_val]
                    # Bias towards smaller values to reduce diameter
                    if random.random() < 0.7:
                        new_val = random.randint(1, max_val)
                    else:
                        new_val = random.randint(max_val, max_val + 20)
                    
                    if new_val not in neighbor:
                        neighbor[idx] = new_val
                        neighbor.sort()
                    else:
                        neighbor = current[:]
                        
            elif move_type == 'move':
                # Move an element to a nearby position
                idxs = [i for i, v in enumerate(neighbor) if v != 0]
                if idxs:
                    idx = random.choice(idxs)
                    old_val = neighbor[idx]
                    # Try shifting by small amount
                    shift = random.randint(-10, 10)
                    new_val = old_val + shift
                    if new_val > 0 and new_val not in neighbor:
                        neighbor[idx] = new_val
                        neighbor.sort()
                    else:
                        neighbor = current[:]

            elif move_type == 'add_remove':
                # Try adding or removing an element
                if len(neighbor) < 32 and random.random() < 0.5:
                    # Add
                    max_val = max(neighbor)
                    new_val = random.randint(1, max_val + 10)
                    if new_val not in neighbor:
                        neighbor.append(new_val)
                        neighbor.sort()
                else:
                    # Remove (if size > 1 and 0 is not removed)
                    if len(neighbor) > 1:
                        idxs = [i for i, v in enumerate(neighbor) if v != 0]
                        if idxs:
                            idx = random.choice(idxs)
                            neighbor.pop(idx)
                            neighbor.sort()
            
            # Evaluate
            if not is_valid(neighbor):
                continue
            
            new_diam = get_diameter(neighbor)
            delta = new_diam - best_diam
            
            if delta < 0 or random.random() < pow(2.71828, -delta / T):
                current = neighbor
                if new_diam < best_diam:
                    best_diam = new_diam
                    best_set = neighbor[:]
                    # If we reach a very small diameter, we can stop or cool faster
                    if best_diam <= 31:
                        # Theoretical minimum is likely around 31-40?
                        # 31 elements in 0..30 is impossible for Sidon.
                        # But if we get close, it's good.
                        pass
            
            T *= cooling
            
        return best_set

    # Main Loop
    best_overall = None
    best_overall_diam = float('inf')
    
    # Run multiple restarts
    # The first restart uses a deterministic greedy to ensure we have a valid set.
    # Subsequent restarts use randomized greedy.
    
    # Initial deterministic greedy
    init_set = generate_greedy_set()
    if is_valid(init_set):
        result = hill_climb(init_set)
        if is_valid(result):
            diam = get_diameter(result)
            if diam < best_overall_diam:
                best_overall_diam = diam
                best_overall = result
    
    # Randomized restarts
    while time.time() < deadline - 15:
        init_set = generate_greedy_set()
        if not is_valid(init_set):
            continue
            
        result = hill_climb(init_set)
        
        if is_valid(result):
            diam = get_diameter(result)
            if diam < best_overall_diam:
                best_overall_diam = diam
                best_overall = result
                # Early exit if we find a very good solution
                if best_overall_diam <= 35:
                    break

    if best_overall is None:
        # Fallback: pure greedy smallest x
        a = [0]
        diffs = set()
        x = 1
        while not all(d in diffs for d in range(1, 31)):
            valid = True
            new_diffs = set()
            for y in a:
                dd = x - y
                if dd in diffs or dd in new_diffs:
                    valid = False
                    break
                new_diffs.add(dd)
            if valid:
                a.append(x)
                diffs.update(new_diffs)
            x += 1
        return a

    return best_overall
# EVOLVE-BLOCK-END
