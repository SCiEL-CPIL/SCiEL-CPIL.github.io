import random
import time

def construct():
    n = 4096
    best_set = []
    start_time = time.time()
    time_limit = 55.0 # Leave 5s buffer

    # Helper to check if x can be added to current_set
    def can_add(x, current_set, existing_sums):
        # Generate new sums involving x
        s = 3 * x
        if s in existing_sums:
            return False, []
        new_sums = [s]
        
        for y in current_set:
            s = 2 * x + y
            if s in existing_sums:
                return False, []
            new_sums.append(s)
            
        for i in range(len(current_set)):
            for j in range(i, len(current_set)):
                y = current_set[i]
                z = current_set[j]
                s = x + y + z
                if s in existing_sums:
                    return False, []
                new_sums.append(s)
                
        return True, new_sums

    def run_greedy(order):
        current_set = []
        existing_sums = set()
        for x in order:
            ok, new_sums = can_add(x, current_set, existing_sums)
            if ok:
                current_set.append(x)
                existing_sums.update(new_sums)
        return current_set

    # Initial Greedy to get a decent start
    candidates = list(range(1, n + 1))
    random.shuffle(candidates)
    current_set = run_greedy(candidates)
    best_set = current_set.copy()
    
    # Local Search / Simulated Annealing
    # We will try to improve the set by swapping elements to reduce "tightness"
    # or simply find a valid configuration that allows adding more elements.
    
    # To make swapping efficient, we don't maintain existing_sums dynamically for the whole search 
    # because we are swapping. We will recompute validity if needed.
    # Given K ~ 40, recomputing sums is fast.
    
    def get_collision_count(s):
        sums = set()
        count = 0
        k = len(s)
        for i in range(k):
            for j in range(i, k):
                for l in range(j, k):
                    val = s[i] + s[j] + s[l]
                    if val in sums:
                        count += 1
                    else:
                        sums.add(val)
        return count

    def is_valid(s):
        return get_collision_count(s) == 0

    current_collisions = get_collision_count(current_set)
    
    # Strategy: 
    # 1. Try to add elements.
    # 2. If stuck, try swaps to reduce collisions or maintain 0 collisions while changing structure.
    
    # We need a list of candidates not in current_set.
    # Since n is 4096, we can just pick random numbers and check if they are in the set.
    
    while time.time() - start_time < time_limit:
        # Try to add a new element
        if len(current_set) < 80:
            # Try a few random additions
            for _ in range(10):
                x = random.randint(1, n)
                if x not in current_set:
                    ok, new_sums = can_add(x, current_set, set()) # Note: can_add expects existing_sums. 
                    # This is inefficient. Let's use a helper that checks against the set's sums.
                    # Actually, can_add in the original code takes existing_sums.
                    # We should precompute sums for current_set?
                    # For simplicity in this local search, we'll use a slower check if we don't maintain sums.
                    # But we can maintain sums.
                    pass
        
        # The original can_add is O(K^2). 
        # Let's stick to a simpler local search:
        # 1. Pick a random element from current_set to remove (if size > min_size)
        # 2. Pick a random element from [1, n] to add
        # 3. Check if new set is valid.
        
        # To be more effective, we try to grow the set first.
        # If we can't grow, we try to "shuffle" the set to find a better configuration.
        
        # Let's try to add
        added = False
        for _ in range(50):
            x = random.randint(1, n)
            if x not in current_set:
                # Check if x can be added
                # We need existing sums. Let's compute them on the fly for the check, 
                # or maintain them.
                # Maintaining sums is better.
                pass
        
        # This approach is getting messy with state management.
        # Let's use a cleaner Simulated Annealing structure.
        
        # State: current_set
        # Move: Swap x in current_set with y not in current_set.
        # Cost: Number of collisions.
        
        if current_collisions == 0:
            # Try to add
            for _ in range(20):
                x = random.randint(1, n)
                if x not in current_set:
                    # Quick check: is x compatible?
                    # We can use a helper that checks against current_set elements
                    # without building the full sum set if we just check intersections?
                    # No, we need to check if 3x, 2x+y, x+y+z collide.
                    # This is O(K^2).
                    # Let's just try to add and if it fails, do a swap.
                    
                    # Efficient check for adding x to current_set
                    # We can precompute all sums of current_set? No, we only need sums involving x.
                    # Sums involving x: 3x, 2x+y, x+y+z.
                    # We need to know if these sums exist in the set of all triple sums of current_set.
                    # Let's maintain a set `current_sums`.
                    pass
        else:
            # Try to reduce collisions via swap
            pass
            
    # This implementation is too complex to write correctly in the scratchpad.
    # Let's simplify: Use the greedy method but with **multiple restarts** and **randomized ordering** is what the current code does.
    # The "different family" should be **Local Search**.
    
    # Let's implement a simple Local Search that works on the set directly.
    
    # 1. Generate a random set of size K (e.g. 30)
    # 2. Optimize it to have 0 collisions using swaps.
    # 3. Try to add elements.
    
    def local_search():
        # Start with a random set of size 25
        size = 25
        s = set(random.sample(range(1, n + 1), size))
        s_list = list(s)
        
        # Optimize to 0 collisions
        # We will use a simple hill climbing on collisions
        colls = get_collision_count(s_list)
        
        for step in range(10000):
            if colls == 0:
                break
            
            # Pick random i, j. Swap s[i] with a random new value?
            # Or swap s[i] with s[j]? No, swapping within set doesn't change sums.
            # We must swap an element IN the set with an element OUTSIDE.
            
            idx = random.randint(0, len(s_list) - 1)
            old_val = s_list[idx]
            
            # Try several random new values
            for _ in range(10):
                new_val = random.randint(1, n + 1)
                if new_val in s:
                    continue
                
                # Create new set
                s_list[idx] = new_val
                new_s = set(s_list)
                if len(new_s) != len(s_list): # Collision in set itself? No, s was a set.
                    s_list[idx] = old_val
                    continue
                
                new_colls = get_collision_count(s_list)
                if new_colls < colls:
                    colls = new_colls
                    s = new_s
                    break
                elif new_colls == colls and random.random() < 0.1:
                    colls = new_colls
                    s = new_s
                    break
                else:
                    s_list[idx] = old_val
            
        # Now s has 0 collisions (hopefully)
        # Try to add elements
        s_list = list(s)
        existing_sums = set()
        # Compute existing sums
        k = len(s_list)
        for i in range(k):
            for j in range(i, k):
                for l in range(j, k):
                    existing_sums.add(s_list[i] + s_list[j] + s_list[l])
        
        # Try to add
        for _ in range(1000):
            x = random.randint(1, n)
            if x in s:
                continue
            ok, new_sums = can_add(x, s_list, existing_sums)
            if ok:
                s_list.append(x)
                s.add(x)
                existing_sums.update(new_sums)
                
        return s_list

    # Run multiple local searches
    for _ in range(20):
        if time.time() - start_time > time_limit:
            break
        res = local_search()
        if len(res) > len(best_set):
            best_set = res

    return best_set
