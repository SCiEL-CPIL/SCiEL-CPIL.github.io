import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s margin
    
    max_val = 4096
    max_sum = 3 * max_val
    
    best_set = []
    best_size = 0
    
    def check_add(x, current_set, sum_set):
        """
        Check if x can be added to current_set.
        Returns (is_valid, new_sums_list) if valid, else (False, None).
        """
        new_sums = []
        
        # 3x
        new_sums.append(3 * x)
        
        # 2x + y
        for y in current_set:
            new_sums.append(2 * x + y)
            
        # x + 2y
        for y in current_set:
            new_sums.append(x + 2 * y)
            
        # x + y + z
        n = len(current_set)
        for i in range(n):
            y = current_set[i]
            for j in range(i + 1, n):
                z = current_set[j]
                new_sums.append(x + y + z)
        
        # Check internal collisions
        if len(set(new_sums)) != len(new_sums):
            return False, None
            
        # Check collision with existing sums
        if not set(new_sums).isdisjoint(sum_set):
            return False, None
            
        return True, new_sums

    def get_sums_for_set(S):
        """Compute all triple sums for a set S."""
        sums = set()
        n = len(S)
        # 3x
        for x in S:
            sums.add(3 * x)
        # 2x + y
        for x in S:
            for y in S:
                if x != y: # Wait, unordered triples allow repetition. 
                    # The problem says "unordered triple, allowing repetitions".
                    # So (x,x,x) is 3x. (x,x,y) is 2x+y. (x,y,z) is x+y+z.
                    # My previous logic in check_add handles this.
                    # But here I need to generate ALL sums for initialization.
                    pass
        
        # Let's just use the same logic as check_add but for the whole set.
        # It's easier to just build it from scratch.
        sums = set()
        S_list = list(S)
        n = len(S_list)
        for i in range(n):
            x = S_list[i]
            sums.add(3 * x)
            for j in range(i + 1, n):
                y = S_list[j]
                sums.add(2 * x + y)
                sums.add(x + 2 * y)
                for k in range(j + 1, n):
                    z = S_list[k]
                    sums.add(x + y + z)
        return sums

    # 1. Greedy Initialization
    # Try a few different greedy orders to get a good start
    candidates = list(range(1, max_val + 1))
    
    def try_greedy(order):
        nonlocal best_set, best_size
        if time.time() > deadline:
            return
        
        current_set = []
        sum_set = set()
        
        for x in order:
            if time.time() > deadline:
                break
            
            valid, new_sums = check_add(x, current_set, sum_set)
            if valid:
                current_set.append(x)
                sum_set.update(new_sums)
                
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
    # Run a few greedy passes
    try_greedy(candidates)
    random.shuffle(candidates)
    try_greedy(candidates)
    
    # 2. Simulated Annealing / Local Search
    if best_set:
        current_set = best_set[:]
        current_sum_set = get_sums_for_set(current_set)
        
        # Temperature schedule
        T = 1.0
        T_min = 0.01
        cooling_rate = 0.999
        
        while time.time() < deadline:
            if T < T_min:
                T = T_min
            
            # Choose a move
            move_type = random.random()
            
            if move_type < 0.7:
                # Try to add a random element
                x = random.randint(1, max_val)
                if x in current_set:
                    continue
                    
                valid, new_sums = check_add(x, current_set, current_sum_set)
                if valid:
                    # Size increases by 1
                    current_set.append(x)
                    current_sum_set.update(new_sums)
                    best_size = max(best_size, len(current_set))
                    if len(current_set) > best_size:
                        best_set = current_set[:]
                        best_size = len(current_set)
                    # No temperature update needed for improvement, or reset?
                    # Let's just keep T.
                else:
                    # Try to swap: remove y, add x
                    # Pick a random y to remove
                    if current_set:
                        y = random.choice(current_set)
                        # Check if x can be added to (current_set - {y})
                        # This is expensive. We can approximate or just do it.
                        # To save time, only try swap if adding x failed.
                        
                        # Construct new set without y
                        new_set = [v for v in current_set if v != y]
                        new_sum_set = get_sums_for_set(new_set)
                        
                        valid_swap, swap_sums = check_add(x, new_set, new_sum_set)
                        if valid_swap:
                            # Size remains same
                            # Accept with probability 1? Or based on T?
                            # Since size is same, we accept to explore.
                            # But we must update state.
                            current_set = new_set
                            current_set.append(x)
                            current_sum_set = new_sum_set
                            current_sum_set.update(swap_sums)
                            # Note: best_set is only updated on size increase.
                            
            else:
                # Try to remove an element (to escape local minima)
                if current_set:
                    y = random.choice(current_set)
                    new_set = [v for v in current_set if v != y]
                    # Accept removal with low probability to allow exploration
                    # Or always accept if it allows a subsequent add?
                    # Let's just accept with prob 0.1 * T
                    if random.random() < 0.1 * T:
                        current_set = new_set
                        current_sum_set = get_sums_for_set(current_set)
                        
            # Cool down
            T *= cooling_rate
            
            # Update best
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set[:]

    return best_set
