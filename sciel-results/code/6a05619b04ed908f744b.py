import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    mandatory = {1, 2, 5, 11}
    best_size = len(mandatory)
    best_set = list(mandatory)
    
    def is_valid(s):
        """Check if set s is a valid Sidon set."""
        sums = set()
        lst = list(s)
        for i in range(len(lst)):
            for j in range(i, len(lst)):
                s_val = lst[i] + lst[j]
                if s_val in sums:
                    return False
                sums.add(s_val)
        return True

    def try_add(s, x):
        """Check if adding x to s keeps it valid. Returns True if valid."""
        lst = list(s)
        existing_sums = set()
        for i in range(len(lst)):
            for j in range(i, len(lst)):
                existing_sums.add(lst[i] + lst[j])
                
        new_sums = set()
        for y in lst:
            new_sums.add(x + y)
        new_sums.add(2 * x)
        
        if new_sums & existing_sums:
            return False
        return True

    def greedy_fill(s, remaining):
        """Greedily add elements from remaining to s."""
        current_s = set(s)
        rem = list(remaining)
        random.shuffle(rem)
        
        for x in rem:
            if x in current_s:
                continue
            if try_add(current_s, x):
                current_s.add(x)
        return current_s

    def local_search(s, remaining, time_limit):
        """
        Try to increase size of s using swaps.
        """
        current_s = set(s)
        curr_rem = set(remaining)
        
        best_local_s = set(current_s)
        best_local_size = len(current_s)
        
        while time.time() < time_limit:
            improved = False
            
            # 1. Try to add any element
            if curr_rem:
                candidates = list(curr_rem)
                random.shuffle(candidates)
                # Try a few random ones to find an addable element
                for x in candidates[:20]:
                    if try_add(current_s, x):
                        current_s.add(x)
                        curr_rem.remove(x)
                        improved = True
                        if len(current_s) > best_local_size:
                            best_local_size = len(current_s)
                            best_local_s = set(current_s)
                        break # Restart loop after improvement
            
            if improved:
                continue
                
            # 2. If stuck, try a swap to escape local minimum
            non_mand = [x for x in current_s if x not in mandatory]
            if non_mand and curr_rem:
                y = random.choice(non_mand)
                x = random.choice(list(curr_rem))
                
                # Check if swap is valid
                # Instead of full is_valid, we can do a targeted check
                # Remove y, Add x
                # But is_valid is simple and safe.
                new_s = (current_s - {y}) | {x}
                if is_valid(new_s):
                    current_s = new_s
                    curr_rem = curr_rem - {x} | {y}
                    # Continue to try adding again
                    continue
            else:
                # Cannot swap
                break

        return best_local_s

    # Main loop: Multiple restarts
    while time.time() < deadline:
        # 1. Greedy phase
        initial_s = set(mandatory)
        remaining = set(range(1, 285)) - initial_s
        
        s = greedy_fill(initial_s, remaining)
        
        if len(s) > best_size:
            best_size = len(s)
            best_set = list(s)
        
        # 2. Local search phase
        curr_rem = set(range(1, 285)) - s
        
        improved_s = local_search(s, curr_rem, deadline)
        
        if len(improved_s) > best_size:
            best_size = len(improved_s)
            best_set = list(improved_s)
            
        if best_size == 20:
            break

    return best_set
