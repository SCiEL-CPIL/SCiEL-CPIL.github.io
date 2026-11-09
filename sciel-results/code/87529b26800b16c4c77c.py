import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    best_size = 0
    best_set = []
    
    # Helper functions defined inside to access local variables if needed, 
    # but passing args is cleaner.
    
    def compute_initial_counts(S):
        counts = {}
        for i, x in enumerate(S):
            for y in S[i:]:
                s = x + y
                counts[s] = counts.get(s, 0) + 1
        return counts

    def can_add(x, S_list, S_set, counts):
        # Check sums with existing elements
        for y in S_list:
            s = x + y
            if counts.get(s, 0) >= 2:
                return False
        s = 2 * x
        if counts.get(s, 0) >= 2:
            return False
        return True

    def add_element(x, S_list, S_set, counts):
        S_list.append(x)
        S_set.add(x)
        for y in S_list[:-1]: # y is all previous elements
            s = x + y
            counts[s] = counts.get(s, 0) + 1
        s = 2 * x
        counts[s] = counts.get(s, 0) + 1

    def remove_element(x, S_list, S_set, counts):
        S_list.remove(x)
        S_set.remove(x)
        for y in S_list: # y is all remaining elements
            s = x + y
            counts[s] -= 1
            if counts[s] == 0:
                del counts[s]
        s = 2 * x
        counts[s] -= 1
        if counts[s] == 0:
            del counts[s]

    def randomized_greedy():
        candidates = list(range(1, 1001))
        random.shuffle(candidates)
        S_list = []
        S_set = set()
        counts = {}
        
        for x in candidates:
            if can_add(x, S_list, S_set, counts):
                add_element(x, S_list, S_set, counts)
        return S_list, S_set, counts

    def local_improve(S_list, S_set, counts):
        # Try to add elements
        while time.time() < deadline:
            added = False
            # Get candidates not in S
            # Instead of filtering 1000 elements every time, we can just iterate 1..1000
            # But we need to randomize order.
            # Creating a list of 1000 elements and shuffling is fast.
            candidates = [x for x in range(1, 1001) if x not in S_set]
            if not candidates:
                break
            
            random.shuffle(candidates)
            
            for x in candidates:
                if can_add(x, S_list, S_set, counts):
                    add_element(x, S_list, S_set, counts)
                    added = True
                    break # Restart loop to re-evaluate with new set
            
            if not added:
                # Try a swap
                if not S_list:
                    break
                
                x_to_remove = random.choice(S_list)
                remove_element(x_to_remove, S_list, S_set, counts)
                
                # Try to add a different element
                candidates = [x for x in range(1, 1001) if x not in S_set]
                random.shuffle(candidates)
                
                swapped = False
                for x in candidates:
                    if can_add(x, S_list, S_set, counts):
                        add_element(x, S_list, S_set, counts)
                        swapped = True
                        break
                
                if not swapped:
                    # Revert
                    add_element(x_to_remove, S_list, S_set, counts)
                    # If we can't add anything and can't swap to something else, 
                    # we are in a local optimum. 
                    # To escape, we might need a larger kick, but for now, stop.
                    break
                else:
                    # We swapped. Size is same. 
                    # Continue loop to see if we can add more now.
                    # We don't mark 'added' as true because size didn't increase, 
                    # but the state changed. The while loop will continue.
                    pass
        
        return S_list, S_set, counts

    # Main loop
    while time.time() < deadline:
        S_list, S_set, counts = randomized_greedy()
        S_list, S_set, counts = local_improve(S_list, S_set, counts)
        
        if len(S_list) > best_size:
            best_size = len(S_list)
            best_set = S_list[:]
            
    return best_set
