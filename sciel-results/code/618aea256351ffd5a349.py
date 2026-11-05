import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    S = set()
    counts = {}

    def can_add(x):
        if counts.get(2*x, 0) >= 2:
            return False
        for s in S:
            if counts.get(x + s, 0) >= 2:
                return False
        return True

    def add(x):
        S.add(x)
        counts[2*x] = counts.get(2*x, 0) + 1
        for s in S:
            if s != x:
                counts[x + s] = counts.get(x + s, 0) + 1

    def remove(x):
        S.remove(x)
        counts[2*x] -= 1
        if counts[2*x] == 0:
            del counts[2*x]
        for s in S:
            counts[x + s] -= 1
            if counts[x + s] == 0:
                del counts[x + s]

    # 1. Randomized Greedy Initialization
    # Shuffle to avoid bias towards small numbers
    candidates = list(range(1, 1001))
    random.shuffle(candidates)
    
    for x in candidates:
        if can_add(x):
            add(x)
        if time.time() > deadline:
            break
            
    best_size = len(S)
    current_size = len(S)
    
    # Track candidates outside S
    outside = [x for x in candidates if x not in S]
    # Note: S contains some from candidates. We need to maintain the list of outside elements.
    # It's easier to just iterate 1..1000 and check membership, or maintain a set of outside.
    outside_set = set(candidates)
    outside_set.difference_update(S)
    
    # To speed up "try add", we can keep a list of outside elements.
    # But "can_add" is O(|S|).
    
    # 2. Local Search with Simulated Annealing-like exploration
    # We will try to increase size. If stuck, we swap to change configuration.
    
    no_improve_count = 0
    max_no_improve = 500 # Try 500 swaps before giving up on this configuration?
    # Actually, we have time. Let's just run until deadline.
    
    # To make swaps effective, we need to pick x from S and y from outside.
    # Random selection is okay but might be slow.
    
    while time.time() < deadline:
        # Phase 1: Try to add
        # Pick a random subset of outside to test to save time? 
        # Or just pick one random outside element.
        if outside_set:
            y = random.choice(list(outside_set))
            if can_add(y):
                add(y)
                outside_set.remove(y)
                current_size += 1
                if current_size > best_size:
                    best_size = current_size
                no_improve_count = 0
                continue
        
        # Phase 2: If can't add, try a swap
        # Pick random x in S, y in outside
        if S and outside_set:
            x = random.choice(list(S))
            y = random.choice(list(outside_set))
            
            # Check if swap is valid
            # Temporarily remove x
            remove(x)
            if can_add(y):
                add(y)
                outside_set.remove(y)
                outside_set.add(x)
                # Size unchanged, but configuration changed
                no_improve_count += 1
            else:
                add(x)
                no_improve_count += 1
                
        # Phase 3: If stuck for too long, try a 2-swap?
        # 2-swap is expensive: O(|S|^2 * |outside|^2) if naive.
        # Let's stick to 1-swap for now but maybe increase frequency or randomize better.
        
        # Optimization: If we haven't improved in a while, maybe restart?
        # Restarting with a different random seed for initialization might help.
        if no_improve_count > 1000:
            # Restart with new random initialization
            no_improve_count = 0
            # Clear S
            for x in list(S):
                remove(x)
            S.clear()
            counts.clear()
            
            candidates = list(range(1, 1001))
            random.shuffle(candidates)
            for x in candidates:
                if can_add(x):
                    add(x)
            outside_set = set(candidates)
            outside_set.difference_update(S)
            current_size = len(S)
            if current_size > best_size:
                best_size = current_size

    return sorted(list(S))
