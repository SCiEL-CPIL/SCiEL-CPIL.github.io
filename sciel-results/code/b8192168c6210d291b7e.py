import time
import random
import math

def construct():
    deadline = time.time() + 55.0  # Safety margin for 60s limit
    n = 256
    
    # Data structures for current state
    A = []
    B = []
    sums_A = set()
    sums_B = set()
    diffs_A = set()
    diffs_B = set()
    
    best_size = 0
    best_A = []
    best_B = []
    
    def is_valid_add_A(x):
        # Check sums
        if 2*x in sums_A:
            return False
        for a in A:
            if x + a in sums_A:
                return False
            d = abs(x - a)
            # Check differences within A (Sidon property implies unique diffs)
            if d in diffs_A:
                return False
            # Check differences against B (Disjointness)
            if d in diffs_B:
                return False
        return True

    def is_valid_add_B(x):
        if 2*x in sums_B:
            return False
        for b in B:
            if x + b in sums_B:
                return False
            d = abs(x - b)
            if d in diffs_B:
                return False
            if d in diffs_A:
                return False
        return True

    def add_to_A(x):
        A.append(x)
        sums_A.add(2*x)
        for a in A[:-1]:
            sums_A.add(x + a)
            d = abs(x - a)
            diffs_A.add(d)

    def add_to_B(x):
        B.append(x)
        sums_B.add(2*x)
        for b in B[:-1]:
            sums_B.add(x + b)
            d = abs(x - b)
            diffs_B.add(d)

    def remove_from_A(x):
        # Remove x from A
        A.remove(x)
        sums_A.remove(2*x)
        for a in A:
            sums_A.remove(x + a)
            d = abs(x - a)
            diffs_A.remove(d)

    def remove_from_B(x):
        B.remove(x)
        sums_B.remove(2*x)
        for b in B:
            sums_B.remove(x + b)
            d = abs(x - b)
            diffs_B.remove(d)

    def update_best():
        nonlocal best_size, best_A, best_B
        current_size = len(A) + len(B)
        if current_size > best_size:
            best_size = current_size
            best_A = list(A)
            best_B = list(B)

    # 1. Greedy Initialization
    # Try to build a large set quickly
    candidates = list(range(1, n+1))
    random.shuffle(candidates)
    for x in candidates:
        if is_valid_add_A(x):
            add_to_A(x)
        elif is_valid_add_B(x):
            add_to_B(x)
    
    update_best()
    
    # 2. Simulated Annealing / Random Walk
    # We alternate between adding and removing to escape local optima.
    # Since we only accept valid states, we don't need temperature for validity,
    # but we need a strategy to decide WHICH move to make.
    
    # Strategy: 
    # - Try to add. If successful, great.
    # - If no adds possible (or random choice says remove), remove a random element.
    # - This allows "swapping" elements effectively.
    
    T_start = 1.0
    T_end = 0.01
    steps = 0
    max_steps = 500000 # Rough estimate, time check will stop it
    
    # Use a simple cooling schedule
    # We will just run until deadline.
    
    while time.time() < deadline:
        steps += 1
        
        # Determine move type
        # Bias towards adding, but allow removal to escape dead ends
        # If we are small, add more. If large, mix.
        # Let's say 70% add, 30% remove.
        
        if random.random() < 0.7:
            # Try to add
            # Pick a random candidate not in A or B? 
            # Or just pick from 1..256?
            # Picking from 1..256 is fine, rejection sampling is cheap.
            x = random.randint(1, n)
            
            if x in A or x in B:
                continue
                
            # Decide which set to try
            # Try A first, then B? Or random?
            # Let's try the set with fewer elements to balance? Or random.
            # Random is simpler.
            if random.random() < 0.5:
                if is_valid_add_A(x):
                    add_to_A(x)
                    update_best()
            else:
                if is_valid_add_B(x):
                    add_to_B(x)
                    update_best()
        else:
            # Remove
            if A:
                x = random.choice(A)
                remove_from_A(x)
            elif B:
                x = random.choice(B)
                remove_from_B(x)
            else:
                # Both empty? Should not happen if we start greedy, but just in case
                pass

    return [best_A, best_B]
