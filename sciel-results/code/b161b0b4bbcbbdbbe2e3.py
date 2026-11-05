import time
import itertools
from itertools import combinations
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50  # safety margin
    
    n = 192
    max_size = 48
    
    def try_add(selected, x):
        """Check if x can be added to selected without creating violations."""
        if x in selected:
            return False
        k = len(selected)
        # Check if x divides sum of any 3 distinct others in selected
        if k >= 3:
            for a, b, c in combinations(selected, 3):
                if (a + b + c) % x == 0:
                    return False
        # Check if any a in selected divides x + b + c for distinct b,c in selected \ {a}
        if k >= 2:
            for a in selected:
                others = [v for v in selected if v != a]
                if len(others) < 2:
                    continue
                for b, c in combinations(others, 2):
                    if (x + b + c) % a == 0:
                        return False
        return True
    
    def repair_and_backfill(initial, deadline):
        """Repair initial set by removing violators, then backfill."""
        current = list(initial)
        current_set = set(current)
        
        # Repair: iteratively remove the most-violating element
        for iteration in range(100):
            if time.time() > deadline:
                break
            
            violation_count = {x: 0 for x in current}
            k = len(current)
            if k < 4:
                break
            
            for i, x in enumerate(current):
                others = [v for j, v in enumerate(current) if j != i]
                if len(others) < 3:
                    continue
                for a, b, c in combinations(others, 3):
                    if (a + b + c) % x == 0:
                        violation_count[x] += 1
            
            if all(v == 0 for v in violation_count.values()):
                break
            
            max_viol = max(violation_count.values())
            if max_viol == 0:
                break
            
            worst = [x for x in current if violation_count[x] == max_viol]
            to_remove = random.choice(worst)
            current.remove(to_remove)
            current_set.discard(to_remove)
        
        # Backfill: try to add remaining candidates
        remaining = [x for x in range(n, 0, -1) if x not in current_set]
        random.shuffle(remaining)
        
        for x in remaining:
            if time.time() > deadline:
                break
            if len(current) >= max_size:
                break
            if try_add(current, x):
                current.append(x)
                current_set.add(x)
        
        return current
    
    best_set = []
    best_size = 0
    
    # Multiple repair+backfill attempts with different initial sets
    for seed in range(10):
        if time.time() > deadline:
            break
        
        if seed == 0:
            # Dense upper half
            initial = list(range(n, n//3, -1))
        elif seed == 1:
            # Upper half with some randomness
            initial = list(range(n, n//3, -1))
            random.shuffle(initial)
            initial = initial[:35]
        elif seed == 2:
            # Numbers > n/4
            initial = list(range(n, n//4, -1))
            random.shuffle(initial)
            initial = initial[:30]
        elif seed == 3:
            # Odd numbers in upper half
            initial = [x for x in range(n, n//3, -1) if x % 2 == 1]
            random.shuffle(initial)
            initial = initial[:30]
        else:
            # Random subset of upper half
            initial = list(range(n, n//3, -1))
            random.shuffle(initial)
            initial = initial[:min(35, len(initial))]
        
        result = repair_and_backfill(initial, deadline)
        
        if len(result) > best_size:
            best_size = len(result)
            best_set = list(result)
    
    # Also try pure greedy as fallback
    if time.time() < deadline - 5:
        for order_seed in range(3):
            if time.time() > deadline:
                break
            
            if order_seed == 0:
                order = list(range(n, 0, -1))
            elif order_seed == 1:
                order = list(range(n, n//3, -1)) + list(range(n//3, 0, -1))
            else:
                order = [x for x in range(n, 0, -1) if x % 2 == 1] + [x for x in range(n, 0, -1) if x % 2 == 0]
            
            selected = []
            selected_set = set()
            
            for x in order:
                if time.time() > deadline:
                    break
                if x in selected_set:
                    continue
                if try_add(selected, x):
                    selected.append(x)
                    selected_set.add(x)
                    if len(selected) >= max_size:
                        break
            
            if len(selected) > best_size:
                best_size = len(selected)
                best_set = list(selected)
    
    return best_set
