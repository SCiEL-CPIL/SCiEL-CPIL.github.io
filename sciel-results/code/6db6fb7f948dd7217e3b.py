import random
import time

def is_valid(set_of_ints):
    if not set_of_ints:
        return True
    
    # Generate all non-empty subset sums
    sums = set()
    current_sums = {0}
    for x in set_of_ints:
        new_sums = set()
        for s in current_sums:
            new_sums.add(s + x)
        current_sums.update(new_sums)
        
    if 0 in current_sums:
        current_sums.remove(0)
        
    sums = current_sums
    max_sum = sum(set_of_ints)
    
    # Check divisibility
    for a in sums:
        if a > max_sum // 2:
            continue
        multiple = 2 * a
        while multiple <= max_sum:
            if multiple in sums:
                return False
            multiple += a
            
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_set = set()
    best_size = 0
    
    while time.time() < deadline:
        # Restart with a random greedy initialization
        current_set = set()
        
        # Greedy: pick random numbers, add if valid
        # Bias towards larger numbers?
        # Let's just shuffle all
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                break
            trial = current_set | {x}
            if is_valid(trial):
                current_set = trial
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
            
        # Local search refinement
        no_improve_count = 0
        max_no_improve = 1000
        
        while time.time() < deadline and no_improve_count < max_no_improve:
            r = random.random()
            
            if r < 0.5:
                # Add
                if len(current_set) >= 256:
                    continue
                available = [x for x in range(1, 257) if x not in current_set]
                if not available:
                    continue
                x = random.choice(available)
                trial = current_set | {x}
                if is_valid(trial):
                    current_set = trial
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_set = set(current_set)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    
            elif r < 0.8:
                # Remove
                if not current_set:
                    continue
                x = random.choice(list(current_set))
                current_set = current_set - {x}
                # no_improve_count logic: removing doesn't improve size, but might help
                # We can reset no_improve_count occasionally to allow exploration
                # But strictly, we want to maximize size.
                
            else:
                # Swap
                if not current_set:
                    continue
                x = random.choice(list(current_set))
                available = [y for y in range(1, 257) if y not in current_set]
                if not available:
                    continue
                y = random.choice(available)
                trial = (current_set - {x}) | {y}
                if is_valid(trial):
                    current_set = trial
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_set = set(current_set)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    
    return list(best_set)
