import random

def construct():
    best_set = []
    best_size = 0
    
    # Reduced restarts to allow for swap logic overhead
    for _ in range(50):
        candidates = list(range(1, 1001))
        random.shuffle(candidates)
        
        current_set = []
        counts = [0] * 2001
        
        def can_add(x, ignore=None):
            # Check 2*x
            if counts[2 * x] >= 2:
                return False
            # Check x + y for y in current_set
            for y in current_set:
                if y == ignore:
                    continue
                if counts[x + y] >= 2:
                    return False
            return True

        def do_add(x):
            current_set.append(x)
            counts[2 * x] += 1
            for y in current_set[:-1]:
                counts[x + y] += 1

        def do_remove(x):
            current_set.remove(x)
            counts[2 * x] -= 1
            for y in current_set:
                counts[x + y] -= 1

        # Greedy with Swap Repair
        for x in candidates:
            if can_add(x):
                do_add(x)
            else:
                # Try to repair by swapping with a random element
                # Limit attempts to 5 to keep speed
                for _ in range(5):
                    if not current_set:
                        break
                    y_idx = random.randint(0, len(current_set) - 1)
                    y = current_set[y_idx]
                    
                    # Check if x can be added if y were removed
                    if can_add(x, ignore=y):
                        # Perform swap
                        do_remove(y)
                        do_add(x)
                        break

        # Post-processing: Try to add any remaining candidates that might fit now
        # We iterate through all 1..1000 to catch any skipped ones
        for x in range(1, 1001):
            if x not in current_set:
                if can_add(x):
                    do_add(x)

        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
    return best_set
