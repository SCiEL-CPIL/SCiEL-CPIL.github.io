import random

def construct():
    best_set = []
    best_size = 0
    
    # We can try a few different strategies for ordering
    # 1. Random
    # 2. Descending
    # 3. Ascending (seed)
    
    # Let's do 200 random restarts
    # And maybe 20 descending restarts?
    
    for _ in range(200):
        candidates = list(range(1, 1001))
        random.shuffle(candidates)
        
        current_set = []
        counts = [0] * 2001 # Max sum is 1000+1000=2000
        
        for x in candidates:
            # Sums to check: x + y for y in current_set, and 2*x
            # Note: y can be x? No, y in current_set, x is new.
            # But the problem says a <= b.
            # If we add x, new pairs are (x, y) where x <= y?
            # Wait, the set is unordered. The representation a+b with a<=b.
            # If we add x, the new sums are x+y for all y in current_set such that x <= y?
            # And 2*x.
            # Actually, it's easier to just consider all pairs (x, y) with y in current_set.
            # If x < y, the pair is (x, y). Sum x+y.
            # If x > y, the pair is (y, x). Sum x+y.
            # If x == y, the pair is (x, x). Sum 2x.
            
            # So we just need to check sums x+y for all y in current_set, and 2x.
            
            valid = True
            new_sums = []
            
            # Check 2*x
            s = 2 * x
            if counts[s] >= 2:
                valid = False
            else:
                new_sums.append(s)
            
            if valid:
                for y in current_set:
                    s = x + y
                    if counts[s] >= 2:
                        valid = False
                        break
                    else:
                        new_sums.append(s)
            
            if valid:
                current_set.append(x)
                for s in new_sums:
                    counts[s] += 1
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            
    return best_set
