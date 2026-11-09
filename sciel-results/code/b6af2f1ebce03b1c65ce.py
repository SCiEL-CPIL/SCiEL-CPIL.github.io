import time
import random

def construct():
    start_time = time.time()
    deadline = 85.0  # 5s margin
    
    # Target sums
    MIN_SUM = 128
    MAX_SUM = 256
    N = 128
    
    # Initialize with full set
    A = set(range(N + 1))
    
    # counts[s] = number of pairs (a,b) in A with a<=b, a+b=s
    # We only care about s in [128, 256]
    counts = [0] * (MAX_SUM + 1)
    
    def update_counts(x, sign):
        """
        Update counts when adding (sign=1) or removing (sign=-1) x.
        A must be the set BEFORE the change for the neighbor iteration?
        If adding: neighbors are in A (old A). x is new.
        If removing: neighbors are in A (old A). x is in A.
        
        Pairs involving x:
        1. (x, x) -> sum 2x
        2. (x, y) where y in A, y != x.
           Since pairs are unordered a<=b:
           If adding x: y must be > x (since x is new, y in old A). If y < x, pair is (y, x) which is same as (x, y).
           Wait, if we define pairs by a<=b:
           When adding x, new pairs are {x, y} for all y in A such that y >= x?
           No, if y < x, the pair is {y, x}. This is a new pair because x wasn't there.
           So for adding x:
           - Pair {x, x} is new.
           - Pair {x, y} is new for all y in A.
             But we must count each unordered pair once.
             If we iterate y in A:
             If y > x, pair is {x, y}.
             If y < x, pair is {y, x}.
             If y == x, impossible since x not in A.
             So we just iterate all y in A and add 1 to counts[x+y].
             Plus add 1 to counts[2x].
             
           When removing x:
           - Pair {x, x} is removed.
           - Pair {x, y} is removed for all y in A \ {x}.
             Iterate y in A:
             If y != x, subtract 1 from counts[x+y].
        """
        if 2 * x >= MIN_SUM and 2 * x <= MAX_SUM:
            counts[2 * x] += sign
            
        for y in A:
            if y == x:
                continue
            s = x + y
            if MIN_SUM <= s <= MAX_SUM:
                counts[s] += sign

    # Initial calculation
    # O(N^2)
    for a in A:
        for b in A:
            if a <= b:
                s = a + b
                if MIN_SUM <= s <= MAX_SUM:
                    counts[s] += 1
    
    def get_peak():
        return max(counts[MIN_SUM:MAX_SUM+1])
        
    def is_valid():
        return min(counts[MIN_SUM:MAX_SUM+1]) >= 1
        
    current_peak = get_peak()
    best_peak = current_peak
    best_A = set(A)
    
    # Peeling Phase
    # Try to remove elements to reduce peak or maintain peak while shrinking
    # We do this in a loop until no improvement or time runs out
    removed_any = True
    while removed_any and time.time() < deadline - 20:
        removed_any = False
        # Shuffle A to avoid bias
        items = list(A)
        random.shuffle(items)
        for x in items:
            if time.time() > deadline - 20:
                break
            # Simulate removal
            old_peak = current_peak
            update_counts(x, -1)
            A.remove(x) # Temporarily remove to update neighbors? No, update_counts uses A.
            # Wait, update_counts(x, -1) iterates over A. If x is in A, it skips y==x.
            # So A should contain x during the update call.
            # Let's re-verify update_counts logic.
            
            # Correct Logic for Removal:
            # 1. Calculate delta on counts using current A (which contains x).
            # 2. Remove x from A.
            
            # Let's rollback
            A.add(x)
            update_counts(x, 1) # Restore counts
            
            # Now properly simulate
            update_counts(x, -1)
            A.remove(x)
            
            if is_valid():
                new_peak = get_peak()
                if new_peak < old_peak:
                    # Accept
                    current_peak = new_peak
                    if current_peak < best_peak:
                        best_peak = current_peak
                        best_A = set(A)
                    removed_any = True
                    break # Restart loop with new A
                elif new_peak == old_peak:
                    # Accept to shrink set, might help later
                    current_peak = new_peak
                    # Don't update best yet, but keep going
                    removed_any = True
                    break
            else:
                # Rollback
                A.add(x)
                update_counts(x, 1)
                current_peak = old_peak # Should be restored
            
    # Local Search Phase
    # Swap x in A with y not in A
    while time.time() < deadline - 5:
        if not A:
            break
            
        x = random.choice(list(A))
        # Pick y not in A
        # To speed up, pick from [0, N]
        y = random.randint(0, N)
        if y in A:
            continue
            
        # Simulate Swap: Remove x, Add y
        old_peak = current_peak
        
        # Remove x
        update_counts(x, -1)
        A.remove(x)
        
        # Add y
        update_counts(y, 1)
        A.add(y)
        
        if is_valid():
            new_peak = get_peak()
            if new_peak < old_peak:
                current_peak = new_peak
                if current_peak < best_peak:
                    best_peak = current_peak
                    best_A = set(A)
            elif new_peak == old_peak:
                # Accept with some probability to explore
                if random.random() < 0.5:
                    current_peak = new_peak
                else:
                    # Rollback
                    A.remove(y)
                    update_counts(y, -1)
                    A.add(x)
                    update_counts(x, 1)
            else:
                # Rollback
                A.remove(y)
                update_counts(y, -1)
                A.add(x)
                update_counts(x, 1)
        else:
            # Rollback
            A.remove(y)
            update_counts(y, -1)
            A.add(x)
            update_counts(x, 1)
            
    return list(best_A)
