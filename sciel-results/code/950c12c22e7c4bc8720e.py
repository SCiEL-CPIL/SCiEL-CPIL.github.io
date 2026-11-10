"""Improved B2[2] set construction for {1..1000}."""
import time
import random
from collections import Counter, defaultdict

N = 1000
MAX_REP = 2  # at most two representations a+b with a<=b

def _sum_counts(chosen):
    """Compute the count of each sum a+b with a<=b from chosen."""
    counts = {}
    for i, a in enumerate(chosen):
        for j in range(i, len(chosen)):
            b = chosen[j]
            s = a + b
            counts[s] = counts.get(s, 0) + 1
    return counts

def _greedy_build(order):
    """Build a B2[2] set by greedily adding elements in the given order."""
    chosen = []
    counts = {}
    for x in order:
        # Check if adding x keeps all sums <= MAX_REP
        # New sums: x + y for y in chosen (y <= x since we process in order? No, order is arbitrary)
        # Actually, we need to check all pairs (x, y) where y in chosen, x <= y OR y <= x
        # Since we're adding x, new pairs are (x, x) and (x, y) for each y in chosen
        # But we need a <= b, so pairs are (min(x,y), max(x,y))
        new_sums = []
        for y in chosen:
            a, b = (x, y) if x <= y else (y, x)
            s = a + b
            new_sums.append(s)
        new_sums.append(2 * x)
        
        # Check if any sum would exceed MAX_REP
        ok = True
        for s in new_sums:
            if counts.get(s, 0) + 1 > MAX_REP:
                ok = False
                break
        
        if ok:
            chosen.append(x)
            for s in new_sums:
                counts[s] = counts.get(s, 0) + 1
    
    return chosen

def _local_search(chosen, counts, max_steps=5000, deadline=None):
    """Improve a B2[2] set via local search: try insertions and swaps."""
    chosen = list(chosen)
    counts = dict(counts)
    
    def try_insert(x):
        """Try to insert x into the set."""
        if x in set(chosen):
            return False
        new_sums = []
        for y in chosen:
            a, b = (x, y) if x <= y else (y, x)
            s = a + b
            new_sums.append(s)
        new_sums.append(2 * x)
        
        for s in new_sums:
            if counts.get(s, 0) + 1 > MAX_REP:
                return False
        
        # Insert
        chosen.append(x)
        for s in new_sums:
            counts[s] = counts.get(s, 0) + 1
        return True
    
    def try_remove(x):
        """Remove x from the set."""
        chosen.remove(x)
        for y in chosen:
            a, b = (x, y) if x <= y else (y, x)
            s = a + b
            counts[s] -= 1
            if counts[s] == 0:
                del counts[s]
        # Remove 2*x
        s = 2 * x
        counts[s] -= 1
        if counts[s] == 0:
            del counts[s]
    
    def try_swap(remove_x, add_y):
        """Try to swap remove_x for add_y."""
        if add_y in set(chosen) and add_y != remove_x:
            return False
        
        # First, temporarily remove remove_x
        idx = chosen.index(remove_x)
        # Compute sums involving remove_x
        sums_to_remove = []
        for y in chosen:
            if y == remove_x:
                continue
            a, b = (remove_x, y) if remove_x <= y else (y, remove_x)
            s = a + b
            sums_to_remove.append(s)
        sums_to_remove.append(2 * remove_x)
        
        # Check if removing remove_x and adding add_y is valid
        # New sums from add_y with remaining elements
        remaining = [y for y in chosen if y != remove_x]
        new_sums = []
        for y in remaining:
            a, b = (add_y, y) if add_y <= y else (y, add_y)
            s = a + b
            new_sums.append(s)
        new_sums.append(2 * add_y)
        
        # Check validity
        ok = True
        # After removing remove_x, counts for sums_to_remove decrease by 1
        # After adding add_y, counts for new_sums increase by 1
        # We need to check that no sum exceeds MAX_REP
        
        # Build a temporary count adjustment
        # For sums in sums_to_remove: count decreases by 1
        # For sums in new_sums: count increases by 1
        # Net change per sum:
        net_changes = {}
        for s in sums_to_remove:
            net_changes[s] = net_changes.get(s, 0) - 1
        for s in new_sums:
            net_changes[s] = net_changes.get(s, 0) + 1
        
        for s, delta in net_changes.items():
            new_count = counts.get(s, 0) + delta
            if new_count < 0:
                ok = False
                break
            if new_count > MAX_REP:
                ok = False
                break
        
        if not ok:
            return False
        
        # Perform the swap
        try_remove(remove_x)
        try_insert(add_y)
        return True
    
    improved = True
    step = 0
    while improved and step < max_steps:
        if deadline and time.time() > deadline:
            break
        improved = False
        step += 1
        
        # Try insertions first
        for x in range(1, N + 1):
            if x not in set(chosen):
                if try_insert(x):
                    improved = True
                    # Don't break, keep trying more insertions
                    if deadline and time.time() > deadline:
                        break
        if deadline and time.time() > deadline:
            break
        
        # Try swaps if no improvement from insertions
        if not improved:
            chosen_set = set(chosen)
            for x in list(chosen):
                if deadline and time.time() > deadline:
                    break
                for y in range(1, N + 1):
                    if y not in chosen_set:
                        if try_swap(x, y):
                            improved = True
                            chosen_set = set(chosen)
                            break
                if deadline and time.time() > deadline:
                    break
    
    return chosen, counts

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    best_size = 0
    best_chosen = []
    
    # Multiple greedy strategies with different orderings
    strategies = []
    
    # 1. Ascending order
    strategies.append(list(range(1, N + 1)))
    
    # 2. Descending order
    strategies.append(list(range(N, 0, -1)))
    
    # 3. Random shuffles
    rng = random.Random(42)
    for _ in range(10):
        order = list(range(1, N + 1))
        rng.shuffle(order)
        strategies.append(order)
    
    # 4. Weighted: prefer numbers that are less likely to create conflicts
    #    Use a heuristic: start with sparse sets
    for _ in range(5):
        # Try to build with a gap-based approach
        order = []
        for x in range(1, N + 1):
            # Add with some randomness
            if rng.random() < 0.5:
                order.append(x)
        # Ensure all numbers are included in some order
        remaining = [x for x in range(1, N + 1) if x not in set(order)]
        order.extend(remaining)
        strategies.append(order)
    
    # 5. Start from a known good structure: multiples with gaps
    #    B2[2] sets can be constructed using quadratic forms or similar
    #    Try: x -> x^2 mod something, or use Beatty sequences
    
    # Run greedy for each strategy
    for order in strategies:
        if time.time() > deadline:
            break
        chosen = _greedy_build(order)
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_chosen = list(chosen)
    
    # Local search to improve
    if best_chosen:
        counts = _sum_counts(best_chosen)
        best_chosen, counts = _local_search(best_chosen, counts, max_steps=2000, deadline=deadline)
    
    # Additional local search restarts
    if time.time() < deadline - 30:
        for restart in range(5):
            if time.time() > deadline - 10:
                break
            # Start from a random greedy solution
            order = list(range(1, N + 1))
            random.Random(restart * 1000).shuffle(order)
            chosen = _greedy_build(order)
            if len(chosen) <= best_size:
                continue
            counts = _sum_counts(chosen)
            chosen, counts = _local_search(chosen, counts, max_steps=1000, deadline=deadline)
            if len(chosen) > best_size:
                best_size = len(chosen)
                best_chosen = list(chosen)
    
    # Final polish: try more aggressive local search
    if best_chosen and time.time() < deadline - 15:
        counts = _sum_counts(best_chosen)
        best_chosen, counts = _local_search(best_chosen, counts, max_steps=3000, deadline=deadline)
    
    return sorted(best_chosen)
