import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    def greedy_with_order(order):
        """Greedy subset selection using bitset, processing elements in given order."""
        mask = (1 << 201) - 1
        selected = []
        reachable_bits = 1  # Bit 0 is set (sum 0 is reachable)
        
        for x in order:
            # Check if adding x would create sum 200
            # This happens if (200 - x) is currently reachable
            if (reachable_bits >> (200 - x)) & 1:
                continue
            
            selected.append(x)
            # Update reachable bits: new sums are old_sums + x
            new_bits = (reachable_bits << x) & mask
            reachable_bits |= new_bits
        
        return selected, reachable_bits
    
    def get_reachable(selected):
        """Compute reachable sums for a given selected set."""
        mask = (1 << 201) - 1
        reachable_bits = 1
        for x in selected:
            new_bits = (reachable_bits << x) & mask
            reachable_bits |= new_bits
        return reachable_bits
    
    def try_add_elements(selected, reachable_bits, candidates):
        """Try to add elements from candidates to selected, updating reachable."""
        mask = (1 << 201) - 1
        improved = False
        for x in candidates:
            if x in selected:
                continue
            if (reachable_bits >> (200 - x)) & 1:
                continue
            # Safe to add
            selected.append(x)
            new_bits = (reachable_bits << x) & mask
            reachable_bits |= new_bits
            improved = True
        return selected, reachable_bits, improved
    
    # Generate diverse orderings
    orders = []
    
    # Largest first
    orders.append(list(range(100, 0, -1)))
    
    # Smallest first
    orders.append(list(range(1, 101)))
    
    # Medium to extremes: 50, 51, 49, 52, 48, 53, ...
    med_order = []
    for d in range(50):
        if 50 - d >= 1:
            med_order.append(50 - d)
        if 51 + d <= 100:
            med_order.append(51 + d)
    orders.append(med_order)
    
    # Random orderings
    for _ in range(20):
        order = list(range(1, 101))
        random.shuffle(order)
        orders.append(order)
    
    # Strategic: prioritize elements that are "harder" to form 200 with
    # Elements > 100 can't pair with anything to make 200 (impossible since max is 100)
    # Actually all elements <= 100. Elements > 100 don't exist.
    # Elements in 101-200 don't exist. So all 1-100 can potentially be in a sum-200 subset.
    # But larger elements require smaller companions.
    # Try: 100, 1, 99, 2, 98, 3, ... (alternating large and small)
    alt_order = []
    for i in range(50):
        alt_order.append(100 - i)
        if i + 1 <= 50:
            alt_order.append(i + 1)
    orders.append(alt_order)
    
    best_set = []
    best_size = 0
    best_reachable = 1
    
    for order in orders:
        if time.time() > deadline:
            break
        
        selected, reachable = greedy_with_order(order)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected[:]
            best_reachable = reachable
        
        # Local search: try to add more elements
        # Candidates are elements not in selected
        selected_set = set(selected)
        candidates = [x for x in range(1, 101) if x not in selected_set]
        
        # Try adding in different orders
        for add_order in [candidates, list(reversed(candidates))]:
            if time.time() > deadline:
                break
            new_selected = selected[:]
            new_reachable = reachable
            new_selected, new_reachable, improved = try_add_elements(
                new_selected, new_reachable, add_order
            )
            if len(new_selected) > best_size:
                best_size = len(new_selected)
                best_set = new_selected[:]
                best_reachable = new_reachable
    
    # Additional local search on the best set found
    if time.time() < deadline - 5:
        # Try multiple rounds of adding
        for round_num in range(10):
            if time.time() > deadline:
                break
            
            selected_set = set(best_set)
            candidates = [x for x in range(1, 101) if x not in selected_set]
            
            if not candidates:
                break
            
            # Try adding candidates in random order
            random.shuffle(candidates)
            new_selected = best_set[:]
            new_reachable = best_reachable
            new_selected, new_reachable, improved = try_add_elements(
                new_selected, new_reachable, candidates
            )
            
            if len(new_selected) > best_size:
                best_size = len(new_selected)
                best_set = new_selected[:]
                best_reachable = new_reachable
            else:
                break  # No improvement, stop
    
    # Ensure nonempty
    if not best_set:
        best_set = [1]
    
    return best_set
