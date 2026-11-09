import time
import random
from itertools import combinations

def _can_add(selected_set, selected_list, candidate):
    """
    Check if candidate can be added to selected_list (and selected_set) without violating constraints.
    selected_set is a set of ints for O(1) lookup.
    selected_list is a list of ints for iteration.
    """
    x = candidate
    n_sel = len(selected_list)
    
    # Condition B: x does not divide sum of any 3 distinct others in selected_list
    # For each pair (b, c) in selected_list, check if there exists d in selected_list
    # such that (b + c + d) % x == 0.
    # This implies d = (-(b + c)) % x.
    # We need d to be in selected_set, and d != b and d != c.
    for i in range(n_sel):
        b = selected_list[i]
        for j in range(i + 1, n_sel):
            c = selected_list[j]
            target_d = (-(b + c)) % x
            if target_d in selected_set:
                if target_d != b and target_d != c:
                    return False
                # If target_d == b or target_d == c, it's not a distinct triplet of OTHERS.
                # Wait, the condition is "sum of any k=3 DISTINCT OTHER selected integers".
                # The others are from selected_list. b, c, d must be distinct.
                # If target_d == b, then b+c+b = 2b+c. Is this sum of 3 distinct others? No, b is repeated.
                # So we only reject if there exist DISTINCT b,c,d in selected_list summing to multiple of x.
                # My check finds a specific d. If d==b, it's not a valid triplet of distinct others.
                # So we continue.
    
    # Condition A: For every a in selected_list, a does not divide (x + b + c)
    # for any distinct b, c in selected_list \ {a, x}.
    # Since x is not in selected_list yet, selected_list \ {a} is just selected_list without a.
    # We need b, c distinct, and b,c != a.
    # Sum = x + b + c. We need (x + b + c) % a != 0.
    # Equivalent to: (b + c) % a != (-x) % a.
    
    for a in selected_list:
        target_sum_mod = (-x) % a
        # Check if there exist distinct b, c in selected_list \ {a} such that (b+c) % a == target_sum_mod
        # Iterate b in selected_list where b != a
        # For each b, we need c in selected_list \ {a, b} such that c % a == (target_sum_mod - b) % a
        # We can precompute residues of selected_list for each a? No, just loop.
        # Optimization: Create a set of residues of (value % a) for values in selected_list \ {a}?
        # But we need distinct b, c.
        
        # Simple O(N^2) check for this a
        # We can break early if found.
        found_pair = False
        # To avoid O(N^2) inner loop for every a, we can use a set of available residues.
        # But we need to ensure b != c.
        
        # Let's just do the double loop, it's small N.
        for i in range(n_sel):
            b = selected_list[i]
            if b == a:
                continue
            # We need c such that (b + c) % a == target_sum_mod
            # c % a == (target_sum_mod - b) % a
            req_res = (target_sum_mod - b) % a
            # Check if there exists c in selected_list, c != a, c != b, such that c % a == req_res
            # We can iterate c
            for j in range(n_sel):
                if j == i:
                    continue
                c = selected_list[j]
                if c == a:
                    continue
                if c % a == req_res:
                    return False
        # If no pair found for this a, continue to next a
        
    return True

def _greedy(order):
    selected_list = []
    selected_set = set()
    for x in order:
        if _can_add(selected_set, selected_list, x):
            selected_list.append(x)
            selected_set.add(x)
    return selected_list

def _local_search(initial_set, max_iters=1000, time_limit=5.0):
    """
    Try to improve the set by adding elements from the complement.
    """
    best_set = list(initial_set)
    best_size = len(best_set)
    current_set = list(best_set)
    current_set_set = set(current_set)
    
    all_numbers = set(range(1, 193))
    complement = list(all_numbers - current_set_set)
    
    start_time = time.time()
    random.shuffle(complement)
    
    for _ in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        if not complement:
            break
            
        # Pick a random candidate from complement
        # We can just iterate through complement and try to add
        # Or pick random index
        idx = random.randint(0, len(complement) - 1)
        x = complement[idx]
        
        if _can_add(current_set_set, current_set, x):
            # Add it
            current_set.append(x)
            current_set_set.add(x)
            complement.pop(idx)
            
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = list(current_set)
                
        # Optional: Try to remove an element and add a different one?
        # This is complex. Let's stick to pure addition first.
        # If we get stuck, we might need a swap.
        
    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    best_set = []
    
    # Strategy 1: Greedy with different orderings
    # Descending
    order1 = list(range(192, 0, -1))
    s1 = _greedy(order1)
    if len(s1) > len(best_set):
        best_set = s1
        
    if time.time() > deadline:
        return best_set
        
    # Ascending
    order2 = list(range(1, 193))
    s2 = _greedy(order2)
    if len(s2) > len(best_set):
        best_set = s2
        
    if time.time() > deadline:
        return best_set
        
    # Random orderings
    random.seed(42)
    for i in range(100):
        if time.time() > deadline:
            break
        order = list(range(1, 193))
        random.shuffle(order)
        s = _greedy(order)
        if len(s) > len(best_set):
            best_set = s
            
    if time.time() > deadline:
        return best_set
        
    # Strategy 2: Local Search on the best set found
    # We can run local search on the best set to see if we can add more
    # The greedy might have missed some additions due to order.
    # Local search tries to add any remaining element.
    
    improved_set = _local_search(best_set, max_iters=2000, time_limit=10.0)
    if len(improved_set) > len(best_set):
        best_set = improved_set
        
    if time.time() > deadline:
        return best_set
        
    # Strategy 3: More aggressive Local Search with Swaps
    # If we are stuck, try swapping an element in the set with one outside.
    # This is harder to implement correctly with the fast check.
    # Let's try a few more random restarts with local search.
    
    for i in range(50):
        if time.time() > deadline:
            break
        order = list(range(1, 193))
        random.shuffle(order)
        s = _greedy(order)
        improved = _local_search(s, max_iters=500, time_limit=2.0)
        if len(improved) > len(best_set):
            best_set = improved
            
    return best_set

# To ensure the program finishes within 60s, we rely on the deadline checks.
# The _can_add function is O(N^3) which is ~10^5 ops for N=40.
# Greedy is 192 * 10^5 = 2 * 10^7 ops. Fast.
# Local search adds more calls.

if __name__ == "__main__":
    # Test
    res = construct()
    print(len(res))
