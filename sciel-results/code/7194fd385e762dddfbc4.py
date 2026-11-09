import time
import random
import numpy as np

N = 512
DEADLINE_MARGIN = 5

def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE_MARGIN

    def elapsed_ok():
        return (time.time() - t0) < deadline

    # 1. Pre-compute conflict masks using bit-parallelism
    # conflict_mask[i] is a bitmask where bit j is 1 if {i, j} cannot be in the same 4-AP-free set?
    # No, that's for 2-APs. For 4-APs, the conflict is hypergraph.
    # However, we can define "conflict" as: if we pick i, we cannot pick any set of 3 others that form a 4-AP with i.
    # A faster approach for local search:
    # Maintain a boolean array `in_set`.
    # To check if x can be added:
    #   For every 4-AP containing x, check if the other 3 are in set.
    #   This is O(deg(x)) checks.
    
    # Let's precompute for each x, the list of 4-APs it is part of.
    # And for each 4-AP, the indices of the 4 elements.
    
    # Optimization: Instead of checking "all 3 others", we can maintain a counter `violations[x]` 
    # which is the number of 4-APs containing x where the other 3 are already in the set.
    # If violations[x] > 0, x cannot be added.
    
    # Precomputation
    ap_index = {i: [] for i in range(1, N+1)}
    all_aps = []
    
    for d in range(1, (N - 1) // 3 + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            ap = (a, a + d, a + 2 * d, a + 3 * d)
            idx = len(all_aps)
            all_aps.append(ap)
            for x in ap:
                ap_index[x].append(idx)
                
    num_aps = len(all_aps)
    
    # 2. Greedy Construction with Violation Counters
    # This is much faster than re-scanning APs.
    
    def run_greedy_and_refine(order, start_set=None, max_swaps=5000):
        in_set = np.zeros(N + 1, dtype=bool)
        # violations[x] = number of APs containing x where the other 3 are in_set
        violations = np.zeros(N + 1, dtype=int)
        
        if start_set:
            for x in start_set:
                in_set[x] = True
            
            # Initialize violations for start set
            # For each AP, if 3 are in set, increment violation for the 4th?
            # No, violations[x] counts how many APs are "blocked" by x?
            # Let's redefine: 
            # `conflict_count[x]` = number of 4-APs containing x such that the other 3 elements are in the set.
            # If we add x, we need conflict_count[x] == 0.
            # When we add y, we must update conflict_count for all x in APs with y.
            
            # Efficient update:
            # For each y in start_set:
            #   For each AP containing y:
            #     Let others = AP \ {y}
            #     For each x in others:
            #        # We need to know if the other 3 (AP \ {x}) are in set.
            #        # This is getting complicated to update incrementally for arbitrary start sets.
            
            # Simpler: Just compute violations from scratch for the current set.
            # O(N * deg(x)) is fine for N=512.
            
        def compute_violations():
            v = np.zeros(N + 1, dtype=int)
            for idx, ap in enumerate(all_aps):
                # Count how many are in set
                # If 3 are in set, the 4th has a "potential conflict" if added?
                # No, if 3 are in set, the 4th CANNOT be added.
                # So if we consider adding x, we check if there is an AP where other 3 are in set.
                # So v[x] should count APs containing x where the other 3 are in set.
                
                # Let's iterate APs.
                # For each AP (a,b,c,d):
                #   if in_set[a] and in_set[b] and in_set[c]: v[d] += 1
                #   if in_set[a] and in_set[b] and in_set[d]: v[c] += 1
                #   ...
                # This is O(num_aps) which is ~ 512^2 / 6 ~ 40k. Very fast.
                
                a, b, c, d = ap
                sa = in_set[a]
                sb = in_set[b]
                sc = in_set[c]
                sd = in_set[d]
                
                if sa and sb and sc: v[d] += 1
                if sa and sb and sd: v[c] += 1
                if sa and sc and sd: v[b] += 1
                if sb and sc and sd: v[a] += 1
            return v
        
        if start_set:
            violations = compute_violations()
        else:
            violations = np.zeros(N + 1, dtype=int)
            
        # Greedy Phase
        for x in order:
            if not elapsed_ok():
                break
            if not in_set[x] and violations[x] == 0:
                in_set[x] = True
                # Update violations for elements in APs with x
                # For each AP containing x:
                #   Let other 3 be y1, y2, y3.
                #   The AP is now "almost full" (x is in).
                #   If any other 2 of y1,y2,y3 are in set, then the remaining 1 cannot be added?
                #   Wait. 
                #   Definition: v[z] = count of APs containing z where other 3 are in set.
                #   If we add x to set:
                #   For any AP containing x and z (so AP = {x, z, u, v}):
                #   The condition for v[z] to increase is if {x, u, v} are all in set.
                #   Since x is just added, we check if u and v are in set.
                #   If in_set[u] and in_set[v]: then v[z] += 1.
                
                for idx in ap_index[x]:
                    ap = all_aps[idx]
                    others = [y for y in ap if y != x]
                    # Check if all others are in set? No, if all others are in set, 
                    # then x would have had violation > 0 and wouldn't be added.
                    # So we are in a state where x was addable.
                    # Now x is in set.
                    # For each z in others:
                    #   The AP is {x, z, u, v}.
                    #   v[z] increases if {x, u, v} are in set.
                    #   x is in set. So we check if u and v are in set.
                    for i, z in enumerate(others):
                        # u and v are the other two in others
                        u, v = others[0], others[1]
                        if z == u: v = others[2]
                        elif z == v: u = others[2]
                        else: # z is the 3rd one
                             u, v = others[0], others[1]
                             # Wait, others has 3 elements.
                             # If z is others[0], u=others[1], v=others[2]
                             # If z is others[1], u=others[0], v=others[2]
                             # If z is others[2], u=others[0], v=others[1]
                        if in_set[u] and in_set[v]:
                            violations[z] += 1
                            
        # Local Search Phase (Swaps)
        # Try to swap x (in set) with y (not in set) if it reduces violations or allows adding more.
        # Heuristic: Remove x if it causes high violations for others?
        # Better: Find a y not in set that has low "cost" and x in set that has high "cost".
        # Cost(x) = number of APs where x is the "blocking" element?
        # Let's use a simple random swap heuristic.
        
        current_size = np.sum(in_set)
        
        # Precompute list of elements not in set for fast sampling
        # And list of elements in set.
        
        for _ in range(max_swaps):
            if not elapsed_ok():
                break
                
            # Pick a random element x in set
            # And a random element y not in set
            # This is slow if done naively.
            
            # Alternative: 
            # 1. Compute "potential" for each y not in set: how many violations would be removed if we added y?
            #    Actually, adding y requires violations[y] == 0.
            #    If violations[y] > 0, we must remove some elements to make it 0.
            
            # Let's try a "repair" step:
            # Find a y not in set with violations[y] > 0.
            # Pick one of the "conflicting" APs. Remove one element from that AP.
            # This is complex.
            
            # Simpler Local Search:
            # 1. Try to add any y not in set with violations[y] == 0. (Greedy missed some due to order?)
            #    The greedy phase already did this for the given order. 
            #    But if we change the order, we might find more.
            #    So, let's just run multiple greedy orders.
            
            pass # Skip complex local search for now, rely on diverse greedy orders.
            
        return {x for x in range(1, N + 1) if in_set[x]}

    # Generate Diverse Orders
    numbers = list(range(1, N + 1))
    
    # Order 1: Min Degree
    degrees = np.array([len(ap_index[i]) for i in range(1, N+1)])
    min_deg_order = sorted(numbers, key=lambda x: degrees[x-1])
    max_deg_order = sorted(numbers, key=lambda x: -degrees[x-1])
    
    # Order 2: Base-4 set first, then rest
    base4_set = set()
    for i in range(1, N + 1):
        x = i
        valid = True
        while x > 0:
            digit = x % 4
            if digit > 1:
                valid = False
                break
            x //= 4
        if valid:
            base4_set.add(i)
            
    base4_list = sorted(base4_set)
    rest = sorted(set(numbers) - base4_set)
    base4_first_order = base4_list + rest
    
    # Order 3: Random permutations
    rng = random.Random(42)
    
    best_set = set()
    best_size = 0
    
    # Run Greedy with different orders
    orders = [
        min_deg_order,
        max_deg_order,
        base4_first_order,
        list(reversed(numbers)),
        list(numbers)
    ]
    
    # Add random orders
    for _ in range(50):
        if not elapsed_ok():
            break
        r_order = numbers[:]
        rng.shuffle(r_order)
        orders.append(r_order)
        
    # Add orders starting with base4, then random rest
    for _ in range(50):
        if not elapsed_ok():
            break
        r_rest = rest[:]
        rng.shuffle(r_rest)
        orders.append(base4_list + r_rest)

    for order in orders:
        if not elapsed_ok():
            break
        s = run_greedy_and_refine(order, start_set=None)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s

    # Try starting with Base4 set and adding rest
    for _ in range(50):
        if not elapsed_ok():
            break
        r_rest = rest[:]
        rng.shuffle(r_rest)
        s = run_greedy_and_refine(r_rest, start_set=base4_set)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s

    return sorted(best_set)
