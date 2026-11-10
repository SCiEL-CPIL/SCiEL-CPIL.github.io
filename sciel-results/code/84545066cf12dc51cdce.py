"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    N = 180
    
    # Precompute forbidden triples and conflict counts
    # triple is (a, b, c) with a < b < c? No, just the set of 3 numbers.
    # We need to know for each x, which pairs (u, v) form a triple with x.
    # Let's store triples as sorted tuples.
    forbidden_triples = set()
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            # c = ab / (b-a)
            # b > a so b-a > 0
            denom = b - a
            num = a * b
            if num % denom != 0:
                continue
            c = num // denom
            if c <= b or c > N:
                continue
            # We have a < b < c
            # Check if distinct (guaranteed by c > b > a)
            triple = (a, b, c)
            forbidden_triples.add(triple)
            
    # Count conflicts for each number
    # conflict_count[x] = number of triples x is in
    conflict_count = [0] * (N + 1)
    # Also store the triples involving x for fast lookup during validation
    # triples_with[x] = set of (u, v) such that {x, u, v} is a forbidden triple
    triples_with = [set() for _ in range(N + 1)]
    
    for t in forbidden_triples:
        a, b, c = t
        conflict_count[a] += 1
        conflict_count[b] += 1
        conflict_count[c] += 1
        
        triples_with[a].add((b, c))
        triples_with[b].add((a, c))
        triples_with[c].add((a, b))
        
    # Order numbers by conflict count (ascending)
    # Tie-breaker: random or value
    ordered_numbers = sorted(range(1, N + 1), key=lambda x: (conflict_count[x], x))
    
    def is_valid_addition(x, S_set):
        """Check if adding x to S_set creates a violation."""
        # Check all pairs (u, v) in triples_with[x]
        # If both u and v are in S_set, then violation.
        for u, v in triples_with[x]:
            if u in S_set and v in S_set:
                return False
        return True
        
    def greedy_order(order):
        S = set()
        for x in order:
            if is_valid_addition(x, S):
                S.add(x)
        return S
        
    # Greedy construction
    best_set = greedy_order(ordered_numbers)
    best_size = len(best_set)
    
    # Local Search
    # Try to swap elements to improve size?
    # Actually, if we use the "min conflict" greedy, we might already be close to optimal.
    # Let's try a simple local search:
    # 1. Try to add any element not in S.
    # 2. If none can be added, try swapping: remove x, add y.
    
    def local_search(S, time_limit):
        S = set(S)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # 1. Try to add any missing element
            added_any = False
            for y in range(1, N + 1):
                if y not in S and is_valid_addition(y, S):
                    S.add(y)
                    improved = True
                    added_any = True
                    break
            if added_any:
                continue
                
            # 2. Try swaps: remove x, add y
            # To be efficient, we only consider y that have low conflict count?
            # Or just iterate all y not in S, and all x in S.
            # Optimization: Only try swaps if it might help.
            # Since we want to maximize size, swaps don't increase size directly, 
            # but they might allow adding more elements later?
            # Actually, if we can't add anything, swapping one for another keeps size same.
            # Unless the swap allows adding a 3rd element?
            # Let's stick to: if we can add, we add. 
            # If we can't add, we are stuck.
            # However, the greedy might have made a bad choice that blocks many elements.
            # A swap might open up space.
            
            # Try random swaps to escape local minima?
            # Or systematic swaps.
            # Let's try systematic: for each y not in S, for each x in S:
            # If is_valid_addition(y, S \ {x}), then swap.
            
            swapped = False
            # Iterate y in order of conflict count (low to high) to prefer "easy" additions
            candidates_y = [y for y in range(1, N + 1) if y not in S]
            candidates_y.sort(key=lambda y: conflict_count[y])
            
            for y in candidates_y:
                if time.time() > time_limit:
                    break
                # We need to check if there exists x in S such that S \ {x} + {y} is valid.
                # This is equivalent to: is there an x in S that "blocks" y?
                # y is blocked by S if there is a triple (y, u, v) with u, v in S.
                # If we remove x, y is unblocked if x is one of the blockers?
                # No, if (y, u, v) is a triple, we need to remove u OR v.
                # So if we can find any x in S such that removing x breaks ALL triples involving y and S\{x},
                # then we can add y.
                
                # Simpler: Just try removing each x and check.
                # But that's expensive.
                
                # Let's just do a few random swaps or stop.
                # Given time constraints, simple addition is the main driver.
                # The greedy with sorted order is likely very good.
                
                pass
            
        return S
        
    # Run local search
    if time.time() < deadline - 5:
        S_improved = local_search(best_set, deadline - 5)
        if len(S_improved) > best_size:
            best_set = S_improved
            best_size = len(best_set)
            
    # Random restarts with the sorted order (shuffled slightly?)
    # Or just return the best greedy result.
    # The sorted greedy is deterministic.
    # Let's try a few random perturbations of the order.
    
    for _ in range(10):
        if time.time() > deadline - 5:
            break
        # Perturb order: swap a few elements
        order = list(ordered_numbers)
        for _ in range(5):
            i, j = random.sample(range(len(order)), 2)
            order[i], order[j] = order[j], order[i]
        
        S_rand = greedy_order(order)
        if len(S_rand) > best_size:
            best_set = S_rand
            best_size = len(best_set)
            
            # Local search on new best
            if time.time() < deadline - 5:
                S_improved = local_search(best_set, deadline - 5)
                if len(S_improved) > best_size:
                    best_set = S_improved
                    best_size = len(best_set)
                    
    return sorted(best_set)
# EVOLVE-BLOCK-END
