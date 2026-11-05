import time
import random
import math

def construct():
    deadline = time.time() + 55  # Safety margin
    
    N = 180
    all_nums = set(range(1, N + 1))
    
    # Precompute all reciprocal triples (x, y, z) with x < y < z
    # 1/x = 1/y + 1/z  =>  (y-x)(z-x) = x^2
    triples = []
    for x in range(1, N + 1):
        x2 = x * x
        d = 1
        while d * d <= x2:
            if x2 % d == 0:
                # Case 1: y - x = d, z - x = x2/d
                y1 = x + d
                z1 = x + x2 // d
                if y1 <= N and z1 <= N and y1 != z1:
                    if y1 < z1:
                        triples.append((x, y1, z1))
                    else:
                        triples.append((x, z1, y1))
                
                # Case 2: y - x = x2/d, z - x = d
                y2 = x + x2 // d
                z2 = x + d
                if y2 <= N and z2 <= N and y2 != z2:
                    if y2 < z2:
                        t = (x, y2, z2)
                        if t not in triples: # Avoid duplicates if d == x2/d
                            triples.append(t)
                    else:
                        t = (x, z2, y2)
                        if t not in triples:
                            triples.append(t)
            d += 1
            
    # Remove duplicates just in case
    triples = list(set(triples))
    
    # Function to count violations
    def count_violations(S):
        count = 0
        for x, y, z in triples:
            if x in S and y in S and z in S:
                count += 1
        return count

    # Function to calculate cost (to be minimized)
    # We want 0 violations and max size.
    # Cost = K * violations - size
    K = 10000
    
    def get_cost(S):
        return K * count_violations(S) - len(S)

    # Initial Greedy Solution
    # Try multiple random orderings to find a good start
    best_initial_S = set()
    best_initial_cost = float('inf')
    
    random.seed(42)
    
    for _ in range(20): # Try 20 random orderings
        order = list(all_nums)
        random.shuffle(order)
        S = set()
        for a in order:
            # Check if adding a creates a violation
            # We can just add and check cost, but greedy is faster if we check conflicts
            # Since triples are precomputed, we can check if any triple containing 'a' is completed
            # But 'a' might be x, y, or z.
            # Simpler: just add and check if count_violations(S | {a}) > count_violations(S)
            # This is O(|T|) which is small.
            new_S = S | {a}
            if count_violations(new_S) == 0:
                S = new_S
        
        cost = get_cost(S)
        if cost < best_initial_cost:
            best_initial_cost = cost
            best_initial_S = set(S)
            
    S = set(best_initial_S)
    
    # Simulated Annealing
    # Start temperature
    T = 1.0
    cooling = 0.99
    min_T = 1e-4
    
    best_S = set(S)
    best_cost = get_cost(S)
    
    while time.time() < deadline:
        if T < min_T:
            T = min_T
            # Restart with perturbation if stuck?
            # For now, just keep cooling or reset T if we want to explore more
            # Let's reset T periodically to allow larger moves
            if random.random() < 0.1:
                T = 0.5
        
        # Pick a random move: add or remove a random element
        # Bias towards adding if size is small?
        # Just pick random element from 1..180
        e = random.randint(1, N)
        
        if e in S:
            S.remove(e)
        else:
            S.add(e)
            
        new_cost = get_cost(S)
        delta_cost = new_cost - best_cost
        
        # Accept if better or with probability exp(-delta/T)
        if delta_cost < 0 or (T > 0 and random.random() < math.exp(-delta_cost / T)):
            if new_cost < best_cost:
                best_cost = new_cost
                best_S = set(S)
        else:
            # Revert
            if e in S:
                S.remove(e)
            else:
                S.add(e)
                
        T *= cooling
        
    # Final Repair: Ensure best_S has 0 violations
    # If best_S has violations, remove elements to fix them
    # Greedy removal
    S = set(best_S)
    while count_violations(S) > 0:
        # Find a violating triple
        viol = None
        for x, y, z in triples:
            if x in S and y in S and z in S:
                viol = (x, y, z)
                break
        if viol is None:
            break
        # Remove one element from the triple randomly or the one with most conflicts
        # Just remove random for now
        x, y, z = viol
        to_remove = random.choice([x, y, z])
        S.remove(to_remove)
        
    # Final Extension: Try to add more elements
    # Local search to increase size
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for a in range(1, N + 1):
            if a not in S:
                new_S = S | {a}
                if count_violations(new_S) == 0:
                    S = new_S
                    changed = True
                    
    return sorted(S)
