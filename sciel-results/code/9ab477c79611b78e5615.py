"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill
    
    N = 180
    # Precompute all forbidden triples (a, b, c) distinct, 1/a = 1/b + 1/c
    # This means c = ab/(a+b), so for given a,b: c = ab/(a+b) must be integer
    # Also need a, b, c all distinct and in [1, N]
    # Note: 1/a = 1/b + 1/c => c = ab/(a+b). Since a,b,c are positive integers,
    # ab must be divisible by a+b.
    
    # Let's enumerate all triples (a, b, c) with a < b < c to avoid duplicates
    # but the condition is symmetric in b and c, and a is the "sum" side.
    # Actually: 1/a = 1/b + 1/c. So a is the smallest in value of 1/x, meaning a is largest.
    # Wait: 1/a = 1/b + 1/c. Since 1/b > 0 and 1/c > 0, we have 1/a < 1/b and 1/a < 1/c,
    # so a > b and a > c. Thus a is the largest element.
    
    # So for each pair (b, c) with b < c (or just all pairs), compute a = bc/(b+c)
    # and check if a is integer, a > c, a <= N, and a, b, c are distinct.
    
    forbidden = []  # list of (a, b, c) where a > b, a > c, 1/a = 1/b + 1/c
    for b in range(1, N + 1):
        for c in range(b + 1, N + 1):
            num = b * c
            den = b + c
            if num % den == 0:
                a = num // den
                if a > c and a <= N:
                    forbidden.append((a, b, c))
    
    # For each element x in [1, N], store the set of triples it participates in
    # triple_index: x -> list of indices into forbidden list
    triple_index = {x: [] for x in range(1, N + 1)}
    for idx, (a, b, c) in enumerate(forbidden):
        triple_index[a].append(idx)
        triple_index[b].append(idx)
        triple_index[c].append(idx)
    
    # We want to maximize |S| such that no triple (a,b,c) has all three in S.
    # This is a 3-uniform hypergraph independent set problem.
    
    # Start with a greedy construction: include elements in order, skipping if it creates a forbidden triple.
    # But first, let's try a smarter initial solution.
    
    # Strategy: 
    # 1. Greedy initial solution
    # 2. Local search with 1-flip and 2-swap moves
    # 3. Simulated annealing or hill climbing
    
    # Greedy: process elements from N down to 1, add if no forbidden triple is completed.
    # Actually, let's think about which elements are "expensive" (participate in many triples).
    
    # Count how many triples each element is in
    triple_count = {x: len(triple_index[x]) for x in range(1, N + 1)}
    
    # Greedy: sort by triple_count descending, add if possible
    # Or sort ascending (add "cheap" elements first)
    
    # Let's try: start with all elements, then remove to fix conflicts.
    # Or: greedy add in order of least constraint.
    
    # I'll use a greedy approach: process elements in order of increasing triple_count,
    # add each if it doesn't complete any forbidden triple with already-added elements.
    
    def is_valid_set(S):
        """Check if set S has no forbidden triple."""
        S_set = set(S)
        for idx, (a, b, c) in enumerate(forbidden):
            if a in S_set and b in S_set and c in S_set:
                return False
        return True
    
    def greedy_initial():
        """Greedy: add elements in order of increasing triple_count."""
        S = set()
        # Track which triples are "active" (all three elements in S)
        # For efficiency, track count of elements in each triple that are in S
        triple_count_in_S = [0] * len(forbidden)
        
        # Order elements by triple_count ascending
        elements = list(range(1, N + 1))
        elements.sort(key=lambda x: triple_count[x])
        
        for x in elements:
            # Check if adding x completes any forbidden triple
            can_add = True
            for tidx in triple_index[x]:
                if triple_count_in_S[tidx] == 2:
                    can_add = False
                    break
            if can_add:
                S.add(x)
                for tidx in triple_index[x]:
                    triple_count_in_S[tidx] += 1
        return S
    
    S = greedy_initial()
    best_S = set(S)
    best_size = len(S)
    
    # Local search: try 1-flip (add/remove) and 2-swap (remove one, add another)
    # Use simulated annealing style to escape local optima
    
    def get_conflicts(S):
        """Return set of triple indices that are fully contained in S."""
        S_set = S
        conflicts = set()
        for idx, (a, b, c) in enumerate(forbidden):
            if a in S_set and b in S_set and c in S_set:
                conflicts.add(idx)
        return conflicts
    
    # For faster checking, maintain a counter per triple
    def count_in_S(S):
        """Return list of counts of elements of each triple that are in S."""
        counts = [0] * len(forbidden)
        for x in S:
            for tidx in triple_index[x]:
                counts[tidx] += 1
        return counts
    
    # Simulated annealing
    def simulate_annealing(S, time_limit):
        S = set(S)
        counts = count_in_S(S)
        current_size = len(S)
        best = set(S)
        best_size = current_size
        
        # Temperature schedule
        T = 5.0
        cooling = 0.9995
        
        while time.time() < time_limit:
            T *= cooling
            if T < 0.01:
                break
            
            # Random move: either add, remove, or swap
            move_type = random.random()
            
            if move_type < 0.33:
                # Add an element not in S
                not_in_S = [x for x in range(1, N + 1) if x not in S]
                if not not_in_S:
                    continue
                x = random.choice(not_in_S)
                # Delta: +1 to size, but may create new conflicts
                new_conflicts = 0
                for tidx in triple_index[x]:
                    if counts[tidx] == 2:
                        new_conflicts += 1
                # Accept if size increases and no new conflicts, or with probability based on T
                if new_conflicts == 0:
                    # Always accept (improves size, no conflicts)
                    S.add(x)
                    for tidx in triple_index[x]:
                        counts[tidx] += 1
                    current_size += 1
                    if current_size > best_size:
                        best_size = current_size
                        best = set(S)
                else:
                    # Accept with probability exp(-new_conflicts / T)
                    if random.random() < math.exp(-new_conflicts / T):
                        S.add(x)
                        for tidx in triple_index[x]:
                            counts[tidx] += 1
                        current_size += 1
                        if current_size > best_size:
                            best_size = current_size
                            best = set(S)
            
            elif move_type < 0.66:
                # Remove an element from S
                if not S:
                    continue
                x = random.choice(list(S))
                # Delta: -1 to size, may remove conflicts
                removed_conflicts = 0
                for tidx in triple_index[x]:
                    if counts[tidx] == 3:
                        removed_conflicts += 1
                # Always accept if it removes conflicts, or with probability
                if removed_conflicts > 0:
                    S.remove(x)
                    for tidx in triple_index[x]:
                        counts[tidx] -= 1
                    current_size -= 1
                else:
                    # Accept with probability exp(1 / T) (since size decreases)
                    if random.random() < math.exp(1.0 / T):
                        S.remove(x)
                        for tidx in triple_index[x]:
                            counts[tidx] -= 1
                        current_size -= 1
            
            else:
                # Swap: remove x from S, add y not in S
                if not S:
                    continue
                x = random.choice(list(S))
                not_in_S = [z for z in range(1, N + 1) if z not in S]
                if not not_in_S:
                    continue
                y = random.choice(not_in_S)
                
                # Compute delta in size: 0 (swap)
                # Compute delta in conflicts
                # Remove x: for each triple containing x, count decreases
                # Add y: for each triple containing y, count increases
                # Net change in number of fully-satisfied triples
                
                # This is complex; approximate by checking
                # For simplicity, just try the swap and check
                S.remove(x)
                for tidx in triple_index[x]:
                    counts[tidx] -= 1
                S.add(y)
                for tidx in triple_index[y]:
                    counts[tidx] += 1
                
                # Check if any new conflict was created
                new_conflict = False
                for tidx in triple_index[y]:
                    if counts[tidx] == 3:
                        new_conflict = True
                        break
                
                if new_conflict:
                    # Revert with probability based on T
                    if random.random() >= math.exp(-1.0 / T):
                        S.remove(y)
                        for tidx in triple_index[y]:
                            counts[tidx] -= 1
                        S.add(x)
                        for tidx in triple_index[x]:
                            counts[tidx] += 1
                else:
                    if current_size > best_size:
                        best_size = current_size
                        best = set(S)
        
        return best, best_size
    
    # Run simulated annealing
    remaining_time = deadline - time.time()
    if remaining_time > 2.0:
        best_S, best_size = simulate_annealing(S, deadline - 1.0)
    
    # Final local search: try all 1-adds and 1-removals greedily
    # First, try to add any element that doesn't create a conflict
    changed = True
    while changed and time.time() < deadline - 1.0:
        changed = False
        counts = count_in_S(best_S)
        for x in range(1, N + 1):
            if x not in best_S:
                can_add = True
                for tidx in triple_index[x]:
                    if counts[tidx] == 2:
                        can_add = False
                        break
                if can_add:
                    best_S.add(x)
                    best_size += 1
                    changed = True
                    counts = count_in_S(best_S)
                    break
    
    # Try swaps
    if time.time() < deadline - 1.0:
        counts = count_in_S(best_S)
        for x in list(best_S):
            for y in range(1, N + 1):
                if y in best_S:
                    continue
                # Try swapping x for y
                # Remove x, add y
                # Check if valid
                S_test = set(best_S)
                S_test.remove(x)
                S_test.add(y)
                # Quick check: does this create any new conflict?
                valid = True
                for tidx in triple_index[y]:
                    a, b, c = forbidden[tidx]
                    if a in S_test and b in S_test and c in S_test:
                        valid = False
                        break
                if valid and len(S_test) == len(best_S):
                    # Same size, but might enable further additions
                    # For now, skip same-size swaps unless they enable growth
                    pass
        
    # Verify the result
    assert is_valid_set(best_S), "Result is not valid!"
    assert len(best_S) > 0, "Result is empty!"
    
    return list(best_S)
# EVOLVE-BLOCK-END
