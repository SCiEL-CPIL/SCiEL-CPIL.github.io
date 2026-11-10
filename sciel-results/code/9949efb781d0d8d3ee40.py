import time
import math
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    N = 256
    
    # Precompute conflicts using bitsets for O(1) intersection checks
    # conflict_bits[i] is an integer where bit j is set if i and j conflict
    conflict_bits = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g = math.gcd(i, j)
            x = i // g
            y = j // g
            if g % (x + y) == 0:
                conflict_bits[i] |= (1 << j)
                conflict_bits[j] |= (1 << i)
    
    # Precompute conflict counts for each element
    conflict_counts = [bin(conflict_bits[i]).count('1') for i in range(N + 1)]
    
    # Helper: check if element e conflicts with any element in set S using bitset
    def has_conflict_bits(e, S_bits):
        return (conflict_bits[e] & S_bits) != 0
    
    # Build S_bits from a set
    def set_to_bits(S):
        bits = 0
        for x in S:
            bits |= (1 << x)
        return bits
    
    # Greedy fill: add non-conflicting elements in order of fewest conflicts
    def greedy_fill(S, S_bits, candidates, deadline):
        """Greedily add non-conflicting elements from candidates."""
        changed = True
        while changed and time.time() < deadline:
            changed = False
            new_candidates = []
            for e in candidates:
                if time.time() > deadline:
                    break
                if not has_conflict_bits(e, S_bits):
                    S.add(e)
                    S_bits |= (1 << e)
                    changed = True
                else:
                    new_candidates.append(e)
            candidates = new_candidates
        return S, S_bits, candidates
    
    # Local search: 1-for-k swaps (remove one element, add multiple)
    def local_search_1_for_k(S, S_bits, evens, deadline, k=3):
        """Try removing one element and adding up to k others."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            S_list = list(S)
            random.shuffle(S_list)
            
            for s_elem in S_list:
                if time.time() > deadline:
                    return S, S_bits, evens
                
                removable_bits = S_bits ^ (1 << s_elem)
                
                # Find elements that are now free
                free_elems = []
                for e in evens:
                    if not has_conflict_bits(e, removable_bits):
                        free_elems.append(e)
                
                if not free_elems:
                    continue
                
                # Sort by conflict count
                free_elems.sort(key=lambda x: conflict_counts[x])
                
                new_S_bits = removable_bits
                added = 0
                for e in free_elems:
                    if time.time() > deadline:
                        break
                    if not has_conflict_bits(e, new_S_bits):
                        new_S_bits |= (1 << e)
                        added += 1
                    if added >= k:
                        break
                
                # If we added more elements than we removed (1), it's an improvement
                if added > 1:
                    new_S = set()
                    for i in range(1, N + 1):
                        if new_S_bits & (1 << i):
                            new_S.add(i)
                    S = new_S
                    S_bits = new_S_bits
                    evens = [e for e in evens if e not in new_S]
                    improved = True
                    break
        
        return S, S_bits, evens
    
    # Local search: 2-for-k swaps (remove two elements, add multiple)
    def local_search_2_for_k(S, S_bits, evens, deadline, k=4):
        """Try removing two elements and adding up to k others."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            S_list = list(S)
            n = len(S_list)
            
            # Limit pairs to avoid excessive runtime
            if n > 150:
                # Sample a subset of pairs
                pairs = random.sample(range(n), min(100, n))
                pair_list = [(pairs[i], pairs[(i+1) % len(pairs)]) for i in range(len(pairs)-1)]
            else:
                pair_list = [(i, j) for i in range(n) for j in range(i+1, n)]
            
            random.shuffle(pair_list)
            
            for idx, (i, j) in enumerate(pair_list):
                if time.time() > deadline:
                    return S, S_bits, evens
                
                s1, s2 = S_list[i], S_list[j]
                removable_bits = S_bits ^ (1 << s1) ^ (1 << s2)
                
                # Find elements that are now free
                free_elems = []
                for e in evens:
                    if not has_conflict_bits(e, removable_bits):
                        free_elems.append(e)
                
                if len(free_elems) < 3:
                    continue
                
                # Sort by conflict count
                free_elems.sort(key=lambda x: conflict_counts[x])
                
                new_S_bits = removable_bits
                added = 0
                for e in free_elems:
                    if time.time() > deadline:
                        break
                    if not has_conflict_bits(e, new_S_bits):
                        new_S_bits |= (1 << e)
                        added += 1
                    if added >= k:
                        break
                
                # If we added more elements than we removed (2), it's an improvement
                if added > 2:
                    new_S = set()
                    for x in range(1, N + 1):
                        if new_S_bits & (1 << x):
                            new_S.add(x)
                    S = new_S
                    S_bits = new_S_bits
                    evens = [e for e in evens if e not in new_S]
                    improved = True
                    break
        
        return S, S_bits, evens
    
    # Run a single attempt with a given initial set
    def run_attempt(initial_S, deadline):
        S = set(initial_S)
        S_bits = set_to_bits(S)
        evens = [i for i in range(1, N + 1) if i not in S]
        evens.sort(key=lambda x: conflict_counts[x])
        
        # Greedy fill
        S, S_bits, evens = greedy_fill(S, S_bits, evens, deadline)
        
        # Local search 1-for-k
        S, S_bits, evens = local_search_1_for_k(S, S_bits, evens, deadline, k=3)
        
        # Local search 2-for-k
        S, S_bits, evens = local_search_2_for_k(S, S_bits, evens, deadline, k=4)
        
        # Final greedy pass
        S, S_bits, evens = greedy_fill(S, S_bits, evens, deadline)
        
        return S
    
    # Strategy 1: Start with all odds (deterministic baseline)
    best_S = run_attempt(set(range(1, N + 1, 2)), deadline)
    best_size = len(best_S)
    
    # Strategy 2: Randomized restarts with different initial seeds
    num_restarts = 5
    for restart in range(num_restarts):
        if time.time() > deadline - 5:  # Leave 5s buffer
            break
        
        # Create a random initial set: start with some odds, add random evens
        initial_S = set(range(1, N + 1, 2))  # Start with all odds
        # Randomly remove some odds and add some evens to create variation
        odds_list = list(range(1, N + 1, 2))
        evens_list = list(range(2, N + 1, 2))
        
        # Remove 10-20 random odds
        num_remove = random.randint(10, 20)
        for _ in range(num_remove):
            if odds_list:
                idx = random.randint(0, len(odds_list) - 1)
                initial_S.discard(odds_list[idx])
                odds_list.pop(idx)
        
        # Add 5-15 random evens
        num_add = random.randint(5, 15)
        for _ in range(num_add):
            if evens_list:
                idx = random.randint(0, len(evens_list) - 1)
                initial_S.add(evens_list[idx])
                evens_list.pop(idx)
        
        # Verify no conflicts in initial set (should be fine since we started with odds)
        # But check just in case
        valid = True
        S_list = list(initial_S)
        for i in range(len(S_list)):
            for j in range(i + 1, len(S_list)):
                if conflict_bits[S_list[i]] & (1 << S_list[j]):
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            continue
        
        S = run_attempt(initial_S, deadline)
        if len(S) > best_size:
            best_size = len(S)
            best_S = S
    
    return list(best_S)
# EVOLVE-BLOCK-END
