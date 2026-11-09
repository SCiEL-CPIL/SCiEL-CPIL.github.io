"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    n = 256
    deadline = time.time() + 85.0  # Safety margin before 90s kill
    
    # The proven safe tail: {k, ..., n} is valid if for all a<b<=c in tail, a ∤ (b+c)
    # For the tail [k, n], the worst case is a=k, b=k+1, c=n (or similar)
    # Actually the constraint for the pure tail [k,n]: 
    #   For a=k, we need k ∤ (b+c) for all k<b<=c<=n
    #   b+c ranges from (k+1)+(k+2)=2k+3 to (n-1)+n=2n-1
    #   So we need k ∤ s for all s in [2k+3, 2n-1]
    #   The multiples of k in that range: ceil((2k+3)/k)*k to floor((2n-1)/k)*k
    #   ceil((2k+3)/k) = 2 + ceil(3/k) = 3 for k>=2
    #   So multiples are 3k, 4k, ..., floor((2n-1)/k)*k
    #   We need 3k > 2n-1, i.e., 3k >= 2n, i.e., k >= ceil(2n/3) = ceil(512/3) = 171
    # So [171, 256] is the maximal pure tail.
    
    tail_start = 171
    base_set = set(range(tail_start, n + 1))
    
    # Precompute conflict structures for fast validity checking
    # For each element a, precompute which values x would conflict with a
    # when x is added to a set containing a.
    
    # Key optimization: precompute for each pair (a, b) whether they conflict
    # in each of the three roles. But that's complex. Instead, use bitset approach.
    
    # For fast checking, precompute for each a:
    #   forbidden_sums[a] = set of multiples of a in range [2, 2*n]
    # Then checking if x can be added involves checking if any existing pair
    # forms a forbidden sum with x.
    
    # Even better: for each a, precompute which c values would make (b+c) % a == 0
    # for each b. This is: c ≡ -b (mod a), so c = (-b) % a + k*a
    
    # Let's use a hybrid: keep the set as a Python set for iteration, but
    # precompute residue classes for fast lookup.
    
    # Precompute: for each a in [1,n], for each residue r in [0,a), 
    # the set of values in [1,n] that are ≡ r (mod a)
    residue_classes = {}
    for a in range(1, n + 1):
        residue_classes[a] = [set() for _ in range(a)]
        for v in range(1, n + 1):
            residue_classes[a][v % a].add(v)
    
    def is_valid_addition(S, x):
        """Check if x can be added to set S (as a sorted set).
        Uses precomputed residue classes for speed."""
        # x as 'a' (smallest): x ∤ (b+c) for all b,c in S with x < b <= c
        # For each b in S with b > x, need no c in S with c >= b and (b+c) % x == 0
        # i.e., c ≡ -b (mod x), c >= b, c in S
        # The values of c ≡ -b (mod x) that are >= b: 
        #   c0 = b if b % x == 0, else b + (x - b % x)
        #   then c0 + x, c0 + 2x, ...
        # We can check if any such c is in S using residue_classes[x][(-b) % x]
        
        for b in S:
            if x < b:
                # Find c in S with c >= b and c ≡ -b (mod x)
                target_res = (-b) % x
                candidates_c = residue_classes[x][target_res]
                # Need c >= b
                for c in candidates_c:
                    if c >= b and c in S:
                        return False
        
        # x as 'b' (middle): a ∤ (x+c) for all a,c in S with a < x <= c
        # For each a in S with a < x, need no c in S with c >= x and (x+c) % a == 0
        # i.e., c ≡ -x (mod a), c >= x, c in S
        for a in S:
            if a < x:
                target_res = (-x) % a
                candidates_c = residue_classes[a][target_res]
                for c in candidates_c:
                    if c >= x and c in S:
                        return False
        
        # x as 'c' (largest): a ∤ (b+x) for all a,b in S with a < b <= x
        # For each a in S, need no b in S with a < b <= x and (b+x) % a == 0
        # i.e., b ≡ -x (mod a), a < b <= x, b in S
        for a in S:
            target_res = (-x) % a
            candidates_b = residue_classes[a][target_res]
            for b in candidates_b:
                if a < b <= x and b in S:
                    return False
        
        return True
    
    current_set = set(base_set)
    best_set = set(current_set)
    best_size = len(current_set)
    
    candidates = list(range(1, tail_start))
    
    # Greedy pass: try adding in ascending order (smaller numbers are harder to place)
    for x in sorted(candidates):
        if is_valid_addition(current_set, x):
            current_set.add(x)
            if len(current_set) > best_size:
                best_set = set(current_set)
                best_size = len(current_set)
    
    # Multiple random restarts with different orderings
    for restart in range(100):
        if time.time() > deadline - 15:
            break
        
        current_set = set(base_set)
        cands = list(range(1, tail_start))
        random.shuffle(cands)
        
        for x in cands:
            if time.time() > deadline - 15:
                break
            if is_valid_addition(current_set, x):
                current_set.add(x)
        
        if len(current_set) > best_size:
            best_set = set(current_set)
            best_size = len(current_set)
    
    # Local search: try swaps more aggressively
    # For each element in best_set, try removing it and adding a better element
    for sweep in range(20):
        if time.time() > deadline - 8:
            break
        
        improved = False
        current = set(best_set)
        
        for x in list(current):
            if time.time() > deadline - 8:
                break
            
            # Try removing x and adding a new element
            current.remove(x)
            found_better = False
            for y in range(1, n + 1):
                if y in current:
                    continue
                if is_valid_addition(current, y):
                    current.add(y)
                    if len(current) > len(best_set):
                        best_set = set(current)
                        improved = True
                        found_better = True
                    current.remove(y)
                    if found_better:
                        break
            if not found_better:
                current.add(x)  # put x back
        
        if not improved:
            break
    
    # Second round of local search: try removing pairs and adding two new elements
    # This can sometimes unlock better configurations
    for sweep in range(10):
        if time.time() > deadline - 5:
            break
        
        improved = False
        current = set(best_set)
        elems = list(current)
        
        for i in range(len(elems)):
            if time.time() > deadline - 5:
                break
            for j in range(i + 1, len(elems)):
                if time.time() > deadline - 5:
                    break
                
                x1, x2 = elems[i], elems[j]
                current.remove(x1)
                current.remove(x2)
                
                found_better = False
                for y1 in range(1, n + 1):
                    if y1 in current or y1 == x1 or y1 == x2:
                        continue
                    if is_valid_addition(current, y1):
                        current.add(y1)
                        for y2 in range(1, n + 1):
                            if y2 in current or y2 == x1 or y2 == x2 or y2 == y1:
                                continue
                            if is_valid_addition(current, y2):
                                current.add(y2)
                                if len(current) > len(best_set):
                                    best_set = set(current)
                                    improved = True
                                    found_better = True
                                current.remove(y2)
                                if found_better:
                                    break
                        current.remove(y1)
                        if found_better:
                            break
                
                if not found_better:
                    current.add(x1)
                    current.add(x2)
        
        if not improved:
            break
    
    # Final verification
    def is_valid_full(S):
        S_list = sorted(S)
        for i, a in enumerate(S_list):
            for j in range(i + 1, len(S_list)):
                b = S_list[j]
                for k in range(j, len(S_list)):
                    c = S_list[k]
                    if a < b <= c and (b + c) % a == 0:
                        return False
        return True
    
    if not is_valid_full(best_set):
        # Fallback to safe tail
        best_set = set(range(171, 257))
    
    return sorted(list(best_set))
# EVOLVE-BLOCK-END
