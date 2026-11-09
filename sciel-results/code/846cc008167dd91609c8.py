import time
import random
from math import gcd
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def _is_bad_triple(a, b, c):
    return _lcm(a, b) == _lcm(a, c) == _lcm(b, c)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    # Precompute LCM table for all pairs in [1, 128]
    N = 128
    lcm_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]
    
    # Precompute bad triples: for each pair (a,b), find all c such that (a,b,c) is bad
    # conflict_mask[x] = bitmask of elements y such that x and y form a "conflict pair"
    # Actually, we need: for a given set S, can we add x? 
    # x conflicts with S if there exist a,b in S such that (a,b,x) is a bad triple.
    # Let's precompute: conflict_with[x] = set of pairs (a,b) where a,b,x form bad triple
    # Better: for each x, compute which pairs (a,b) in S would be violated by adding x.
    # 
    # Alternative approach using bitsets:
    # For each element x, and for each potential pair (a,b), we know if (a,b,x) is bad.
    # When checking if x can be added to S, we need: no pair (a,b) in S has (a,b,x) bad.
    # 
    # Let's define: for each x, bad_pairs_with_x = set of (a,b) such that (a,b,x) is bad triple
    # Then x can be added to S iff S contains no pair from bad_pairs_with_x
    # 
    # To check this fast: for each x, precompute a "conflict mask" - 
    # but pairs are not single elements. 
    # 
    # Better formulation: 
    # Define conflict[x][y] = set of z such that (x,y,z) is a bad triple.
    # Then when we have set S, x can be added iff for all pairs (a,b) in S, x not in conflict[a][b]
    # 
    # For SA, we want fast add/remove. Let's use a different representation:
    # For each element x, define "forbidden_by[x]" = set of pairs (a,b) that would be violated
    # Actually, let's think of it as: adding x to S is valid iff 
    #   for all a in S: for all b in S, a != b: x not in bad_pair[a][b]
    # 
    # Let's precompute for each x: the set of "witnesses" - pairs that would block x.
    # But for fast checking, let's use: 
    #   For each x, precompute conflict_mask[x] as a bitmask over elements.
    #   But the conflict is pairwise, not unary.
    #
    # Let me use a different approach: 
    # For each pair (a,b), bad_pair[a][b] is the set of c that complete a bad triple.
    # To check if x can be added to S: 
    #   For each a in S, for each b in S where a < b: check x not in bad_pair[a][b]
    # This is O(|S|^2) per check.
    #
    # For SA with many iterations, we need faster. Let's use:
    #   For each x, precompute: which elements y are "compatible" with x in the sense that
    #   they don't form a bad triple with any other element. But this is still pairwise.
    #
    # Alternative: Use the fact that N=128 is small. We can represent S as a bitmask (128 bits).
    # For each x, precompute: block_mask[x] = bitmask of all y such that there exists z in [1,128]
    # where (x,y,z) is a bad triple. Wait, that's not quite right either.
    #
    # Let me think more carefully. The condition for adding x to S is:
    #   No pair (a,b) in S has x in bad_pair[a][b]
    # Equivalently: For all a in S, for all b in S, a<b: x not in bad_pair[a][b]
    #
    # Let's precompute for each x: 
    #   incompatible_pairs[x] = {(a,b) : (a,b,x) is bad triple, a<b}
    # Then x can be added to S iff S contains no pair from incompatible_pairs[x]
    #
    # To check this fast, for each x we can precompute a "conflict signature":
    #   For each a, let conflict_with_a[x] = set of b such that (a,b,x) is bad
    #   Then x can be added to S iff for all a in S: conflict_with_a[x] ∩ S = empty
    #
    # This is still O(|S|) per check if we do it naively.
    #
    # For SA, let's just use the O(|S|^2) check since |S| is at most ~100 and we have 85 seconds.
    # Actually, let's estimate: if we do 1 million SA steps, each O(100^2) = 10000 ops, that's 10^10 ops - too slow.
    #
    # We need faster. Let's use bitsets.
    # Represent S as a Python int (bitmask). 
    # For each x, precompute: 
    #   For each a, conflict[a][x] = bitmask of b such that (a,b,x) is bad triple
    # Then x can be added to S iff: for all a in S: (conflict[a][x] & S) == 0
    # But checking "for all a in S" is still O(|S|).
    #
    # Alternative: precompute for each x: 
    #   total_conflict_mask[x] = OR over all a of (conflict[a][x] shifted or something)
    # This doesn't directly work.
    #
    # Let me try a different angle. Since N=128, we can use Python's int as a 128-bit bitmask.
    # For each x, precompute: 
    #   For each a, let C[a][x] = bitmask of all b such that (a,b,x) is a bad triple
    # Then x is compatible with S iff: for all a in S, C[a][x] & S == 0
    # 
    # We can rewrite: x is incompatible with S iff there exists a in S such that C[a][x] & S != 0
    # 
    # To check this in O(1) or O(log N), we can precompute:
    #   For each x, define D[x] = bitmask of all (a,b) pairs... but pairs don't fit in a single bitmask easily.
    #
    # Let's just use the O(|S|) check with bitmasks. For each a in S (iterate set bits), check C[a][x] & S.
    # This is O(|S|) bit operations, which is fast in Python for |S| up to 100.
    #
    # Actually, let's precompute it differently:
    # For each x, precompute: 
    #   incompat_mask[x] = bitmask of all y such that there exists z where (x,y,z) is bad
    # No, that's not the right condition.
    #
    # OK, let me just use the straightforward approach with the bad_pair table and optimize the inner loop.
    
    # Precompute bad_pair[a][b] = set of c such that (a,b,c) is a bad triple
    bad_pair = [[set() for _ in range(N + 1)] for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            lab = lcm_table[a][b]
            for c in range(1, N + 1):
                if c == a or c == b:
                    continue
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    bad_pair[a][b].add(c)
                    bad_pair[b][a].add(c)
    
    # For fast checking, precompute for each x: 
    #   conflict_with[x] = list of (a, b) pairs such that (a,b,x) is bad
    # Then to check if x can be added to S: check no (a,b) in conflict_with[x] has both a,b in S
    
    # Even better: for each x, precompute a bitmask-based structure
    # Let's use: for each x, and for each a, C[x][a] = bitmask of b such that (a,b,x) is bad
    # Then x can be added to S iff for all a in S: C[x][a] & S == 0
    
    C = [[0] * (N + 1) for _ in range(N + 1)]  # C[x][a] = bitmask of b
    for x in range(1, N + 1):
        for a in range(1, N + 1):
            mask = 0
            for b in range(1, N + 1):
                if b != a and b != x and a != x:
                    if x in bad_pair[a][b]:
                        mask |= (1 << b)
            C[x][a] = mask
    
    def _can_add_bitmask(x, S_mask):
        """Check if x can be added to set represented by S_mask."""
        for a in range(1, N + 1):
            if S_mask & (1 << a):
                if C[x][a] & S_mask:
                    return False
        return True
    
    def _can_add(x, S):
        """Check if x can be added to set S."""
        S_list = list(S)
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i + 1, n):
                b = S_list[j]
                if x in bad_pair[a][b]:
                    return False
        return True
    
    # Build base set
    base_set = set()
    base_set.add(1)
    
    for p in range(2, 129):
        is_prime = True
        if p < 2:
            is_prime = False
        else:
            for i in range(2, int(p**0.5) + 1):
                if p % i == 0:
                    is_prime = False
                    break
        if is_prime:
            power = p
            while power <= 128:
                base_set.add(power)
                power *= p
    
    # Greedy to build initial solution
    remaining = [x for x in range(2, 129) if x not in base_set]
    S = set(base_set)
    for x in sorted(remaining):
        if _can_add(x, S):
            S.add(x)
    
    best_set = set(S)
    best_size = len(S)
    
    # Try random orderings for greedy
    all_candidates = list(remaining)
    for _ in range(20):
        if time.time() > deadline:
            break
        random.shuffle(all_candidates)
        S_current = set(base_set)
        for x in all_candidates:
            if _can_add(x, S_current):
                S_current.add(x)
        if len(S_current) > best_size:
            best_size = len(S_current)
            best_set = set(S_current)
    
    # Simulated Annealing
    if time.time() < deadline - 10:
        # Convert to bitmask representation
        def set_to_mask(S):
            mask = 0
            for x in S:
                mask |= (1 << x)
            return mask
        
        def mask_to_set(mask):
            S = set()
            for x in range(1, N + 1):
                if mask & (1 << x):
                    S.add(x)
            return S
        
        current_mask = set_to_mask(best_set)
        current_set = set(best_set)
        best_sa_mask = current_mask
        best_sa_set = set(best_set)
        best_sa_size = len(best_set)
        
        T = 10.0  # Initial temperature
        T_min = 0.01
        cooling_rate = 0.999
        max_iterations = 500000
        
        for it in range(max_iterations):
            if time.time() > deadline - 5:
                break
            
            # Random move: add or remove a random element
            r = random.random()
            if r < 0.5:
                # Try to add a random element not in current set
                candidates = [x for x in range(1, N + 1) if not (current_mask & (1 << x))]
                if not candidates:
                    continue
                x = random.choice(candidates)
                new_mask = current_mask | (1 << x)
                # Check validity
                if _can_add_bitmask(x, current_mask):
                    new_set = mask_to_set(new_mask)
                    delta = 1  # Size increases
                    # Always accept improving moves
                    current_mask = new_mask
                    current_set = new_set
                    if len(current_set) > best_sa_size:
                        best_sa_size = len(current_set)
                        best_sa_set = set(current_set)
                        best_sa_mask = current_mask
                else:
                    # Invalid move, reject
                    continue
            else:
                # Try to remove a random element from current set
                if not current_set:
                    continue
                x = random.choice(list(current_set))
                new_mask = current_mask & ~(1 << x)
                new_set = mask_to_set(new_mask)
                delta = -1  # Size decreases
                # Accept with probability exp(delta / T)
                if random.random() < (2.718281828 ** (delta / T)):
                    current_mask = new_mask
                    current_set = new_set
                    if len(current_set) > best_sa_size:
                        best_sa_size = len(current_set)
                        best_sa_set = set(current_set)
                        best_sa_mask = current_mask
            
            T = max(T_min, T * cooling_rate)
        
        if best_sa_size > best_size:
            best_size = best_sa_size
            best_set = best_sa_set
    
    # Final validation
    S_list = sorted(best_set)
    valid = True
    for a, b, c in combinations(S_list, 3):
        if _is_bad_triple(a, b, c):
            valid = False
            break
            
    if not valid:
        return list(base_set)
        
    if not best_set:
        return [1]
        
    return list(best_set)
