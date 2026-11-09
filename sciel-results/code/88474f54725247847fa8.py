"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from itertools import combinations

N = 128
ALL_MASK = (1 << N) - 1

def _lcm(a, b):
    return a * b // gcd(a, b)

def _is_bad_triple(a, b, c):
    return _lcm(a, b) == _lcm(a, c) == _lcm(b, c)

# Precompute conflict data
# For each x, and each potential pair member a, we want a bitmask of b's such that {x,a,b} is bad.
# Actually, simpler: For each x, store a list of (a, mask_b) where mask_b has bits for all b such that {x,a,b} is bad.
# But we need to check if ANY (a,b) in S is bad.
# Condition: x conflicts with S if there exist a, b in S such that {x,a,b} is bad.
# Let's precompute: bad_triples = list of (a,b,c) sorted.
# For fast checking, let's create for each x, a list of pairs (a, b) that form a bad triple with x.
# To speed up checking "does S contain any pair from P_x", we can use the bitmask trick.
# For a fixed x, let P_x be the set of pairs (a,b).
# We can precompute for each x, an array conflict_mask[x][a] which is a bitmask of all b such that (a,b) in P_x.
# Then x conflicts with S if:
#   exists a in S such that (conflict_mask[x][a] & S) != 0

conflict_masks = [[0] * N for _ in range(N)]

def _precompute_conflicts():
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                if _is_bad_triple(a + 1, b + 1, c + 1):
                    # Indices are 0-based internally
                    # Add (b,c) to conflict list for a
                    # Add (a,c) to conflict list for b
                    # Add (a,b) to conflict list for c
                    
                    # For a: pair (b,c)
                    conflict_masks[a][b] |= (1 << c)
                    conflict_masks[a][c] |= (1 << b)
                    
                    # For b: pair (a,c)
                    conflict_masks[b][a] |= (1 << c)
                    conflict_masks[b][c] |= (1 << a)
                    
                    # For c: pair (a,b)
                    conflict_masks[c][a] |= (1 << b)
                    conflict_masks[c][b] |= (1 << a)

_precompute_conflicts()

def _conflicts_with_set(x, S_mask):
    """
    Check if adding x to S_mask creates a bad triple.
    x is 0-based index. S_mask is an integer bitmask.
    """
    # We need to check if there exist a, b in S_mask such that {x, a, b} is bad.
    # This is equivalent to: exists a in S_mask such that (conflict_masks[x][a] & S_mask) != 0
    # To avoid iterating all N, we can iterate only set bits in S_mask?
    # Or iterate all a? N=128 is small.
    # Let's iterate all a from 0 to N-1.
    # Optimization: Precompute list of 'a's for each x that have non-zero conflict masks.
    pass

# Let's refine _conflicts_with_set for speed
# Precompute for each x, the list of 'a' such that conflict_masks[x][a] != 0
conflict_keys = [[] for _ in range(N)]
for x in range(N):
    for a in range(N):
        if conflict_masks[x][a] != 0:
            conflict_keys[x].append(a)

def _conflicts_with_set_fast(x, S_mask):
    """
    Fast check using precomputed keys and bitmasks.
    """
    for a in conflict_keys[x]:
        if (S_mask >> a) & 1:
            # If a is in S, check if any b in conflict_masks[x][a] is in S
            if conflict_masks[x][a] & S_mask:
                return True
    return False

def _greedy_init(order):
    S_mask = 0
    for x in order:
        if not _conflicts_with_set_fast(x, S_mask):
            S_mask |= (1 << x)
    return S_mask

def _local_search(S_mask, deadline):
    """
    Perform local search on S_mask.
    Tries to add elements, then 1-swaps, then 2-swaps.
    Returns improved S_mask.
    """
    current_size = bin(S_mask).count('1')
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Phase 1: Add any element
        for x in range(N):
            if not (S_mask >> x) & 1:
                if not _conflicts_with_set_fast(x, S_mask):
                    S_mask |= (1 << x)
                    current_size += 1
                    improved = True
                    break
        
        if improved:
            continue

        # Phase 2: 1-Swap (remove one, add one)
        # Try removing each element and see if we can add two? No, 1-swap is remove 1 add 1.
        # To increase size, we need to remove 0 and add 1 (handled in Phase 1).
        # Or remove 1 and add 2 (net +1).
        # Or remove 2 and add 3 (net +1).
        
        # Let's try "Remove 1, Add 2" (net +1)
        found_swap = False
        # Get list of elements in S
        S_list = [i for i in range(N) if (S_mask >> i) & 1]
        
        for a in S_list:
            S_mask_no_a = S_mask & ~(1 << a)
            # Try to add two elements x, y
            # This is O(N^2) per removal. 128 * 128^2 ~ 2M ops. Might be slow in Python loop.
            # Let's try "Remove 2, Add 3" later.
            # For now, simple 1-swap (remove 1, add 1) doesn't increase size.
            # We want to increase size.
            
            # Try adding one element first to S_mask_no_a?
            # If we can add one element to S_mask_no_a, and that element is not a, we are back to size-1+1 = size.
            # We need to add 2 elements to S_mask_no_a to get size+1.
            
            # This is getting complex. Let's use a simpler heuristic:
            # Try all pairs (x,y) not in S. If we can remove a set of elements R such that x,y can be added, and |R| < 2, we gain.
            # Specifically, if we remove 1 element a, and can add x,y, we gain 1.
            
            # To speed up, we can just try to find any x not in S that can be added to S_mask_no_a.
            # If we find one, we are at size-1+1=size. We need to find another one y that can be added to S_mask_no_a | (1<<x).
            
            # This nested search is expensive. Let's stick to 1-swap (replace a with x) for stability, 
            # and rely on Phase 1 for growth. If Phase 1 fails, we are in a local optimum for additions.
            # We need to break out.
            
            # Let's try a "2-opt" style: Remove 2 elements, add 3? Or Remove 1, Add 2.
            # Given time constraints, let's implement a simple "Remove 1, Add 2" check.
            
            # For each a in S:
            #   For each x not in S:
            #     If x can be added to S \ {a}:
            #       For each y not in S, y != x:
            #         If y can be added to S \ {a} \ {x? no, S \ {a} + x}:
            #           Update S = S \ {a} + {x, y}
            
            # This is O(N * N * N) = 128^3 ~ 2M. In Python, 2M simple ops is ~0.1s.
            # We have 60s. We can do this.
            
            for x in range(N):
                if (S_mask_no_a >> x) & 1: continue
                if _conflicts_with_set_fast(x, S_mask_no_a): continue
                
                # Try to add y to S_mask_no_a | (1<<x)
                S_mask_temp = S_mask_no_a | (1 << x)
                for y in range(N):
                    if (S_mask_temp >> y) & 1: continue
                    if _conflicts_with_set_fast(y, S_mask_temp): continue
                    
                    # Found a valid move: remove a, add x, y
                    S_mask = S_mask_temp | (1 << y)
                    current_size += 1
                    found_swap = True
                    break
                if found_swap: break
            
            if found_swap: break
        
        if found_swap:
            improved = True
            continue
        
        # If no improvement, we might be stuck. 
        # For this version, we stop local search if no size increase is found.
        # In a more advanced version, we would do simulated annealing steps here.
        break
        
    return S_mask

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    all_elements = list(range(N))
    
    # Generate multiple orderings
    orderings = []
    
    # 1. Natural order
    orderings.append(all_elements)
    
    # 2. Reverse order
    orderings.append(list(reversed(all_elements)))
    
    # 3. Primes first
    def _is_prime(n):
        if n < 2: return False
        if n < 4: return True
        if n % 2 == 0 or n % 3 == 0: return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0: return False
            i += 6
        return True
        
    primes = [i for i in all_elements if _is_prime(i+1)]
    non_primes = [i for i in all_elements if not _is_prime(i+1)]
    orderings.append(primes + non_primes)
    orderings.append(non_primes + primes)
    
    # 4. Random orderings
    random.seed(42)
    for _ in range(100):
        orderings.append(random.sample(all_elements, N))
    
    best_mask = 0
    best_size = 0
    
    for order in orderings:
        if time.time() > deadline:
            break
        
        S_mask = _greedy_init(order)
        
        # Local search
        S_mask = _local_search(S_mask, deadline)
        
        size = bin(S_mask).count('1')
        if size > best_size:
            best_size = size
            best_mask = S_mask
    
    # If we have time, do some random perturbations (ILS)
    while time.time() < deadline - 5:
        # Perturb best_mask
        S_mask = best_mask
        # Randomly remove 2-3 elements
        S_list = [i for i in range(N) if (S_mask >> i) & 1]
        if len(S_list) < 3:
            break
        for _ in range(random.randint(2, 3)):
            rem = random.choice(S_list)
            S_mask &= ~(1 << rem)
            S_list.remove(rem)
        
        # Run local search on perturbed set
        S_mask = _local_search(S_mask, deadline)
        
        size = bin(S_mask).count('1')
        if size > best_size:
            best_size = size
            best_mask = S_mask
    
    # Convert best_mask to list of 1-based indices
    result = []
    for i in range(N):
        if (best_mask >> i) & 1:
            result.append(i + 1)
            
    if not result:
        result = [1]
        
    return result
# EVOLVE-BLOCK-END
