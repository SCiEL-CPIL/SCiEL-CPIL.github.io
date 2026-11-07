import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin
    
    N = 128
    
    # Strategy: Start with high numbers {64..128} which is valid (size 65).
    # Then greedily add smaller numbers 1..63.
    # Then local search.
    
    S = set(range(64, 129))
    
    def is_valid_with_new(x, S):
        """
        Check if adding x to S creates a violation.
        Only need to check triples containing x.
        Triple {x, u, v} with u < v, u,v in S.
        Sums: x+u, x+v, u+v must all be in S_new = S U {x}.
        """
        S_new = S | {x}
        # Optimization: 
        # If u, v >= 64, u+v > 128, so u+v not in S_new. No violation.
        # So at least one of u, v must be < 64.
        # Since we are adding x in increasing order (1 to 63), 
        # any 'small' u, v in S are < x.
        
        # Iterate over u in S
        # We need u < v.
        # To be fast, we can just iterate all pairs? 
        # |S| is at most 128. Pairs ~ 8000. 
        # For each pair, 3 lookups. 24k ops. 
        # Doing this 63 times is ~1.5M ops. Fast enough.
        
        for u in S:
            for v in S:
                if u >= v:
                    continue
                # Check if {x, u, v} is a valid triple that causes violation
                # Condition: x+u in S_new, x+v in S_new, u+v in S_new
                # And x, u, v distinct. (They are, since x not in S, u!=v)
                if (x + u) in S_new and (x + v) in S_new and (u + v) in S_new:
                    return False
        return True

    # Greedy Add
    for x in range(1, 64):
        if time.time() > deadline - 10:
            break
        if is_valid_with_new(x, S):
            S.add(x)
            
    best_S = set(S)
    best_size = len(S)
    
    # Local Search: Try to add remaining numbers
    remaining = [x for x in range(1, 129) if x not in S]
    
    def try_add(S, x):
        return is_valid_with_new(x, S)
        
    # Iterate a few times to see if order matters or if we can add more
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for x in remaining:
            if x in S:
                continue
            if try_add(S, x):
                S.add(x)
                remaining.remove(x)
                improved = True
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = set(S)
                    
    # Swap Search: Try to replace an element to allow adding another?
    # Or just swap to improve "potential"? 
    # Since we maximize size, and we are at a local max for additions,
    # a swap might allow a larger set? No, swap keeps size same.
    # We want to increase size. 
    # If we can't add anything, we are stuck unless we remove something to add two things?
    # That's a 1-out 2-in move.
    
    # Let's try 1-out 2-in moves.
    # Remove y, add x1, x2.
    # This is expensive.
    
    # Alternative: Random Restart with different orderings.
    
    def run_greedy(order):
        S_curr = set(range(64, 129))
        for x in order:
            if x >= 64:
                continue # Already in S_curr
            if is_valid_with_new(x, S_curr):
                S_curr.add(x)
        return S_curr

    # Try a few random orders for the small numbers
    smalls = list(range(1, 64))
    random.seed(42)
    
    for _ in range(50):
        if time.time() > deadline - 10:
            break
        random.shuffle(smalls)
        S_rand = run_greedy(smalls)
        if len(S_rand) > best_size:
            best_size = len(S_rand)
            best_S = S_rand
            
    # Final verification of best_S
    # Full check to be safe
    S_check = best_S
    valid = True
    # O(N^3) check is ~2M ops, takes ~0.2s.
    # We have ~5s left. Safe.
    elems = list(S_check)
    for i in range(len(elems)):
        for j in range(i+1, len(elems)):
            for k in range(j+1, len(elems)):
                a, b, c = elems[i], elems[j], elems[k]
                if (a+b in S_check) and (a+c in S_check) and (b+c in S_check):
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
            
    if not valid:
        # Fallback to a known valid set
        best_S = set(range(64, 129))
        
    result = [int(x) for x in best_S]
    return result
