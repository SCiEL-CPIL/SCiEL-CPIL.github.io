"""Combined program: optimized greedy with local search for no-divisor-of-triple-sums."""
import time
import random
from itertools import combinations

N = 192
MAX_CARD = 48

def _is_valid_add(c, S):
    """
    Check if c can be added to set S (list of distinct ints).
    1. c must not divide sum of any 3 distinct elements in S.
    2. For each a in S, a must not divide (c + x + y) for any distinct x, y in S\{a}.
    """
    n = len(S)
    if n < 2:
        # If n < 2, we can't form a triple from S, and for check 2 we need 2 others.
        # If n < 3, check 1 is vacuously true (no 3 distinct others).
        # If n < 2, check 2 is vacuously true (no 2 distinct others excluding a).
        # Wait, if n=2, S={a,b}. Check 2 for a: others={b}. Need 2 distinct from others. Impossible.
        # So for n < 2, valid. For n=2, valid.
        # Actually, if n=3, S={a,b,c}. Check 1 for c: sum(a,b) needs 3rd? No, sum of 3 distinct from S.
        # If S has 3 elements, sum of 3 distinct is a+b+c. Does c divide a+b+c? i.e. c divides a+b.
        # Check 2 for a: others={b,c}. Sum c+b. Does a divide c+b?
        
        # Let's just run the general loops, they will handle small n correctly by not finding violations.
        pass

    # Check 1: c divides x+y+z for x,y,z in S, x!=y!=z
    # Optimization: Precompute pair sums? Or just iterate.
    # Since n is small (<=48), O(n^3) is ~ 17k ops. Fast enough.
    # But we can optimize by breaking early.
    
    # A faster check for Check 1:
    # For each pair (x, y) in S, we need to know if there exists z in S, z!=x, z!=y, such that (x+y+z) % c == 0.
    # This is equivalent to: does S contain an element z such that z % c == -(x+y) % c?
    # And z != x, z != y.
    
    S_set = set(S)
    
    # Check 1
    for i in range(n):
        x = S[i]
        for j in range(i + 1, n):
            y = S[j]
            target = (-(x + y)) % c
            # We need z in S such that z % c == target, z != x, z != y
            # Iterate through S to find such z? That's O(n) per pair -> O(n^3).
            # Alternatively, precompute a map from residue to list of elements?
            # Residues are 0..c-1. c can be up to 192.
            # Let's just do the triple loop for simplicity and correctness first, then optimize if needed.
            # Given the deadline, let's try the triple loop with early exit.
            
            # Actually, the triple loop is:
            # for i in range(n):
            #   for j in range(i+1, n):
            #     for k in range(j+1, n):
            #        if (S[i]+S[j]+S[k]) % c == 0: return False
            # This is ~ C(n,3) operations. For n=48, C(48,3) = 17296.
            # 17296 * 192 candidates * 2 checks ~ 6.6 million ops. Very fast in Python? Maybe 0.1s.
            # Let's stick to the simple triple loop for Check 1.
            pass
            
    # Re-implementing Check 1 with triple loop for clarity and correctness
    for i in range(n):
        x = S[i]
        for j in range(i + 1, n):
            y = S[j]
            for k in range(j + 1, n):
                z = S[k]
                if (x + y + z) % c == 0:
                    return False

    # Check 2: For each a in S, a divides (c + x + y) for x, y in S\{a}, x!=y
    # Iterate a in S
    for i in range(n):
        a = S[i]
        # Iterate pairs (x, y) from S excluding a
        # x, y distinct, x!=a, y!=a
        # We can iterate j, k in range(n) with j<k, and skip if S[j]==a or S[k]==a
        for j in range(n):
            if j == i: continue
            x = S[j]
            for k in range(j + 1, n):
                if k == i: continue
                y = S[k]
                if (c + x + y) % a == 0:
                    return False

    return True

def _greedy_select(order, deadline):
    """
    Greedily select elements from 'order' (a list of ints) respecting the constraint.
    Stops if deadline is exceeded or MAX_CARD is reached.
    """
    selected = []
    for c in order:
        if time.time() > deadline:
            break
        if len(selected) >= MAX_CARD:
            break
        if _is_valid_add(c, selected):
            selected.append(c)
    return selected

def construct():
    start_time = time.time()
    # Safety margin: 90s limit, leave 5s for overhead
    deadline = start_time + 85.0
    
    # Strategy 1: Descending Greedy
    # Large numbers are less likely to divide sums of smaller numbers (which are bounded).
    # Also, sums of smaller numbers are less likely to be multiples of large numbers.
    best_set = []
    
    # Primary attempt: Descending order
    order1 = list(range(N, 0, -1))
    best_set = _greedy_select(order1, deadline)
    
    # If we already hit the max, return immediately
    if len(best_set) >= MAX_CARD:
        return best_set

    # Strategy 2: Local Search / Random Restarts
    # Try different orderings to see if we can find a larger set.
    # Since the greedy is deterministic and order-dependent, different orders might yield different results.
    
    rng = random.Random(42)
    attempts = 0
    max_attempts = 100 # Cap attempts to avoid infinite loop if time is abundant, but time guard is primary
    
    while time.time() < deadline - 2.0 and attempts < max_attempts:
        attempts += 1
        
        # Generate a random order
        # Bias towards larger numbers? Or pure random?
        # Let's try pure random first.
        cand = list(range(1, N + 1))
        rng.shuffle(cand)
        
        S = _greedy_select(cand, deadline)
        
        if len(S) > len(best_set):
            best_set = S
            if len(best_set) >= MAX_CARD:
                break

    return best_set

# EVOLVE-BLOCK-END
