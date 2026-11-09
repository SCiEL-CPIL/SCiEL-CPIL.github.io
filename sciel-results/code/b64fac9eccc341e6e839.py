import time
import random
import math
from fractions import Fraction

N = 256
TIME_LIMIT = 50.0  # Safety margin before 60s kill

# Precompute AP pairs for each x
# ap_pairs[x] is a list of tuples (a, b) such that {x, a, b} is a 3-AP
# and a, b != x
ap_pairs = [[] for _ in range(N + 1)]

def precompute_ap_pairs():
    for x in range(1, N + 1):
        pairs = set()
        # Case 1: x is first term: x, x+d, x+2d
        for d in range(1, (N - x) // 2 + 1):
            y = x + d
            z = x + 2 * d
            if y <= N and z <= N:
                pairs.add((y, z))
        
        # Case 2: x is middle term: x-d, x, x+d
        for d in range(1, min(x - 1, N - x) + 1):
            y = x - d
            z = x + d
            if y >= 1 and z <= N:
                pairs.add((y, z))
        
        # Case 3: x is last term: x-2d, x-d, x
        for d in range(1, x // 2 + 1):
            y = x - 2 * d
            z = x - d
            if y >= 1 and z <= N:
                pairs.add((y, z))
        
        ap_pairs[x] = list(pairs)

precompute_ap_pairs()

def can_add(bitset, x):
    """
    Check if adding x to the set represented by bitset creates a 3-AP.
    Returns True if x can be added without creating a 3-AP.
    """
    for a, b in ap_pairs[x]:
        # Check if both a and b are in the bitset
        if (bitset & (1 << (a - 1))) and (bitset & (1 << (b - 1))):
            return False
    return True

def greedy_construct(order):
    """
    Greedy construction: add elements in the given order if they don't create a 3-AP.
    Returns (bitset, score) where score is the exact sum of reciprocals as a Fraction.
    """
    bitset = 0
    score = Fraction(0, 1)
    for x in order:
        if can_add(bitset, x):
            bitset |= (1 << (x - 1))
            score += Fraction(1, x)
    return bitset, score

# Precompute suffix sums of reciprocals for bounding in B&B
suffix_sums = [0.0] * (N + 2)
for i in range(N, 0, -1):
    suffix_sums[i] = suffix_sums[i + 1] + 1.0 / i

# Precompute reciprocal Fractions for speed
recip_frac = [Fraction(0, 1)] * (N + 1)
for i in range(1, N + 1):
    recip_frac[i] = Fraction(1, i)

def branch_and_bound(deadline):
    """
    Branch and Bound search to maximize the reciprocal sum.
    """
    best_score = Fraction(0, 1)
    best_bitset = 0
    best_score_float = 0.0
    
    # Use a list to hold mutable state for the recursive function
    state = {
        'best_score': best_score,
        'best_bitset': best_bitset,
        'best_score_float': best_score_float,
        'deadline': deadline
    }
    
    def dfs(x, bitset, score, score_float):
        # Time check
        if time.time() > state['deadline']:
            return
            
        # Bound check
        # If even adding all remaining reciprocals doesn't beat the best, prune
        if score_float + suffix_sums[x] <= state['best_score_float'] + 1e-15:
            return
            
        if x > N:
            if score > state['best_score']:
                state['best_score'] = score
                state['best_bitset'] = bitset
                state['best_score_float'] = score_float
            return
        
        # Option 1: Include x (if possible)
        # We try to include first because 1/x is larger for smaller x
        if can_add(bitset, x):
            new_bitset = bitset | (1 << (x - 1))
            new_score = score + recip_frac[x]
            new_score_float = score_float + 1.0 / x
            dfs(x + 1, new_bitset, new_score, new_score_float)
            
        # Option 2: Exclude x
        dfs(x + 1, bitset, score, score_float)

    # Initial greedy to get a good starting bound
    # Try a few random greedy orders
    best_initial_score = Fraction(0, 1)
    best_initial_bitset = 0
    best_initial_float = 0.0
    
    for _ in range(10):
        order = list(range(1, N + 1))
        random.shuffle(order)
        b, s = greedy_construct(order)
        s_float = float(s)
        if s > best_initial_score:
            best_initial_score = s
            best_initial_bitset = b
            best_initial_float = s_float
            
    # Also try sorted order
    b, s = greedy_construct(list(range(1, N + 1)))
    s_float = float(s)
    if s > best_initial_score:
        best_initial_score = s
        best_initial_bitset = b
        best_initial_float = s_float
        
    state['best_score'] = best_initial_score
    state['best_bitset'] = best_initial_bitset
    state['best_score_float'] = best_initial_float
    
    # Start DFS from x=1
    dfs(1, 0, Fraction(0, 1), 0.0)
    
    return state['best_bitset'], state['best_score']

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Run Branch and Bound
    best_bitset, best_score = branch_and_bound(deadline)
    
    # Final polish: try to add any remaining numbers that can be added
    # This is a local improvement step
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in range(1, N + 1):
            if not (best_bitset & (1 << (x - 1))):
                if can_add(best_bitset, x):
                    best_bitset |= (1 << (x - 1))
                    best_score += recip_frac[x]
                    improved = True
    
    # Convert bitset to list
    result = []
    for x in range(1, N + 1):
        if best_bitset & (1 << (x - 1)):
            result.append(x)
    
    return result

if __name__ == "__main__":
    res = construct()
    print(len(res))
    print(sum(1.0/x for x in res))
