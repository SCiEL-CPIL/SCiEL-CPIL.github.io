import time
import sys
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    n = 256
    
    # We use 0-indexed for bitmasks. Bit i corresponds to number i+1.
    # Weights are 1/(i+1).
    # We want to maximize sum of weights of selected bits, subject to no 3-AP.
    
    # 1. Precomputation
    # Elements are processed in order of decreasing weight.
    # Weight of element i (0-indexed) is 1/(i+1).
    # So order is 0, 1, 2, ..., 255 (values 1, 2, ..., 256).
    
    # Precompute conflict masks.
    # conflict_mask[i] is a bitmask where bit j is 1 if selecting element i 
    # creates a 3-AP with element j and some other element k.
    # Wait, the standard B&B for independent sets on graphs uses:
    # If we select vertex i, we cannot select any neighbor of i.
    # Here, the "graph" is not fixed. The constraint is "no 3-AP".
    # This is an independent set in a 3-uniform hypergraph.
    
    # However, we can reformulate:
    # Let S be the set of selected elements.
    # Condition: No a,b,c in S such that a+c=2b.
    
    # B&B Strategy:
    # Sort elements by weight descending: e_1, e_2, ..., e_n. (Here e_k = k, weight 1/k)
    # We make decisions on whether to include e_k or not.
    # State:
    #   idx: current index in sorted list (0 to n)
    #   selected_mask: bitmask of selected elements so far
    #   available_mask: bitmask of elements that can still be selected.
    #       Initially all 1s.
    #       When we select element v (index i in 0-based, value v=i+1):
    #           1. Remove v from available.
    #           2. For any pair (a, b) in available, if {a, v, b} is a 3-AP, 
    #              we cannot select BOTH a and b.
    #              This is not a simple "remove neighbors" step.
    
    # Alternative B&B for 3-AP:
    # Process elements in order 1, 2, ..., 256.
    # At step k, we decide if k is in S.
    # If k is in S:
    #   - For any j < k already in S, the third term 2k - j is forbidden.
    #   - For any j < k already in S, if k is the middle term, then j + (k-j) is forbidden? No.
    #   - If k is the middle term of a 3-AP (a, k, c), then a and c cannot both be in S.
    #     If a is already in S, then c is forbidden.
    #     If c is already in S, then a is forbidden.
    #     If neither is in S, we have a constraint on future choices.
    
    # This "future constraint" makes simple bitmask pruning hard unless we track constraints.
    
    # Let's use a different, more robust B&B approach for small n=256.
    # Since weights are 1/x, the optimal set will heavily favor small numbers.
    # The maximum possible sum is H_256 ~ 6.3.
    # The Moser set sum is ~ 0.99 (score 0.9939 is the ratio? No, score is value/seed. Seed is 1.0? 
    # "score is its floating serialization... search score is value/seed". 
    # If seed program returns a set with sum 1.0, score is 1.0.
    # Current program score 0.9939 implies its sum is 0.9939 * seed_sum.
    # If seed_sum is 1.0, then current sum is 0.9939.
    # Wait, the Moser set for n=256:
    # Numbers with base-3 digits 0,1.
    # Let's estimate the sum.
    
    # Let's try a different structural approach: **Integer Linear Programming (ILP)** is not available.
    # **Simulated Annealing** with bitmask operations is often effective for these combinatorial problems.
    # The current program uses local search. Let's try a more aggressive **Bitmask-based Simulated Annealing** 
    # with a good initialization (Moser set) and efficient move evaluation.
    
    # But the directive asks for a STRUCTURAL change.
    # Let's try **Branch and Bound with Recursive Pruning based on "Forced Exclusions"**.
    
    # Actually, for n=256, a well-optimized B&B might be too slow in Python.
    # Let's look at the "Inspiration 2": "Bitmask-based Greedy Max-Weight Independent Set".
    # Let's try a **Greedy Heuristic with Local Repair** but optimized with NumPy/Bitmasks.
    
    # Better idea: **Exact B&B for the first K elements, then Greedy for the rest?**
    # No, the interactions are global.
    
    # Let's try **Dynamic Programming on subsets**? 2^256 is too big.
    
    # Let's go with **Bitmask Simulated Annealing** but with a crucial improvement:
    # Instead of checking validity of the whole set on every move, we maintain a "conflict count" array.
    # conflict_count[x] = number of 3-APs containing x that are currently "active" (i.e., formed by selected elements).
    # Wait, a 3-AP is invalid if all 3 elements are selected.
    # So, we want to select a set S such that for every 3-AP {a,b,c}, NOT (a in S AND b in S AND c in S).
    # This is equivalent to: Minimize (or Maximize weight) subject to constraints.
    
    # Let's use **Greedy with Backtracking** limited to a small depth?
    
    # Let's try **Bitmask B&B** but optimized.
    # Key insight: The set of 3-AP free numbers is sparse.
    # We can precompute all 3-APs.
    # There are O(n^2) 3-APs.
    
    # Let's implement a **Bitmask B&B** that works on the "complement" or uses a specific ordering.
    
    # Actually, the most promising structural change for Python within 90s for n=256 is 
    # **Bitmask Simulated Annealing** with **O(1) move evaluation** using precomputed masks.
    
    # Precompute for each element x (0..255):
    #   partners[x] = list of (y, z) such that {x, y, z} is a 3-AP.
    #   Actually, we need to know: if I add x, which pairs (y, z) become forbidden?
    #   If I add x, then for any y, z in S, if {x, y, z} is 3-AP, it's invalid.
    #   So if I add x, I must ensure that for all y in S, the partner z (if exists) is NOT in S.
    #   And for all z in S, the partner y is NOT in S.
    
    # Let's maintain:
    #   S: current set (bitmask)
    #   score: current sum
    #   
    # Move: Add x, Remove y.
    # Check if S - {y} + {x} is valid.
    # This requires checking if x forms a 3-AP with any pair in S - {y}.
    # This is O(|S|) per move check if done naively.
    # With n=256, |S| is ~100. 256*100 = 25600 ops per move.
    # 100k moves = 2.5M ops. This is fast enough in Python if optimized.
    
    # Let's implement a **Fast Local Search / Simulated Annealing** hybrid.
    # 1. Start with Moser set.
    # 2. Precompute for each x, the list of "conflicting pairs" (y, z) such that {x, y, z} is 3-AP.
    #    Actually, for each x, precompute a bitmask `conflict_mask[x]`?
    #    No, the conflict depends on the current S.
    
    # Let's precompute:
    #   For each x, and each y, does there exist a z such that {x, y, z} is 3-AP?
    #   If yes, what is z?
    #   There are at most 2 such z's for a given pair (x, y).
    #   z1 = 2x - y (if x is middle) -> wait, x, y, z are just a set.
    #   The 3-AP is defined by (a, b, c) with a+c=2b.
    #   Given x and y, they can form a 3-AP if:
    #   1. x is middle: a=y, c=2x-y. Check if c in [1, 256].
    #   2. y is middle: a=x, c=2y-x. Check if c in [1, 256].
    #   3. x is end, y is end: middle = (x+y)/2. Check if integer and in [1, 256].
    
    # So for a pair (x, y), we can precompute the set of "third elements" T(x, y).
    # If we add x to S, we must ensure that for all y in S, no element in T(x, y) is in S.
    # AND for all y in S, if y is in T(x, z) for some z in S, that's the same condition.
    
    # Let's precompute `thirds[x][y]` as a bitmask of elements z such that {x, y, z} is a 3-AP.
    # Size: 256x256 bitmasks. 256*256*4 bytes = 256KB. Manageable.
    
    thirds = [[0] * 256 for _ in range(256)]
    for x in range(256):
        for y in range(256):
            if x == y:
                continue
            mask = 0
            # Case 1: x is middle. y is end. z = 2x - y.
            z = 2 * x - y
            if 0 <= z < 256 and z != x and z != y:
                mask |= (1 << z)
            # Case 2: y is middle. x is end. z = 2y - x.
            z = 2 * y - x
            if 0 <= z < 256 and z != x and z != y:
                mask |= (1 << z)
            # Case 3: x and y are ends. Middle m = (x+y)/2.
            s = x + y
            if s % 2 == 0:
                m = s // 2
                if 0 <= m < 256 and m != x and m != y:
                    mask |= (1 << m)
            thirds[x][y] = mask

    # Now, to check if x can be added to S:
    # We need to ensure that for all y in S, (thirds[x][y] & S) == 0.
    # This is equivalent to: (OR of thirds[x][y] for all y in S) & S == 0.
    # Let's precompute `union_thirds[x][S]`? No, S changes.
    # But we can compute `conflict_mask = 0`
    # for each y in S: conflict_mask |= thirds[x][y]
    # if conflict_mask & S: invalid.
    
    # This is O(|S|) bit operations.
    
    # Let's implement the SA.
    
    def is_moser(x):
        # x is 1-based value
        while x > 0:
            if x % 3 == 2:
                return False
            x //= 3
        return True

    # Initialize with Moser set
    S_mask = 0
    for i in range(1, 257):
        if is_moser(i):
            S_mask |= (1 << (i - 1))
            
    # Calculate initial score
    def get_score(mask):
        score = 0.0
        for i in range(256):
            if mask & (1 << i):
                score += 1.0 / (i + 1)
        return score
        
    current_score = get_score(S_mask)
    best_score = current_score
    best_mask = S_mask
    
    # Precompute weights
    weights = [1.0 / (i + 1) for i in range(256)]
    
    # SA Parameters
    T = 1.0
    T_min = 0.001
    alpha = 0.999
    
    # Helper to get set bits
    def get_set_bits(mask):
        bits = []
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            bits.append(idx)
            m ^= lsb
        return bits

    # Precompute list of all indices for random selection
    all_indices = list(range(256))
    
    import random
    random.seed(42)
    
    steps = 0
    while time.time() < deadline:
        # Pick a random move: Add x, Remove y
        # x not in S, y in S
        # Or just swap x and y.
        
        # To be efficient, pick random x from all, random y from all.
        # Ensure x not in S, y in S.
        
        x = random.choice(all_indices)
        y = random.choice(all_indices)
        
        if (S_mask & (1 << x)) or not (S_mask & (1 << y)):
            continue
            
        # Try move: remove y, add x
        new_mask = (S_mask & ~(1 << y)) | (1 << x)
        
        # Check validity of new_mask
        # We need to check if x can be added to S - {y}.
        # Let S_temp = S_mask & ~(1 << y)
        # Check if x is compatible with S_temp.
        
        S_temp = S_mask & ~(1 << y)
        
        # Check compatibility of x with S_temp
        # conflict = 0
        # for each bit z in S_temp:
        #   conflict |= thirds[x][z]
        # if conflict & S_temp: invalid
        
        # Optimization: Only check if x actually creates conflicts.
        # If x is "isolated" from S_temp, it's valid.
        
        # Iterate through set bits of S_temp
        valid = True
        m = S_temp
        while m:
            lsb = m & -m
            z = lsb.bit_length() - 1
            if thirds[x][z] & S_temp:
                valid = False
                break
            m ^= lsb
            
        if not valid:
            continue
            
        # Calculate delta
        delta = weights[x] - weights[y]
        new_score = current_score + delta
        
        # Accept if better, or with prob exp(delta/T)
        if delta > 0:
            S_mask = new_mask
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_mask = S_mask
        else:
            # Metropolis criterion
            if random.random() < pow(2.718281828, delta / T):
                S_mask = new_mask
                current_score = new_score
                
        T *= alpha
        steps += 1
        
        # If T is too low, restart? Or just stop.
        # For 90s, we can run many iterations.
        
    # Return the best set found
    result = []
    for i in range(256):
        if best_mask & (1 << i):
            result.append(i + 1)
            
    return result

# EVOLVE-BLOCK-END
