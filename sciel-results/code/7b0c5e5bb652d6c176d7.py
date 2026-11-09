import time
import numpy as np
import random

# Precompute all nonconstant 4-term progressions (a, a+d, a+2d, a+3d), 1-indexed a,d
# Convert to 0-indexed internally.
_ALL_APS = []
for a in range(1, 125):  # a+3d <= 128 => a <= 128-3
    max_d = (128 - a) // 3
    for d in range(1, max_d + 1):
        _ALL_APS.append((a-1, a+d-1, a+2*d-1, a+3*d-1))

# Convert to numpy array for vectorized operations
_APS_ARRAY = np.array(_ALL_APS, dtype=np.int32)
_NUM_APS = len(_APS_ARRAY)

# Map index to list of AP indices containing it for local search delta calculation
_idx_to_aps = [[] for _ in range(128)]
for i, ap in enumerate(_ALL_APS):
    for idx in ap:
        _idx_to_aps[idx].append(i)

def _count_bad_colors_vectorized(colors):
    """
    Counts the number of 4-term APs that use <= 2 distinct colors.
    colors: numpy array of shape (128,)
    Returns: integer count of bad APs.
    """
    # Extract colors for each position in each AP
    # _APS_ARRAY shape: (N_APS, 4)
    c0 = colors[_APS_ARRAY[:, 0]]
    c1 = colors[_APS_ARRAY[:, 1]]
    c2 = colors[_APS_ARRAY[:, 2]]
    c3 = colors[_APS_ARRAY[:, 3]]
    
    # An AP is bad if it has <= 2 distinct colors.
    # This is equivalent to:
    # 1. All 4 are same: c0==c1==c2==c3
    # 2. Exactly 2 colors. The patterns of 2 colors in 4 slots are:
    #    A A A B, A A B A, A A B B, A B A A, A B A B, A B B A, A B B B
    #    A A B B (c0==c1, c2==c3, c0!=c2)
    #    A B A B (c0==c2, c1==c3, c0!=c1)
    #    A B B A (c0==c3, c1==c2, c0!=c1)
    #    And permutations where the 'minority' color is in different positions.
    
    # Simpler check: 
    # Distinct count <= 2 iff:
    # (c0 == c1 and c2 == c3) OR # A A B B or A A A A
    # (c0 == c2 and c1 == c3) OR # A B A B or A A A A
    # (c0 == c3 and c1 == c2) OR # A B B A or A A A A
    # Wait, this logic covers "at most 2 distinct" if we combine correctly?
    # Let's test:
    # A A A B: c0==c1 (T), c2!=c3 (F). c0!=c2 (F). c0==c3 (F). c1!=c2 (F). 
    # The condition above would fail for A A A B.
    
    # Better logic:
    # Bad if:
    # (c0 == c1 and c2 == c3) # Covers A A B B, A A A A
    # OR (c0 == c2 and c1 == c3) # Covers A B A B, A A A A
    # OR (c0 == c3 and c1 == c2) # Covers A B B A, A A A A
    # OR (c0 == c1 and c1 == c2 and c2 != c3) # A A A B
    # OR (c0 == c1 and c1 != c2 and c2 == c3) # A A B B (already covered by first)
    # ... This is getting messy.
    
    # Alternative: 
    # Count distinct values.
    # A tuple has <= 2 distinct values if:
    # max(c0,c1,c2,c3) - min(c0,c1,c2,c3) <= 1 is NOT sufficient (e.g., 0, 0, 1, 2 has range 2 but 3 colors).
    
    # Let's use a set-based check but vectorized? Hard.
    # Let's use the property:
    # #distinct <= 2 iff (c0==c1 and c2==c3) or (c0==c2 and c1==c3) or (c0==c3 and c1==c2) 
    # IS NOT TRUE for A A A B.
    
    # Correct Condition for <= 2 distinct colors in {a,b,c,d}:
    # It is NOT the case that there are 3 or 4 distinct colors.
    # 3 or 4 distinct colors implies that no two pairs are "fully matching" in the specific ways?
    
    # Let's just compute the number of distinct values using a trick?
    # Actually, for N=128 and ~2000 APs, a Python loop is fast enough for SA iterations if we are careful, 
    # but vectorization is better.
    
    # Let's try:
    # bad = (c0 == c1) & (c2 == c3)  # AA BB
    # bad |= (c0 == c2) & (c1 == c3) # AB AB
    # bad |= (c0 == c3) & (c1 == c2) # AB BA
    # bad |= (c0 == c1) & (c1 == c2) & (c2 != c3) # AAA B
    # bad |= (c0 == c1) & (c1 != c2) & (c2 == c3) # AA BB (duplicate)
    # bad |= (c0 != c1) & (c1 == c2) & (c2 == c3) # A BBB
    # bad |= (c0 != c1) & (c1 == c2) & (c2 != c3) & (c0 == c3) # A B B A (duplicate)
    # bad |= (c0 == c1) & (c1 != c2) & (c2 != c3) & (c0 == c3) # A A B A? No, c0==c3 means A A B A.
    # Let's list all patterns with <= 2 colors:
    # 1 color: AAAA
    # 2 colors:
    #   3-1 split: AAAB, AABA, ABAA, BAAA (4 patterns)
    #   2-2 split: AABB, ABAB, ABBA (3 patterns)
    
    # Vectorized check:
    # is_1_color = (c0 == c1) & (c1 == c2) & (c2 == c3)
    # 
    # is_2_color_3_1:
    #   (c0==c1 & c1==c2 & c2!=c3) # AAAB
    #   (c0==c1 & c1!=c2 & c2==c3) # AABB -> wait, this is 2-2.
    #   Let's stick to:
    #   p1 = (c0==c1) & (c1==c2) & (c2!=c3) # AAAB
    #   p2 = (c0==c1) & (c1!=c2) & (c2==c3) # A A B B (2-2)
    #   p3 = (c0!=c1) & (c1==c2) & (c2==c3) # A B B B
    #   p4 = (c0==c1) & (c1!=c2) & (c2!=c3) & (c0==c3) # A A B A
    #   p5 = (c0!=c1) & (c1==c2) & (c2!=c3) & (c0==c3) # A B B A
    #   p6 = (c0==c2) & (c1!=c2) & (c1==c3) # A B A B
    #   p7 = (c0!=c2) & (c0==c1) & (c1==c3) # A A B A (duplicate of p4? No, p4 is c0==c1, c0==c3. c1!=c2. c2!=c3. This is A A B A. p7 is c0==c1, c1==c3. c0!=c2. This is also A A B A? No.
    #       If c0==c1 and c1==c3, then c0==c1==c3. If c0!=c2, then A B A A.
    #   Let's simplify.
    
    # A simpler vectorized approach:
    # distinct_count <= 2
    # We can compute:
    # d1 = (c0 != c1)
    # d2 = (c0 != c2)
    # d3 = (c0 != c3)
    # d4 = (c1 != c2)
    # d5 = (c1 != c3)
    # d6 = (c2 != c3)
    
    # If all are 0, 1 color.
    # If exactly 1 is 1? Impossible for 4 items? 
    # If c0!=c1 and all others same? A B B B. d1=1, d2=1, d3=1, d4=0, d5=0, d6=0. Sum=3.
    # If A A B B. d1=0, d2=1, d3=1, d4=0, d5=1, d6=0. Sum=3.
    # If A B A B. d1=1, d2=0, d3=1, d4=1, d5=0, d6=1. Sum=4.
    # If A B B A. d1=1, d2=1, d3=0, d4=0, d5=1, d6=1. Sum=4.
    # If A A A B. d1=0, d2=0, d3=1, d4=0, d5=0, d6=1. Sum=2.
    
    # So:
    # 1 color: Sum = 0
    # 2 colors: Sum can be 2, 3, or 4.
    # 3 colors: 
    #   A A B C. d1=0, d2=1, d3=1, d4=0, d5=1, d6=1. Sum=4.
    #   A B C D. Sum=6.
    #   A B A C. d1=1, d2=0, d3=1, d4=1, d5=1, d6=1. Sum=5.
    #   A B C B. d1=1, d2=1, d3=1, d4=1, d5=0, d6=1. Sum=5.
    #   A B C A. d1=1, d2=1, d3=0, d4=1, d5=1, d6=1. Sum=5.
    #   A B A B (2 colors, Sum=4).
    
    # It seems Sum=2,3,4 covers 1 and 2 colors?
    # But A A B C has Sum=4.
    # A B A B has Sum=4.
    # So Sum=4 is ambiguous.
    
    # Let's use the explicit pattern matching which is safer.
    # bad = is_1_color | is_2_color
    
    is_1 = (c0 == c1) & (c1 == c2) & (c2 == c3)
    
    # 2 colors:
    # Patterns:
    # AAA B: (c0==c1) & (c1==c2) & (c2!=c3)
    # AA B A: (c0==c1) & (c1!=c2) & (c2==c3) # This is AABB. Wait. A A B A means c0=A, c1=A, c2=B, c3=A.
    #          c0==c1 (T), c1!=c2 (T), c2!=c3 (T), c0==c3 (T).
    #          My previous p4 was (c0==c1) & (c1!=c2) & (c2!=c3) & (c0==c3).
    # AA B B: (c0==c1) & (c1!=c2) & (c2==c3)
    # A AA B: (c0!=c1) & (c1==c2) & (c2!=c3) # A B B B? No. A B B B is c0=A, c1=B, c2=B, c3=B.
    #          c0!=c1 (T), c1==c2 (T), c2==c3 (T).
    # A B AA: (c0!=c1) & (c1!=c2) & (c2==c3) # A B B B? No. A B B B is A B B B.
    #          A B A B: (c0==c2) & (c1==c3) & (c0!=c1)
    # A B B A: (c0==c3) & (c1==c2) & (c0!=c1)
    # A A A B: (c0==c1) & (c1==c2) & (c2!=c3)
    # A B B B: (c0!=c1) & (c1==c2) & (c2==c3)
    # B A A A: (c0!=c1) & (c1==c2) & (c2==c3) # Same as above
    # B B A A: (c0==c1) & (c1!=c2) & (c2==c3) # Same as AABB
    
    # Let's just use the property: 
    # #distinct <= 2 iff (c0==c1 and c2==c3) or (c0==c2 and c1==c3) or (c0==c3 and c1==c2) 
    # IS NOT CORRECT for AAAB.
    
    # Okay, I will use a Python loop for the cost function in SA, but optimized with local variables.
    # N=128, ~2000 APs. 2000 * 4 ops is 8000 ops. 
    # 10000 iterations * 8000 ops = 80M ops. Python might be slow.
    # Let's stick to vectorization but do it right.
    
    # Distinct count <= 2:
    # Let's compute:
    # x0 = c0
    # x1 = c1
    # x2 = c2
    # x3 = c3
    
    # min_val = np.minimum(np.minimum(c0, c1), np.minimum(c2, c3))
    # max_val = np.maximum(np.maximum(c0, c1), np.maximum(c2, c3))
    # This doesn't help directly.
    
    # Let's use the sum of pairwise differences?
    # No.
    
    # Let's use the explicit boolean OR of all 7 patterns for 2 colors + 1 for 1 color.
    # It's 8 boolean arrays of size N_APS.
    
    p1 = (c0 == c1) & (c1 == c2) & (c2 != c3) # AAAB
    p2 = (c0 == c1) & (c1 != c2) & (c2 == c3) # AABB
    p3 = (c0 != c1) & (c1 == c2) & (c2 == c3) # ABBB
    p4 = (c0 == c1) & (c1 != c2) & (c2 != c3) & (c0 == c3) # AABA
    p5 = (c0 != c1) & (c1 == c2) & (c2 != c3) & (c0 == c3) # ABBA
    p6 = (c0 == c2) & (c1 == c3) & (c0 != c1) # ABAB
    p7 = (c0 != c1) & (c1 == c2) & (c2 != c3) & (c1 == c3) # ABAB? No.
    #    If c1==c2 and c1==c3, then c1==c2==c3. If c0!=c1, then A B B B. Covered by p3.
    #    What about B A B A? c0=B, c1=A, c2=B, c3=A.
    #    c0==c2 (T), c1==c3 (T), c0!=c1 (T). Covered by p6.
    #    What about B A A B? c0=B, c1=A, c2=A, c3=B.
    #    c0==c3 (T), c1==c2 (T), c0!=c1 (T). Covered by p5?
    #    p5: c0!=c1 (T), c1==c2 (T), c2!=c3 (T, A!=B), c0==c3 (T). Yes.
    
    # So p1..p6 covers all 2-color cases?
    # AAAB (p1), AABB (p2), ABBB (p3), AABA (p4), ABBA (p5), ABAB (p6).
    # Are there any others?
    # BAAA (covered by p3? No. p3 is ABBB. BAAA is c0!=c1, c1==c2, c2==c3. Yes, p3 covers BAAA if we swap A/B roles? 
    #    p3: (c0!=c1) & (c1==c2) & (c2==c3). 
    #    If c0=B, c1=A, c2=A, c3=A. c0!=c1 (T), c1==c2 (T), c2==c3 (T). Yes.
    # BABA (covered by p6? c0==c2 (B==B), c1==c3 (A==A), c0!=c1 (B!=A). Yes.
    # BBAA (covered by p2? c0==c1 (B==B), c1!=c2 (B!=A), c2==c3 (A==A). Yes.
    # BABB (covered by p4? c0==c1? No. c0=B, c1=A.
    #    BABB: c0=B, c1=A, c2=B, c3=B.
    #    p4: c0==c1 (F).
    #    p5: c0!=c1 (T), c1==c2 (F).
    #    p6: c0==c2 (T), c1==c3 (F).
    #    Is BABB covered?
    #    BABB has colors {A, B}.
    #    c0=B, c1=A, c2=B, c3=B.
    #    p1: c0==c1 (F).
    #    p2: c0==c1 (F).
    #    p3: c0!=c1 (T), c1==c2 (F).
    #    p4: c0==c1 (F).
    #    p5: c0!=c1 (T), c1==c2 (F).
    #    p6: c0==c2 (T), c1==c3 (F).
    #    None match!
    #    Why?
    #    BABB: c0=B, c1=A, c2=B, c3=B.
    #    It is a 2-color pattern.
    #    My patterns:
    #    p1: AAA B
    #    p2: AA BB
    #    p3: A BBB
    #    p4: AA BA
    #    p5: A BBA
    #    p6: AB AB
    #    Missing:
    #    B B A A (p2 covers if A/B swapped? p2: c0==c1, c1!=c2, c2==c3. 
    #             If c0=B, c1=B, c2=A, c3=A. c0==c1 (T), c1!=c2 (T), c2==c3 (T). Yes, p2 covers BBAA.
    #    B A B B (BABB).
    #             c0=B, c1=A, c2=B, c3=B.
    #             This is "A B A B" with A=B, B=A? No.
    #             This is "B A B B".
    #             Is it covered by p3? p3: A B B B. If A=B, B=A. Then B A A A. 
    #             BABB is not BAAA.
    #             Is it covered by p1? p1: A A A B. If A=B, B=A. B B B A.
    #             BABB is not BBBB.
    #             
    #    I missed the case where the first two are different, and the last two are the same as the first?
    #    B A B B: c0=B, c1=A, c2=B, c3=B.
    #    c0==c2 (T), c2==c3 (T). c1!=c2.
    #    Condition: (c0 == c2) & (c2 == c3) & (c1 != c2)
    #    Let's add p7: (c0 == c2) & (c2 == c3) & (c1 != c2)
    #    Check BABB: T & T & T. Yes.
    #    Check ABAB: c0=A, c1=B, c2=A, c3=B. c0==c2 (T), c2==c3 (F). No.
    #    Check AABA: c0=A, c1=A, c2=B, c3=A. c0==c2 (F). No.
    
    # So add p7.
    p7 = (c0 == c2) & (c2 == c3) & (c1 != c2)
    
    bad = is_1 | p1 | p2 | p3 | p4 | p5 | p6 | p7
    
    return np.sum(bad)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    n = 128
    
    # Try k = 3, 4, 5...
    for k in range(3, 10):
        if time.time() > deadline - 10:
            break
            
        # Simulated Annealing
        best_solution = None
        best_cost = float('inf')
        
        max_restarts = 20
        for r in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            # Random initialization
            colors = np.random.randint(0, k, n, dtype=np.int32)
            current_cost = _count_bad_colors_vectorized(colors)
            
            if current_cost == 0:
                best_solution = colors.copy()
                best_cost = 0
                break
                
            T = 1.0
            no_improve = 0
            max_iters = 5000
            
            for _ in range(max_iters):
                if time.time() > deadline - 5:
                    break
                    
                if current_cost == 0:
                    best_solution = colors.copy()
                    best_cost = 0
                    break
                    
                # Pick random index
                i = np.random.randint(n)
                old_color = colors[i]
                new_color = np.random.randint(0, k)
                if new_color == old_color:
                    continue
                    
                # Calculate delta cost
                # We need to check APs containing i
                delta = 0
                for ap_idx in _idx_to_aps[i]:
                    ap = _ALL_APS[ap_idx]
                    # Get old colors
                    old_c = [colors[j] for j in ap]
                    old_bad = 1 if len(set(old_c)) <= 2 else 0
                    
                    # Get new colors
                    new_c = old_c[:]
                    pos_in_ap = ap.index(i)
                    new_c[pos_in_ap] = new_color
                    new_bad = 1 if len(set(new_c)) <= 2 else 0
                    
                    delta += new_bad - old_bad
                    
                if delta < 0:
                    colors[i] = new_color
                    current_cost += delta
                    no_improve = 0
                else:
                    # Accept with probability exp(-delta/T)
                    if random.random() < np.exp(-delta / T):
                        colors[i] = new_color
                        current_cost += delta
                    else:
                        no_improve += 1
                        if no_improve > 100:
                            break
                            
                # Cool down
                T *= 0.999
                if T < 0.01:
                    T = 0.01

            if current_cost < best_cost:
                best_cost = current_cost
                best_solution = colors.copy()
                
            if best_cost == 0:
                break

        if best_cost == 0:
            # Found a valid solution with k colors
            # Convert to list of ints
            return list(best_solution)
            
    # Fallback: Greedy construction similar to seed
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color} | {colors[x-1-j*d] for j in (1,2,3)}) >= 3 
                    for d in range(1, (x-1)//3 + 1) if x-1-3*d >= 0)
                and all(any(colors[x-1-j*d] != color for j in (1,2)) 
                        for d in range(1, (x-1)//2 + 1) if x-1-2*d >= 0)):
                colors.append(color)
                break
        else:
            colors.append(0)
    return colors
