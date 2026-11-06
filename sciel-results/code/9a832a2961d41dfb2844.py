"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Find a nonempty subset of [1, 512] with lcm(a,b) <= 512 for every pair.
    Maximize cardinality.
    
    Key Insight:
    If we select a set S such that for all a, b in S, lcm(a, b) <= 512,
    does that imply there exists a common multiple M <= 512 divisible by all elements in S?
    
    Let's analyze:
    Let M be the LCM of the entire set S.
    For any pair a, b in S, lcm(a, b) divides M.
    So lcm(a, b) <= M.
    The condition is lcm(a, b) <= 512 for all pairs.
    This does NOT strictly imply M <= 512. 
    Example: S = {2, 3, 5}. 
    lcm(2,3)=6 <= 512.
    lcm(2,5)=10 <= 512.
    lcm(3,5)=15 <= 512.
    But LCM(S) = 30 <= 512.
    
    Consider S = {6, 10, 15}.
    lcm(6,10) = 30.
    lcm(6,15) = 30.
    lcm(10,15) = 30.
    All <= 512.
    LCM(S) = 30.
    
    Consider S = {12, 18, 24}.
    lcm(12,18) = 36.
    lcm(12,24) = 24.
    lcm(18,24) = 72.
    LCM(S) = 72.
    
    Is it possible that all pairwise lcms are <= 512, but the global LCM is > 512?
    Let S = {a_1, ..., a_k}.
    Let p be a prime. Let v_p(x) be the exponent of p in x.
    Let M_p = max_{a in S} v_p(a).
    The global LCM has exponent M_p for prime p.
    The condition is: for all i != j, sum over p of min(v_p(a_i), v_p(a_j)) ... no, lcm exponent is max(v_p(a_i), v_p(a_j)).
    So for all pairs i, j: product_p p^{max(v_p(a_i), v_p(a_j))} <= 512.
    
    This implies that for any prime p, and any pair i,j, p^{max(v_p(a_i), v_p(a_j))} <= 512.
    This means max(v_p(a_i), v_p(a_j)) <= log_p(512).
    Let E_p = max_{a in S} v_p(a).
    Then for any pair, the exponent of p in their lcm is <= E_p.
    Wait, if we take the pair that achieves the maximum exponent for p, say a_i has v_p(a_i) = E_p.
    For any other a_j, the lcm(a_i, a_j) has exponent max(E_p, v_p(a_j)) = E_p.
    So p^{E_p} must divide the lcm(a_i, a_j).
    Since lcm(a_i, a_j) <= 512, we must have p^{E_p} <= 512.
    
    Does this imply the global LCM <= 512?
    The global LCM L has prime factorization product p^{E_p}.
    We know that for each p, p^{E_p} <= 512.
    But the product of these prime powers might exceed 512.
    Example:
    Let S contain numbers such that:
    For p=2, max exponent is 9 (2^9 = 512).
    For p=3, max exponent is 5 (3^5 = 243).
    If there is a number in S with 2^9 and another with 3^5, their lcm is 2^9 * 3^5 which is huge.
    But the condition is pairwise lcm <= 512.
    If a has 2^9, and b has 3^5, then lcm(a,b) is divisible by 2^9 * 3^5 > 512.
    So such a pair cannot exist in S.
    
    This suggests a structure.
    Let's look at the "common multiple" insight from the knowledge:
    "The key insight is that pairwise lcm <= 512 is equivalent to the lcm of the entire set being <= 512"
    
    Let's verify this equivalence.
    (=>) Assume pairwise lcm <= 512.
    Let L = LCM(S).
    For any prime p, let E_p be the max exponent in S.
    There exists a in S with v_p(a) = E_p.
    For any other b in S, lcm(a,b) has exponent E_p for p.
    So p^{E_p} divides lcm(a,b) <= 512.
    Thus p^{E_p} <= 512.
    This holds for all p.
    However, L = product p^{E_p}.
    Do we know L <= 512?
    Consider S = {6, 10, 15}.
    E_2 = 1 (from 6,10). 2^1 = 2 <= 512.
    E_3 = 1 (from 6,15). 3^1 = 3 <= 512.
    E_5 = 1 (from 10,15). 5^1 = 5 <= 512.
    L = 2*3*5 = 30 <= 512.
    
    Consider a case where L > 512.
    We need product p^{E_p} > 512.
    This requires at least two primes with significant exponents or many small primes.
    Suppose E_2 >= 1, E_3 >= 1, E_5 >= 1, E_7 >= 1.
    Then there are a,b,c,d in S with these max exponents.
    If they are distinct numbers, we must check pairwise lcms.
    If a=2, b=3, c=5, d=7.
    lcm(2,3)=6, lcm(2,5)=10, lcm(2,7)=14, lcm(3,5)=15, lcm(3,7)=21, lcm(5,7)=35.
    All <= 512.
    LCM(S) = 210 <= 512.
    
    Try to push L higher.
    Add 11. S={2,3,5,7,11}.
    Pairwise lcms are products of two primes. Max is 11*7=77 <= 512.
    LCM = 2310 > 512.
    Wait!
    S = {2, 3, 5, 7, 11}.
    lcm(2,3)=6
    lcm(2,5)=10
    lcm(2,7)=14
    lcm(2,11)=22
    lcm(3,5)=15
    lcm(3,7)=21
    lcm(3,11)=33
    lcm(5,7)=35
    lcm(5,11)=55
    lcm(7,11)=77
    All <= 512.
    But LCM(S) = 2*3*5*7*11 = 2310 > 512.
    
    So the equivalence "pairwise lcm <= 512 <=> global lcm <= 512" is FALSE.
    The set of distinct primes <= sqrt(512) approx 22.
    Primes: 2, 3, 5, 7, 11, 13, 17, 19.
    19*17 = 323 <= 512.
    23*19 = 437 <= 512.
    29*19 = 551 > 512.
    So we can include primes up to 23?
    Let's check the set of primes P = {2, 3, 5, 7, 11, 13, 17, 19, 23}.
    Max pair product: 23 * 19 = 437 <= 512.
    So all pairwise lcms of distinct primes in this set are <= 512.
    Size = 9.
    
    Can we add more numbers?
    We can add 1. Size 10.
    Can we add 4?
    lcm(4, 2) = 4.
    lcm(4, 3) = 12.
    lcm(4, 5) = 20.
    lcm(4, 7) = 28.
    lcm(4, 11) = 44.
    lcm(4, 13) = 52.
    lcm(4, 17) = 68.
    lcm(4, 19) = 76.
    lcm(4, 23) = 92.
    All <= 512.
    So S = {1, 2, 3, 4, 5, 7, 11, 13, 17, 19, 23} is valid. Size 11.
    
    Can we add 8?
    lcm(8, 2)=8, lcm(8,3)=24, ..., lcm(8,23)=184.
    All <= 512.
    Size 12.
    
    Can we add 9?
    lcm(9, 2)=18, lcm(9,4)=36, lcm(9,5)=45, lcm(9,7)=63, lcm(9,8)=72, lcm(9,11)=99, lcm(9,13)=117, lcm(9,17)=153, lcm(9,19)=171, lcm(9,23)=207.
    All <= 512.
    Size 13.
    
    Can we add 16?
    lcm(16, 9) = 144.
    lcm(16, 23) = 368.
    All <= 512.
    Size 14.
    
    Can we add 25?
    lcm(25, 2)=50, lcm(25,3)=75, lcm(25,4)=100, lcm(25,7)=175, lcm(25,8)=200, lcm(25,9)=225, lcm(25,11)=275, lcm(25,13)=325, lcm(25,16)=400, lcm(25,17)=425, lcm(25,19)=475, lcm(25,23)=575 > 512.
    So 25 is incompatible with 23.
    If we drop 23, we can add 25.
    Current set without 23: {1, 2, 3, 4, 5, 7, 8, 9, 11, 13, 16, 17, 19}. Size 13.
    Add 25: Size 14.
    Check 25 against others:
    25*19 = 475 <= 512.
    25*17 = 425 <= 512.
    25*16 = 400 <= 512.
    25*13 = 325 <= 512.
    25*11 = 275 <= 512.
    25*9 = 225 <= 512.
    25*8 = 200 <= 512.
    25*7 = 175 <= 512.
    25*5 = 25.
    25*4 = 100.
    25*3 = 75.
    25*2 = 50.
    25*1 = 25.
    All good.
    
    Can we add 27?
    27 * 19 = 513 > 512.
    So 27 incompatible with 19.
    
    Let's formulate this as a Maximum Clique problem on a graph where vertices are 1..512 and edges connect a,b if lcm(a,b) <= 512.
    The graph is dense for small numbers.
    
    The "common multiple" strategy suggests looking for divisors of a number M.
    If we pick M=512, divisors are powers of 2. Size 10.
    If we pick M=420 (2^2 * 3 * 5 * 7).
    Divisors of 420:
    1, 2, 3, 4, 5, 6, 7, 10, 12, 14, 15, 20, 21, 28, 30, 35, 42, 60, 70, 84, 105, 140, 210, 420.
    Count: 24 divisors.
    For any a,b dividing 420, lcm(a,b) divides 420, so lcm(a,b) <= 420 <= 512.
    So the set of divisors of 420 is a valid clique of size 24.
    
    Is there a number with more divisors?
    d(n) for n <= 512.
    360: 2^3 * 3^2 * 5. d(360) = 4*3*2 = 24.
    420: 2^2 * 3 * 5 * 7. d(420) = 3*2*2*2 = 24.
    480: 2^5 * 3 * 5. d(480) = 6*2*2 = 24.
    504: 2^3 * 3^2 * 7. d(504) = 4*3*2 = 24.
    512: 2^9. d(512) = 10.
    
    Is there any n <= 512 with d(n) > 24?
    Next highly composite numbers:
    1: 1
    2: 2
    4: 3
    6: 4
    12: 6
    24: 8
    36: 9
    48: 10
    60: 12
    120: 16
    180: 18
    240: 20
    360: 24
    720: 30 (too big)
    
    So max divisors for n <= 512 is 24.
    This gives a clique of size 24.
    
    Can we get a larger clique that is NOT a divisor set of a single number?
    The set of primes {2,3,5,7,11,13,17,19,23} plus powers of 2,3,5...
    Let's try to build a clique greedily starting from the divisor set of 420.
    Divisors of 420: D420.
    Can we add a number x not dividing 420?
    x must be compatible with all d in D420.
    This means lcm(x, d) <= 512 for all d in D420.
    This is a strong condition.
    
    Let's check if we can add 11 to D420.
    D420 contains 7, 5, 3, 2, etc.
    lcm(11, 7) = 77.
    lcm(11, 5) = 55.
    lcm(11, 3) = 33.
    lcm(11, 2) = 22.
    lcm(11, 4) = 44.
    lcm(11, 6) = 66.
    lcm(11, 10) = 110.
    lcm(11, 12) = 132.
    lcm(11, 14) = 154.
    lcm(11, 15) = 165.
    lcm(11, 20) = 220.
    lcm(11, 21) = 231.
    lcm(11, 28) = 308.
    lcm(11, 30) = 330.
    lcm(11, 35) = 385.
    lcm(11, 42) = 462.
    lcm(11, 60) = 660 > 512.
    So 11 is incompatible with 60 (which is in D420).
    
    What if we remove 60 from D420 and add 11?
    D420 \ {60} has size 23.
    Add 11 -> 24.
    Can we add more?
    Maybe 13?
    lcm(13, 42) = 546 > 512.
    42 is in D420.
    So 13 incompatible with 42.
    
    It seems the divisor sets are quite maximal or close to it.
    
    Let's write a program to find the maximum clique using a branch and bound or greedy heuristic, but leveraging the structure.
    Since N=512 is small, we can build the compatibility matrix and use a max clique solver.
    Python's networkx has a max_clique function, but it might be slow for N=512 if the graph is dense.
    However, the graph is not that dense.
    Actually, let's just use a simple greedy + local search or a specific branch and bound.
    
    Given the 60s limit, a robust greedy approach with multiple restarts and local search (swap moves) should work well.
    Or, we can exploit the fact that if we pick a "core" number M, its divisors form a clique.
    We can iterate over all M <= 512, compute the divisor set, and take the largest.
    Then try to expand that set.
    
    Let's try the expansion strategy.
    1. Find M* maximizing d(M). Let S0 be divisors of M*.
    2. Try to add elements x not in S0.
    3. If x is compatible with all in S0, add it.
    4. Repeat until no more can be added.
    5. Try swapping: remove y in S0, add x not in S0 if it increases size or allows more additions.
    
    Actually, the maximum clique in this specific graph might be exactly 24 or slightly more.
    Let's check if we can get 25.
    If we take divisors of 360 (24 elements).
    Can we add 11?
    360 = 2^3 * 3^2 * 5.
    Divisors include 8, 9, 5.
    lcm(11, 8) = 88.
    lcm(11, 9) = 99.
    lcm(11, 5) = 55.
    lcm(11, 24) = 264.
    lcm(11, 45) = 495.
    lcm(11, 72) = 792 > 512.
    72 is a divisor of 360.
    So 11 incompatible with 72.
    
    What if we use a number with 4 distinct prime factors?
    2*3*5*7 = 210. d(210) = 16.
    2*3*5*11 = 330. d(330) = 16.
    2*3*5*13 = 390. d(390) = 16.
    2*3*5*17 = 510. d(510) = 16.
    
    It seems 24 is a strong candidate for the maximum.
    
    Let's implement a solver that:
    1. Generates all divisor sets for M in 1..512.
    2. Finds the largest one.
    3. Attempts to augment it with local search (add/swap).
    
    To ensure speed, we precompute the compatibility matrix.
    
    Code structure:
    - Precompute compat matrix (512x512 boolean).
    - Find best divisor set.
    - Local search:
      - Current set S.
      - Loop:
        - Try to add any x not in S if compatible with all in S.
        - If none, try swap: for each y in S, for each x not in S, if (S - {y} + {x}) is valid and potentially leads to larger set?
          Actually, simple swap: if (S - {y}) is compatible with x, and len(S - {y} + {x}) == len(S), it doesn't help size immediately.
          We want to increase size.
          So we look for x that is compatible with S \ {y} for some y, but not S.
          Then S' = S \ {y} U {x}. Size same.
          But maybe S' allows adding z?
          This is getting complex.
          
    Given the small N, a simple branch and bound for max clique might be feasible if well-implemented.
    Or, since we suspect the answer is 24, we can just return the best divisor set found.
    The seed score 1.0909 corresponds to size 24 if seed size is 22.
    If the true max is 24, and we return 24, we match the best.
    If the true max is higher, we need a better search.
    
    Let's assume the divisor set approach is the key.
    I will implement the divisor set search and a simple augmentation.
    
    """
    import time
    from math import gcd
    
    deadline = time.time() + 50  # Safety margin
    
    N = 512
    
    # 1. Build compatibility matrix
    # compat[a][b] is True if lcm(a,b) <= N
    compat = [bytearray(N + 1) for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a, N + 1):
            g = gcd(a, b)
            if a * b // g <= N:
                compat[a][b] = 1
                compat[b][a] = 1
    
    # 2. Find best divisor set
    best_set = set()
    best_size = 0
    
    for M in range(1, N + 1):
        # Find divisors of M
        divs = []
        for i in range(1, int(M**0.5) + 1):
            if M % i == 0:
                divs.append(i)
                if i != M // i:
                    divs.append(M // i)
        
        if len(divs) > best_size:
            best_size = len(divs)
            best_set = set(divs)
            
        if time.time() > deadline - 20:
            break
    
    # 3. Local Search / Augmentation
    # Try to add elements to best_set
    current_set = list(best_set)
    in_set = set(current_set)
    
    improved = True
    while improved:
        if time.time() > deadline - 5:
            break
        improved = False
        
        # Try to add
        for v in range(1, N + 1):
            if v not in in_set:
                if all(compat[v][s] for s in in_set):
                    in_set.add(v)
                    current_set.append(v)
                    improved = True
        
        # Try to swap (remove one, add one) to enable further additions?
        # Simple swap: if removing y allows adding x, and x is "better" or allows more?
        # For now, just try to see if we can increase size via swaps.
        # A swap keeps size constant. We need a sequence of swaps that allows an addition.
        # This is a 1-step look-ahead.
        
        # Let's try a simpler heuristic:
        # Sort candidates by degree in the remaining graph?
        # Or just try random restarts on the augmentation?
        
        # If no direct addition, try removing a "bottleneck" element.
        # Identify elements that block the most other elements.
        # For each y in in_set, count how many x not in in_set are compatible with in_set - {y} but not in_set.
        # This is O(|S| * N * |S|) which might be slow.
        # |S| ~ 24, N = 512. 24 * 512 * 24 ~ 300k ops. Fine.
        
        if not improved:
            best_swap_gain = -1
            best_swap_y = None
            best_swap_x = None
            
            for y in in_set:
                temp_set = in_set - {y}
                for x in range(1, N + 1):
                    if x not in temp_set:
                        # Check if x is compatible with temp_set
                        # If yes, and x was NOT compatible with in_set (otherwise it would have been added)
                        # Then swapping y for x is a valid move.
                        # Does it help? It keeps size same.
                        # But maybe x is more "central" and allows adding more later.
                        # For now, let's just do the swap if it doesn't decrease size (it doesn't)
                        # and hope that the next iteration adds something.
                        # To avoid cycles, we can keep a history or just do a limited number of swaps.
                        
                        # Optimization: only check compatibility with temp_set
                        if all(compat[x][s] for s in temp_set):
                            # x is compatible with temp_set.
                            # Check if x was blocked by y in the original set.
                            # If x was compatible with in_set, it would have been added in the "Try to add" phase.
                            # So x is definitely incompatible with y (or some other element, but y is the only one removed).
                            # So x is incompatible with y.
                            
                            # Heuristic: Prefer x with high degree in the "outside" graph?
                            # Or just pick the first one?
                            # Let's pick the one with the largest value? Or smallest?
                            # Let's try to pick x that has the most neighbors in the "outside" set.
                            # This is expensive.
                            # Let's just pick the first one found to keep it fast, or randomize.
                            
                            # To break ties and explore, let's use a deterministic order but allow multiple swaps per iteration.
                            # Actually, if we swap, we might get stuck.
                            # Let's limit swaps to 100 per "stall".
                            
                            best_swap_y = y
                            best_swap_x = x
                            break # Found a valid swap for this y
                if best_swap_y is not None:
                    break
                    
            if best_swap_y is not None:
                in_set.remove(best_swap_y)
                in_set.add(best_swap_x)
                current_set = list(in_set)
                improved = True # Force another loop to try adding
                
    return list(in_set)
# EVOLVE-BLOCK-END
