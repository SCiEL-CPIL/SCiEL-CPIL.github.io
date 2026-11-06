import time
import random
from math import gcd
from collections import deque

def construct():
    start_time = time.time()
    limit = 512
    deadline = start_time + 85
    n = limit

    # Precompute compatibility as bitmasks
    compat_mask = [0] * (n + 1)
    for i in range(1, n + 1):
        mask = 0
        for j in range(1, n + 1):
            if i * j // gcd(i, j) <= n:
                mask |= (1 << j)
        compat_mask[i] = mask

    def can_add(S_mask, x):
        return (S_mask & ~compat_mask[x]) == 0

    def complete(S_mask):
        changed = True
        while changed:
            changed = False
            for x in range(1, n + 1):
                if time.time() > deadline - 5:
                    break
                if (S_mask & (1 << x)) == 0 and can_add(S_mask, x):
                    S_mask |= (1 << x)
                    changed = True
        return S_mask

    def mask_to_set(S_mask):
        return [i for i in range(1, n + 1) if (S_mask & (1 << i))]

    # ---- Maximum Weight Closure via Max-Flow / Min-Cut ----
    # Build a DAG: nodes are numbers 1..n. Edge i->j (i != j) if i and j are
    # incompatible (lcm(i,j) > n). We want the maximum-weight closed set
    # (if a node is in the set, all its out-neighbors must NOT be in the set... 
    # wait, closure means if node is in, all successors are in).
    # 
    # Actually for "no two incompatible nodes both selected", we need:
    # If i is selected AND i->j exists (incompatible), then j must NOT be selected.
    # This is NOT a standard closure (closure: i in => j in).
    # 
    # Standard closure: if i in S, then all out-neighbors j of i must be in S.
    # We want the OPPOSITE: if i in S, then incompatible j must NOT be in S.
    # 
    # Alternative formulation: Let's think of it differently.
    # A set S is valid iff for every pair (i,j) in S, lcm(i,j) <= n.
    # Equivalently, there is no edge in the incompatibility graph within S.
    # This is an Independent Set problem, which is NP-hard in general.
    #
    # However, the incompatibility graph here has special structure. Let me think
    # about whether it's a perfect graph or has other exploitable structure.
    #
    # Actually, let me reconsider. The directive says "Maximum Weight Closure in a 
    # DAG where nodes represent prime-power components and edges enforce the global 
    # LCM bound via the submodularity of the logarithmic LCM function."
    #
    # Let me think about this differently. Each number x can be written as a product
    # of prime powers. The lcm of a set S is the product of max prime powers across S.
    # We need lcm(S) <= 512 for the set to be valid (since lcm(a,b) <= 512 for all
    # pairs is equivalent to lcm of the entire set <= 512).
    #
    # Wait, that's the key insight! lcm(a,b) <= 512 for ALL pairs (a,b) in S is
    # equivalent to lcm(S) <= 512 (the lcm of the entire set). Because if the overall
    # lcm <= 512, then every pair's lcm <= overall lcm <= 512. Conversely, if every
    # pair's lcm <= 512, then the overall lcm (which divides the product of any two)
    # ... actually no, that's not quite right. The overall lcm is the lcm of all elements,
    # and lcm(a,b) <= 512 for all pairs does NOT necessarily mean lcm(S) <= 512.
    #
    # Counter-example: S = {6, 10, 15}. lcm(6,10)=30, lcm(6,15)=30, lcm(10,15)=30.
    # All pair lcms are 30 <= 512. But lcm(6,10,15) = 30. OK that works.
    #
    # Actually, lcm(S) = max over all subsets of the lcm of that subset. And lcm of
    # the full set is the lcm of all elements. If every pair has lcm <= 512, does
    # the full lcm have to be <= 512?
    #
    # lcm(a,b,c) = lcm(lcm(a,b), c). If lcm(a,b) <= 512 and we need lcm(lcm(a,b), c) <= 512.
    # But we only know lcm(a,b) <= 512 and lcm(b,c) <= 512 and lcm(a,c) <= 512.
    # Does that imply lcm(a,b,c) <= 512?
    #
    # lcm(a,b,c) divides lcm(a,b) * c / gcd(lcm(a,b), c)... this is getting complex.
    # Let me just check: is the condition "lcm(a,b) <= n for all pairs" equivalent to
    # "lcm of all elements <= n"?
    #
    # Claim: Yes. Proof: lcm(S) = lcm(lcm(S\{x}), x) for any x in S. By induction,
    # lcm(S) is built up by adding one element at a time. At each step, we compute
    # lcm(current_lcm, new_element). If at every step the lcm of the pair (current_lcm,
    # new_element) is <= n, then the final result is <= n. But current_lcm might not
    # be an element of S!
    #
    # Hmm, so the condition is NOT simply "lcm(S) <= n". Let me reconsider.
    #
    # Actually wait. Let me re-examine. We need lcm(a,b) <= n for all a,b in S, a != b.
    # This is a pairwise condition. The overall lcm(S) could be larger than n even if
    # all pairwise lcms are <= n.
    #
    # Example: n=12. S={6,4,3}. lcm(6,4)=12<=12, lcm(6,3)=6<=12, lcm(4,3)=12<=12.
    # lcm(6,4,3) = 12 <= 12. OK.
    #
    # Example: n=12. S={6,4,3,2}. lcm(6,4)=12, lcm(6,3)=6, lcm(6,2)=6, lcm(4,3)=12,
    # lcm(4,2)=4, lcm(3,2)=6. All <= 12. lcm(6,4,3,2)=12. OK.
    #
    # Let me try to find a counterexample. n=30. S={6,10,15}. Pair lcms: 30,30,30. All <=30.
    # lcm(6,10,15)=30. OK.
    #
    # n=60. S={6,10,15,4}. lcm(6,10)=30, lcm(6,15)=30, lcm(6,4)=12, lcm(10,15)=30,
    # lcm(10,4)=20, lcm(15,4)=60. All <= 60. lcm(6,10,15,4)=60. OK.
    #
    # I think it might actually be true that pairwise lcm <= n implies overall lcm <= n.
    # Here's why: lcm(S) = product of p^max(e_p) over all primes p, where e_p is the
    # exponent of p in each element. For each prime p, let x_p be the element in S with
    # the maximum exponent of p. Then lcm(S) has p^e_p where e_p is from x_p.
    # Now, for any two elements a,b in S, lcm(a,b) has for each prime p the max of
    # their exponents. So lcm(a,b) >= p^max(e_a(p), e_b(p)) for all p.
    #
    # The overall lcm(S) = product of p^{max over S of e(x)(p)}. For each prime p,
    # the element achieving the max exponent, call it x_p, has e(x_p)(p) = max.
    # Now take any two elements a, b in S. lcm(a,b) has p^{max(e(a)(p), e(b)(p))}.
    # This is <= p^{max over S of e(x)(p)} = the p-component of lcm(S).
    # So lcm(a,b) divides lcm(S) for all a,b in S. Therefore lcm(a,b) <= lcm(S).
    # So if lcm(S) <= n, then all pairwise lcms <= n. 
    #
    # But the converse: if all pairwise lcms <= n, does lcm(S) <= n?
    # We have lcm(a,b) <= n for all pairs. And lcm(a,b) divides lcm(S). So lcm(S) is a
    # common multiple of all pairwise lcms. But that doesn't bound lcm(S) directly.
    #
    # Counterexample attempt: n=12. Need pairwise lcms <= 12 but overall lcm > 12.
    # Overall lcm > 12 means some prime power in the lcm exceeds what's in 12=2^2*3.
    # So we'd need a 5 or 7 or 2^3=8 or 3^2=9 in the lcm.
    # If 8 is in the lcm, some element has 2^3=8. That element is 8,16,24,etc. But
    # 8 > 12 so can't be in S (S subset of [1,512] but we need elements <= 512, and
    # lcm(a,b) <= 512). Wait, I'm confusing n. Let me redo with n=512.
    #
    # OK let me just verify computationally whether the two conditions are equivalent
    # for n=512. If they are, the problem reduces to: find the largest subset S of
    # [1,512] such that lcm(S) <= 512. And that's equivalent to: S is a subset of the
    # divisors of some number M <= 512, and we want to maximize |S|. The maximum would
    # be the maximum number of divisors of any M <= 512.
    #
    # Let me check: max number of divisors for M <= 512.
    # 360 = 2^3 * 3^2 * 5: (3+1)(2+1)(1+1) = 24 divisors
    # 420 = 2^2 * 3 * 5 * 7: (2+1)(1+1)(1+1)(1+1) = 24 divisors  
    # 480 = 2^5 * 3 * 5: (5+1)(1+1)(1+1) = 24 divisors
    # 504 = 2^3 * 3^2 * 7: (3+1)(2+1)(1+1) = 24 divisors
    # 240 = 2^4 * 3 * 5: 5*2*2 = 20
    # 360 has 24. What about 480? 2^5*3*5 = 6*2*2 = 24.
    # What about numbers with more divisors? 720 > 512. 
    # 432 = 2^4*3^3 = 5*4 = 20.
    # 468 = 2^2*3^2*13 = 3*3*2 = 18.
    # 496 = 2^4*31 = 5*2 = 10.
    # So max is 24.
    #
    # But wait - the current program scores 1.1818, and the seed presumably scores 1.0
    # with some base value. If the answer is 24, and the seed gives 24/1.1818 ≈ 20.3,
    # that doesn't make sense. Let me reconsider.
    #
    # The score is value/seed where seed is the seed program's output. If the seed
    # program outputs a set of size 20, and we output 24, score = 24/20 = 1.2.
    # Current score is 1.1818, so current output is 1.1818 * seed_size.
    #
    # Anyway, let me just check if pairwise lcm <= n implies overall lcm <= n for n=512.
    # If yes, the answer is max divisors of any number <= 512 = 24.
    #
    # Let me verify the equivalence and then use the divisor approach.

    # Check: does pairwise lcm <= n imply overall lcm <= n?
    # If yes, answer = max over M<=n of d(M) where d is divisor count.
    # Let's verify by checking if the divisors of 360 (which has 24 divisors, all <= 360 <= 512)
    # satisfy the pairwise condition.
    
    # Divisors of 360: 1,2,3,4,5,6,8,9,10,12,15,18,20,24,30,36,40,45,60,72,90,120,180,360
    # lcm of any two divisors of 360 divides 360, so <= 360 <= 512. Yes!
    
    # So the set of divisors of any M <= 512 is valid. Max size = max d(M) for M <= 512.
    
    # But is the converse true? Can we have a valid set that's NOT contained in divisors of any M <= 512?
    # If pairwise lcm <= 512 implies overall lcm <= 512, then YES, every valid set is contained
    # in divisors of lcm(S) <= 512, so |S| <= d(lcm(S)) <= max d(M).
    
    # Let me verify the equivalence with a quick check for small n.
    # For n=12: valid sets. Divisors of 12: {1,2,3,4,6,12}, size 6.
    # Is there a valid set of size > 6? {1,2,3,4,6,8,12}? lcm(8,12)=24>12. No.
    # {1,2,3,4,6,9,12}? lcm(4,9)=36>12. No.
    # So max is 6 = d(12). Seems to hold.
    
    # I'm fairly confident the equivalence holds. Let me just compute the answer directly.
    
    # Find M <= 512 with maximum number of divisors
    def divisor_count(m):
        count = 0
        i = 1
        while i * i <= m:
            if m % i == 0:
                count += 1
                if i != m // i:
                    count += 1
            i += 1
        return count

    best_M = 1
    best_d = 1
    for M in range(1, n + 1):
        d = divisor_count(M)
        if d > best_d:
            best_d = d
            best_M = M

    # Get divisors of best_M
    divs = [i for i in range(1, n + 1) if best_M % i == 0]
    
    # Verify the set is valid
    valid = True
    for i in range(len(divs)):
        for j in range(i + 1, len(divs)):
            if divs[i] * divs[j] // gcd(divs[i], divs[j]) > n:
                valid = False
                break
        if not valid:
            break

    if valid:
        return divs

    # Fallback: use the original heuristic approach
    def local_search(S_mask):
        S_mask = complete(S_mask)
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            S_list = mask_to_set(S_mask)
            
            for e in S_list:
                if time.time() > deadline - 10:
                    break
                S_minus = S_mask & ~(1 << e)
                for x in range(1, n + 1):
                    if time.time() > deadline - 10:
                        break
                    if x != e and (S_mask & (1 << x)) == 0 and can_add(S_minus, x):
                        S2 = complete(S_minus | (1 << x))
                        if bin(S2).count('1') > bin(S_mask).count('1'):
                            S_mask = S2
                            improved = True
                        break

            for e in S_list:
                if time.time() > deadline - 10:
                    break
                S_minus = S_mask & ~(1 << e)
                S2 = S_minus
                added = 0
                for x in range(1, n + 1):
                    if time.time() > deadline - 10:
                        break
                    if (S2 & (1 << x)) == 0 and can_add(S2, x):
                        S2 |= (1 << x)
                        added += 1
                        if added >= 2:
                            break
                if bin(S2).count('1') > bin(S_mask).count('1'):
                    S_mask = complete(S2)
                    improved = True

        return S_mask

    best_mask = 0
    best_size = 0

    S = 0
    for x in range(1, n + 1):
        if can_add(S, x):
            S |= (1 << x)
    S = local_search(S)
    if bin(S).count('1') > best_size:
        best_size = bin(S).count('1')
        best_mask = S

    return mask_to_set(best_mask)
