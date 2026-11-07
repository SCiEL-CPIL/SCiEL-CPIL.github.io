"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
from collections import defaultdict

P = {'n': 210, 'maximum_window': 10000, 'maximum_size': 100}

def construct():
    n = P['n']
    max_val = 10210
    max_size = P['maximum_size']
    
    # We need to select k distinct integers from (210, 10210]
    # such that their product * 210 is a perfect square.
    # This is equivalent to: the product of the squarefree kernels of the selected
    # integers, multiplied by the squarefree kernel of 210, must be 1 (i.e., even exponents for all primes).
    # Actually, let's be precise.
    # Let S be the set of selected integers.
    # Product P_S = prod_{x in S} x.
    # We require P_S * 210 = y^2 for some integer y.
    # This means the prime factorization of P_S * 210 has all even exponents.
    # Let v_p(m) be the exponent of prime p in m.
    # For all primes p, v_p(P_S) + v_p(210) must be even.
    # So v_p(P_S) mod 2 must be equal to v_p(210) mod 2.
    # Let sf(m) be the squarefree kernel of m (product of primes with odd exponent).
    # The condition is equivalent to: the product of sf(x) for x in S, multiplied by sf(210), must be a perfect square.
    # Wait, that's not quite right. The product of squarefree kernels isn't simply the squarefree kernel of the product.
    # Let's use vector space over GF(2).
    # Each integer x maps to a vector v(x) in {0,1}^Pi, where Pi is the set of relevant primes.
    # v(x)_p = v_p(x) mod 2.
    # We need sum_{x in S} v(x) + v(210) = 0 (mod 2).
    # So sum_{x in S} v(x) = v(210) (mod 2).
    #
    # The primes involved are those up to sqrt(10210) ~ 101, plus possibly primes up to 10210 themselves if they appear.
    # However, since we're multiplying by 210 = 2 * 3 * 5 * 7, the primes 2, 3, 5, 7 are definitely relevant.
    # Any other prime p that appears in any selected x must appear in an even number of selected x's (counting multiplicity in the vector sum), 
    # OR if it appears in 210 (which it doesn't, except 2,3,5,7), then it must appear in an odd number.
    # Actually, for primes p not in {2,3,5,7}, v_p(210)=0, so we need sum v_p(x) = 0 mod 2.
    # This means any prime p > 7 that appears in any selected x must appear in an even number of the selected x's (in the sense of the vector sum).
    #
    # Strategy:
    # We want to minimize max(selected) - 210.
    # This suggests we should try to use small numbers.
    # Let's iterate over possible max values M from 211 to 10210.
    # For each M, we consider the set of available numbers A_M = {211, 212, ..., M}.
    # We want to find the smallest subset S of A_M such that sum_{x in S} v(x) = v(210) mod 2.
    # And |S| <= 100.
    # We want to minimize M.
    #
    # This is a linear algebra problem over GF(2). We need to find a linear combination of the vectors v(x) for x in A_M that equals v(210).
    # We want to minimize the number of terms (since we also have a constraint on size, but primarily we want to find *a* solution).
    # Actually, the objective is to minimize M, the maximum value. The size constraint is secondary (just needs to be <= 100).
    #
    # Let's reformulate:
    # For a fixed M, does there exist a subset S of {211, ..., M} with |S| <= 100 such that sum v(x) = v(210)?
    # We want the smallest such M.
    #
    # How to solve for a fixed M?
    # This is a subset sum problem in GF(2)^d, where d is the number of primes up to M.
    # d can be large (~1000 primes up to 10000). This makes direct linear algebra hard.
    #
    # Alternative approach:
    # Notice that v(x) is determined by the squarefree part of x.
    # If x = k^2 * s, where s is squarefree, then v(x) = v(s).
    # So we can group numbers by their squarefree kernel.
    # Let C_s be the set of numbers in {211, ..., M} with squarefree kernel s.
    # If we pick one number from C_s, we add v(s) to the sum.
    # If we pick two numbers from C_s, we add v(s) + v(s) = 0.
    # So, effectively, for each squarefree kernel s, we can choose to "activate" it (add v(s) to the sum) by picking an odd number of elements from C_s, or not activate it (add 0) by picking an even number (including 0).
    # To minimize the maximum value, we should pick the smallest available numbers.
    # But we also have a constraint on the total number of selected integers (<= 100).
    #
    # Let's think differently.
    # We want to find a set of squarefree kernels {s_1, ..., s_k} such that sum v(s_i) = v(210), and for each s_i, there is at least one number in {211, ..., M} with that kernel.
    # Then we can pick one such number for each s_i.
    # The size of the selected set will be k. We need k <= 100.
    # The maximum value will be max(x_i) where x_i is the chosen number for s_i.
    # To minimize the maximum value, for each s_i, we should pick the smallest number in {211, ..., M} with kernel s_i.
    # Let min_val(s, M) be the smallest number in {211, ..., M} with squarefree kernel s, or infinity if none.
    # Then for a given set of kernels {s_1, ..., s_k}, the maximum value is max(min_val(s_i, M)).
    # We want to minimize this maximum.
    #
    # This is still complex. Let's try a greedy or heuristic approach.
    #
    # Simpler idea:
    # We know that 210 = 2 * 3 * 5 * 7.
    # We need the product of selected numbers to have the same parity of exponents as 210 for primes 2,3,5,7, and even parity for all other primes.
    #
    # Consider picking numbers that are squares times a divisor of 210.
    # Let d be a divisor of 210. Then d * k^2 has the same squarefree kernel as d.
    # If we pick numbers of the form d_i * k_i^2, the sum of their kernels is sum d_i (in the vector space sense, i.e., symmetric difference of prime sets).
    # We want sum d_i = 210 (in the kernel sense, i.e., the product of d_i's, simplified, is 210).
    # Wait, the product of the kernels must be 210 (up to squares).
    # So if we pick kernels s_1, ..., s_k, we need s_1 * ... * s_k = 210 * (square).
    # Since s_i are squarefree, this means the symmetric difference of their prime sets must be {2,3,5,7}.
    #
    # Let's try to find a small set of squarefree numbers s_1, ..., s_k such that their product is 210 times a square.
    # And then find small numbers of the form s_i * k_i^2.
    #
    # Example:
    # If we pick s_1 = 210, then we need just one number: 210 * k^2.
    # Smallest such number > 210 is 210 * 2^2 = 840.
    # Max selected = 840. Value = 840 - 210 = 630.
    # This is a valid solution with 1 number.
    #
    # Can we do better?
    # Try 2 numbers: s_1 * s_2 = 210 * square.
    # Since s_1, s_2 are squarefree, s_1 * s_2 / 210 must be a square.
    # Let s_1 = d_1, s_2 = d_2.
    # d_1 * d_2 = 210 * m^2.
    # Since d_1, d_2 are squarefree, m^2 must be 1 (if d_1, d_2 are divisors of 210) or...
    # Actually, if d_1 and d_2 are squarefree, their product is squarefree iff they are coprime.
    # If they share a prime, that prime appears with exponent 2 in the product, so it's not squarefree.
    # So d_1 * d_2 is not necessarily squarefree.
    # We need d_1 * d_2 = 210 * m^2.
    # This means d_1 * d_2 / 210 is a perfect square.
    # Let d_1 = 2 * 3 = 6, d_2 = 5 * 7 = 35.
    # d_1 * d_2 = 210. 210 / 210 = 1 = 1^2. So this works.
    # We need numbers x_1 = 6 * k_1^2, x_2 = 35 * k_2^2, both > 210 and <= 10210.
    # Minimize max(x_1, x_2).
    # Smallest x_1 > 210: 6 * k_1^2 > 210 => k_1^2 > 35 => k_1 >= 6 => x_1 = 6*36 = 216.
    # Smallest x_2 > 210: 35 * k_2^2 > 210 => k_2^2 > 6 => k_2 >= 3 => x_2 = 35*9 = 315.
    # Max = 315. Value = 315 - 210 = 105.
    # This is better than 630.
    #
    # Try other pairs:
    # d_1 = 2, d_2 = 105.
    # x_1 = 2 * k_1^2 > 210 => k_1^2 > 105 => k_1 >= 11 => x_1 = 2*121 = 242.
    # x_2 = 105 * k_2^2 > 210 => k_2^2 > 2 => k_2 >= 2 => x_2 = 105*4 = 420.
    # Max = 420. Value = 210. Worse.
    #
    # d_1 = 3, d_2 = 70.
    # x_1 = 3 * k_1^2 > 210 => k_1^2 > 70 => k_1 >= 9 => x_1 = 3*81 = 243.
    # x_2 = 70 * k_2^2 > 210 => k_2^2 > 3 => k_2 >= 2 => x_2 = 70*4 = 280.
    # Max = 280. Value = 70. Better!
    #
    # d_1 = 5, d_2 = 42.
    # x_1 = 5 * k_1^2 > 210 => k_1^2 > 42 => k_1 >= 7 => x_1 = 5*49 = 245.
    # x_2 = 42 * k_2^2 > 210 => k_2^2 > 5 => k_2 >= 3 => x_2 = 42*9 = 378.
    # Max = 378. Value = 168.
    #
    # d_1 = 7, d_2 = 30.
    # x_1 = 7 * k_1^2 > 210 => k_1^2 > 30 => k_1 >= 6 => x_1 = 7*36 = 252.
    # x_2 = 30 * k_2^2 > 210 => k_2^2 > 7 => k_2 >= 3 => x_2 = 30*9 = 270.
    # Max = 270. Value = 60. Even better!
    #
    # d_1 = 10, d_2 = 21.
    # x_1 = 10 * k_1^2 > 210 => k_1^2 > 21 => k_1 >= 5 => x_1 = 10*25 = 250.
    # x_2 = 21 * k_2^2 > 210 => k_2^2 > 10 => k_2 >= 4 => x_2 = 21*16 = 336.
    # Max = 336. Value = 126.
    #
    # d_1 = 14, d_2 = 15.
    # x_1 = 14 * k_1^2 > 210 => k_1^2 > 15 => k_1 >= 4 => x_1 = 14*16 = 224.
    # x_2 = 15 * k_2^2 > 210 => k_2^2 > 14 => k_2 >= 4 => x_2 = 15*16 = 240.
    # Max = 240. Value = 30. Very good!
    #
    # Let's check if we can do better with 3 numbers.
    # Or maybe other pairs.
    # d_1 = 1, d_2 = 210.
    # x_1 = 1 * k_1^2 > 210 => k_1 >= 15 => x_1 = 225.
    # x_2 = 210 * k_2^2 > 210 => k_2 >= 2 => x_2 = 840.
    # Max = 840.
    #
    # What about d_1 = 2*5=10, d_2=3*7=21? Already did that.
    # d_1 = 2*3*5=30, d_2=7.
    # x_1 = 30 * k_1^2 > 210 => k_1^2 > 7 => k_1 >= 3 => x_1 = 270.
    # x_2 = 7 * k_2^2 > 210 => k_2^2 > 30 => k_2 >= 6 => x_2 = 252.
    # Max = 270. Same as before.
    #
    # d_1 = 2*3*7=42, d_2=5.
    # x_1 = 42 * k_1^2 > 210 => k_1^2 > 5 => k_1 >= 3 => x_1 = 378.
    # x_2 = 5 * k_2^2 > 210 => k_2^2 > 42 => k_2 >= 7 => x_2 = 245.
    # Max = 378.
    #
    # d_1 = 3*5*7=105, d_2=2.
    # x_1 = 105 * k_1^2 > 210 => k_1^2 > 2 => k_1 >= 2 => x_1 = 420.
    # x_2 = 2 * k_2^2 > 210 => k_2^2 > 105 => k_2 >= 11 => x_2 = 242.
    # Max = 420.
    #
    # d_1 = 2*5*7=70, d_2=3.
    # x_1 = 70 * k_1^2 > 210 => k_1^2 > 3 => k_1 >= 2 => x_1 = 280.
    # x_2 = 3 * k_2^2 > 210 => k_2^2 > 70 => k_2 >= 9 => x_2 = 243.
    # Max = 280.
    #
    # d_1 = 2*3*5*7=210, d_2=1.
    # Already did.
    #
    # So the best pair so far is (14, 15) giving max 240, value 30.
    #
    # Can we get lower with 3 numbers?
    # We need s_1 * s_2 * s_3 = 210 * square.
    # Let's try small squarefree numbers.
    # s_1=2, s_2=3, s_3=35. Product = 210.
    # x_1 = 2 * k_1^2 > 210 => k_1 >= 11 => x_1 = 242.
    # x_2 = 3 * k_2^2 > 210 => k_2 >= 9 => x_2 = 243.
    # x_3 = 35 * k_3^2 > 210 => k_3 >= 3 => x_3 = 315.
    # Max = 315. Value = 105. Worse.
    #
    # s_1=2, s_2=5, s_3=21.
    # x_1 = 242.
    # x_2 = 5 * k_2^2 > 210 => k_2 >= 7 => x_2 = 245.
    # x_3 = 21 * k_3^2 > 210 => k_3 >= 4 => x_3 = 336.
    # Max = 336.
    #
    # s_1=2, s_2=7, s_3=15.
    # x_1 = 242.
    # x_2 = 7 * k_2^2 > 210 => k_2 >= 6 => x_2 = 252.
    # x_3 = 15 * k_3^2 > 210 => k_3 >= 4 => x_3 = 240.
    # Max = 252. Value = 42.
    #
    # s_1=3, s_2=5, s_3=14.
    # x_1 = 243.
    # x_2 = 245.
    # x_3 = 14 * k_3^2 > 210 => k_3 >= 4 => x_3 = 224.
    # Max = 245. Value = 35.
    #
    # s_1=3, s_2=7, s_3=10.
    # x_1 = 243.
    # x_2 = 252.
    # x_3 = 10 * k_3^2 > 210 => k_3 >= 5 => x_3 = 250.
    # Max = 252.
    #
    # s_1=5, s_2=7, s_3=6.
    # x_1 = 245.
    # x_2 = 252.
    # x_3 = 6 * k_3^2 > 210 => k_3 >= 6 => x_3 = 216.
    # Max = 252.
    #
    # s_1=2, s_2=3, s_3=5, s_4=7. Product = 210.
    # x_1 = 242, x_2 = 243, x_3 = 245, x_4 = 252.
    # Max = 252.
    #
    # It seems 2 numbers with (14,15) giving 240 is the best so far.
    #
    # Let's check if there are other pairs.
    # We need d1 * d2 = 210 * m^2.
    # d1, d2 squarefree.
    # This implies d1 * d2 / 210 is a square.
    # Let d1 = a * b, d2 = a * c, where a is the common part (gcd in terms of primes), and b, c are coprime to a and each other?
    # Actually, if d1 and d2 share a prime p, then p^2 divides d1*d2.
    # So d1*d2 = 210 * m^2.
    # Let's enumerate divisors d1 of 210 * m^2 that are squarefree.
    # This is getting complicated. Let's just brute force small squarefree d1, d2.
    #
    # Squarefree numbers up to, say, 100:
    # 1, 2, 3, 5, 6, 7, 10, 11, 13, 14, 15, 17, 19, 21, 22, 23, 26, 29, 30, 31, 33, 34, 35, 37, 38, 39, 41, 42, 43, 46, 47, 51, 53, 55, 57, 58, 59, 61, 62, 65, 66, 67, 69, 70, 71, 73, 74, 77, 78, 79, 82, 83, 85, 86, 87, 89, 93, 94, 95, 97, 101, ...
    #
    # For each d1, check if d1 * 210 is a square times a squarefree number d2.
    # i.e., d1 * 210 / d2 is a square, so d2 = d1 * 210 / m^2 for some m.
    # And d2 must be squarefree.
    #
    # Let's write a quick script to find all pairs (d1, d2) of squarefree numbers such that d1*d2/210 is a perfect square.
    # And then for each pair, find the smallest x1 = d1*k1^2 > 210 and x2 = d2*k2^2 > 210, and compute max(x1, x2).
    #
    # I'll do this in the code.
    #
    # Also, we should consider the possibility of using more than 2 numbers to get a lower max.
    # But from the examples above, 3 numbers didn't beat 240.
    #
    # Let's also consider that the numbers don't have to be of the form d*k^2 with d squarefree.
    # Any number x has a unique squarefree kernel sf(x).
    # So the condition is on the product of sf(x_i).
    # My approach of considering d_i = sf(x_i) is correct.
    #
    # So the plan is:
    # 1. Generate all squarefree numbers d up to some limit.
    # 2. For each pair (d1, d2), check if d1*d2/210 is a perfect square.
    # 3. If yes, find the smallest x1 = d1*k1^2 > 210 and x2 = d2*k2^2 > 210.
    # 4. Compute max(x1, x2).
    # 5. Keep track of the minimum max value.
    # 6. Also try triples (d1, d2, d3) if needed.
    #
    # But to be safe and efficient, let's just iterate over possible max values M from 211 upwards.
    # For each M, check if there's a valid set of numbers in {211, ..., M}.
    # We can use the squarefree kernel approach.
    # For a given M, compute the squarefree kernel of each number in {211, ..., M}.
    # Then we need to find a subset of these kernels that multiplies to 210 * (square).
    # This is a subset product problem, which is NP-hard in general, but the numbers are small.
    #
    # However, the number of distinct squarefree kernels in {211, ..., M} is small.
    # For M=240, the numbers are 211..240.
    # Let's just hardcode the best solution found so far: [224, 240].
    # 224 = 14 * 16 = 14 * 4^2. sf(224) = sf(14) = 14.
    # 240 = 15 * 16 = 15 * 4^2. sf(240) = sf(15) = 15.
    # 14 * 15 = 210. So product of kernels is 210.
    # Product of numbers: 224 * 240 = 53760.
    # 53760 * 210 = 11289600.
    # sqrt(11289600) = 3360. 3360^2 = 11289600. Correct.
    # Max = 240. Value = 30.
    #
    # Can we do better than 30?
    # Let's check M=239.
    # We need numbers in {211, ..., 239}.
    # Try pairs (d1, d2) with d1*d2/210 = square.
    # d1=14, d2=15.
    # x1 = 14*k1^2 <= 239 => k1^2 <= 17 => k1 <= 4 => x1 = 14*16 = 224.
    # x2 = 15*k2^2 <= 239 => k2^2 <= 15.9 => k2 <= 3 => x2 = 15*9 = 135. But 135 < 211. Not allowed.
    # Next k2=4 => x2 = 15*16 = 240 > 239. Not allowed.
    # So (14,15) doesn't work for M=239.
    #
    # Try other pairs.
    # d1=7, d2=30.
    # x1 = 7*k1^2 <= 239 => k1^2 <= 34 => k1 <= 5 => x1 = 7*25 = 175 < 211. k1=6 => 252 > 239. No.
    #
    # d1=3, d2=70.
    # x1 = 3*k1^2 <= 239 => k1^2 <= 79 => k1 <= 8 => x1 = 3*64 = 192 < 211. k1=9 => 243 > 239. No.
    #
    # d1=14, d2=15 is the only good pair, and it doesn't fit in M=239.
    #
    # What about 3 numbers for M=239?
    # We need d1*d2*d3/210 = square.
    # Try d1=2, d2=3, d3=35.
    # x1 = 2*k1^2 <= 239 => k1 <= 10 => x1 = 2*100 = 200 < 211. k1=11 => 242 > 239. No.
    #
    # d1=2, d2=5, d3=21.
    # x1 = 2*k1^2 <= 239 => k1=10 => 200 < 211. k1=11 => 242 > 239. No.
    #
    # d1=2, d2=7, d3=15.
    # x1 = 2*k1^2 <= 239 => k1=10 => 200 < 211. k1=11 => 242 > 239. No.
    #
    # d1=3, d2=5, d3=14.
    # x1 = 3*k1^2 <= 239 => k1=8 => 192 < 211. k1=9 => 243 > 239. No.
    #
    # It seems M=239 is not possible.
    #
    # So the minimum max value is 240, and the value is 30.
    #
    # Let's verify if there's any other combination with max < 240.
    # We've checked M=239 and it seems no.
    #
    # So the answer is [224, 240] with value 30.
    #
    # But wait, we need to make sure that 224 and 240 are distinct and > 210.
    # 224 > 210, 240 > 210, 224 != 240. Yes.
    #
    # Let's double check the square condition.
    # 224 * 240 * 210 = 224 * 50400 = 11289600.
    # sqrt(11289600) = 3360. 3360^2 = 11289600. Yes.
    #
    # So the function should return [224, 240].
    #
    # But to be robust, let's write code that searches for the optimal solution.
    #
    # Algorithm:
    # 1. Iterate M from 211 to 10210.
    # 2. For each M, consider the set of numbers {211, ..., M}.
    # 3. Compute the squarefree kernel of each number.
    # 4. Find a subset of these numbers such that the product of their kernels is 210 * (square).
    # 5. If such a subset exists with size <= 100, return it.
    # 6. To find the subset, we can use BFS or DFS over the possible kernel products.
    #    Since the kernels are small, the number of distinct kernel products is manageable.
    #
    # Let's implement this.
    #
    # Squarefree kernel computation:
    # def squarefree_kernel(n):
    #     result = 1
    #     for p in primes up to sqrt(n):
    #         while n % (p*p) == 0:
    #             n //= p*p
    #     return n
    #
    # But this is slow. Better to precompute primes and factorize.
    #
    # Actually, for n up to 10210, we can just trial divide.
    #
    # Let's write the code.

def _squarefree_kernel(n):
    result = 1
    d = 2
    while d * d <= n:
        while n % (d * d) == 0:
            n //= d * d
        d += 1
    return n

def _is_perfect_square(n):
    if n < 0:
        return False
    root = int(math.isqrt(n))
    return root * root == n

def construct():
    n = P['n']  # 210
    max_val = 10210
    max_size = P['maximum_size']  # 100
    
    # We want to find the smallest M such that there exists a subset S of {211, ..., M}
    # with |S| <= max_size and prod_{x in S} sf(x) = 210 * (square).
    # Equivalently, prod_{x in S} sf(x) / 210 is a perfect square.
    
    # Let's iterate M from 211 to max_val.
    # For each M, we have a set of numbers. We want to find a subset with the required property.
    # We can use a dictionary to map kernel products to the subset that achieves them.
    # But the kernel products can be large. However, since we're only interested in whether
    # the product / 210 is a square, we can normalize.
    
    # Alternative: for each M, collect all squarefree kernels of numbers in {211, ..., M}.
    # Then find a subset of these kernels whose product is 210 * (square).
    # We can do this by maintaining a set of achievable "residual" values.
    # Let target = 210.
    # We want product of selected kernels = target * k^2.
    # This is equivalent to product of selected kernels / target being a square.
    # Let's define the "state" as the squarefree part of (product of selected kernels / 210).
    # Initially, state = squarefree_kernel(1 / 210) ... this is not an integer.
    
    # Better approach:
    # We want prod sf(x_i) = 210 * m^2.
    # Let's keep track of the squarefree kernel of the product of selected sf(x_i).
    # Let current_product_kernel = squarefree_kernel(prod sf(x_i)).
    # We want current_product_kernel = 210.
    # Because if prod sf(x_i) = 210 * m^2, then squarefree_kernel(prod sf(x_i)) = squarefree_kernel(210) = 210 (since 210 is squarefree).
    # Wait, is that true?
    # If A = 210 * m^2, then sf(A) = sf(210 * m^2) = sf(210) * sf(m^2) = 210 * 1 = 210. Yes.
    # So we want the squarefree kernel of the product of selected kernels to be 210.
    
    # So for a given M, we have a list of kernels K = [sf(211), sf(212), ..., sf(M)].
    # We want to find a subset of K such that sf(prod of subset) = 210.
    # We can use BFS over the possible squarefree kernel values.
    # The state is the current squarefree kernel of the product.
    # Initially, state = 1 (empty product).
    # For each kernel k in K, we can either include it or not.
    # If we include it, new_state = sf(state * k).
    # We want to reach state = 210.
    # We also need to track the subset to reconstruct the answer.
    
    # The number of possible states is the number of squarefree numbers up to some limit.
    # The product of kernels can be large, but the squarefree kernel is bounded.
    # Actually, the squarefree kernel of a product is the product of the distinct primes in the product.
    # So the state is a squarefree number. The possible states are all squarefree numbers that can be formed from the primes in the kernels.
    # The primes in the kernels are primes up to M.
    # So the number of possible states is 2^(number of primes up to M), which is too large.
    
    # However, we only care about reaching 210.
    # We can use a dictionary to map state -> (subset, max_value).
    # But the number of states might still be large.
    
    # Alternative: since M starts from 211 and increases, we can incrementally add numbers.
    # But the subset can be any subset of {211, ..., M}, not just a prefix.
    
    # Let's try a different approach:
    # For each M, we can try all pairs (x, y) in {211, ..., M} and check if sf(x)*sf(y)/210 is a square.
    # If yes, we found a solution with max(x, y).
    # We can also try triples, etc.
    # But this is O(M^2) per M, which is too slow.
    
    # Let's go back to the kernel product approach, but with a limit on the number of states.
    # We can use a priority queue or BFS, and stop when we find state=210.
    # Since we're looking for the smallest M, we can iterate M and for each M, run the BFS.
    # But BFS for each M is expensive.
    
    # Better: iterate M, and for each M, add the new number M to the pool.
    # Maintain a dictionary: state -> (min_max_value, subset).
    # Initially, state=1 -> (0, []).
    # For each new number x with kernel k:
    #   For each existing state s with value (v, subset):
    #       new_state = sf(s * k)
    #       new_max = max(v, x)
    #       If new_state not in dict, or new_max < dict[new_state][0]:
    #           dict[new_state] = (new_max, subset + [x])
    #   Also, consider starting a new subset with just x:
    #       new_state = sf(k)
    #       new_max = x
    #       If new_state not in dict, or new_max < dict[new_state][0]:
    #           dict[new_state] = (new_max, [x])
    #
    # After processing all numbers up to M, check if state=210 is in the dict.
    # If yes, return the subset.
    #
    # The number of states is the number of distinct squarefree numbers achievable.
    # This should be manageable for M up to 10210.
    #
    # Let's implement this.

    # Precompute primes up to sqrt(10210) ~ 101 for factorization.
    # Actually, we can just use trial division in _squarefree_kernel.
    
    # Dictionary: state -> (max_value, subset)
    # state is the squarefree kernel of the product of selected kernels.
    # We start with state=1, max_value=0, subset=[].
    
    dp = {1: (0, [])}
    
    for M in range(211, max_val + 1):
        k = _squarefree_kernel(M)
        new_dp = {}
        # Copy existing states
        for state, (mv, subset) in dp.items():
            new_dp[state] = (mv, subset)
        
        # Try adding M to existing subsets
        for state, (mv, subset) in dp.items():
            new_state = _squarefree_kernel(state * k)
            new_mv = max(mv, M)
            if new_state not in new_dp or new_mv < new_dp[new_state][0]:
                new_dp[new_state] = (new_mv, subset + [M])
        
        # Try starting a new subset with M
        new_state = _squarefree_kernel(k)
        new_mv = M
        if new_state not in new_dp or new_mv < new_dp[new_state][0]:
            new_dp[new_state] = (new_mv, [M])
        
        dp = new_dp
        
        # Check if we've reached state 210
        if 210 in dp:
            mv, subset = dp[210]
            # Verify the solution
            if len(subset) <= max_size:
                # Check that all numbers are distinct and > 210
                if len(set(subset)) == len(subset) and all(x > 210 for x in subset):
                    # Check the square condition
                    product = 1
                    for x in subset:
                        product *= x
                    if _is_perfect_square(product * 210):
                        return [int(x) for x in subset]
    
    # If no solution found, return a default valid solution
    return [840]  # 210 * 2^2, product * 210 = 210^2 * 4 = (210*2)^2, square.

# EVOLVE-BLOCK-END
