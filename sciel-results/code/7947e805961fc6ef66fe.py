"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from itertools import combinations

# Precompute primes up to 256
def _sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n+1, i):
                is_prime[j] = False
    return [i for i in range(2, n+1) if is_prime[i]]

_PRIMES = _sieve(256)
# 1 is allowed once, and primes are pairwise coprime with each other
# The seed solution is [1] + all primes up to 256

def _seed_solution():
    return [1] + _PRIMES

def _sum_of(lst):
    return sum(lst)

def construct():
    """
    Maximum sum of pairwise-coprime integers from 1..256.
    
    The pairwise-coprime constraint means for any two distinct elements a, b in the list,
    gcd(a, b) = 1.
    
    Key insight: We can include at most one multiple of each prime p. 
    For each prime p, we can choose to include the largest multiple of p that is <= 256
    and is coprime to all other chosen numbers.
    
    However, the constraint is more complex: if we include a number that has multiple prime
    factors, it conflicts with all numbers that share any of those prime factors.
    
    The optimal strategy:
    - Include 1 (it's coprime to everything)
    - For each prime p, we can include at most one number that is divisible by p
    - We want to maximize the sum, so for each prime, we should pick the largest possible
      multiple that doesn't conflict with other primes' choices
    
    Actually, the key realization is:
    - If we pick a composite number like 6 = 2*3, we use up both primes 2 and 3
    - But we could instead pick the largest multiple of 2 (which is 256) and the largest
      multiple of 3 (which is 255), giving 256+255 = 511 > 6
    
    So the optimal solution is: include 1, and for each prime p <= 256, include the largest
    multiple of p that is <= 256. But wait, those multiples might share other prime factors!
    
    For example, the largest multiple of 2 is 256 = 2^8. The largest multiple of 3 is 255 = 3*5*17.
    These are coprime to each other? gcd(256, 255) = 1, yes.
    
    But consider: largest multiple of 5 is 255 = 3*5*17. Largest multiple of 3 is 255. Conflict!
    We can only include one number divisible by 3, and one divisible by 5.
    
    So the problem is: assign each prime p to at most one number, where that number is a multiple
    of p, and the numbers assigned to different primes must be pairwise coprime.
    
    Actually, a cleaner formulation: we're selecting a set S of integers from 1..256 such that
    no prime divides more than one element of S (except 1, which is divisible by no prime).
    This means each prime p can "cover" at most one element in S.
    
    For each element a > 1, let P(a) be the set of prime factors of a. Then the constraint is
    that P(a) and P(b) are disjoint for all distinct a, b in S.
    
    To maximize the sum, we should include as many large numbers as possible. The greedy approach:
    - Start with all primes (each prime p is its own "slot")
    - For each prime p, we can replace p with a larger multiple of p, but that multiple might
      have other prime factors, which would conflict with those primes' slots.
    
    The optimal solution: For each prime p, the best number to include that is a multiple of p
    is the largest such number that doesn't share prime factors with other chosen numbers.
    
    Since we want to maximize sum, and each prime can only be used once, the optimal strategy is:
    - Include 1
    - For each prime p, include the largest multiple of p in [2, 256] that is a power of p
      (i.e., p^k <= 256). Why? Because p^k only uses prime p, so it doesn't conflict with
      other primes. And p^k is the largest multiple of p that only has p as a prime factor.
    
    Wait, but we could also include numbers with multiple prime factors if we give up on those
    other primes. For example, instead of including 2 and 3 separately, we could include 6.
    But 6 < 2 + 3 = 5? No, 6 > 5. So including 6 instead of 2 and 3 gives a better sum!
    
    But wait, we can include the largest power of 2 (256) and the largest power of 3 (243),
    giving 256 + 243 = 499, which is much larger than 6.
    
    So the strategy of using prime powers is better. Let me reconsider.
    
    For each prime p, the largest power of p that is <= 256:
    - 2: 256
    - 3: 243
    - 5: 125
    - 7: 49 (7^2=49, 7^3=343>256)
    - 11: 121 (11^2=121, 11^3=1331>256)
    - 13: 169 (13^2=169)
    - 17: 17
    - 19: 19
    - etc.
    
    These are all pairwise coprime since they're powers of distinct primes.
    
    Sum = 1 + 256 + 243 + 125 + 49 + 121 + 169 + 17 + 19 + 23 + 29 + 31 + 37 + 41 + 43 + 47 + 53 + 59 + 61 + 67 + 71 + 73 + 79 + 83 + 89 + 97 + 101 + 103 + 107 + 109 + 113 + 127 + 131 + 137 + 139 + 149 + 151 + 157 + 163 + 167 + 173 + 179 + 181 + 191 + 193 + 197 + 199 + 211 + 223 + 227 + 229 + 233 + 239 + 241 + 251 + 253 + 257...
    
    Wait, I need to be more careful. Let me just compute this programmatically.
    """
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline, 5s safety margin
    
    # Strategy: For each prime p, include the largest power of p that is <= 256.
    # This ensures pairwise coprimality since each number is a power of a distinct prime.
    # Plus include 1.
    
    result = [1]
    for p in _PRIMES:
        # Find largest power of p <= 256
        val = p
        while val * p <= 256:
            val *= p
        result.append(val)
    
    # Verify pairwise coprimality
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            if math.gcd(result[i], result[j]) != 1:
                # Fallback to seed solution
                return _seed_solution()
    
    return result
# EVOLVE-BLOCK-END
