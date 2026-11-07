import time
from math import isqrt

# EVOLVE-BLOCK-START
def construct():
    """
    Build a set of distinct integers in 2..256 such that for every three DISTINCT
    members a, b, c, a does not divide b*c. Maximize cardinality.
    
    Key structural insight:
    - All primes in [2, 256] form a valid set (54 primes).
    - If a set contains prime p, it cannot contain any other multiple of p.
      Proof: if p, x, y are distinct members and p|x, then p | x*y, violating the condition.
    - So for each prime p, we can choose at most one element from {p, multiples of p in [2,256]}.
    - A composite c = p1^a * p2^b * ... (with multiple distinct primes) cannot be in a set
      that contains all of p1, p2, etc., because for any other prime q in the set:
      c | p1 * p2 * q? No wait, we need to check if c divides some pair product.
      
    Actually the constraint is: for all distinct a,b,c in S, a ∤ b*c.
    
    If S contains primes p1, p2, ..., pn and we want to add composite c:
    - For each prime pi in S (pi ≠ c): we need pi ∤ b*c for all b ≠ pi, b ≠ c in S.
      Since c might be divisible by pi, if c is divisible by pi, then pi | c, so pi | b*c for any b.
      Wait: pi | b*c means b*c ≡ 0 (mod pi). If pi | c, then yes pi | b*c for any b.
      So if c is divisible by prime pi and pi is in S, we have a violation (take a=pi, b=any other, c=c).
      Unless... wait, we need THREE DISTINCT members. So a=pi, b=some third element, c=c.
      If pi | c, then pi | b*c for any b. So this violates the condition.
      
    Therefore: if c is composite and divisible by prime p, and p is in S, then c CANNOT be in S.
    This means: for each prime p, we can include at most one element divisible by p.
    
    So the problem reduces to: for each prime p in [2,256], choose at most one number from
    the set of multiples of p in [2,256]. But also, if we choose a number c that is a multiple
    of multiple primes p1, p2, ..., we "use up" the slot for all those primes.
    
    To maximize count, we want to choose as many distinct primes' representatives as possible.
    The best strategy: for each prime p, choose p itself (the prime), giving 54 elements.
    
    Can we do better? Only if we can include a composite c that is a multiple of p, while
    NOT including p, and this doesn't reduce the total count. But replacing p with c keeps
    the count the same. However, if c is a multiple of p AND q (two primes), then we're using
    up slots for both p and q, which is worse than using p and q separately.
    
    So the maximum is 54 (all primes), UNLESS there's a way to include extra elements.
    
    Wait - what about numbers that are powers of a single prime? Like 4 = 2^2.
    If we include 4 instead of 2: 4 is divisible by 2. For the constraint:
    - a=4, b and c are other elements: 4 | b*c? 
    - If b and c are both odd (not divisible by 2), then b*c is odd, so 4 ∤ b*c. Good.
    - If one of b,c is even... but we can't have any other even number in S if we have 4,
      because 4 is divisible by 2, and any other even number x: 2 | x, so for a=4... 
      wait, let me re-check.
    
    If S contains 4 and some other even number x (x≠4): 
    - Take a=4, b=x, c=some odd prime p. Then 4 | x*p? x is even, so x*p is even but not necessarily divisible by 4.
    - If x=6: 4 | 6*p? 6*p = 2*3*p. For 4|6p, we need 4|2*3*p, i.e., 2|3*p, i.e., 2|p. But p is an odd prime, so no.
    - If x=10: 4 | 10*p? 10*p = 2*5*p. 4|10p means 2|5p, i.e., 2|p. No.
    - If x=14: similar.
    - But if x=8: 4 | 8*p? 8*p = 8p. 4|8p is always true! So 4 and 8 cannot coexist with any third element.
    
    So 4 and 8 conflict. What about 4 and 6?
    - a=4, b=6, c=p (odd prime): 4 | 6p? 6p=2*3*p. Need 4|2*3*p → 2|3*p → 2|p. No (p odd).
    - a=6, b=4, c=p: 6 | 4p? 4p=4*p. 6|4p means 6|4p. 4p/6 = 2p/3. For this to be integer, 3|2p, so 3|p. If p≠3, no.
    - a=p, b=4, c=6: p | 4*6=24? Only if p|24, so p∈{2,3}. p=2: 2|24 yes! Violation!
    
    So 4 and 6 and prime 2... wait, if we have 4 instead of 2, we don't have 2 in S.
    Let's say S = {4, 6, 3, 5, 7, ...} (primes except 2, plus 4 and 6).
    - a=3, b=4, c=6: 3 | 4*6=24? Yes! 3|24. Violation!
    
    So 4 and 6 conflict with prime 3.
    
    What about just 4 (replacing 2)? S = {4} ∪ {primes except 2}.
    - a=4, b=p, c=q (p,q odd primes, p≠q): 4 | p*q? p*q is odd*odd=odd, 4 is even, so 4 ∤ odd. Good.
    - a=p (odd prime), b=4, c=q (odd prime, q≠p): p | 4*q? 4q = 2^2 * q. p | 4q means p|4 or p|q. p is odd prime > 2, so p∤4. p≠q. So no.
    - a=q, b=4, c=p: similar, no.
    
    So {4} ∪ {all odd primes} is valid! Same size as all primes (54 elements).
    
    Can we add MORE? What if we use 4 AND 9 (replacing 2 and 3)?
    S = {4, 9} ∪ {primes except 2,3}.
    - a=4, b=9, c=p (odd prime ≠2,3): 4 | 9p? 9p is odd, 4 is even. No.
    - a=9, b=4, c=p: 9 | 4p? 4p = 4p. 9|4p means 9|4 or 9|p. 9∤4, 9∤p (p prime≠3). No.
    - a=p, b=4, c=9: p | 36? p|36 means p∈{2,3}. But p≠2,3. No.
    - a=4, b=p, c=q (p,q odd primes ≠2,3): 4 | pq? pq odd. No.
    - a=p, b=4, c=q: p | 4q? p|4q, p odd prime≠2,3. p|4? No. p|q? No (distinct). No.
    - a=p, b=q, c=4: p | 4q? Same as above. No.
    - a=9, b=p, c=q: 9 | pq? 9|pq means 9|p or 9|q. p,q primes≠3. No.
    
    So {4, 9} ∪ {primes except 2,3} is valid! Still 54 elements.
    
    Can we get 55? We'd need to find a composite that doesn't conflict. But any composite is divisible by at least one prime, and that prime's slot is "used." So we can't exceed 54.
    
    Actually wait - what about a number like 25=5^2? If we use 25 instead of 5, same count.
    
    The maximum is 54. Let me just return all primes, which is the simplest valid solution.
    """
    # Sieve of Eratosthenes to find all primes up to 256
    limit = 256
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    
    for i in range(2, isqrt(limit) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    return primes
# EVOLVE-BLOCK-END
