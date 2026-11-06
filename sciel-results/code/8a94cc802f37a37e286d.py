import random
import math
from itertools import combinations

def construct():
       # Seed solution
       best_A = [1, 7, 13]
       best_B = [4, 10, 16]
       best_score = 5
       
       # Precompute primes
       is_prime = [True] * 257
       is_prime[0] = is_prime[1] = False
       for i in range(2, int(256**0.5) + 1):
           if is_prime[i]:
               for j in range(i*i, 257, i):
                   is_prime[j] = False
               
       def score(A, B):
           sums = set()
           for a in A:
               for b in B:
                   sums.add(a + b)
           distinct_sums = list(sums)
           # Check pairwise coprime
           # We want the largest subset of distinct_sums that is pairwise coprime?
           # No, the task says "All DISTINCT values in A+B must be pairwise coprime".
           # This implies the SET of distinct values must be pairwise coprime.
           # If not, the configuration is INVALID?
           # "Return two lists... All DISTINCT values in A+B must be pairwise coprime."
           # This is a constraint on the returned A, B.
           # "Maximize the cardinality of A+B".
           # So if A+B has a pair of non-coprime distinct values, is it invalid?
           # "The evaluator measures the raw value... an invalid result... scores 0".
           # So we must return A, B such that the distinct sums ARE pairwise coprime.
           # If they are not, we cannot return them.
           # So score(A, B) is valid only if the condition holds.
           # If condition holds, score is len(distinct_sums).
           # Else, score is 0 (or invalid).
           
           # Check pairwise coprime for all pairs in distinct_sums
           n = len(distinct_sums)
           for i in range(n):
               for j in range(i+1, n):
                   if math.gcd(distinct_sums[i], distinct_sums[j]) != 1:
                       return 0
           return n

       # Check seed
       s = score(best_A, best_B)
       if s > best_score:
           best_score = s
           
       # Random Search
       # To maximize chance, we can bias towards primes in sums.
       # But simple random might find 6 or 7.
       
       # Let's try a structured search for 9 primes.
       # Iterate B from a small list of "good" B's?
       # Or just random.
       
       # Time limit 60s.
       # Let's do 500,000 random iterations.
       
       for _ in range(500000):
           A = list(random.sample(range(1, 129), 3))
           B = list(random.sample(range(1, 129), 3))
           s = score(A, B)
           if s > best_score:
               best_score = s
               best_A = A
               best_B = B
               if best_score == 9:
                   break
                   
       return [best_A, best_B]
