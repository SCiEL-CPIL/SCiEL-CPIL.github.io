import itertools
import time

def construct():
       start_time = time.time()
       deadline = start_time + 60 # Safety margin for 90s limit
       
       # Primes in [2, 256]
       is_prime = [True] * 257
       is_prime[0] = is_prime[1] = False
       for i in range(2, int(256**0.5) + 1):
           if is_prime[i]:
               for j in range(i*i, 257, i):
                   is_prime[j] = False
       
       range_max = 128
       # We will search in increasing order of max(A) and max(B)
       # This biases towards smaller sums, which have higher prime density.
       
       best_score = 0
       best_A = [1, 2, 3]
       best_B = [1, 2, 3]
       
       # To handle the combinatorial explosion, we can limit the search range initially
       # and expand if needed, or just search [1, 128] but break early on success?
       # The task says "Maximize cardinality". 9 is the max.
       # If we find a 9-distinct-prime sumset, we can return it immediately as it's optimal.
       
       # Strategy: Search for a 9-prime sumset.
       # Iterate A, then B.
       
       # Optimization:
       # If we fix A, the condition is:
       # For all a in A, b in B: a+b is prime.
       
       # Let's iterate A.
       # We can restrict B to be "compatible" with A.
       
       # Since 1.8 score was achieved, a solution exists.
       # We just need to find it fast.
       
       # Let's try searching A, B in [1, 80].
       # If not found, expand to 128.
       
       def search(limit):
           for A in itertools.combinations(range(1, limit+1), 3):
               # Check time
               if time.time() > deadline:
                   return None
               
               # For this A, we need to find B such that A+B are all prime.
               # We can iterate B.
               # To speed up, we can pre-filter B candidates?
               
               # Let's just iterate B.
               # If limit is 80, C(80,3) = 82160.
               # Pairs: 82160^2 is too big.
               
               # We need a smarter way.
               # Notice that if a+b is prime, and a+b' is prime, etc.
               
               # Let's try to find B for a given A.
               # This is a constraint satisfaction problem.
               
               # Alternative: Iterate over pairs of B?
               
               # Let's stick to a simpler heuristic:
               # Random sampling of A and B from [1, 128] with replacement? No, distinct required.
               # Random sampling might be faster if the density of valid pairs is high enough.
               
               # But systematic is safer.
               
               # Let's optimize the inner loop.
               # For a fixed A, we can iterate B but break early.
               # Most B will fail on the first sum.
               
               # Let's try to iterate B and check.
               # We can vectorize the check for a single B against A.
               
               for B in itertools.combinations(range(1, limit+1), 3):
                   if time.time() > deadline:
                       return None
                   
                   # Check if all 9 sums are prime
                   # Sums: A[0]+B[0], A[0]+B[1], ...
                   valid = True
                   for a in A:
                       for b in B:
                           if not is_prime[a+b]:
                               valid = False
                               break
                       if not valid:
                           break
                   
                   if valid:
                       # Check distinctness
                       sums = [a+b for a in A for b in B]
                       if len(set(sums)) == 9:
                           return (list(A), list(B))
           
           return None
       
       # Try small limits first
       for limit in [20, 30, 40, 50, 60, 80, 100, 128]:
           if time.time() > deadline:
               break
           res = search(limit)
           if res:
               return res
       
       # If no 9-prime sumset found, try to maximize cardinality with coprimality
       # This is harder. But given the hint, a 9-prime solution likely exists in small range.
       
       return [1, 7, 13], [4, 10, 16]
