       def construct():
           import math
           from collections import defaultdict
           
           # Limit for powerful numbers
           L = 10**8
           
           powerfuls = []
           
           # Max b such that b^3 <= L
           max_b = int(round(L ** (1/3)))
           
           # Precompute square-free numbers up to max_b
           # Or just check on the fly
           
           for b in range(1, max_b + 1):
               # Check square free
               is_sf = True
               x = b
               p = 2
               while p * p <= x:
                   if x % (p*p) == 0:
                       is_sf = False
                       break
                   x //= p
                   p += 1
               if not is_sf:
                   continue
               
               b3 = b ** 3
               max_a = int(math.isqrt(L // b3))
               for a in range(1, max_a + 1):
                   p = a * a * b3
                   powerfuls.append(p)
           
           powerfuls.sort()
           # unique?
           # The representation a^2 b^3 with b square-free is unique.
           # So powerfuls contains unique numbers.
           
           M = len(powerfuls)
           P = powerfuls
           
           sum_counts = defaultdict(int)
           
           # Counting loop
           # We can optimize by breaking if ai + P[j] > some limit? No, we want all.
           
           for i in range(M):
               ai = P[i]
               for j in range(i, M):
                   sum_counts[ai + P[j]] += 1
           
           best_N = max(sum_counts, key=sum_counts.get)
           best_count = sum_counts[best_N]
           
           P_set = set(powerfuls)
           pairs = []
           for a in powerfuls:
               b = best_N - a
               if b < a:
                   continue
               if b in P_set:
                   pairs.append([a, b])
           
           return {'N': best_N, 'pairs': pairs}
       