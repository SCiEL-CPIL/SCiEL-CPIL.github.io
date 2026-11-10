import time
from fractions import Fraction

def construct():
       start_time = time.time()
       timeout = 55
       
       target = Fraction(5, 31)
       
       best_solution = [8, 28, 1736]
       best_max = 1736
       
       # Search for 3-term solutions
       # 1/d1 + 1/d2 + 1/d3 = 5/31
       # d1 < d2 < d3
       
       # d1 bounds
       # 3/d1 > 5/31 => d1 < 93/5 = 18.6
       # 1/d1 < 5/31 => d1 > 31/5 = 6.2
       
       for d1 in range(7, 19):
           if time.time() - start_time > timeout:
               break
           
           # R1 = 5/31 - 1/d1
           R1 = target - Fraction(1, d1)
           if R1 <= 0:
               continue
           
           # 1/d2 + 1/d3 = R1
           # d2 > d1
           # 1/d2 < R1 => d2 > 1/R1
           # 1/d2 > R1/2 => d2 < 2/R1 (since 1/d3 < 1/d2)
           
           min_d2 = d1 + 1
           max_d2 = int(2 / float(R1))
           
           for d2 in range(min_d2, max_d2 + 1):
               if time.time() - start_time > timeout:
                   break
               
               R2 = R1 - Fraction(1, d2)
               if R2 <= 0:
                   continue
               
               # d3 = 1/R2
               # R2 = num / den
               # d3 = den / num
               if R2.numerator != 1:
                   # Check if 1/R2 is integer
                   # R2 = a/b. 1/R2 = b/a. Integer if a divides b.
                   if R2.denominator % R2.numerator != 0:
                       continue
                   
               d3 = R2.denominator // R2.numerator
               
               if d3 <= d2:
                   continue
               
               # Verify
               if Fraction(1, d1) + Fraction(1, d2) + Fraction(1, d3) == target:
                   current_max = d3
                   if current_max < best_max:
                       best_max = current_max
                       best_solution = [d1, d2, d3]
       
       return best_solution
