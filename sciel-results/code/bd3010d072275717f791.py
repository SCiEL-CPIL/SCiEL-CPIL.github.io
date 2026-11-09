import time
from fractions import Fraction
import random

def construct():
       start_time = time.time()
       deadline = start_time + 55.0
       
       denominators = list(range(2, 49))
       
       # Greedy initialization
       selected = []
       total = Fraction(0)
       for i, d in enumerate(denominators):
           if total + Fraction(1, d) < 1:
               selected.append(i)
               total += Fraction(1, d)
       
       best_indices = selected
       best_sum = total
       
       # Local Search: Try swapping one selected index with one unselected index
       # Or just checking if removing one and adding two helps?
       # Given the time, simple swap is safer.
       
       unselected = [i for i in range(len(denominators)) if i not in set(best_indices)]
       
       improved = True
       while improved and time.time() < deadline:
           improved = False
           # Try single swaps
           for i in list(best_indices):
               # Remove i
               sum_without = best_sum - Fraction(1, denominators[i])
               for j in unselected:
                   new_sum = sum_without + Fraction(1, denominators[j])
                   if new_sum < 1 and new_sum > best_sum:
                       best_indices.remove(i)
                       best_indices.append(j)
                       unselected.remove(j)
                       unselected.append(i)
                       best_sum = new_sum
                       improved = True
                       break
               if improved:
                   break
                   
       return best_indices
