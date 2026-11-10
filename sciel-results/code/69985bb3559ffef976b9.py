import numpy as np
import random

def construct():
       n = 30
       w = [1.0 / i for i in range(1, n + 1)]
       
       best_coeffs = None
       best_val = float('inf')
       
       for _ in range(1000): # 1000 restarts
           coeffs = np.array([random.choice([-1, 0, 1]) for _ in range(n)])
           val = abs(np.sum(coeffs * w))
           
           improved = True
           while improved:
               improved = False
               for i in range(n):
                   for new_c in [-1, 0, 1]:
                       if new_c == coeffs[i]:
                           continue
                       old_c = coeffs[i]
                       coeffs[i] = new_c
                       new_val = abs(np.sum(coeffs * w))
                       if new_val < val and new_val > 0:
                           val = new_val
                           improved = True
                           break
                       coeffs[i] = old_c
                   if improved:
                       break
           
           if val < best_val:
               best_val = val
               best_coeffs = coeffs.copy()
               
           if best_val < 1e-10: # Close to optimal?
               break
               
       return [int(c) for c in best_coeffs]
