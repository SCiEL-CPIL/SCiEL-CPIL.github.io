import numpy as np
from scipy.special import roots_laguerre
import time

def construct():
       start_time = time.time()
       deadline = start_time + 600.0 # 600s limit, use 630s guard? "KILLS ... after 630s". Use 600s safety margin.
       
       try:
           # Get roots of Laguerre polynomial L_12^0(x)
           # alpha = 0 for n=1? 
           # Standard Laguerre L_n^alpha. For 1D radial, alpha=0.
           # The prompt says "dimension n=1".
           # In quantum mechanics 1D, Laguerre is not standard, but in the context of the Cohn-Goncalves theorem for R^n, 
           # the weight is often x^{n-1} e^{-x/r^2} or similar.
           # For n=1, weight is e^{-x}. alpha=0.
           
           roots = roots_laguerre(12)
           # roots.x contains the 12 roots.
           # Ensure they are positive and sorted.
           r = sorted(roots.x)
           
           # Check time
           if time.time() > deadline:
               # Fallback to seed if somehow slow (unlikely)
               return [4.0 * (i + 1) for i in range(12)]
               
           return r
       except Exception:
           # Fallback to seed
           return [4.0 * (i + 1) for i in range(12)]
