import time
import numpy as np

def construct():
       # Paley tournament of order 67
       p = 67
       # Quadratic residues mod 67
       squares = {x*x % p for x in range(1, p)}
       
       orientation = []
       for i in range(p):
           for j in range(i+1, p):
               # i -> j if (j-i) % p is a square
               if (j - i) % p in squares:
                   orientation.append(0)
               else:
                   orientation.append(1)
                   
       return {'n': p, 'orientation': orientation}
