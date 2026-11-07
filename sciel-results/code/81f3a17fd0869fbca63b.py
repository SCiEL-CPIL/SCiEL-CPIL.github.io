import time
import sys

def construct():
    n = 10
    
    # Generate L1 and L2 using Z5 x Z2 construction
    # i -> (x, u) where x = i % 5, u = i // 5
    # j -> (y, v) where y = j % 5, v = j // 5
    
    L1 = [[0 for _ in range(n)] for _ in range(n)]
    L2 = [[0 for _ in range(n)] for _ in range(n)]
    
    for i in range(n):
        x = i % 5
        u = i // 5
        for j in range(n):
            y = j % 5
            v = j // 5
            
            # L1: (x+y, u+v)
            a1 = (x + y) % 5
            b1 = (u + v) % 2
            L1[i][j] = 5 * a1 + b1
            
            # L2: (x-y, v)
            a2 = (x - y) % 5
            b2 = v
            L2[i][j] = 5 * a2 + b2
            
    return [L1, L2]
