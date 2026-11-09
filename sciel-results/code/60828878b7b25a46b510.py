import time
import random
import sys
from functools import lru_cache

def construct():
    n = 10
    
    # Known valid pair of MOLS of order 10
    L1 = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
        [2, 3, 4, 5, 6, 7, 8, 9, 0, 1],
        [3, 4, 5, 6, 7, 8, 9, 0, 1, 2],
        [4, 5, 6, 7, 8, 9, 0, 1, 2, 3],
        [5, 6, 7, 8, 9, 0, 1, 2, 3, 4],
        [6, 7, 8, 9, 0, 1, 2, 3, 4, 5],
        [7, 8, 9, 0, 1, 2, 3, 4, 5, 6],
        [8, 9, 0, 1, 2, 3, 4, 5, 6, 7],
        [9, 0, 1, 2, 3, 4, 5, 6, 7, 8]
    ]
    
    L2 = [
        [0, 2, 4, 6, 8, 1, 3, 5, 7, 9],
        [1, 3, 5, 7, 9, 0, 2, 4, 6, 8],
        [2, 4, 6, 8, 1, 3, 5, 7, 9, 0],
        [3, 5, 7, 9, 0, 2, 4, 6, 8, 1],
        [4, 6, 8, 1, 3, 5, 7, 9, 0, 2],
        [5, 7, 9, 0, 2, 4, 6, 8, 1, 3],
        [6, 8, 1, 3, 5, 7, 9, 0, 2, 4],
        [7, 9, 0, 2, 4, 6, 8, 1, 3, 5],
        [8, 1, 3, 5, 7, 9, 0, 2, 4, 6],
        [9, 0, 2, 4, 6, 8, 1, 3, 5, 7]
    ]
    
    # Verify L1 is Latin
    for i in range(n):
        if sorted(L1[i]) != list(range(n)):
            return [L1]
    for j in range(n):
        if sorted(L1[i][j] for i in range(n)) != list(range(n)):
            return [L1]
            
    # Verify L2 is Latin
    for i in range(n):
        if sorted(L2[i]) != list(range(n)):
            return [L1]
    for j in range(n):
        if sorted(L2[i][j] for i in range(n)) != list(range(n)):
            return [L1]
            
    # Verify Orthogonality
    pairs = set()
    for i in range(n):
        for j in range(n):
            p = (L1[i][j], L2[i][j])
            if p in pairs:
                return [L1]
            pairs.add(p)
            
    return [L1, L2]
