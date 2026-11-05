import time
import sys
from itertools import combinations

# Increase recursion limit just in case, though depth is small
sys.setrecursionlimit(10000)

def construct():
    # The S(4,5,11) design is unique.
    # It can be generated from 6 base blocks modulo 11.
    # Base blocks:
    base_blocks = [
        (0, 1, 3, 4, 5),
        (0, 1, 2, 7, 8),
        (0, 1, 4, 6, 9),
        (0, 1, 5, 7, 10),
        (0, 2, 3, 8, 9),
        (0, 2, 5, 6, 10)
    ]
    
    # Generate all 66 blocks of S(4,5,11)
    s4511_blocks = []
    for base in base_blocks:
        for shift in range(11):
            block = tuple((x + shift) % 11 for x in base)
            s4511_blocks.append(block)
            
    # Construct S(5,6,12) blocks
    # Points are 0..11
    # Type 1: {11} U B for B in S(4,5,11)
    blocks = []
    for b in s4511_blocks:
        blocks.append(tuple(list(b) + [11]))
        
    # Type 2: Complement of B in {0..10}
    # The complement of a 5-set in 11 points is a 6-set.
    for b in s4511_blocks:
        comp = tuple(x for x in range(11) if x not in b)
        blocks.append(comp)
        
    return blocks
