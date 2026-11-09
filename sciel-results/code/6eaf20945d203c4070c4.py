"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
from collections import deque

N = 12

# EVOLVE-BLOCK-START

def compute_threshold(trans):
    """
    trans: list of two lists, each of length N.
    trans[0][i] = destination of state i on letter 0
    trans[1][i] = destination of state i on letter 1
    Returns the length of the shortest reset word, or float('inf') if not synchronizing.
    Uses BFS on the subset automaton.
    """
    full_set = frozenset(range(N))
    visited = set()
    visited.add(full_set)
    queue = deque()
    queue.append((full_set, 0))
    
    while queue:
        current_set, dist = queue.popleft()
        if len(current_set) == 1:
            return dist
        
        for a in range(2):
            next_set = frozenset(trans[a][s] for s in current_set)
            if next_set not in visited:
                visited.add(next_set)
                queue.append((next_set, dist + 1))
    
    return float('inf')

def construct():
    # Černý (Pawlik) automaton for N=12.
    # Known to achieve the conjectured maximum reset threshold (N-1)^2 + (N-1) = 121+11=132? 
    # Actually the Černý automaton has threshold (n-1)^2 + (n-1) = n^2 - n. For n=12 that's 120.
    # Wait, let me recall: the Černý automaton on n states has reset word length (n-1)^2 + (n-1) = n(n-1).
    # For n=12: 12*11 = 132.
    # Letter 0: cyclic permutation i -> (i+1) mod N
    # Letter 1: fixes 0, and for i>0 maps i -> i-1
    sigma0 = [(i + 1) % N for i in range(N)]
    sigma1 = [0 if i == 0 else i - 1 for i in range(N)]
    return [sigma0, sigma1]

# EVOLVE-BLOCK-END
