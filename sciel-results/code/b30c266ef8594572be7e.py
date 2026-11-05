"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations
import random

def construct():
    deadline = time.time() + 85
    N = 128
    
    # Precompute GCD table
    G = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            G[i][j] = g
            G[j][i] = g

    def can_add(S, x):
        """Check if x can be added to set S (list) without creating a bad triple."""
        n = len(S)
        for i in range(n):
            a = S[i]
            ga_x = G[a][x]
            for j in range(i + 1, n):
                b = S[j]
                if G[a][b] == ga_x == G[b][x]:
                    return False
        return True

    def greedy_build(order):
        """Build a set by iterating through order and adding if valid."""
        S = []
        for x in order:
            if can_add(S, x):
                S.append(x)
        return S

    def local_search(S):
        """Improve set S using addition and swaps."""
        S = list(S)
        S_set = set(S)
        improved = True
        while improved:
            improved = False
            
            # Try to add any element not in S
            for x in range(1, N + 1):
                if x not in S_set:
                    if can_add(S, x):
                        S.append(x)
                        S_set.add(x)
                        improved = True
                        # Restart loop to check for more additions
                        break 
            
            # Try swaps
            if not improved:
                for x in S:
                    for y in range(1, N + 1):
                        if y not in S_set:
                            # Check if swapping x with y is valid
                            # S without x
                            S_temp = [z for z in S if z != x]
                            if can_add(S_temp, y):
                                # Perform swap
                                S = S_temp + [y]
                                S_set.remove(x)
                                S_set.add(y)
                                improved = True
                                break
                    if improved:
                        break
        
        return S

    best_set = []
    
    # Deterministic Orders
    orders = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        sorted(range(1, N + 1), key=lambda x: -len([d for d in range(1, x+1) if x % d == 0])), # Desc divisors
        sorted(range(1, N + 1), key=lambda x: -sum(1 for i in range(1, N+1) if gcd(x,i) > 1)) # Desc non-coprime count
    ]
    
    for order in orders:
        S = greedy_build(order)
        S = local_search(S)
        if len(S) > len(best_set):
            best_set = S
        if time.time() > deadline - 10:
            break

    # Randomized Restart
    if time.time() < deadline - 5:
        random.seed(42)
        while time.time() < deadline - 2:
            order = list(range(1, N + 1))
            random.shuffle(order)
            S = greedy_build(order)
            S = local_search(S)
            if len(S) > len(best_set):
                best_set = S

    # Final safety check
    if not best_set:
        best_set = [1]
        
    return best_set
# EVOLVE-BLOCK-END
