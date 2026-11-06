# EVOLVE-BLOCK-START
import random
from math import gcd
from collections import defaultdict

def construct():
    N = 128
    # Precompute lcm for all pairs (1..128)
    lcms = [[0]*N for _ in range(N)]
    for i in range(1, N):
        for j in range(i+1, N):
            lcms[i-1][j-1] = (i*j)//gcd(i,j)
            lcms[j-1][i-1] = lcms[i-1][j-1]
    
    # Precompute conflicts
    # conflicts[x] is a set of frozensets of pairs (a,b) such that {a,b,x} is a bad triple
    # a,b,x are 0-indexed
    conflicts = [set() for _ in range(N)]
    for a in range(1, N):
        for b in range(a+1, N):
            L = lcms[a-1][b-1]
            for c in range(1, N):
                if c == a or c == b:
                    continue
                if lcms[a-1][c-1] == L and lcms[b-1][c-1] == L:
                    ac, bc, cc = a-1, b-1, c-1
                    conflicts[ac].add(frozenset((bc, cc)))
                    conflicts[bc].add(frozenset((ac, cc)))
                    conflicts[cc].add(frozenset((ac, bc)))
    
    # Compute degrees
    degree = [len(conflicts[i]) for i in range(N)]
    
    def greedy_solve(order):
        selected = set()
        for idx in order:
            can_add = True
            for pair in conflicts[idx]:
                a, b = tuple(pair)
                if a in selected and b in selected:
                    can_add = False
                    break
            if can_add:
                selected.add(idx)
        return selected

    # Generate multiple orderings
    orderings = []
    # 1. Degree ascending
    orderings.append(sorted(range(N), key=lambda i: (degree[i], i)))
    # 2. Degree descending
    orderings.append(sorted(range(N), key=lambda i: (-degree[i], i)))
    # 3. Value ascending
    orderings.append(sorted(range(N), key=lambda i: i))
    # 4. Value descending
    orderings.append(sorted(range(N), key=lambda i: -i))
    
    # 5. Random permutations
    random.seed(42) # Fixed seed for reproducibility in evaluation? 
                    # Actually, better to use random without seed or seed based on time?
                    # The evaluator runs the function. If we use random, results vary.
                    # It's better to be deterministic or use a fixed set of random seeds.
    for seed in range(10):
        random.seed(seed)
        order = list(range(N))
        random.shuffle(order)
        orderings.append(order)

    best_set = set()
    for order in orderings:
        s = greedy_solve(order)
        if len(s) > len(best_set):
            best_set = s

    # If still small, try a simple local search on the best set?
    # For now, return the best greedy result.
    result = sorted(i+1 for i in best_set)
    if not result:
        result = [1]
    return result
# EVOLVE-BLOCK-END
