import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    best_set = []
    best_size = 0
    
    def greedy_construct(order):
        S = []
        residues = {}  # a -> set of (u+v) % a for u,v in S\{a}, u!=v
        sums3 = set()  # set of b+c+d for b,c,d in S, distinct
        
        for x in order:
            if time.time() > deadline:
                break
            
            # Check validity of x
            valid = True
            
            # 1. Check if x divides any sum in sums3
            if sums3:
                for s in sums3:
                    if s % x == 0:
                        valid = False
                        break
            
            if valid:
                # 2. Check if any a in S divides x + u + v
                # i.e., (x + u + v) % a == 0  =>  (u+v) % a == (-x) % a
                for a in S:
                    target = (-x) % a
                    if target in residues[a]:
                        valid = False
                        break
            
            if valid:
                # Add x to S
                # Update sums3: add x + u + v for all u,v in S, u!=v
                for u, v in combinations(S, 2):
                    sums3.add(x + u + v)
                
                # Update residues for existing a
                for a in S:
                    for u in S:
                        if u != a:
                            residues[a].add((x + u) % a)
                
                # Initialize residues[x]
                residues[x] = set()
                for u, v in combinations(S, 2):
                    residues[x].add((u + v) % x)
                
                S.append(x)
        
        return S

    candidates = list(range(1, 193))
    
    # Initial attempt: Descending
    order1 = sorted(candidates, reverse=True)
    S1 = greedy_construct(order1)
    if len(S1) > best_size:
        best_set = S1
        best_size = len(S1)
        
    # Randomized attempts
    attempts = 0
    while time.time() < deadline - 5.0:
        if attempts % 3 == 0:
            order = sorted(candidates, reverse=True)
            mid = len(order) // 2
            random.shuffle(order[:mid])
        elif attempts % 3 == 1:
            order = candidates[:]
            random.shuffle(order)
        else:
            order = sorted(candidates)
            random.shuffle(order[:10])
            
        S = greedy_construct(order)
        if len(S) > best_size:
            best_set = S
            best_size = len(S)
            
        attempts += 1
        
    return best_set
