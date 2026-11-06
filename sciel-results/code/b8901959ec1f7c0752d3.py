import time
import random
from itertools import combinations

def construct():
    start = time.time()
    DEADLINE = start + 55.0  # Safety margin

    # All 4-subsets as bitmasks
    all_sets = []
    for combo in combinations(range(16), 4):
        m = 0
        for x in combo:
            m |= (1 << x)
        all_sets.append(m)

    def mask_to_list(m):
        return sorted((i+1) for i in range(16) if (m >> i) & 1)

    best_family = []

    def is_valid_add(family, inter, C):
        # Check if C can be added to family without forming a sunflower
        # Sunflower: A, B, C such that A&B = A&C = B&C
        for A in family:
            v = A & C
            if v in inter[A]:
                for B in inter[A][v]:
                    if (B & C) == v:
                        return False
        return True

    def add_set(family, inter, C):
        family.append(C)
        inter[C] = {}
        for A in family[:-1]:
            v = A & C
            if v not in inter[A]:
                inter[A][v] = []
            inter[A][v].append(C)
            if v not in inter[C]:
                inter[C][v] = []
            inter[C][v].append(A)

    def remove_set(family, inter, C):
        family.remove(C)
        for A in family:
            v = A & C
            if v in inter[A]:
                if C in inter[A][v]:
                    inter[A][v].remove(C)
                if not inter[A][v]:
                    del inter[A][v]
            if v in inter[C]:
                if A in inter[C][v]:
                    inter[C][v].remove(A)
                if not inter[C][v]:
                    del inter[C][v]
        del inter[C]

    def greedy_pass(order):
        family = []
        inter = {}
        
        for C in order:
            if is_valid_add(family, inter, C):
                add_set(family, inter, C)
                
        return family, inter

    # Multiple restarts with different random orders
    attempts = 0
    while time.time() < DEADLINE:
        attempts += 1
        order = all_sets[:]
        random.shuffle(order)
        fam, inter = greedy_pass(order)
        
        if len(fam) > len(best_family):
            best_family = fam[:]
        
        if best_family and len(best_family) >= 250:
            break
            
    return [mask_to_list(m) for m in best_family]
