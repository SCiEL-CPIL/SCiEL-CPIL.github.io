"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start = time.time()
    DEADLINE = start + 55.0  # Safety margin before 90s limit
    
    # All 4-subsets as bitmasks (16 bits, each bit represents element 0..15)
    all_sets = []
    for combo in combinations(range(16), 4):
        m = 0
        for x in combo:
            m |= (1 << x)
        all_sets.append(m)
    
    def mask_to_list(m):
        return sorted((i + 1) for i in range(16) if (m >> i) & 1)
    
    def is_valid_add(family, idx, C):
        """Check if C can be added without forming a sunflower triple."""
        for A in family:
            v = A & C
            if v in idx[A]:
                for B in idx[A][v]:
                    if (B & C) == v:
                        return False
        return True
    
    def add_set(family, idx, C):
        family.append(C)
        idx[C] = {}
        for A in family[:-1]:
            v = A & C
            if v not in idx[A]:
                idx[A][v] = []
            idx[A][v].append(C)
            if v not in idx[C]:
                idx[C][v] = []
            idx[C][v].append(A)
    
    def remove_set(family, idx, C):
        """Remove C from family and update indices."""
        family.remove(C)
        for A in family:
            v = A & C
            if v in idx[A] and C in idx[A][v]:
                idx[A][v].remove(C)
            if v in idx.get(C, {}):
                pass  # We'll delete idx[C] below
        if C in idx:
            del idx[C]
    
    def greedy_pass(order):
        family = []
        idx = {}
        for C in order:
            if is_valid_add(family, idx, C):
                add_set(family, idx, C)
        return family, idx
    
    def local_search(family, idx, all_sets, deadline):
        """Try to add more sets or swap to improve family size."""
        family = list(family)
        idx = {k: {kk: list(vv) for kk, vv in v.items()} for k, v in idx.items()}
        in_family = set(family)
        candidates = [s for s in all_sets if s not in in_family]
        random.shuffle(candidates)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for s in candidates:
                if time.time() >= deadline:
                    break
                if s in in_family:
                    continue
                if is_valid_add(family, idx, s):
                    add_set(family, idx, s)
                    in_family.add(s)
                    improved = True
                    break
                # Try swap: remove one member, add s
                # Only try a limited number of swaps
                for rem_idx in range(min(5, len(family))):
                    if time.time() >= deadline:
                        break
                    old = family[rem_idx]
                    # Check if family without old + s is valid
                    # Temporarily remove old
                    remove_set(family, idx, old)
                    in_family.discard(old)
                    if is_valid_add(family, idx, s):
                        add_set(family, idx, s)
                        in_family.add(s)
                        improved = True
                        break
                    else:
                        # Put old back
                        add_set(family, idx, old)
                        in_family.add(old)
                if improved:
                    break
            # Rebuild candidates
            in_family = set(family)
            candidates = [s for s in all_sets if s not in in_family]
            random.shuffle(candidates)
        
        return family, idx
    
    best_family = []
    random.seed(42)
    
    num_passes = 0
    while time.time() < DEADLINE - 5.0:  # Leave 5s for local search
        num_passes += 1
        order = all_sets[:]
        random.shuffle(order)
        fam, idx = greedy_pass(order)
        
        if len(fam) > len(best_family):
            best_family = fam
        
        if len(best_family) >= 250:
            break
    
    # Local search on best family if we have time
    if best_family and time.time() < DEADLINE - 2.0:
        # Rebuild idx for best_family
        idx = {}
        for i, C in enumerate(best_family):
            idx[C] = {}
            for A in best_family[:i]:
                v = A & C
                if v not in idx[A]:
                    idx[A][v] = []
                idx[A][v].append(C)
                if v not in idx[C]:
                    idx[C][v] = []
                idx[C][v].append(A)
        
        improved_fam, _ = local_search(best_family, idx, all_sets, DEADLINE - 1.0)
        if len(improved_fam) > len(best_family):
            best_family = improved_fam
    
    return [mask_to_list(m) for m in best_family]
# EVOLVE-BLOCK-END
