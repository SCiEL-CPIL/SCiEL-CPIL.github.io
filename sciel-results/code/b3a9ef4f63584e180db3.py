"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit
    
    best_score = 0
    best_set = [0, 1, 3]  # seed fallback
    
    def compute_score(U):
        U = sorted(set(U))
        if not U or 0 not in U:
            return -1
        max_u = max(U)
        if max_u == 0:
            return 0
        
        # Compute sumset and diffset efficiently
        sumset = set()
        diffset = set()
        n = len(U)
        for i in range(n):
            a = U[i]
            for j in range(i, n):
                b = U[j]
                sumset.add(a + b)
            for j in range(n):
                b = U[j]
                diffset.add(a - b)
        
        n_sum = len(sumset)
        n_diff = len(diffset)
        
        if n_sum == 0:
            return -1
        
        base = 2 * max_u + 1
        if base <= 1:
            return 0
        ratio = n_diff / n_sum
        if ratio <= 1:
            return 1.0
        score = 1 + math.log(ratio) / math.log(base)
        return score
    
    # Pre-compute seed
    best_score = compute_score([0, 1, 3])
    
    def try_set(cand):
        nonlocal best_score, best_set
        cand = sorted(set(cand))
        if not cand or 0 not in cand:
            return
        if max(cand) == 0:
            return
        s = compute_score(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)
    
    # Phase 1: Generate diverse structured candidates
    # 1a. Small exhaustive search
    for max_u in range(1, 25):
        if time.time() > deadline - 30:
            break
        elements = list(range(1, max_u + 1))
        for size in range(2, min(7, max_u + 1)):
            if len(elements) < size:
                continue
            for comb in combinations(elements, size):
                cand = sorted([0] + list(comb))
                try_set(cand)
    
    # 1b. Geometric sequences
    for max_u in range(10, 300):
        if time.time() > deadline - 30:
            break
        for r in [2, 3, 4, 5, 6]:
            geo = [0, 1]
            while geo[-1] * r <= max_u:
                geo.append(geo[-1] * r)
            try_set(geo)
        
        # 1c. Two blocks: {0..a} U {b..b+a}
        for a in range(1, 8):
            for b in range(a + 1, max_u - a + 1):
                if time.time() > deadline - 30:
                    break
                block1 = list(range(0, a + 1))
                block2 = list(range(b, b + a + 1))
                cand = sorted(set(block1 + block2))
                if max(cand) <= max_u:
                    try_set(cand)
        
        # 1d. Mixed: {0..a} U {b, c, d}
        for a in range(1, 6):
            for b in range(a + 1, max_u + 1):
                for c in range(b + 1, max_u + 1):
                    for d in range(c + 1, max_u + 1):
                        cand = sorted(list(range(0, a + 1)) + [b, c, d])
                        try_set(cand)
    
    # 1e. Two scaled APs with coprime steps
    for max_u in range(20, 200, 5):
        if time.time() > deadline - 30:
            break
        for da in range(1, 15):
            for db in range(1, 15):
                if da == db or da % db == 0 or db % da == 0:
                    continue
                A = []
                v = 0
                while v <= max_u:
                    A.append(v)
                    v += da
                for b0 in range(1, min(da + db, max_u) + 1):
                    B = []
                    v = b0
                    while v <= max_u:
                        B.append(v)
                        v += db
                    cand = sorted(set(A + B))
                    if 0 in cand and len(cand) >= 3:
                        try_set(cand)
    
    # Phase 2: Intensive local search from multiple starting points
    starting_points = [list(best_set)]
    
    # Add some diverse starting points
    for r in [2, 3, 4]:
        geo = [0, 1]
        while geo[-1] * r <= 100:
            geo.append(geo[-1] * r)
        starting_points.append(list(geo))
    
    for a in [2, 3, 4]:
        for b in [a+2, a+5, a+10]:
            if b + a <= 100:
                block1 = list(range(0, a + 1))
                block2 = list(range(b, b + a + 1))
                starting_points.append(sorted(set(block1 + block2)))
    
    for start in starting_points:
        if time.time() > deadline - 15:
            break
        
        current = list(start)
        current_score = compute_score(current)
        no_improve = 0
        max_no_improve = 100
        
        while time.time() < deadline - 15 and no_improve < max_no_improve:
            improved = False
            max_cur = max(current) if current else 0
            
            # Try adding an element
            if len(current) < 12:
                candidates = set()
                for x in current:
                    for dx in [-3, -2, -1, 1, 2, 3]:
                        if x + dx >= 1:
                            candidates.add(x + dx)
                candidates.update(range(max_cur + 1, max_cur + 8))
                candidates.discard(0)
                
                for new_el in sorted(candidates):
                    if new_el in current:
                        continue
                    cand = sorted(current + [new_el])
                    s = compute_score(cand)
                    if s > best_score:
                        best_score = s
                        best_set = list(cand)
                        current = list(cand)
                        current_score = s
                        improved = True
                        break
            
            # Try replacing an element
            if len(current) >= 3:
                for i in range(len(current)):
                    if current[i] == 0:
                        continue
                    for new_el in range(max(1, current[i] - 4), current[i] + 5):
                        if new_el in current:
                            continue
                        cand = list(current)
                        cand[i] = new_el
                        cand = sorted(set(cand))
                        if 0 not in cand:
                            continue
                        s = compute_score(cand)
                        if s > best_score:
                            best_score = s
                            best_set = list(cand)
                            current = list(cand)
                            current_score = s
                            improved = True
                            break
                    if improved:
                        break
            
            # Try removing an element
            if len(current) > 3:
                for i in range(len(current)):
                    if current[i] == 0:
                        continue
                    cand = [x for j, x in enumerate(current) if j != i]
                    s = compute_score(cand)
                    if s > best_score:
                        best_score = s
                        best_set = list(cand)
                        current = list(cand)
                        current_score = s
                        improved = True
                        break
            
            if not improved:
                no_improve += 1
            else:
                no_improve = 0
    
    # Phase 3: Final polish - try more aggressive moves on best set
    if time.time() < deadline - 5:
        current = list(best_set)
        no_improve = 0
        max_no_improve = 50
        
        while time.time() < deadline - 5 and no_improve < max_no_improve:
            improved = False
            max_cur = max(current) if current else 0
            
            # Wider replacement range
            for i in range(len(current)):
                if current[i] == 0:
                    continue
                for new_el in range(max(1, current[i] - 8), current[i] + 9):
                    if new_el in current:
                        continue
                    cand = list(current)
                    cand[i] = new_el
                    cand = sorted(set(cand))
                    if 0 not in cand:
                        continue
                    s = compute_score(cand)
                    if s > best_score:
                        best_score = s
                        best_set = list(cand)
                        current = list(cand)
                        improved = True
                        break
                if improved:
                    break
            
            # Try adding multiple elements
            if len(current) < 10:
                max_cur = max(current)
                for new_el in range(max_cur + 1, max_cur + 15):
                    cand = sorted(current + [new_el])
                    s = compute_score(cand)
                    if s > best_score:
                        best_score = s
                        best_set = list(cand)
                        current = list(cand)
                        improved = True
                        break
            
            if not improved:
                no_improve += 1
            else:
                no_improve = 0
    
    return list(best_set)
# EVOLVE-BLOCK-END
