import math
import time
import random

def _is_disjoint(r1, m1, r2, m2):
    g = math.gcd(m1, m2)
    return (r1 - r2) % g != 0

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    # Use all even moduli: 2, 4, 6, ..., 128 (64 moduli)
    moduli = list(range(2, 129, 2))
    n = len(moduli)
    
    # Precompute gcds
    gcds = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                gcds[i][j] = math.gcd(moduli[i], moduli[j])
    
    # Order moduli by ascending value (smaller moduli have fewer residues, more constrained)
    # Actually, let's use a smarter ordering: process moduli with fewer "flexible" residues first
    # For now, use ascending order which naturally puts smaller moduli first
    
    residues = [None] * n
    assigned = [False] * n
    count = 0
    
    # Try greedy first with randomization
    def greedy_random():
        nonlocal start_time
        for attempt in range(500):
            if time.time() > deadline:
                return None
            res = [None] * n
            # Random order of assignment
            order = list(range(n))
            random.shuffle(order)
            # But process smaller moduli first for constraint propagation
            order.sort(key=lambda i: moduli[i])
            ok = True
            for idx in order:
                m_k = moduli[idx]
                # Find valid residues
                valid_res = []
                for r in range(m_k):
                    valid = True
                    for j in range(n):
                        if assigned[j] and res[j] is not None:
                            if not _is_disjoint(r, m_k, res[j], moduli[j]):
                                valid = False
                                break
                    if valid:
                        valid_res.append(r)
                if not valid_res:
                    ok = False
                    break
                res[idx] = random.choice(valid_res)
                assigned[idx] = True
            if ok:
                return res
        return None
    
    # Try greedy
    result = greedy_random()
    if result is not None:
        return [[result[i], moduli[i]] for i in range(n)]
    
    # Backtracking with MRV heuristic
    def backtrack():
        nonlocal start_time, count
        if time.time() > deadline:
            return False
        
        # Find unassigned variable with fewest valid values (MRV)
        best_idx = -1
        best_count = float('inf')
        best_candidates = []
        
        for i in range(n):
            if residues[i] is not None:
                continue
            m_i = moduli[i]
            candidates = []
            for r in range(m_i):
                valid = True
                for j in range(n):
                    if residues[j] is not None and i != j:
                        if not _is_disjoint(r, m_i, residues[j], moduli[j]):
                            valid = False
                            break
                if valid:
                    candidates.append(r)
            if len(candidates) < best_count:
                best_count = len(candidates)
                best_idx = i
                best_candidates = candidates
                if best_count == 0:
                    return False
                if best_count == 1:
                    break
        
        if best_idx == -1:
            return True  # All assigned
        
        m_k = moduli[best_idx]
        for r in best_candidates:
            residues[best_idx] = r
            count += 1
            if backtrack():
                return True
            residues[best_idx] = None
            count -= 1
        
        return False
    
    if backtrack():
        return [[residues[i], moduli[i]] for i in range(n)]
    
    # Final fallback: return a small valid set
    return [[0, 2], [1, 4]]
