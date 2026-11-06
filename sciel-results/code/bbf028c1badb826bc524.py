import random
import time

def construct():
    n = 192
    candidates = list(range(1, n + 1))
    
    # Helper: Check if x can be added to S
    def can_add(S, x):
        # 1. x must not divide sum of any 3 in S
        nS = len(S)
        for i in range(nS):
            for j in range(i + 1, nS):
                for k in range(j + 1, nS):
                    if (S[i] + S[j] + S[k]) % x == 0:
                        return False, []
        
        # 2. No z in S should divide (x + a + b) for any a,b in S
        for idx_z, z in enumerate(S):
            for i in range(nS):
                if i == idx_z: continue
                for j in range(i + 1, nS):
                    if j == idx_z: continue
                    if (x + S[i] + S[j]) % z == 0:
                        return False, []
                        
        return True, []

    # Helper: Find conflicts to guide swaps
    def find_conflicts(S, x):
        conflicts = []
        nS = len(S)
        # Type 1: x divides sum of 3 in S
        for i in range(nS):
            for j in range(i + 1, nS):
                for k in range(j + 1, nS):
                    if (S[i] + S[j] + S[k]) % x == 0:
                        conflicts.append({S[i], S[j], S[k]})
        
        # Type 2: z in S divides x + a + b
        for idx_z, z in enumerate(S):
            for i in range(nS):
                if i == idx_z: continue
                for j in range(i + 1, nS):
                    if j == idx_z: continue
                    if (x + S[i] + S[j]) % z == 0:
                        conflicts.append({z, S[i], S[j]})
                        
        return conflicts

    # Helper: Check if swap y -> x is valid
    def can_swap(S, y, x):
        R = [v for v in S if v != y]
        nR = len(R)
        
        # 1. x vs R: No 3 in R sum to mult of x
        for i in range(nR):
            for j in range(i + 1, nR):
                for k in range(j + 1, nR):
                    if (R[i] + R[j] + R[k]) % x == 0:
                        return False
        
        # 2. R vs x: No z in R divides x + a + b
        for idx_z, z in enumerate(R):
            for i in range(nR):
                if i == idx_z: continue
                for j in range(i + 1, nR):
                    if j == idx_z: continue
                    if (x + R[i] + R[j]) % z == 0:
                        return False
                        
        return True

    # Initial Greedy Construction
    S = []
    for x in range(n, 0, -1):
        ok, _ = can_add(S, x)
        if ok:
            S.append(x)
            
    # Local Search Phase
    improved = True
    while improved:
        improved = False
        
        # Try to add any element
        for x in candidates:
            if x in S:
                continue
            ok, _ = can_add(S, x)
            if ok:
                S.append(x)
                improved = True
                break
        
        if not improved:
            # Try swaps
            for x in candidates:
                if x in S:
                    continue
                # Check if x can be added directly (already failed, but maybe S changed? No, we broke if added)
                # So x is not addable.
                conflicts = find_conflicts(S, x)
                # Try to resolve conflicts by swapping
                for group in conflicts:
                    for y in group:
                        if y not in S:
                            continue
                        if can_swap(S, y, x):
                            # Perform swap
                            S = [v for v in S if v != y]
                            S.append(x)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
                    
    return S
