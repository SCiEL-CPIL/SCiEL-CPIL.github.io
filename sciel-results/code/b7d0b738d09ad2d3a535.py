import random
import time
from itertools import combinations

def construct():
    # Generate all 4-subsets of {1..16} as bitmasks
    all_sets = []
    for c in combinations(range(1, 17), 4):
        mask = 0
        for x in c:
            mask |= (1 << (x - 1))
        all_sets.append(mask)
    
    # Shuffle to randomize order for greedy init
    random.shuffle(all_sets)
    
    def can_add(S, F):
        """
        Check if S can be added to F without creating a sunflower.
        F is a list of masks.
        Sunflower condition for A, B in F and S:
        A & S == B & S == A & B
        """
        # Group elements of F by their intersection with S
        groups = {}
        for A in F:
            K = A & S
            if K in groups:
                groups[K].append(A)
            else:
                groups[K] = [A]
        
        # Check conflicts within each group
        for K, members in groups.items():
            if len(members) < 2:
                continue
            
            # We need to check if there exist A, B in members such that A & B == K
            # This is equivalent to (A ^ K) & (B ^ K) == 0
            # Since A & S = K and B & S = K, A \ K and B \ K are subsets of ~S.
            # Actually A \ K = A & ~K.
            # Let's just check pairs.
            # Optimization: If K has many bits, members are likely few.
            # If K is small, members might be many.
            
            # To speed up, we can check if any two "residuals" are disjoint.
            residuals = [A & ~K for A in members]
            n = len(residuals)
            for i in range(n):
                r_i = residuals[i]
                for j in range(i + 1, n):
                    r_j = residuals[j]
                    if (r_i & r_j) == 0:
                        return False
        return True

    # Initial Greedy Construction
    F = []
    for S in all_sets:
        if can_add(S, F):
            F.append(S)
    
    best_F = F
    best_size = len(F)
    
    # Simulated Annealing
    # Time limit management
    start_time = time.time()
    time_limit = 55.0 # Leave 5s for output conversion
    
    T = 1.0
    cooling = 0.999
    
    # Precompute indices for all_sets not in F? 
    # It's easier to just pick random from all_sets and check if in F.
    # Or maintain a list of candidates.
    # Given 1820 sets, picking random is fine.
    
    while time.time() - start_time < time_limit:
        if T < 1e-4:
            break
            
        move_type = random.random()
        
        if move_type < 0.5:
            # Try to Add
            S = random.choice(all_sets)
            if S not in F:
                if can_add(S, F):
                    F.append(S)
                    # Acceptance: Size increased, so always accept?
                    # In SA for maximization, we accept if better.
                    # This is a "hill climbing" step.
                    # To allow backtracking, we need removals.
                    pass
        elif move_type < 0.8:
            # Try to Remove
            if F:
                idx = random.randrange(len(F))
                A = F[idx]
                # Delta E = -1 (size decreases)
                # Acceptance prob = exp(-Delta E / T) = exp(1/T)
                # Since T > 0, exp(1/T) > 1? 
                # Standard SA: E is cost. Minimize cost.
                # Here we Maximize Size. Cost = -Size.
                # Remove: Size decreases by 1. Cost increases by 1.
                # Delta Cost = 1.
                # Accept with prob exp(-1/T).
                if random.random() < (2.71828 ** (-1.0 / T)):
                    F.pop(idx)
        else:
            # Try to Swap
            if F:
                idx = random.randrange(len(F))
                A = F[idx]
                S = random.choice(all_sets)
                if S not in F:
                    # Check if S can be added to F \ {A}
                    F_temp = F[:idx] + F[idx+1:]
                    if can_add(S, F_temp):
                        # Size remains same. 
                        # We should accept with some probability to explore.
                        # Or just accept if it's "better" in some heuristic?
                        # For now, accept with prob 0.5 to mix.
                        if random.random() < 0.5:
                            F[idx] = S
                            # Note: F_temp was a copy, so we just replace in F
                            # But we need to ensure F is consistent.
                            # F[idx] = S is correct.
                            pass

        if len(F) > best_size:
            best_size = len(F)
            best_F = F[:]
            
        T *= cooling

    # Convert best_F back to lists
    result = []
    for mask in best_F:
        s = []
        for i in range(16):
            if mask & (1 << i):
                s.append(i + 1)
        result.append(sorted(s))
        
    return result
