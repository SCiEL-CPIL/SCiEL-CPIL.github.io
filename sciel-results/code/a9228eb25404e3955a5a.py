# EVOLVE-BLOCK-START
import time
from fractions import Fraction

N = 256
DEADLINE = 85  # 90s wall-clock limit, 5s safety margin

def construct():
    start_time = time.time()
    
    # Helper to check if adding z to S keeps it 3-AP-free
    def is_compatible(z, S, in_S):
        for a in S:
            if a == z:
                continue
            # Case 1: z is middle. a, z, b => b = 2z - a
            b = 2 * z - a
            if b != a and b != z and 1 <= b <= N and in_S[b]:
                return False
            # Case 2: z is largest. a, b, z => b = (a+z)/2
            if (a + z) % 2 == 0:
                b = (a + z) // 2
                if b != a and b != z and 1 <= b <= N and in_S[b]:
                    return False
            # Case 3: z is smallest. z, a, b => b = 2a - z
            b = 2 * a - z
            if b != a and b != z and 1 <= b <= N and in_S[b]:
                return False
        return True

    # Greedy initialization: add elements in order 1, 2, ..., N if compatible
    S = set()
    in_S = [False] * (N + 1)
    
    for x in range(1, N + 1):
        if is_compatible(x, S, in_S):
            S.add(x)
            in_S[x] = True
    
    # Pass 1: Try to add any missing element that's compatible (free gain)
    improved = True
    while improved:
        if time.time() - start_time > DEADLINE:
            break
        improved = False
        for z in range(1, N + 1):
            if not in_S[z]:
                if is_compatible(z, S, in_S):
                    S.add(z)
                    in_S[z] = True
                    improved = True
    
    # Pass 2: Swap larger elements with smaller ones to increase reciprocal sum
    improved = True
    while improved:
        if time.time() - start_time > DEADLINE:
            break
        improved = False
        
        # For each y in S (largest first), try to find best z < y not in S
        # such that (S - {y}) ∪ {z} is 3-AP-free and 1/z > 1/y
        for y in sorted(S, reverse=True):
            if time.time() - start_time > DEADLINE:
                break
            
            # Create temporary set without y
            S_temp = S - {y}
            in_S_temp = in_S[:]
            in_S_temp[y] = False
            
            # Find best z < y not in S that is compatible with S_temp
            best_z = None
            for z in range(1, y):
                if in_S[z]:
                    continue
                if is_compatible(z, S_temp, in_S_temp):
                    # 1/z > 1/y since z < y, so any valid z improves
                    best_z = z
                    break  # Smallest z gives largest 1/z
            
            if best_z is not None:
                S.remove(y)
                S.add(best_z)
                in_S[y] = False
                in_S[best_z] = True
                improved = True
    
    # Pass 3: Final "try to add" pass after swaps
    improved = True
    while improved:
        if time.time() - start_time > DEADLINE:
            break
        improved = False
        for z in range(1, N + 1):
            if not in_S[z]:
                if is_compatible(z, S, in_S):
                    S.add(z)
                    in_S[z] = True
                    improved = True
    
    return sorted(S)
# EVOLVE-BLOCK-END
