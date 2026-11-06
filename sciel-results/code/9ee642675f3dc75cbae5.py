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
    
    # Pass 2.5: 2-swap moves (remove 2, add 2) to escape local optima
    improved = True
    while improved:
        if time.time() - start_time > DEADLINE:
            break
        improved = False
        
        # Try removing pairs (y1, y2) and adding (z1, z2) where
        # 1/z1 + 1/z2 > 1/y1 + 1/y2
        for y1 in sorted(S, reverse=True):
            if time.time() - start_time > DEADLINE:
                break
            for y2 in sorted(S, reverse=True):
                if y2 >= y1:
                    continue
                if time.time() - start_time > DEADLINE:
                    break
                
                # Temporarily remove y1 and y2
                S_temp = S - {y1, y2}
                in_S_temp = in_S[:]
                in_S_temp[y1] = False
                in_S_temp[y2] = False
                
                # Find best pair (z1, z2) not in S, z1 < y1 or z2 < y2,
                # such that both are compatible with S_temp and each other
                # and 1/z1 + 1/z2 > 1/y1 + 1/y2
                base_val = 1.0/y1 + 1.0/y2
                best_pair = None
                best_val = base_val
                
                # Try all z1 < max(y1, y2) not in S
                for z1 in range(1, max(y1, y2)):
                    if in_S[z1]:
                        continue
                    if not is_compatible(z1, S_temp, in_S_temp):
                        continue
                    
                    # Now try z2 > z1, z2 < max(y1,y2), not in S, compatible with S_temp ∪ {z1}
                    in_S_temp2 = in_S_temp[:]
                    in_S_temp2[z1] = True
                    S_temp2 = S_temp | {z1}
                    
                    for z2 in range(z1 + 1, max(y1, y2)):
                        if in_S[z2]:
                            continue
                        if not is_compatible(z2, S_temp2, in_S_temp2):
                            continue
                        
                        val = 1.0/z1 + 1.0/z2
                        if val > best_val + 1e-12:
                            best_val = val
                            best_pair = (z1, z2)
                    
                    # Early exit: if we found a very good pair, stop
                    if best_pair is not None and best_val > base_val + 0.05:
                        break
                
                if best_pair is not None:
                    S.discard(y1)
                    S.discard(y2)
                    S.add(best_pair[0])
                    S.add(best_pair[1])
                    in_S[y1] = False
                    in_S[y2] = False
                    in_S[best_pair[0]] = True
                    in_S[best_pair[1]] = True
                    improved = True
                    break  # Restart the outer loop with updated S
            
            if improved:
                break
    
    # Pass 3: Re-run single swaps after 2-swaps
    improved = True
    while improved:
        if time.time() - start_time > DEADLINE:
            break
        improved = False
        
        for y in sorted(S, reverse=True):
            if time.time() - start_time > DEADLINE:
                break
            
            S_temp = S - {y}
            in_S_temp = in_S[:]
            in_S_temp[y] = False
            
            best_z = None
            for z in range(1, y):
                if in_S[z]:
                    continue
                if is_compatible(z, S_temp, in_S_temp):
                    best_z = z
                    break
            
            if best_z is not None:
                S.remove(y)
                S.add(best_z)
                in_S[y] = False
                in_S[best_z] = True
                improved = True
    
    # Pass 4: Final "try to add" pass
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
    
    # Pass 5: One more round of single swaps (polish)
    if time.time() - start_time < DEADLINE - 5:
        improved = True
        while improved:
            if time.time() - start_time > DEADLINE:
                break
            improved = False
            for y in sorted(S, reverse=True):
                if time.time() - start_time > DEADLINE:
                    break
                S_temp = S - {y}
                in_S_temp = in_S[:]
                in_S_temp[y] = False
                best_z = None
                for z in range(1, y):
                    if in_S[z]:
                        continue
                    if is_compatible(z, S_temp, in_S_temp):
                        best_z = z
                        break
                if best_z is not None:
                    S.remove(y)
                    S.add(best_z)
                    in_S[y] = False
                    in_S[best_z] = True
                    improved = True
        
        # Final add pass
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
