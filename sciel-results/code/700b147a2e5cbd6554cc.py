import time
import random
from itertools import combinations

def construct():
        start_time = time.time()
        deadline = start_time + 55.0 # 5s margin
        
        # Generate all 4-subsets of 1..16 as bitmasks
        # Bit i represents element i+1? Or just 0..15.
        # Task says [1, 16].
        elements = range(16)
        all_sets = []
        for combo in combinations(elements, 4):
            mask = 0
            for x in combo:
                mask |= (1 << x)
            all_sets.append(mask)
        
        N = len(all_sets)
        best_family = []
        best_size = 0
        
        # Precompute nothing complex, just rely on fast bitwise ops.
        
        while time.time() < deadline:
            random.shuffle(all_sets)
            current_family = []
            
            for m in all_sets:
                # Check if m can be added
                ok = True
                n = len(current_family)
                for i in range(n):
                    a = current_family[i]
                    for j in range(i+1, n):
                        b = current_family[j]
                        k = a & b
                        if (a & m) == k and (b & m) == k:
                            ok = False
                            break
                    if not ok:
                        break
                
                if ok:
                    current_family.append(m)
                    
            if len(current_family) > best_size:
                best_size = len(current_family)
                best_family = current_family[:]
                
                # If we found a very large family, we might want to try to improve it
                # via local search, but for now, let's just keep restarting.
                # The greedy is fast enough to do many restarts.
                
        # Convert masks back to sorted lists of 1..16
        result = []
        for m in best_family:
            s = []
            for i in range(16):
                if m & (1 << i):
                    s.append(i + 1)
            result.append(sorted(s))
            
        return result
