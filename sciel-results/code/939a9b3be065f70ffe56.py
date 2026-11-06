import time
import random
import math

N = 256
MAX_EL = 64
DEADLINE = 55.0  # Safety margin

def _sum_bitset(arr):
    """Return a bitmask where bit s is set if s can be written as a+b (a<=b) from arr."""
    bs = 0
    n = len(arr)
    for i in range(n):
        ai = arr[i]
        for j in range(i, n):
            bs |= 1 << (ai + arr[j])
    return bs

def _diff_bitset(arr):
    """Return a bitmask where bit d is set if d = |a-b| (a!=b) occurs in arr."""
    bs = 0
    n = len(arr)
    for i in range(n):
        ai = arr[i]
        for j in range(n):
            if i != j:
                d = abs(ai - arr[j])
                bs |= 1 << d
    return bs

def _cost(A, B, sA, dA, sB, dB):
    """Calculate total conflict cost."""
    nA = len(A)
    nB = len(B)
    
    # Sum conflicts
    total_sums_A = nA * (nA + 1) // 2
    unique_sums_A = sA.bit_count()
    conf_A = total_sums_A - unique_sums_A
    
    total_sums_B = nB * (nB + 1) // 2
    unique_sums_B = sB.bit_count()
    conf_B = total_sums_B - unique_sums_B
    
    # Diff conflicts
    conf_diff = (dA & dB).bit_count()
    
    return conf_A + conf_B + conf_diff

def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE
    
    best_A = []
    best_B = []
    best_size = 0
    
    # We will try to build sets of increasing size
    # Start with small sizes to find valid configurations quickly, then grow
    for target_size in range(2, MAX_EL + 1):
        if time.time() > deadline - 5:
            break
            
        # Run Simulated Annealing for this target size
        # We allow the size to fluctuate slightly during SA, but aim for target_size
        # Actually, strictly maintaining size is harder. Let's allow size changes but penalize deviation?
        # Simpler: Start with random sets of size target_size.
        
        # If target_size is large, random init is very likely to have high cost.
        # We can use the previous best valid set (if size < target_size) as a starting point and try to add elements.
        
        if best_size > 0 and best_size < target_size:
            A = list(best_A)
            B = list(best_B)
            # We want to reach target_size for A and B combined? Or each?
            # The problem says "at most 64 elements each". Maximize |A|+|B|.
            # So we want to grow both.
            # Let's aim for |A| + |B| = target_size * 2? No, target_size is just a counter.
            # Let's define target_total = target_size * 2.
            # If we have best_size total, we want to add 2 more elements?
            # Let's just run SA to minimize cost while trying to increase total size.
            
            pass # Logic below handles this
            
        # Initialize random A and B
        # To make it easier, let's pick random subsets
        all_elements = list(range(1, N + 1))
        random.shuffle(all_elements)
        
        # Split into A and B
        # Let's start with small sets if target is small
        kA = target_size // 2
        kB = target_size - kA
        
        if kA < 1: kA = 1
        if kB < 1: kB = 1
        
        A = all_elements[:kA]
        B = all_elements[kA:kA+kB]
        
        sA = _sum_bitset(A)
        dA = _diff_bitset(A)
        sB = _sum_bitset(B)
        dB = _diff_bitset(B)
        curr_cost = _cost(A, B, sA, dA, sB, dB)
        
        best_cost = curr_cost
        best_A_local = list(A)
        best_B_local = list(B)
        
        T = 10.0
        T_min = 0.01
        cooling = 0.999
        
        # Time budget for this target size
        start_time = time.time()
        max_time_for_target = 2.0 # seconds per target size
        
        while time.time() < start_time + max_time_for_target and T > T_min:
            if time.time() > deadline - 2:
                break
                
            # Generate a move
            move_type = random.random()
            
            new_A = list(A)
            new_B = list(B)
            
            if move_type < 0.2:
                # Add to A
                if len(A) < MAX_EL:
                    x = random.randint(1, N)
                    if x not in A:
                        new_A.append(x)
            elif move_type < 0.4:
                # Add to B
                if len(B) < MAX_EL:
                    x = random.randint(1, N)
                    if x not in B:
                        new_B.append(x)
            elif move_type < 0.6:
                # Remove from A
                if len(A) > 1:
                    idx = random.randint(0, len(A)-1)
                    new_A.pop(idx)
            elif move_type < 0.8:
                # Remove from B
                if len(B) > 1:
                    idx = random.randint(0, len(B)-1)
                    new_B.pop(idx)
            elif move_type < 0.9:
                # Move A -> B
                if len(A) > 1 and len(B) < MAX_EL:
                    idx = random.randint(0, len(A)-1)
                    x = new_A.pop(idx)
                    if x not in new_B:
                        new_B.append(x)
            else:
                # Move B -> A
                if len(B) > 1 and len(A) < MAX_EL:
                    idx = random.randint(0, len(B)-1)
                    x = new_B.pop(idx)
                    if x not in new_A:
                        new_A.append(x)
            
            if new_A == A and new_B == B:
                continue
                
            # Calculate new cost
            sA_new = _sum_bitset(new_A)
            dA_new = _diff_bitset(new_A)
            sB_new = _sum_bitset(new_B)
            dB_new = _diff_bitset(new_B)
            new_cost = _cost(new_A, new_B, sA_new, dA_new, sB_new, dB_new)
            
            delta = new_cost - curr_cost
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                A = new_A
                B = new_B
                sA = sA_new
                dA = dA_new
                sB = sB_new
                dB = dB_new
                curr_cost = new_cost
                
                if curr_cost < best_cost:
                    best_cost = curr_cost
                    best_A_local = list(A)
                    best_B_local = list(B)
                    
                    # If we found a valid solution with larger total size, update global best
                    total_size = len(A) + len(B)
                    if best_cost == 0 and total_size > best_size:
                        best_size = total_size
                        best_A = list(A)
                        best_B = list(B)
            
            T *= cooling
            
        # After SA for this target size, if we found a valid solution, it's in best_A/best_B
        # If not, we can try to use the best local solution to seed the next run?
        # The next loop iteration will generate new random sets.
        # To improve, we could seed the next iteration with the best local solution if it was close.
        
    # Final check: ensure nonempty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
        
    # Final validation check just in case
    sA = _sum_bitset(best_A)
    dA = _diff_bitset(best_A)
    sB = _sum_bitset(best_B)
    dB = _diff_bitset(best_B)
    
    nA = len(best_A)
    nB = len(best_B)
    
    valid_A = sA.bit_count() == nA * (nA + 1) // 2
    valid_B = sB.bit_count() == nB * (nB + 1) // 2
    valid_diff = (dA & dB) == 0
    
    if not (valid_A and valid_B and valid_diff):
        # Fallback to a simple valid pair if SA failed completely
        # {1, 3, 12, 28} is a Sidon set? 
        # 1,3: sums 2,4,6. diffs 2.
        # 12: sums 13,15,24. diffs 11,9.
        # 28: sums 29,31,40. diffs 27,25,16.
        # All sums unique? 2,4,6,13,15,24,29,31,40. Yes.
        # Diffs: 2, 11,9, 27,25,16. All unique.
        # Let's use two such sets with disjoint diffs.
        # Set 1: {1, 3, 12, 28} -> Diffs: {2, 9, 11, 16, 25, 27}
        # Set 2: Need disjoint diffs.
        # Try {100, 105, 120, 150}?
        # Diffs: 5, 20, 50, 15, 45, 30.
        # Intersection with Set 1 diffs? 
        # Set 1: 2, 9, 11, 16, 25, 27
        # Set 2: 5, 15, 20, 30, 45, 50
        # No intersection.
        # Check Sidon for Set 2:
        # 100,105,120,150
        # Sums: 200, 205, 250, 210, 255, 270, 240, 270 (Wait, 120+150=270, 105+165? No. 100+150=250, 105+145? No.
        # Pairs:
        # 100+100=200
        # 100+105=205
        # 100+120=220
        # 100+150=250
        # 105+105=210
        # 105+120=225
        # 105+150=255
        # 120+120=240
        # 120+150=270
        # 150+150=300
        # Sums: 200, 205, 220, 250, 210, 225, 255, 240, 270, 300. All unique.
        # So [1, 3, 12, 28] and [100, 105, 120, 150] are valid.
        
        best_A = [1, 3, 12, 28]
        best_B = [100, 105, 120, 150]
        
    return [best_A, best_B]
