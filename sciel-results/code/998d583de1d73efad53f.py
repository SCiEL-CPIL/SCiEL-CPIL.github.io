import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin
    
    N = 128
    TARGETS = list(range(128, 257))  # 129 targets
    T_START = 128
    T_END = 256
    NUM_TARGETS = T_END - T_START + 1
    
    def get_mults(A_set):
        """Return array of multiplicities for all targets."""
        A_list = sorted(A_set)
        mults = np.zeros(NUM_TARGETS, dtype=int)
        for i, t in enumerate(TARGETS):
            count = 0
            for a in A_list:
                if a > t // 2:
                    break
                b = t - a
                if b in A_set:
                    count += 1
            mults[i] = count
        return mults

    def local_search(A_set, max_iters=300):
        A_set = set(A_set)
        mults = get_mults(A_set)
        if np.min(mults) == 0:
            return A_set, float('inf')
            
        current_max = np.max(mults)
        current_size = len(A_set)
        stuck_count = 0
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            improved = False
            
            # 1. Try removing an element
            for a in list(A_set):
                new_mults = mults.copy()
                for i, t in enumerate(TARGETS):
                    b = t - a
                    if 0 <= b <= N and b in A_set:
                        new_mults[i] -= 1
                
                if np.min(new_mults) == 0:
                    continue
                    
                new_max = np.max(new_mults)
                new_size = current_size - 1
                
                if new_max < current_max:
                    A_set.remove(a)
                    mults = new_mults
                    current_max = new_max
                    current_size = new_size
                    improved = True
                    stuck_count = 0
                    break
                elif new_max == current_max and new_size < current_size:
                    A_set.remove(a)
                    mults = new_mults
                    current_max = new_max
                    current_size = new_size
                    improved = True
                    stuck_count = 0
                    break
            
            if improved:
                continue
                
            # 2. Try swapping an element
            a_in = random.choice(list(A_set))
            for _ in range(50):
                a_out = random.randint(0, N)
                if a_out in A_set:
                    continue
                    
                temp_mults = mults.copy()
                # Remove a_in
                for i, t in enumerate(TARGETS):
                    b = t - a_in
                    if 0 <= b <= N and b in A_set:
                        temp_mults[i] -= 1
                        
                # Add a_out
                for i, t in enumerate(TARGETS):
                    b = t - a_out
                    if 0 <= b <= N:
                        if b == a_out:
                            temp_mults[i] += 1
                        else:
                            if (b in A_set and b != a_in) or b == a_out:
                                temp_mults[i] += 1
                                
                if np.min(temp_mults) == 0:
                    continue
                    
                new_max = np.max(temp_mults)
                
                if new_max < current_max:
                    A_set.remove(a_in)
                    A_set.add(a_out)
                    mults = temp_mults
                    current_max = new_max
                    improved = True
                    stuck_count = 0
                    break
                elif new_max == current_max:
                    A_set.remove(a_in)
                    A_set.add(a_out)
                    mults = temp_mults
                    improved = True
                    stuck_count += 1
                    if stuck_count > 10:
                        break
                    break
                        
            if not improved:
                break
                
        return A_set, current_max

    best_A = None
    best_mult = float('inf')
    
    seeds = []
    
    # Seed 1: The seed program's output (likely complete, large)
    seeds.append(set(range(64, 129)))
    
    # Seed 2: Random
    for _ in range(5):
        k = random.randint(20, 40)
        A_set = set(random.sample(range(0, N+1), k))
        seeds.append(A_set)
        
    # Seed 3: Structured
    for d in [2, 3, 4]:
        seeds.append(set(range(0, N+1, d)))
        
    # Seed 4: Intervals
    for k in [20, 30, 40]:
        seeds.append(set(range(0, k+1)) | set(range(N-k, N+1)))

    for A_set in seeds:
        if time.time() > deadline - 10:
            break
            
        # Ensure completeness
        mults = get_mults(A_set)
        if np.min(mults) == 0:
            # Greedy completion
            changed = True
            while changed and time.time() < deadline - 5:
                changed = False
                mults = get_mults(A_set)
                missing = [t for t, m in zip(TARGETS, mults) if m == 0]
                if not missing:
                    break
                    
                t = missing[0]
                best_x = None
                best_gain = 0
                for x in range(0, N+1):
                    if x in A_set:
                        continue
                    y = t - x
                    if y < 0 or y > N:
                        continue
                    gain = 0
                    if y in A_set:
                        gain = 1
                    for t2 in missing[1:]:
                        y2 = t2 - x
                        if 0 <= y2 <= N and y2 in A_set:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                        best_y = y
                        
                if best_gain > 0:
                    A_set.add(best_x)
                    if best_y not in A_set:
                        A_set.add(best_y)
                    changed = True
                else:
                    x = t // 2
                    y = t - x
                    A_set.add(x)
                    A_set.add(y)
                    changed = True
        
        mults = get_mults(A_set)
        if np.min(mults) == 0:
            continue
            
        A_set, mult = local_search(A_set, max_iters=300)
        
        if mult < best_mult:
            best_mult = mult
            best_A = set(A_set)

    if best_A is None:
        best_A = set(range(64, 129))
        best_mult = np.max(get_mults(best_A))
        
    return list(best_A)
