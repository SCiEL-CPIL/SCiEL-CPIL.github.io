import random
import time

def construct():
    start_time = time.time()
    timeout = 85
    
    target_min = 128
    target_max = 256
    targets = list(range(target_min, target_max + 1))
    num_targets = len(targets)
    
    def get_multiplicities(A):
        A_set = set(A)
        mults = [0] * num_targets
        for a in A:
            for b in A:
                if a > b:
                    continue
                k = a + b
                if target_min <= k <= target_max:
                    idx = k - target_min
                    mults[idx] += 1
        return mults

    def is_valid(mults):
        return all(m > 0 for m in mults)

    def get_score(mults):
        if not is_valid(mults):
            return float('inf')
        return max(mults)

    best_A = list(range(64, 129))
    best_mults = get_multiplicities(best_A)
    best_score = get_score(best_mults)
    
    sizes_to_try = [65, 70, 80, 90, 100]
    
    for M in sizes_to_try:
        for restart in range(5):
            if time.time() - start_time > timeout:
                break
                
            # Random init
            A = random.sample(range(129), M)
            A_set = set(A)
            
            # Repair
            repair_attempts = 0
            while time.time() - start_time < timeout and repair_attempts < 1000:
                mults = get_multiplicities(A)
                if is_valid(mults):
                    break
                
                # Find missing k
                missing_ks = [i for i, m in enumerate(mults) if m == 0]
                if not missing_ks:
                    break
                k_miss_idx = random.choice(missing_ks)
                k_miss = targets[k_miss_idx]
                
                # Pick c in A
                c = random.choice(A)
                b = k_miss - c
                if 0 <= b <= 128 and b not in A_set:
                    # Swap a random a with b
                    a_idx = random.randint(0, len(A) - 1)
                    a_out = A[a_idx]
                    A[a_idx] = b
                    A_set.remove(a_out)
                    A_set.add(b)
                    repair_attempts += 1
                else:
                    # Try another c or just random swap
                    repair_attempts += 1
                    # If stuck, do a random swap to escape
                    a_idx = random.randint(0, len(A) - 1)
                    a_out = A[a_idx]
                    b_in = random.randint(0, 128)
                    if b_in != a_out and b_in not in A_set:
                        A[a_idx] = b_in
                        A_set.remove(a_out)
                        A_set.add(b_in)
            
            if not is_valid(get_multiplicities(A)):
                continue
                
            current_mults = get_multiplicities(A)
            current_score = get_score(current_mults)
            
            # Optimization
            no_improve = 0
            while time.time() - start_time < timeout and no_improve < 200:
                if current_score < best_score:
                    best_score = current_score
                    best_A = A[:]
                    best_mults = current_mults[:]
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve > 200:
                        break
                
                # Try a swap
                i = random.randint(0, len(A) - 1)
                a_out = A[i]
                a_in = random.randint(0, 128)
                if a_in == a_out or a_in in A_set:
                    continue
                
                A[i] = a_in
                A_set.remove(a_out)
                A_set.add(a_in)
                
                new_mults = get_multiplicities(A)
                new_score = get_score(new_mults)
                
                if new_score < current_score:
                    current_score = new_score
                    current_mults = new_mults
                    no_improve = 0
                else:
                    # Revert
                    A[i] = a_out
                    A_set.remove(a_in)
                    A_set.add(a_out)
            
    return best_A
