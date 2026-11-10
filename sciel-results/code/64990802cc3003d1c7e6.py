import numpy as np
import random

def construct():
    N = 129
    TARGETS = np.arange(128, 257)
    
    def get_mults(A_set):
        vec = np.zeros(N)
        for a in A_set:
            vec[a] = 1.0
        conv = np.convolve(vec, vec)
        # conv indices 0 to 256
        # We need indices 128 to 256
        target_conv = conv[128:257]
        
        diag = np.zeros(129)
        # S is in 128..256
        # S even -> a = S/2
        # Check if a in A
        # We can just loop or use vectorization
        # S % 2 == 0
        even_S = TARGETS[TARGETS % 2 == 0]
        for S in even_S:
            a = S // 2
            if 0 <= a < N and vec[a] == 1.0:
                # Map S to index in diag array? 
                # diag array index corresponds to S?
                # Let's just create an array of same shape as target_conv
                pass
        
        # Simpler:
        diag_target = np.zeros(len(TARGETS))
        for i, S in enumerate(TARGETS):
            if S % 2 == 0:
                a = S // 2
                if 0 <= a < N and vec[a] == 1.0:
                    diag_target[i] = 1.0
                    
        mults = (target_conv + diag_target) / 2.0
        return mults

    def is_covered(mults):
        return np.all(mults > 0)

    def max_mult(mults):
        return np.max(mults)

    best_A = None
    best_val = float('inf')
    
    # Strategy 1: Prune from Full
    A = set(range(N))
    mults = get_mults(A)
    curr_max = max_mult(mults)
    
    improved = True
    while improved:
        improved = False
        # Find sum with max multiplicity
        idx_max = np.argmax(mults)
        S_max = TARGETS[idx_max]
        
        # Candidates: a in A such that S_max - a in A and a <= S_max - a
        candidates = []
        for a in A:
            b = S_max - a
            if b in A and a <= b:
                candidates.append(a)
        
        best_candidate = None
        best_new_max = curr_max
        
        for a in candidates:
            A_temp = A.copy()
            A_temp.remove(a)
            
            mults_temp = get_mults(A_temp)
            if not is_covered(mults_temp):
                continue
                
            new_max = max_mult(mults_temp)
            if new_max < best_new_max:
                best_new_max = new_max
                best_candidate = a
        
        if best_candidate is not None:
            A.remove(best_candidate)
            mults = get_mults(A)
            curr_max = max_mult(mults)
            improved = True
            
    if curr_max < best_val:
        best_val = curr_max
        best_A = A.copy()
        
    # Strategy 2: Random Restart with Local Search (Swaps)
    for _ in range(20): # Reduced restarts
        K = len(A) + random.randint(-5, 5)
        K = max(20, min(128, K))
        A_curr = set(random.sample(range(N), K))
        
        mults = get_mults(A_curr)
        while not is_covered(mults):
            missing_idx = np.where(mults == 0)[0]
            if len(missing_idx) == 0: break
            
            best_add = None
            best_cover = 0
            # Heuristic: pick a that covers most missing
            # To speed up, maybe just pick random a?
            # Or pick a that is in the "middle" of missing sums?
            # Let's try a few random candidates
            for _ in range(10):
                a = random.randint(0, N-1)
                if a in A_curr: continue
                A_temp = A_curr.copy()
                A_temp.add(a)
                mults_temp = get_mults(A_temp)
                # Count how many missing are now covered
                covered_count = np.sum(mults_temp[missing_idx] > 0)
                if covered_count > best_cover:
                    best_cover = covered_count
                    best_add = a
                    best_mults_temp = mults_temp
            
            if best_add is None:
                # Fallback: add random
                a = random.randint(0, N-1)
                if a not in A_curr:
                    A_curr.add(a)
                    mults = get_mults(A_curr)
                else:
                    break
            else:
                A_curr.add(best_add)
                mults = best_mults_temp
            
        curr_max = max_mult(mults)
        improved = True
        while improved:
            improved = False
            idx_max = np.argmax(mults)
            S_max = TARGETS[idx_max]
            
            candidates_a = [a for a in A_curr if (S_max - a in A_curr and a <= S_max - a) or (S_max - a == a)]
            
            found_swap = False
            for a in candidates_a:
                for b in range(N):
                    if b in A_curr: continue
                    A_temp = A_curr.copy()
                    A_temp.remove(a)
                    A_temp.add(b)
                    
                    mults_temp = get_mults(A_temp)
                    if not is_covered(mults_temp):
                        continue
                        
                    new_max = max_mult(mults_temp)
                    if new_max < curr_max:
                        A_curr = A_temp
                        mults = mults_temp
                        curr_max = new_max
                        improved = True
                        found_swap = True
                        break
                if found_swap:
                    break
            
            if not improved:
                # Try removal
                for a in list(A_curr):
                    A_temp = A_curr.copy()
                    A_temp.remove(a)
                    mults_temp = get_mults(A_temp)
                    if is_covered(mults_temp):
                        new_max = max_mult(mults_temp)
                        if new_max < curr_max:
                            A_curr = A_temp
                            mults = mults_temp
                            curr_max = new_max
                            improved = True
                            break

        if curr_max < best_val:
            best_val = curr_max
            best_A = A_curr.copy()

    return list(best_A)
