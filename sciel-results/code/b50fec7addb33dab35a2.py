import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    N = 4096
    
    def check_addition(S, sums_set, x):
        """
        Fast check if x can be added to S.
        Returns True if valid.
        """
        n = len(S)
        S_arr = np.array(S, dtype=np.int64)
        
        # 1. (x, x, x)
        s3 = 3 * x
        if s3 in sums_set:
            return False
        
        # 2. (x, x, s_i): 2*x + S
        vals2 = 2 * x + S_arr
        if np.any(np.isin(vals2, list(sums_set))):
            return False
        
        # 3. (x, s_i, s_j) with i <= j: x + S_i + S_j
        # Generate all pairwise sums S_i + S_j for i <= j
        # For efficiency, use broadcasting for small n
        if n > 0:
            # Create upper triangle indices
            idx_i, idx_j = np.triu_indices(n)
            pair_sums = S_arr[idx_i] + S_arr[idx_j]
            vals3 = x + pair_sums
            if np.any(np.isin(vals3, list(sums_set))):
                return False
            # Also check within new sums
            if np.any(np.isin(vals2, vals3)):
                return False
            if len(vals3) > 1:
                # Check for duplicates within vals3
                if len(set(vals3.tolist())) != len(vals3):
                    return False
            # Check s3 against vals2 and vals3
            if s3 in vals2 or s3 in vals3:
                return False
        
        return True

    def get_new_sums(S, x):
        """Compute new triple sums when adding x to S."""
        n = len(S)
        new_sums = set()
        S_arr = np.array(S, dtype=np.int64)
        
        new_sums.add(3 * x)
        
        vals2 = 2 * x + S_arr
        new_sums.update(vals2.tolist())
        
        if n > 0:
            idx_i, idx_j = np.triu_indices(n)
            pair_sums = S_arr[idx_i] + S_arr[idx_j]
            vals3 = x + pair_sums
            new_sums.update(vals3.tolist())
        
        return new_sums

    def greedy_search(order, sums_set=None):
        """
        Performs a greedy search given a specific order of candidates.
        """
        S = []
        if sums_set is None:
            sums_set = set()
        else:
            sums_set = set(sums_set)
        
        for x in order:
            if time.time() > deadline:
                break
            if check_addition(S, sums_set, x):
                S.append(x)
                new_sums = get_new_sums(S[:-1], x)
                sums_set.update(new_sums)
                
        return S, sums_set

    def compute_all_sums(S):
        """Recomputes the full set of triple sums for a given set S."""
        sums = set()
        n = len(S)
        S_arr = np.array(S, dtype=np.int64)
        for i in range(n):
            si = S_arr[i]
            for j in range(i, n):
                sj = S_arr[j]
                for k in range(j, n):
                    sk = S_arr[k]
                    sums.add(int(si + sj + sk))
        return sums

    best_set = []
    best_len = 0
    best_sums = set()
    
    # Strategy 1: Natural Order (Smallest to Largest)
    order_natural = list(range(1, N + 1))
    S_nat, sums_nat = greedy_search(order_natural)
    if len(S_nat) > best_len:
        best_set = S_nat
        best_len = len(S_nat)
        best_sums = sums_nat
        
    if time.time() > deadline - 5.0:
        return best_set

    # Strategy 2: Randomized Greedy Restarts with diverse orderings
    random.seed(42)
    
    # Pre-generate some diverse orderings
    while time.time() < deadline - 15.0:
        # Choose ordering strategy
        strategy = random.random()
        
        if strategy < 0.4:
            # Uniform shuffle
            order = list(range(1, N + 1))
            random.shuffle(order)
        elif strategy < 0.6:
            # Bias toward smaller numbers (weighted shuffle)
            order = list(range(1, N + 1))
            # Partial shuffle: only shuffle first half
            half = N // 2
            random.shuffle(order[:half])
            # Keep second half in order
        elif strategy < 0.8:
            # Sort by residue modulo a prime
            p = random.choice([7, 11, 13, 17, 19, 23, 29, 31])
            order = list(range(1, N + 1))
            order.sort(key=lambda x: (x % p, random.random()))
        else:
            # Reverse-biased: start with larger numbers
            order = list(range(N, 0, -1))
            random.shuffle(order[:N//2])
        
        S_try, sums_try = greedy_search(order)
        
        if len(S_try) > best_len:
            best_set = S_try
            best_len = len(S_try)
            best_sums = sums_try

    # Strategy 3: Local Improvement - try to add elements
    if best_len > 0 and time.time() < deadline - 5.0:
        S_cur = list(best_set)
        sums_cur = set(best_sums)
        S_cur_set = set(S_cur)
        
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for x in range(1, N + 1):
                if x in S_cur_set:
                    continue
                if check_addition(S_cur, sums_cur, x):
                    S_cur.append(x)
                    S_cur_set.add(x)
                    new_sums = get_new_sums(S_cur[:-1], x)
                    sums_cur.update(new_sums)
                    best_len = len(S_cur)
                    best_set = S_cur
                    best_sums = sums_cur
                    improved = True
                    break
        
        # Strategy 4: Kick moves (remove one, add one)
        if time.time() < deadline - 3.0:
            for kick_iter in range(20):
                if time.time() > deadline - 3.0:
                    break
                
                S_kick = list(best_set)
                sums_kick = set(best_sums)
                S_kick_set = set(S_kick)
                
                if not S_kick:
                    break
                
                idx_to_remove = random.randint(0, len(S_kick) - 1)
                removed_elem = S_kick.pop(idx_to_remove)
                S_kick_set.discard(removed_elem)
                
                # Recompute sums after removal
                sums_kick = compute_all_sums(S_kick)
                
                # Try to add any element
                found_addition = False
                for x in range(1, N + 1):
                    if x in S_kick_set:
                        continue
                    if check_addition(S_kick, sums_kick, x):
                        S_kick.append(x)
                        S_kick_set.add(x)
                        new_sums = get_new_sums(S_kick[:-1], x)
                        sums_kick.update(new_sums)
                        found_addition = True
                        break
                        
                if found_addition:
                    # Try to add more
                    improved2 = True
                    while improved2 and time.time() < deadline - 3.0:
                        improved2 = False
                        for x in range(1, N + 1):
                            if x in S_kick_set:
                                continue
                            if check_addition(S_kick, sums_kick, x):
                                S_kick.append(x)
                                S_kick_set.add(x)
                                new_sums = get_new_sums(S_kick[:-1], x)
                                sums_kick.update(new_sums)
                                improved2 = True
                                break
                                
                    if len(S_kick) > best_len:
                        best_set = S_kick
                        best_len = len(S_kick)
                        best_sums = sums_kick

    return best_set
