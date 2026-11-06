"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin
    
    best_set = []
    best_len = 0
    
    def can_add(S, sums, x):
        """
        Check if adding x to S preserves unique triple sums.
        S is a list of integers.
        sums is a set of all current triple sums.
        Returns True if valid, False otherwise.
        """
        new_sums = []
        
        # 1. (x, x, x)
        new_sums.append(3 * x)
        
        # 2. (x, x, s) for s in S
        for s in S:
            new_sums.append(2 * x + s)
            
        # 3. (x, s1, s2) for s1, s2 in S, s1 <= s2
        n = len(S)
        for i in range(n):
            s1 = S[i]
            for j in range(i, n):
                s2 = S[j]
                new_sums.append(x + s1 + s2)
                
        # Check for duplicates among new_sums
        if len(new_sums) != len(set(new_sums)):
            return False
            
        # Check if any new_sum is already in sums
        for ns in new_sums:
            if ns in sums:
                return False
                
        return True

    def greedy_search(skip_prob, order='ascending'):
        S = []
        sums = set()
        
        # Create candidate list based on order
        if order == 'ascending':
            candidates = list(range(1, 4097))
        elif order == 'descending':
            candidates = list(range(4096, 0, -1))
        else:  # shuffled
            candidates = list(range(1, 4097))
            random.shuffle(candidates)
        
        for x in candidates:
            # Check time
            if time.time() > deadline:
                return S, sums
            
            # Random skip
            if random.random() < skip_prob:
                continue
                
            if can_add(S, sums, x):
                S.append(x)
                # Update sums
                sums.add(3 * x)
                for s in S[:-1]:
                    sums.add(2 * x + s)
                old_S = S[:-1]
                n = len(old_S)
                for i in range(n):
                    s1 = old_S[i]
                    for j in range(i, n):
                        s2 = old_S[j]
                        sums.add(x + s1 + s2)
                        
        return S, sums

    def local_improve(S, sums):
        """
        Try to swap elements in S for larger ones to increase the set size
        or improve the set.
        """
        S = list(S)
        sums = set(sums)
        improved = True
        
        while improved and time.time() < deadline - 3.0:
            improved = False
            n = len(S)
            
            # Try to find a replacement for each element
            for i in range(n):
                if time.time() > deadline - 3.0:
                    break
                    
                old_val = S[i]
                
                # Try to remove old_val and add a larger value
                # First, compute sums without old_val
                old_sums = set(sums)
                
                # Remove sums involving old_val
                # We need to be careful here - let's just try adding new values
                # and see if we can find a better set
                
                # Try values from 4096 down to old_val+1
                for new_val in range(4096, old_val, -1):
                    if new_val in S:
                        continue
                    
                    # Check if we can replace old_val with new_val
                    # This is complex - let's use a simpler approach
                    # Try adding new_val to a set that excludes old_val
                    
                    S_temp = [s for s in S if s != old_val]
                    
                    # Recompute sums for S_temp
                    sums_temp = set()
                    n_temp = len(S_temp)
                    for a in range(n_temp):
                        for b in range(a, n_temp):
                            for c in range(b, n_temp):
                                sums_temp.add(S_temp[a] + S_temp[b] + S_temp[c])
                    
                    if can_add(S_temp, sums_temp, new_val):
                        # Success! Replace old_val with new_val
                        S[i] = new_val
                        S.sort()
                        # Recompute all sums
                        sums = set()
                        n_new = len(S)
                        for a in range(n_new):
                            for b in range(a, n_new):
                                for c in range(b, n_new):
                                    sums.add(S[a] + S[b] + S[c])
                        improved = True
                        break
                        
                if time.time() > deadline - 3.0:
                    break
                    
        return S, sums

    def try_add_more(S, sums):
        """
        Try to add more elements to the set.
        """
        S = list(S)
        sums = set(sums)
        
        for x in range(1, 4097):
            if time.time() > deadline - 3.0:
                break
            if x in S:
                continue
            if can_add(S, sums, x):
                S.append(x)
                sums.add(3 * x)
                for s in S[:-1]:
                    sums.add(2 * x + s)
                old_S = S[:-1]
                n = len(old_S)
                for i in range(n):
                    s1 = old_S[i]
                    for j in range(i, n):
                        s2 = old_S[j]
                        sums.add(x + s1 + s2)
                        
        return S, sums

    # Run standard greedy first
    S, sums = greedy_search(0.0, 'ascending')
    if len(S) > best_len:
        best_len = len(S)
        best_set = S
    
    # Try to improve with local improvement
    S_imp, sums_imp = local_improve(S, sums)
    if len(S_imp) > best_len:
        best_len = len(S_imp)
        best_set = S_imp
    
    # Run randomized greeds with different orders and skip probabilities
    orders = ['ascending', 'descending', 'shuffled']
    
    while time.time() < deadline - 5.0:
        order = random.choice(orders)
        p = random.uniform(0.0, 0.15)
        S, sums = greedy_search(p, order)
        
        # Try to improve
        S_imp, sums_imp = local_improve(S, sums)
        if len(S_imp) > best_len:
            best_len = len(S_imp)
            best_set = S_imp
    
    # Final attempt to add more elements to the best set
    if best_set and time.time() < deadline - 2.0:
        # Recompute sums for best_set
        best_sums = set()
        n = len(best_set)
        for a in range(n):
            for b in range(a, n):
                for c in range(b, n):
                    best_sums.add(best_set[a] + best_set[b] + best_set[c])
        
        best_set2, best_sums2 = try_add_more(best_set, best_sums)
        if len(best_set2) > best_len:
            best_len = len(best_set2)
            best_set = best_set2
    
    return best_set
# EVOLVE-BLOCK-END
