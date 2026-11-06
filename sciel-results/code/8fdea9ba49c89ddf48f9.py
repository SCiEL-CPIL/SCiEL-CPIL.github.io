"""
B_3[1] set construction using Simulated Annealing.
Strategy: Treat the set as a binary vector x of length N=4096.
Objective: Maximize sum(x) subject to all unordered triple sums being unique.
Energy: E = -len(S) + lambda * (num_conflicts)
We minimize E.
Vectorized collision check using NumPy to evaluate moves quickly.
"""
import time
import random
import numpy as np

DEADLINE = 85.0  # Safety margin before 90s limit

def _count_conflicts_vectorized(S):
    """
    Given a sorted list S, returns the number of distinct triple sums that are duplicated.
    Specifically, if a sum appears k times, it contributes (k-1) to the conflict count.
    This allows us to know how many "fixes" are needed to reach 0 conflicts.
    """
    if len(S) < 3:
        return 0
    
    # Generate all unordered triples (i, j, k) with i <= j <= k
    # This is O(n^3) in the number of elements, but we only do this for the current state.
    # To speed up, we can generate indices and sums.
    n = len(S)
    S_arr = np.array(S)
    
    # We need to be careful with memory if n is large.
    # Max n is ~80. Number of triples ~ 80^3/6 ~ 85k. This is small.
    # However, generating the full array every step in SA might be slow in Python loops.
    # We will generate indices once and update sums incrementally? 
    # No, S changes. 
    # Let's just compute sums directly.
    
    # Efficient way: 
    # i from 0 to n-1
    # j from i to n-1
    # k from j to n-1
    # Sum = S[i] + S[j] + S[k]
    
    # Using broadcasting might be faster if we can vectorize the triple generation.
    # But triple generation is tricky to fully vectorize without huge intermediate arrays.
    # Given n <= 80, a python loop over i, and vectorized j,k might be a good balance.
    
    sums = []
    for i in range(n):
        si = S_arr[i]
        # j >= i
        # We need sums of S[j] + S[k] for j >= i, k >= j
        # This is equivalent to pairs from S[i:]
        
        # Let's just use itertools for simplicity and speed in C, then numpy for counting?
        # Actually, pure python set operations are often faster for small n.
        pass
        
    # Let's try a pure python approach with sets first, it might be fast enough for n~80.
    # If not, we optimize.
    
    seen = {}
    conflicts = 0
    # Optimized triple iteration
    for i in range(n):
        si = S[i]
        for j in range(i, n):
            sj = S[j]
            # k >= j
            # Sums are si + sj + S[k] for k in j..n-1
            # We can slice S[j:]
            for k in range(j, n):
                s_val = si + sj + S[k]
                if s_val in seen:
                    conflicts += 1
                else:
                    seen[s_val] = True
                    
    return conflicts

def construct():
    start_time = time.time()
    
    # Initial solution: Greedy construction to get a decent start
    # Greedy: add smallest possible numbers
    S_init = []
    sum_set = set()
    
    def try_add_greedy(x, S, sum_set):
        # Check if x can be added to S
        # New sums involving x:
        # 3x
        # 2x + s for s in S
        # x + 2s for s in S
        # x + s1 + s2 for s1, s2 in S, s1 <= s2
        n = len(S)
        new_sums = [3*x]
        for s in S:
            new_sums.append(2*x + s)
            new_sums.append(x + 2*s)
        for i in range(n):
            for j in range(i, n):
                new_sums.append(x + S[i] + S[j])
        
        for s in new_sums:
            if s in sum_set:
                return False
        return True

    # Greedy build
    for x in range(1, 4097):
        if try_add_greedy(x, S_init, sum_set):
            S_init.append(x)
            # Add new sums to sum_set
            n = len(S_init) - 1
            for s in [3*x]:
                sum_set.add(s)
            for s in S_init[:-1]:
                sum_set.add(2*x + s)
                sum_set.add(x + 2*s)
            for i in range(n):
                for j in range(i, n):
                    sum_set.add(x + S_init[i] + S_init[j])
            
    S_best = S_init
    best_len = len(S_best)
    
    # Simulated Annealing
    # State: S (list of ints)
    # We need to evaluate moves quickly.
    # Move 1: Remove an element
    # Move 2: Add an element (from 1..4096 not in S)
    # Move 3: Swap an element
    
    # To speed up evaluation, we maintain the set of all current triple sums.
    # When we remove x, we remove all sums involving x.
    # When we add x, we add all sums involving x.
    
    # Let's implement a class-like structure or just careful functions.
    
    def get_sums_involving(x, S):
        """Returns list of triple sums involving x, given S does NOT contain x."""
        sums = [3*x]
        for s in S:
            sums.append(2*x + s)
            sums.append(x + 2*s)
        n = len(S)
        for i in range(n):
            for j in range(i, n):
                sums.append(x + S[i] + S[j])
        return sums

    # Current state
    S = list(S_best)
    S_set = set(S)
    current_sums = set()
    
    # Populate current_sums
    # We can reuse the logic from greedy, but let's just recompute to be safe
    # Since S_best is small, recomputing is fine.
    for i in range(len(S)):
        for j in range(i, len(S)):
            for k in range(j, len(S)):
                current_sums.add(S[i] + S[j] + S[k])
                
    best_S = list(S)
    best_conflicts = 0 # Since it's a valid set from greedy, conflicts should be 0
    
    # Actually, greedy ensures 0 conflicts.
    # SA will try to increase size.
    # Energy function:
    # If conflicts == 0: Energy = -len(S)
    # If conflicts > 0: Energy = conflicts * 100 - len(S)  (Penalty for invalidity)
    
    def get_energy(S, conflicts):
        if conflicts == 0:
            return -len(S)
        else:
            return conflicts * 100 - len(S)
    
    current_energy = get_energy(S, 0)
    best_energy = current_energy
    
    T = 10.0
    cooling_rate = 0.999
    
    rng = random.Random(42)
    
    while time.time() - start_time < DEADLINE:
        if T < 0.01:
            T = 10.0 # Reheat occasionally? Or just stop. Let's stop or reheat.
            # Reheating can help escape deep local minima.
            pass
        
        # Choose a move
        move_type = rng.random()
        
        if move_type < 0.3:
            # Remove an element
            if len(S) == 0:
                continue
            idx = rng.randrange(len(S))
            x = S[idx]
            # Calculate new energy
            # Remove x from S
            S_new = S[:idx] + S[idx+1:]
            
            # We need to know how many sums are removed and if any become unique again?
            # This is complex. Easier to just compute conflicts for the new set?
            # But computing conflicts is O(n^3).
            # For n=80, O(80^3) = 512k ops. In Python, this is ~0.05s.
            # If we do 1000 steps/sec, that's 50s for 1000 steps. Too slow.
            
            # We need a faster conflict check.
            # Let's maintain the frequency of sums.
            
            pass # This approach is too slow for pure Python re-evaluation.
            
            # Let's try a different approach:
            # Maintain a dictionary: sum -> count
            # When removing x:
            #   For each sum s involving x, decrement count[s].
            #   If count[s] becomes 1, it was a conflict (count>=2), now it's not.
            #   So conflicts decreases by (count[s] - 1) if count[s] was >= 2?
            #   Actually, conflicts = sum(max(0, count[s]-1)).
            #   Let delta_conflicts = new_conflicts - old_conflicts.
            #   For each sum s involving x:
            #      old_c = count[s]
            #      new_c = old_c - 1
            #      old_contrib = max(0, old_c - 1)
            #      new_contrib = max(0, new_c - 1)
            #      delta += new_contrib - old_contrib
            
            # This is O(num_sums_involving_x) = O(n^2).
            # This is fast!
            
            # Let's restart the SA loop with this efficient delta calculation.
            break
        
        # We broke out to implement the efficient version.
        # Let's rewrite the SA loop properly.
        
        # State:
        # S: list of elements
        # S_set: set of elements
        # sum_counts: dict {sum: count}
        # current_conflicts: int
        
        # Re-initialize state for SA
        S = list(S_best)
        S_set = set(S)
        sum_counts = {}
        for i in range(len(S)):
            for j in range(i, len(S)):
                for k in range(j, len(S)):
                    s_val = S[i] + S[j] + S[k]
                    sum_counts[s_val] = sum_counts.get(s_val, 0) + 1
        
        current_conflicts = sum(max(0, c - 1) for c in sum_counts.values())
        current_energy = get_energy(S, current_conflicts)
        best_energy = current_energy
        best_S = list(S)
        
        T = 5.0
        cooling_rate = 0.9995
        steps = 0
        
        while time.time() - start_time < DEADLINE:
            steps += 1
            
            # Choose move
            r = rng.random()
            
            if r < 0.2 and len(S) > 0:
                # Remove
                idx = rng.randrange(len(S))
                x = S[idx]
                
                # Calculate delta
                # Sums involving x
                # We need to iterate through all triples involving x.
                # Since S is small, we can just generate them.
                # But we need to know which sums are affected.
                
                # Helper to get sums involving x given S (excluding x)
                # Wait, S includes x.
                S_without_x = S[:idx] + S[idx+1:]
                
                affected_sums = []
                # 3x
                affected_sums.append(3*x)
                # 2x + s
                for s in S_without_x:
                    affected_sums.append(2*x + s)
                    affected_sums.append(x + 2*s)
                # x + s1 + s2
                n_wo = len(S_without_x)
                for i in range(n_wo):
                    for j in range(i, n_wo):
                        affected_sums.append(x + S_without_x[i] + S_without_x[j])
                
                delta_conf = 0
                for s_val in affected_sums:
                    old_c = sum_counts.get(s_val, 0)
                    new_c = old_c - 1
                    old_contrib = max(0, old_c - 1)
                    new_contrib = max(0, new_c - 1)
                    delta_conf += new_contrib - old_contrib
                    sum_counts[s_val] = new_c
                    if new_c == 0:
                        del sum_counts[s_val]
                
                new_len = len(S) - 1
                new_conf = current_conflicts + delta_conf
                new_energy = get_energy(S, new_conf) # Note: len(S) is used, so pass new_len? 
                # get_energy takes S and conflicts. It uses len(S).
                # So we should calculate energy based on new_len and new_conf.
                new_energy = -new_len if new_conf == 0 else new_conf * 100 - new_len
                
                if new_energy < current_energy or rng.random() < np.exp((current_energy - new_energy) / T):
                    # Accept
                    S.pop(idx)
                    S_set.remove(x)
                    current_conflicts = new_conf
                    current_energy = new_energy
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_S = list(S)
                else:
                    # Reject: undo changes
                    for s_val in affected_sums:
                        old_c = sum_counts.get(s_val, 0)
                        sum_counts[s_val] = old_c + 1
                        
            elif r < 0.8:
                # Add
                # Pick a candidate x not in S
                # To make it efficient, we can pick a random x from 1..4096
                # and check if it's in S_set.
                # Or maintain a list of available candidates.
                # Given 4096 is small, random pick is fine.
                x = rng.randint(1, 4096)
                if x in S_set:
                    continue
                
                # Calculate delta
                S_current = S
                affected_sums = []
                affected_sums.append(3*x)
                for s in S_current:
                    affected_sums.append(2*x + s)
                    affected_sums.append(x + 2*s)
                n_cur = len(S_current)
                for i in range(n_cur):
                    for j in range(i, n_cur):
                        affected_sums.append(x + S_current[i] + S_current[j])
                
                delta_conf = 0
                for s_val in affected_sums:
                    old_c = sum_counts.get(s_val, 0)
                    new_c = old_c + 1
                    old_contrib = max(0, old_c - 1)
                    new_contrib = max(0, new_c - 1)
                    delta_conf += new_contrib - old_contrib
                    sum_counts[s_val] = new_c
                
                new_len = len(S) + 1
                new_conf = current_conflicts + delta_conf
                new_energy = -new_len if new_conf == 0 else new_conf * 100 - new_len
                
                if new_energy < current_energy or rng.random() < np.exp((current_energy - new_energy) / T):
                    # Accept
                    S.append(x)
                    S_set.add(x)
                    current_conflicts = new_conf
                    current_energy = new_energy
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_S = list(S)
                else:
                    # Reject: undo
                    for s_val in affected_sums:
                        old_c = sum_counts.get(s_val, 0)
                        new_c = old_c - 1
                        sum_counts[s_val] = new_c
                        if new_c == 0:
                            del sum_counts[s_val]
                        
            else:
                # Swap
                if len(S) == 0:
                    continue
                idx = rng.randrange(len(S))
                x_old = S[idx]
                x_new = rng.randint(1, 4096)
                if x_new in S_set:
                    continue
                if x_new == x_old:
                    continue
                
                # Remove x_old, Add x_new
                # We can do this as two steps or combined.
                # Combined is better for delta.
                
                S_without_old = S[:idx] + S[idx+1:]
                
                # Sums involving x_old (to be removed)
                sums_remove = []
                sums_remove.append(3*x_old)
                for s in S_without_old:
                    sums_remove.append(2*x_old + s)
                    sums_remove.append(x_old + 2*s)
                n_wo = len(S_without_old)
                for i in range(n_wo):
                    for j in range(i, n_wo):
                        sums_remove.append(x_old + S_without_old[i] + S_without_old[j])
                        
                # Sums involving x_new (to be added)
                # Note: S_without_old does not contain x_old.
                # Does it contain x_new? No, x_new not in S_set.
                sums_add = []
                sums_add.append(3*x_new)
                for s in S_without_old:
                    sums_add.append(2*x_new + s)
                    sums_add.append(x_new + 2*s)
                for i in range(n_wo):
                    for j in range(i, n_wo):
                        sums_add.append(x_new + S_without_old[i] + S_without_old[j])
                
                # Apply removals
                delta_conf = 0
                for s_val in sums_remove:
                    old_c = sum_counts.get(s_val, 0)
                    new_c = old_c - 1
                    old_contrib = max(0, old_c - 1)
                    new_contrib = max(0, new_c - 1)
                    delta_conf += new_contrib - old_contrib
                    sum_counts[s_val] = new_c
                    if new_c == 0:
                        del sum_counts[s_val]
                        
                # Apply additions
                for s_val in sums_add:
                    old_c = sum_counts.get(s_val, 0)
                    new_c = old_c + 1
                    old_contrib = max(0, old_c - 1)
                    new_contrib = max(0, new_c - 1)
                    delta_conf += new_contrib - old_contrib
                    sum_counts[s_val] = new_c
                    
                new_len = len(S) # Length unchanged
                new_conf = current_conflicts + delta_conf
                new_energy = -new_len if new_conf == 0 else new_conf * 100 - new_len
                
                if new_energy < current_energy or rng.random() < np.exp((current_energy - new_energy) / T):
                    # Accept
                    S[idx] = x_new
                    S_set.remove(x_old)
                    S_set.add(x_new)
                    current_conflicts = new_conf
                    current_energy = new_energy
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_S = list(S)
                else:
                    # Reject: Undo
                    # Undo additions
                    for s_val in sums_add:
                        old_c = sum_counts.get(s_val, 0)
                        new_c = old_c - 1
                        sum_counts[s_val] = new_c
                        if new_c == 0:
                            del sum_counts[s_val]
                    # Undo removals
                    for s_val in sums_remove:
                        old_c = sum_counts.get(s_val, 0)
                        sum_counts[s_val] = old_c + 1
                        
            # Cool
            T *= cooling_rate
            if T < 0.01:
                T = 5.0 # Reheat
                
    # Final verification of best_S
    # Ensure it's valid
    if best_S:
        # Quick verify
        sums = set()
        valid = True
        for i in range(len(best_S)):
            for j in range(i, len(best_S)):
                for k in range(j, len(best_S)):
                    s_val = best_S[i] + best_S[j] + best_S[k]
                    if s_val in sums:
                        valid = False
                        break
                    sums.add(s_val)
                if not valid: break
            if not valid: break
        
        if valid:
            return best_S
        else:
            # Fallback to initial greedy
            return S_init
    else:
        return S_init

if __name__ == "__main__":
    res = construct()
    print(f"Length: {len(res)}")
