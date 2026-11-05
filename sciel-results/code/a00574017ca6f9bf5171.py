import time
import random
import bisect

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin

    def check_valid(S_list):
        """Check if S_list (sorted) satisfies the property."""
        for i, a in enumerate(S_list):
            for j in range(i + 1, len(S_list)):
                b = S_list[j]
                for k in range(j, len(S_list)):
                    c = S_list[k]
                    if (b + c) % a == 0:
                        return False
        return True

    def is_valid_add(S_list, m):
        """Check if adding m to S_list (sorted) maintains the property."""
        pos = bisect.bisect_left(S_list, m)
        L = len(S_list)

        # Case 1: m is smallest (a=m), need m ∤ (b+c) for all b,c in S with b > m, c >= b
        for j in range(pos, L):
            b = S_list[j]
            for k in range(j, L):
                c = S_list[k]
                if (b + c) % m == 0:
                    return False

        # Case 2: m is middle (b=m), need a ∤ (m+c) for all a < m, c >= m
        for i in range(0, pos):
            a = S_list[i]
            for k in range(pos, L):
                c = S_list[k]
                if (m + c) % a == 0:
                    return False

        # Case 3: m is largest (c=m), need a ∤ (b+m) for all a < b <= m
        for i in range(0, pos):
            a = S_list[i]
            for j in range(i + 1, pos):
                b = S_list[j]
                if (b + m) % a == 0:
                    return False

        return True

    def is_valid_remove_add(S_list, to_remove, to_add):
        """Check if S_list \ to_remove + to_add is valid."""
        # Create a set for fast lookup
        S_set = set(S_list)
        for r in to_remove:
            S_set.discard(r)
        for a in to_add:
            if a in S_set:
                return False # Should not happen if to_add is from complement
            S_set.add(a)
        
        S_new = sorted(list(S_set))
        return check_valid(S_new)

    # Strategy: Start with upper tail
    base_set = list(range(171, n + 1))
    best_S = list(base_set)
    best_size = len(best_S)
    
    random.seed(42)

    # Local Search with Swaps
    # We will try to improve the set by removing small subsets and adding larger subsets.
    
    # To speed up, we can precompute "conflict scores" or just use the fast is_valid_add.
    # But is_valid_add is for adding ONE element to a valid set.
    # If we remove elements, the set is still valid.
    # So we can use is_valid_add sequentially for the "add" part.
    
    # Move: Remove R (list), Add A (list).
    # 1. Remove R from S. S is still valid.
    # 2. Add A one by one using is_valid_add.
    # If all succeed, the move is valid.
    
    def try_move(S, to_remove, to_add):
        # Check if to_remove is in S
        S_set = set(S)
        for r in to_remove:
            if r not in S_set:
                return None
        for a in to_add:
            if a in S_set:
                return None
            
        # Remove
        S_temp = [x for x in S if x not in to_remove]
        
        # Add
        for a in to_add:
            if not is_valid_add(S_temp, a):
                return None
            pos = bisect.bisect_left(S_temp, a)
            S_temp.insert(pos, a)
            
        return S_temp

    # Search Loop
    improved = True
    while improved and time.time() < deadline:
        improved = False
        S = list(best_S)
        in_S = set(S)
        not_in_S = [m for m in range(1, n + 1) if m not in in_S]
        
        # Try 1-for-2 swaps
        # Randomly select an element to remove? Or iterate?
        # Iterating all elements is 256 * (cost of trying to add 2).
        # Cost of trying to add 2:
        # For each candidate pair (a,b) in not_in_S, check validity.
        # This is too slow if we do it for all pairs.
        
        # Heuristic: Try to remove elements that are "small" (more likely to cause conflicts)
        # and add elements that are "large" (less likely to cause conflicts).
        
        # Sort not_in_S descending (larger first, easier to add)
        not_in_S.sort(reverse=True)
        
        # Try removing each element in S
        for i in range(len(S)):
            if time.time() > deadline:
                break
            r = S[i]
            # Try to add 2 elements from not_in_S
            # We want to find ANY pair that works.
            # Try pairs from the "easiest" (largest) first.
            found = False
            for idx1 in range(len(not_in_S)):
                if time.time() > deadline:
                    break
                a1 = not_in_S[idx1]
                for idx2 in range(idx1 + 1, len(not_in_S)):
                    a2 = not_in_S[idx2]
                    # Quick check: if a1 and a2 are very large, likely valid.
                    # Try the move
                    new_S = try_move(S, [r], [a1, a2])
                    if new_S is not None:
                        if len(new_S) > best_size:
                            best_S = new_S
                            best_size = len(new_S)
                            improved = True
                            found = True
                            break
            if found:
                break # Restart search with new best_S
        
        # If no 1-for-2 found, try 2-for-3?
        # Or just restart with a different random seed/perturbation?
        
        # To avoid getting stuck, if no improvement in a full pass, we can try a random restart
        # or a more aggressive move.
        
        # Let's add a random restart mechanism if stagnated
        if not improved and random.random() < 0.1:
            # Perturb current best
            S_perturbed = list(best_S)
            # Remove 1 random element
            if S_perturbed:
                idx = random.randint(0, len(S_perturbed) - 1)
                S_perturbed.pop(idx)
            # Add 1 random element from not in S
            not_in_S_perturbed = [m for m in range(1, n + 1) if m not in set(S_perturbed)]
            if not_in_S_perturbed:
                add_m = random.choice(not_in_S_perturbed)
                if is_valid_add(S_perturbed, add_m):
                    pos = bisect.bisect_left(S_perturbed, add_m)
                    S_perturbed.insert(pos, add_m)
            # Try to improve this perturbed set?
            # For now, just use it as a new starting point for the next iteration
            # This might not be better, but it changes the landscape.
            # Actually, let's just continue. The "improved" flag will be False, loop ends.
            # We need to force the loop to continue if we want to try more.
            # Let's change the condition to run for a fixed number of iterations or time.

    # Final verification
    S_final = sorted(best_S)
    if not check_valid(S_final):
        return list(range(171, 257))

    return S_final
