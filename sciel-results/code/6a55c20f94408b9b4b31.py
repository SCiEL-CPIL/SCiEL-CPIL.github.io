"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit

    n = 30
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, range(1, n + 1))
    
    # We want to minimize |sum(c_i * (L/i))| != 0
    # c_i in {-1, 0, 1}
    
    best_val = float('inf')
    best_coefs = None
    
    # Strategy: Meet-in-the-Middle (MITM)
    # Split 30 terms into two sets of 15.
    # Compute all possible sums for each set.
    # Each set has 3^15 = 14,348,907 possible combinations.
    # This is large but feasible in C++ with optimization, but in Python it might be tight.
    # Let's estimate: 14M iterations * 2 sets.
    # Each iteration involves a few integer operations.
    # Python is slow. 14M * 2 * ~100ns = 2.8 seconds? No, Python is ~100x slower than C.
    # 14M * 2 * ~1us = 28 seconds.
    # This is risky but potentially doable if optimized.
    # However, we can prune or use a smarter generation.
    
    # Alternative:
    # Since we just need the minimum non-zero sum, we can generate sums for the first half,
    # sort them, and for each sum in the second half, binary search for the closest value in the first half
    # such that the total sum is non-zero and minimized.
    
    # But generating 3^15 sums in Python might take too long.
    # Let's try a smaller split: 12 and 18?
    # 3^12 = 531,441. 3^18 = 387,420,489. Too big.
    # 15 and 15 is the most balanced.
    
    # Let's check if we can generate 3^15 sums within time.
    # We can represent the state as a list of sums.
    # Start with [0].
    # For each term t, new_sums = [s - val, s, s + val for s in sums].
    # This creates 3x the list.
    # 3^15 = 14M elements. List of 14M integers.
    # Memory: 14M * 8 bytes = 112 MB. Acceptable.
    # Time: Generating 14M elements.
    # Let's try to optimize the generation using NumPy if available, or just pure Python lists.
    # Pure Python list comprehension is quite fast.
    
    def generate_sums(indices):
        sums = [0]
        for idx in indices:
            val = L // idx
            # Create new list
            # [s - val for s in sums] + sums + [s + val for s in sums]
            # This is efficient in Python.
            sums = [s - val for s in sums] + sums + [s + val for s in sums]
        return sums
    
    half1 = list(range(1, 16))  # 1..15
    half2 = list(range(16, 31)) # 16..30
    
    # Check time before generating
    if time.time() > deadline - 20:
        # Fallback to local search if we are already running out of time
        pass
    else:
        # Generate sums for half 1
        sums1 = generate_sums(half1)
        
        if time.time() > deadline - 10:
            # Can't even sort and search, fallback
            pass
        else:
            # Sort sums1 for binary search
            sums1.sort()
            
            # Generate sums for half 2
            sums2 = generate_sums(half2)
            
            if time.time() > deadline - 5:
                pass
            else:
                # For each sum in sums2, find best match in sums1
                # We want to minimize |s1 + s2| != 0
                # Since sums1 is sorted, for a given s2, the best s1 is near -s2.
                
                import bisect
                
                best_found_val = float('inf')
                best_s1 = None
                best_s2 = None
                
                # Iterate over sums2
                # To speed up, we can skip if abs(s2) is already larger than best_found_val? No, s1 can cancel it.
                # But we can prune s2 if abs(s2) - max_abs_s1 > best_found_val?
                # Max abs s1 is sum(L/i for i in 1..15).
                # This is a loose bound.
                
                # Optimization: Only check a window of s1 around -s2.
                # Since we just need the global min, we can iterate.
                
                # However, 14M iterations in Python for the binary search part might be slow.
                # bisect is C-speed, but the loop is Python.
                # 14M * 1us = 14s. This is tight.
                
                # Let's try.
                for s2 in sums2:
                    if time.time() > deadline - 2:
                        break
                    
                    # Target is -s2
                    target = -s2
                    # Find insertion point
                    pos = bisect.bisect_left(sums1, target)
                    
                    # Check neighbors
                    candidates = []
                    if pos < len(sums1):
                        candidates.append(sums1[pos])
                    if pos > 0:
                        candidates.append(sums1[pos-1])
                    
                    for s1 in candidates:
                        total = s1 + s2
                        if total == 0:
                            continue
                        abs_total = abs(total)
                        if abs_total < best_found_val:
                            best_found_val = abs_total
                            best_s1 = s1
                            best_s2 = s2
                            
                            if best_found_val == 1:
                                break
                    if best_found_val == 1:
                        break
                
                # Reconstruct coefficients
                if best_s1 is not None:
                    # We need to find which combination in half1 gives best_s1
                    # And which combination in half2 gives best_s2
                    # We didn't store the parent pointers in the simple generate_sums.
                    # We need to modify generate_sums to also return the choices, or reconstruct.
                    # Reconstructing 3^15 is hard without parent pointers.
                    
                    # Let's change strategy:
                    # Instead of just sums, store (sum, tuple_of_choices) ?
                    # Tuple of 15 ints for 14M entries? Memory will explode.
                    
                    # Alternative:
                    # Use a dictionary mapping sum -> one valid tuple of choices?
                    # Collisions are likely. We just need ONE valid tuple for the best sum.
                    
                    # Let's redo generate_sums to return a dict {sum: choices_tuple}
                    # Or, since we found best_s1 and best_s2, we can run a second pass
                    # to find the specific coefficients that yield those sums.
                    
                    # A faster way:
                    # Since we have the list of sums, we can just re-run the generation
                    # but track the path for the specific target sum.
                    # This is like a "reverse search" or "meet in the middle with reconstruction".
                    
                    # Actually, let's just use a dictionary for the first half.
                    # {sum: choices}
                    # Memory: 14M entries. Dict overhead is high. ~100 bytes per entry? 1.4 GB? Too much.
                    
                    # Let's stick to the list of sums, and if we find a candidate,
                    # we can do a DFS/BFS to find the coefficients?
                    # No, that's 3^15 again.
                    
                    # Better:
                    # Modify generate_sums to return list of (sum, parent_index) or similar?
                    # Or simply:
                    # Since we only need the minimum non-zero sum, and we found it,
                    # we can just use a local search starting from a random config
                    # to try to match that specific sum?
                    
                    # Actually, the most robust way in Python with this time limit:
                    # 1. Generate sums1 (list) and sort.
                    # 2. Generate sums2 (list).
                    # 3. Find best pair (s1, s2).
                    # 4. Now, we need to find c1 (15 coeffs) such that sum(c1_i * L/i) = s1.
                    # 5. And c2 (15 coeffs) such that sum(c2_i * L/i) = s2.
                    
                    # How to find c1 for s1?
                    # We can use the same MITM logic recursively? No.
                    # We can use a dictionary for the smaller half of the half?
                    # 15 terms -> 7 and 8.
                    # 3^7 = 2187, 3^8 = 6561.
                    # This is very fast.
                    
                    def find_coeffs_for_target(target, indices):
                        # indices is a list of 15 numbers
                        # Split into two halves: 7 and 8
                        mid = len(indices) // 2
                        left_idx = indices[:mid]
                        right_idx = indices[mid:]
                        
                        left_sums = generate_sums(left_idx)
                        left_sums.sort()
                        right_sums = generate_sums(right_idx)
                        
                        # Find pair (ls, rs) such that ls + rs = target
                        # Iterate over smaller list (right_sums)
                        for rs in right_sums:
                            if time.time() > deadline - 1:
                                return None
                            ls_needed = target - rs
                            pos = bisect.bisect_left(left_sums, ls_needed)
                            if pos < len(left_sums) and left_sums[pos] == ls_needed:
                                # Found match!
                                # Now reconstruct left coeffs for ls_needed
                                # And right coeffs for rs
                                # We need the actual coefficients, not just sums.
                                # So generate_sums must return the coefficients too?
                                # Or we can do a second recursive search to find the coeffs.
                                pass
                        return None
                        # This is getting complex.
                        # Let's simplify:
                        # If we find the best sum value V, we can just run a local search
                        # to try to find a configuration that sums to V.
                        # But local search might not find it.
                        
                        # Alternative:
                        # Just return the best value found by MITM, and then
                        # use a heuristic to construct a valid config?
                        # No, we must return the exact coefficients.
                        
                        # Let's go back to the dictionary approach but with a smaller split.
                        # Split 30 into 10 and 20?
                        # 3^10 = 59049.
                        # 3^20 is too big.
                        
                        # Split 30 into 15 and 15.
                        # Generate sums1 (list of 14M).
                        # Generate sums2 (list of 14M).
                        # Find best pair.
                        # To get coefficients, we can store the first occurrence of each sum
                        # as a tuple of choices.
                        # But 14M tuples of length 15 is too much memory.
                        
                        # Compromise:
                        # Store only the sums in a list.
                        # When we find the best (s1, s2), we re-generate the sums
                        # but this time we stop as soon as we find the target sum?
                        # No, we need to search the space.
                        
                        # Let's use a different approach:
                        # 1. Generate all sums for half 1, store in a dict: {sum: coeffs_tuple}
                        #    But only keep the first one found for each sum to save memory?
                        #    No, we might need a specific one.
                        #    Actually, any coeffs that sum to s1 are valid.
                        #    So we can just store ONE coeffs tuple for each unique sum.
                        #    How many unique sums?
                        #    The sums are integers. Range is roughly +/- L * sum(1/i).
                        #    L is huge.
                        #    But the number of unique sums is bounded by 3^15.
                        #    So the dict will have up to 14M entries.
                        #    Key: int, Value: tuple of 15 ints.
                        #    Memory: 14M * (28 bytes for int + 15*28 for tuple) = 14M * 448 bytes = 6 GB.
                        #    Too much.
                        
                        # We must avoid storing the tuples.
                        
                        # New Idea:
                        # 1. Generate sums1 (list).
                        # 2. Generate sums2 (list).
                        # 3. Find best (s1, s2).
                        # 4. Now, use a DFS with pruning to find coefficients for s1.
                        #    Prune if the remaining possible sum range cannot reach s1.
                        #    This is a subset sum variant.
                        #    With 15 items, it's feasible.
                        #    We can use the sorted list of sums for the remaining items to prune.
                        
                        # Let's implement a helper:
                        # find_coeffs(target, indices, current_sum, current_coeffs, remaining_sums_sorted)
                        # This is complex to implement correctly and fast in Python.
                        
                        # Simplest Fallback:
                        # If MITM finds a good value, but we can't reconstruct,
                        # we can just use that value as a guide for local search?
                        # No, local search is weak.
                        
                        # Let's try the Dict approach with a smaller half.
                        # Split 30 into 12 and 18.
                        # 3^12 = 531,441.
                        # Dict for 12 terms: 531k entries. Memory is fine.
                        # Generate sums for 18 terms: 3^18 = 387M. Too big for list.
                        
                        # So MITM with 15/15 is the only viable option for exact search,
                        # but reconstruction is the bottleneck.
                        
                        # What if we store the sums1 in a list, and ALSO store the coefficients
                        # in a parallel list of tuples?
                        # 14M tuples. Still too much memory.
                        
                        # Okay, let's just use the local search approach but with a better start.
                        # The current program's local search is weak because it starts from greedy.
                        # We can start from a random configuration and do a simulated annealing or
                        # more aggressive hill climbing.
                        
                        # But the task says "Minimize its absolute value".
                        # The theoretical minimum is 1.
                        # If we can find 1, we are done.
                        # Probability of finding 1 is high if we search enough.
                        
                        # Let's stick to the local search but make it much more aggressive.
                        # 1. Start with random config.
                        # 2. Hill climb: flip any coefficient if it reduces |sum|.
                        # 3. If stuck, do a random jump (change 2-3 coeffs) and hill climb again.
                        # 4. Repeat until time limit.
                        
                        # This is a standard stochastic local search.
                        
                        pass # End of MITM analysis
            
            # Since reconstruction is hard, let's implement a robust Stochastic Local Search.
            # It is simpler and likely to find a small value (1 or 2) if given enough time.
            
            def random_config():
                return [random.choice([-1, 0, 1]) for _ in range(n)]
            
            def calculate_sum(coefs):
                return sum(coefs[i] * (L // (i + 1)) for i in range(n))
            
            best_found = float('inf')
            best_config = None
            
            for _ in range(1000):  # Iterations
                if time.time() > deadline:
                    break
                
                # Start with a random config
                coefs = random_config()
                curr_sum = calculate_sum(coefs)
                
                if curr_sum == 0:
                    # Flip a random coefficient to make it non-zero
                    for i in range(n):
                        if coefs[i] == 0:
                            coefs[i] = random.choice([-1, 1])
                        else:
                            coefs[i] = -coefs[i]
                        curr_sum = calculate_sum(coefs)
                        if curr_sum != 0:
                            break
                
                if abs(curr_sum) < best_found:
                    best_found = abs(curr_sum)
                    best_config = coefs[:]
                
                # Local Search (Hill Climbing)
                improved = True
                while improved:
                    improved = False
                    for i in range(n):
                        old_c = coefs[i]
                        val = L // (i + 1)
                        for new_c in [-1, 0, 1]:
                            if new_c == old_c:
                                continue
                            new_sum = curr_sum + (new_c - old_c) * val
                            if new_sum == 0:
                                continue
                            if abs(new_sum) < abs(curr_sum):
                                coefs[i] = new_c
                                curr_sum = new_sum
                                improved = True
                                if abs(curr_sum) < best_found:
                                    best_found = abs(curr_sum)
                                    best_config = coefs[:]
                                break # Restart inner loop? Or just update and continue?
                        if improved:
                            break # Restart outer loop to pick a new index
                    # If no improvement, break
                if best_found == 1:
                    break
            
            if best_config is not None:
                return best_config
    
    # Fallback: Simple deterministic construction
    # Just return a known small sum config if search fails
    # e.g. 1/1 - 1/2 - 1/3 ... 
    # Let's just return a random one or the seed.
    return [0] * 28 + [1, -1]

# EVOLVE-BLOCK-END
