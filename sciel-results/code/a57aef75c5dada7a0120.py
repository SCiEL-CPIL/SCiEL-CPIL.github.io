import random
import time
import math
import itertools

def construct(n):
    if n <= 1:
        return "1"
    
    # Generate all permutations
    perms = list(itertools.permutations(range(1, n + 1)))
    N = len(perms)
    
    # Helper: Calculate overlap between two permutations
    def get_overlap(a, b):
        # a, b are tuples of length n
        # max k is n-1
        k = 0
        # a[n-1-k] ... a[n-1] == b[0] ... b[k-1]
        for i in range(n - 1):
            if a[n - 1 - i] == b[i]:
                k += 1
            else:
                break
        return k

    # Helper: Calculate cost (length added)
    def cost(a, b):
        return n - get_overlap(a, b)

    # 1. Greedy Initialization
    # Start with a random permutation
    start_idx = random.randint(0, N - 1)
    path = [start_idx]
    unused = set(range(N))
    unused.remove(start_idx)
    
    # Build path greedily
    while unused:
        last_idx = path[-1]
        last_perm = perms[last_idx]
        
        best_idx = -1
        best_ov = -1
        
        # Find next permutation with max overlap
        # This is O(N) per step, total O(N^2)
        for idx in unused:
            ov = get_overlap(last_perm, perms[idx])
            if ov > best_ov:
                best_ov = ov
                best_idx = idx
                # If we found max possible overlap (n-1), we can break early?
                # Max overlap is n-1.
                if best_ov == n - 1:
                    break
        
        path.append(best_idx)
        unused.remove(best_idx)
        
    # 2. Local Search: Swap improvements
    # Precompute current costs for faster delta?
    # cost[i] = cost(path[i], path[i+1])
    # Total energy = sum(cost[i])
    
    def calc_path_energy(path):
        e = 0
        for i in range(N - 1):
            e += cost(perms[path[i]], perms[path[i + 1]])
        return e

    current_energy = calc_path_energy(path)
    best_energy = current_energy
    best_path = path[:]
    
    start_time = time.time()
    timeout = 100 # 100 seconds limit
    
    # Local Search Loop
    # We will do passes of swapping i, j
    # To speed up, we can try random swaps or sequential swaps
    # Sequential 2-opt is O(N^2) per pass.
    # Let's do a few passes.
    
    passes = 0
    while time.time() - start_time < timeout:
        improved = False
        # Randomize order of i to avoid bias? Or just sequential.
        # Sequential is deterministic.
        for i in range(N - 1):
            if time.time() - start_time > timeout - 5:
                break
            
            # Try swapping path[i] with path[j] for j > i
            # To limit time, we might not check all j
            # But for N=5040, N^2 is large.
            # Let's try a limited set of j, e.g., j in [i+1, i+100] or random j
            
            # Strategy: Try a few random j for each i
            for _ in range(10): # 10 random swaps per i
                if time.time() - start_time > timeout - 5:
                    break
                j = random.randint(i + 1, N - 1)
                
                # Calculate delta energy for swapping i and j
                # Neighbors:
                # i-1, i, i+1
                # j-1, j, j+1
                
                # Case 1: i and j are not adjacent (j > i+1)
                if j == i + 1:
                    # Adjacent swap (reverse 2)
                    # Old: (i-1, i), (i, i+1), (i+1, i+2)
                    # New: (i-1, i+1), (i+1, i), (i, i+2)
                    left_i = path[i - 1] if i > 0 else None
                    right_j = path[j + 1] if j + 1 < N else None
                    
                    old_e = 0
                    new_e = 0
                    
                    if i > 0:
                        old_e += cost(perms[left_i], perms[path[i]])
                    if j + 1 < N:
                        old_e += cost(perms[path[j]], perms[right_j])
                    old_e += cost(perms[path[i]], perms[path[j]])
                    
                    if i > 0:
                        new_e += cost(perms[left_i], perms[path[j]])
                    if j + 1 < N:
                        new_e += cost(perms[path[i]], perms[right_j])
                    new_e += cost(perms[path[j]], perms[path[i]])
                    
                    delta = new_e - old_e
                    if delta < 0:
                        path[i], path[j] = path[j], path[i]
                        current_energy += delta
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_path = path[:]
                        improved = True
                else:
                    # Non-adjacent
                    left_i = path[i - 1] if i > 0 else None
                    right_i = path[i + 1] if i + 1 < N else None
                    left_j = path[j - 1] if j > 0 else None
                    right_j = path[j + 1] if j + 1 < N else None
                    
                    old_e = 0
                    new_e = 0
                    
                    if i > 0:
                        old_e += cost(perms[left_i], perms[path[i]])
                    if i + 1 < N:
                        old_e += cost(perms[path[i]], perms[right_i])
                    if j > 0:
                        old_e += cost(perms[left_j], perms[path[j]])
                    if j + 1 < N:
                        old_e += cost(perms[path[j]], perms[right_j])
                        
                    if i > 0:
                        new_e += cost(perms[left_i], perms[path[j]])
                    if i + 1 < N:
                        new_e += cost(perms[path[j]], perms[right_i])
                    if j > 0:
                        new_e += cost(perms[left_j], perms[path[i]])
                    if j + 1 < N:
                        new_e += cost(perms[path[i]], perms[right_j])
                        
                    delta = new_e - old_e
                    if delta < 0:
                        path[i], path[j] = path[j], path[i]
                        current_energy += delta
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_path = path[:]
                        improved = True
        
        passes += 1
        if not improved:
            # If no improvement in a pass, break? Or continue with random restarts?
            # Let's break if no improvement to save time for other strategies?
            # Or just break.
            break
            
    # Construct final string from best_path
    final_state = best_path
    word = list(perms[final_state[0]])
    for k in range(1, N):
        prev = perms[final_state[k - 1]]
        curr = perms[final_state[k]]
        ov = get_overlap(prev, curr)
        word.extend(curr[ov:])
        
    return "".join(map(str, word))
