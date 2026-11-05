import time
import numpy as np
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    base = 51975
    
    # 1. Get all odd divisors of base greater than 1
    divisors = []
    for d in range(3, base + 1, 2):
        if base % d == 0:
            divisors.append(d)
    divisors.sort()
    
    # 2. Precompute indices for each modulus and residue
    indices = {}
    for m in divisors:
        indices[m] = []
        for r in range(m):
            arr = np.arange(r, base, m, dtype=np.int32)
            indices[m].append(arr)
            
    # 3. Greedy Initialization
    counts = np.zeros(base, dtype=np.int32)
    classes = [] # List of [m, r]
    
    # Sort divisors by ascending order (smaller moduli first)
    # This allows high-coverage classes to be placed first
    for m in divisors:
        if time.time() > deadline - 15:
            break
            
        best_r = 0
        max_gain = -1
        uncovered = (counts == 0).astype(np.int32)
        
        # Vectorized gain calculation for all residues of this modulus
        # gains[r] = sum(uncovered[indices[m][r]])
        # We can do this by summing the uncovered array over the specific indices
        # Since indices[m] is a list of arrays, we can stack them? No, different lengths? No, same length N/m.
        # Actually, we can just loop r, it's small.
        # Or: 
        # gain_vec = np.zeros(m, dtype=np.int32)
        # for r in range(m):
        #     gain_vec[r] = np.dot(uncovered, (indices[m][r] is not efficient for dot))
        
        # Let's stick to the loop, it's fast enough.
        gains = np.array([np.sum(uncovered[indices[m][r]]) for r in range(m)])
        best_r = int(np.argmax(gains))
        
        classes.append([m, best_r])
        counts[indices[m][best_r]] += 1
        
    # 4. Simulated Annealing
    current_uncovered = np.sum(counts == 0)
    best_uncovered = current_uncovered
    best_classes = [c[:] for c in classes]
    
    T = 5.0
    T_min = 0.01
    cooling_rate = 0.9995
    
    # Main SA Loop
    while time.time() < deadline - 10:
        idx = random.randint(0, len(classes) - 1)
        m, old_r = classes[idx]
        
        new_r = random.randint(0, m - 1)
        if new_r == old_r:
            continue
            
        old_indices = indices[m][old_r]
        new_indices = indices[m][new_r]
        
        loss = np.sum(counts[old_indices] == 1)
        gain = np.sum(counts[new_indices] == 0)
        
        delta = loss - gain
        
        if delta < 0:
            counts[old_indices] -= 1
            counts[new_indices] += 1
            classes[idx][1] = new_r
            current_uncovered += delta
            if current_uncovered < best_uncovered:
                best_uncovered = current_uncovered
                best_classes = [c[:] for c in classes]
        else:
            if random.random() < math.exp(-delta / T):
                counts[old_indices] -= 1
                counts[new_indices] += 1
                classes[idx][1] = new_r
                current_uncovered += delta
                if current_uncovered < best_uncovered:
                    best_uncovered = current_uncovered
                    best_classes = [c[:] for c in classes]
        
        T *= cooling_rate
        if T < T_min:
            T = T_min

    # 5. Final Polish (Hill Climbing)
    # Try to improve further with only accepting better moves
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        for idx in range(len(classes)):
            if time.time() > deadline - 2:
                break
            m, old_r = classes[idx]
            
            # Find best residue for this modulus given current counts
            # We need to check all r
            # This is O(m * N/m) = O(N) per class.
            # 47 classes * 51975 = 2.4M ops. Very fast.
            
            # Temporarily remove
            counts[indices[m][old_r]] -= 1
            
            best_r = old_r
            best_gain = -1
            uncovered = (counts == 0).astype(np.int32)
            
            # Calculate gain for each r
            # gain[r] = sum(uncovered[indices[m][r]])
            # We want to maximize gain, which is equivalent to minimizing new uncovered.
            # Actually, the delta formula is loss - gain.
            # If we pick r, the new uncovered is current_uncovered - gain[r] (since we removed the class, 
            # the "loss" part is handled by the fact that we are re-adding it).
            # Wait, the delta formula assumed we are changing from old_r to new_r.
            # Here we are removing old_r and adding new_r.
            # So we want to maximize the number of newly covered spots (gain).
            
            gains = np.array([np.sum(uncovered[indices[m][r]]) for r in range(m)])
            best_r = int(np.argmax(gains))
            
            # Apply best_r
            counts[indices[m][best_r]] += 1
            classes[idx][1] = best_r
            
            # Update current_uncovered
            # The change in uncovered is:
            # - (spots in best_r that were 0) + (spots in old_r that were 1)
            # We already subtracted old_r.
            # Let's just recalculate or update delta.
            # Delta = (spots in old_r that were 1 before removal) - (spots in best_r that were 0 before addition)
            # This is tricky to track without storing intermediate states.
            # Easier: just recalculate current_uncovered? No, too slow.
            # Let's use the delta logic:
            # Before removal: U_old.
            # After removal: U_mid = U_old + (count of x in old_r with C[x]==1)
            # After addition of best_r: U_new = U_mid - (count of x in best_r with C[x]==0 in U_mid state)
            # Note: C[x] in U_mid state is C[x] - 1 if x in old_r, else C[x].
            # This is getting complicated.
            
            # Simpler approach for Polish:
            # Just check if the new configuration is better than the old one for this class.
            # We can calculate the "cost" of having residue r for modulus m, given the rest of the system.
            # Cost(r) = number of x in indices[m][r] such that (counts_without_m[x] == 0)
            # Where counts_without_m = counts - [x in old_r]
            # We want to minimize Cost(r).
            
            # Let's just do a full recalculation of uncovered for the whole system if we change?
            # No, that's O(N) per class change. 47 * 51975 = 2.4M. 
            # If we do this for every class in a pass, it's 2.4M * 47 = 112M ops. 
            # In Python/NumPy, this might take ~1-2 seconds. 
            # So we can do a few passes.
            
            # Let's optimize the polish:
            # We already have `uncovered` (counts == 0) BEFORE removing the class.
            # Let's re-evaluate the polish step properly.
            
            # Re-do polish logic:
            # For each class i:
            # 1. Remove class i.
            # 2. Calculate U_mid (uncovered count).
            # 3. For each r, calculate gain (spots in r that are 0 in U_mid).
            # 4. Pick best r.
            # 5. Add class i with best r.
            # 6. Update U.
            
            # This is what I'll implement in the polish loop.
            
            pass # Placeholder, will rewrite below
            
    return best_classes
