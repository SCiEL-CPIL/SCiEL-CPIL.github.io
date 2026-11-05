"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    """
    Construct a large subset of [1, 512] with no 4-term AP.
    
    Strategy:
    1. Represent the candidate set S as a Python integer bitmask.
    2. Precompute for each element i, a list of 'conflict masks'. 
       A conflict mask is a bitmask representing the other 3 elements of a 4-AP containing i.
       Actually, to check if adding i is valid, we need to ensure no 4-AP is fully formed.
       A 4-AP is (a, a+d, a+2d, a+3d).
       If we have S, and we want to add x, we must ensure that for no d, 
       {x-d, x-2d, x-3d} subset S, {x-d, x-2d, x+3d} is not a 4AP structure... 
       
       Let's refine the bitmask check.
       Let S be the current set.
       We want to add x.
       This is invalid if there exists a 4-AP containing x such that the other 3 elements are in S.
       
       We can precompute for each x, a list of bitmasks M such that M represents the set of 3 elements 
       that, combined with x, form a 4-AP.
       Then, adding x is valid iff for all such masks M, (S & M) != M.
       
       Since N=512, we can use Python integers (arbitrary precision) as bitmasks.
       Bit i corresponds to element i+1 (or i-1, let's use 0-indexed for bits: bit i is element i+1).
       
       Precomputation:
       For each x in [1, 512]:
           For each 4-AP containing x:
               Let the other 3 elements be y1, y2, y3.
               Create a mask m = (1 << (y1-1)) | (1 << (y2-1)) | (1 << (y3-1)).
               Add m to conflicts[x-1].
               
       Check if x can be added to S (bitmask S):
           for m in conflicts[x-1]:
               if (S & m) == m:
                   return False
           return True
           
       This is O(K) per check, where K is the number of 4-APs containing x. K is ~300-400.
       This is fast enough for Simulated Annealing.
       
    3. Use Simulated Annealing to maximize the cardinality of S.
       State: Bitmask S.
       Move: Flip a random bit (add or remove an element).
       Energy: -|S| (we want to minimize energy, so maximize size).
       Constraint: S must be valid (no 4-AP).
       
       To ensure validity during SA, we can either:
       a) Only allow moves that keep the set valid (rejection sampling).
       b) Allow invalid states but penalize them heavily.
       
       Option (a) is safer and simpler to implement correctly.
       However, if we start with a valid set and only make valid moves, we might get stuck.
       To escape local optima, we can occasionally allow "invalid" moves? No, that breaks the constraint.
       
       Better: Use a local search that maintains validity.
       If we remove an element, the set remains valid (subset of valid set is valid).
       If we add an element, we must check validity.
       
       So the moves are:
       1. Remove a random element from S. (Always valid)
       2. Add a random element not in S, if valid. (Conditionally valid)
       
       To explore more, we can use "k-exchange" moves:
       Remove k elements, add l elements.
       But simple 1-exchange (add or remove) is a good start.
       
       To improve on greedy, we can start with multiple random valid initial sets 
       (using the bitmask greedy) and then run SA.
       
    4. Optimization:
       - Use Python integers for bitmasks.
       - Precompute conflicts as lists of integers.
       - Use a fast validity check.
    """
    
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 512
    # Bit i corresponds to element i+1
    
    # Precompute conflicts
    # conflicts[i] is a list of masks for element i+1
    conflicts = [[] for _ in range(N)]
    
    # Iterate all 4-APs
    for a in range(1, N - 2):
        max_d = (N - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            
            # Elements are a, b, c, e
            # For a: others are b, c, e
            mask_a = (1 << (b - 1)) | (1 << (c - 1)) | (1 << (e - 1))
            conflicts[a - 1].append(mask_a)
            
            # For b: others are a, c, e
            mask_b = (1 << (a - 1)) | (1 << (c - 1)) | (1 << (e - 1))
            conflicts[b - 1].append(mask_b)
            
            # For c: others are a, b, e
            mask_c = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (e - 1))
            conflicts[c - 1].append(mask_c)
            
            # For e: others are a, b, c
            mask_e = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))
            conflicts[e - 1].append(mask_e)
    
    def is_valid_addition(x, S):
        """Check if element x (1-indexed) can be added to bitmask S."""
        idx = x - 1
        for m in conflicts[idx]:
            if (S & m) == m:
                return False
        return True
    
    def greedy_construct_bitmask():
        """
        Greedy construction using bitmask.
        Returns a valid bitmask and its size.
        """
        S = 0
        # Random order
        order = list(range(1, N + 1))
        random.shuffle(order)
        for x in order:
            if is_valid_addition(x, S):
                S |= (1 << (x - 1))
        return S
    
    # Find a good initial set
    best_S = 0
    best_size = 0
    
    # Run a few greedy restarts to find a decent starting point
    for _ in range(50):
        if time.time() > deadline - 10:
            break
        S = greedy_construct_bitmask()
        size = bin(S).count('1')
        if size > best_size:
            best_size = size
            best_S = S
    
    current_S = best_S
    current_size = best_size
    
    # Simulated Annealing
    # Temperature starts high, cools down.
    # Move: 
    #   50% chance: try to add a random element not in S.
    #   50% chance: try to remove a random element in S.
    #   If adding fails, try a different random element (up to a few times).
    
    T = 1.0
    cooling = 0.9999
    min_T = 0.01
    
    # We need to track which elements are in S for fast removal.
    # We can just pick a random index and check if bit is set.
    
    while time.time() < deadline - 2 and T > min_T:
        # Random move
        # Try to add
        if random.random() < 0.5:
            # Pick random x not in S
            # To pick efficiently, we can pick a random index and check
            # Or maintain a list of missing elements? 
            # For N=512, picking random index is fine.
            added = False
            for _ in range(10): # Try up to 10 times to find a valid addition
                x = random.randint(1, N)
                if not (current_S & (1 << (x - 1))):
                    if is_valid_addition(x, current_S):
                        new_S = current_S | (1 << (x - 1))
                        new_size = current_size + 1
                        # Accept if it improves (which it always does for size)
                        # In SA, we might accept worse, but here "worse" means smaller size.
                        # We only want to increase size.
                        # However, sometimes removing one to add two later might be needed.
                        # But simple SA with only size as objective:
                        # If new_size > current_size, accept.
                        # If new_size == current_size (swap), accept with prob exp(-delta/T).
                        # Here delta = 0. So always accept swaps? 
                        # But we don't have swaps, only adds and removes.
                        # Remove decreases size. Add increases size.
                        
                        # If we only do adds, we get stuck at local optimum where no single add is possible.
                        # We need to allow removals to enable future additions.
                        # So, we should accept removals with some probability.
                        
                        current_S = new_S
                        current_size = new_size
                        added = True
                        break
        else:
            # Try to remove
            # Pick random x in S
            removed = False
            for _ in range(10):
                x = random.randint(1, N)
                if current_S & (1 << (x - 1)):
                    new_S = current_S & ~(1 << (x - 1))
                    new_size = current_size - 1
                    # Accept with probability exp(-(new_size - current_size) / T)
                    # delta = -1
                    # prob = exp(1/T)
                    # Wait, standard SA: accept if new_energy < old_energy OR with prob exp(-(new-old)/T)
                    # Energy = -size.
                    # new_energy = -new_size, old_energy = -current_size.
                    # new - old = -new_size - (-current_size) = current_size - new_size = 1.
                    # prob = exp(-1/T).
                    if random.random() < np.exp(-1.0 / T):
                        current_S = new_S
                        current_size = new_size
                        removed = True
                        break
            
        # Cool down
        T *= cooling
        
        # Update best
        if current_size > best_size:
            best_size = current_size
            best_S = current_S
    
    # Final local search: try to add any remaining elements
    # Iterate through all elements and try to add
    for x in range(1, N + 1):
        if not (best_S & (1 << (x - 1))):
            if is_valid_addition(x, best_S):
                best_S |= (1 << (x - 1))
                best_size += 1
    
    # Convert bitmask to list
    result = []
    for i in range(N):
        if best_S & (1 << i):
            result.append(i + 1)
            
    # Verification
    def has_4ap(s):
        s_set = set(s)
        for a in s_set:
            for b in s_set:
                if a >= b:
                    continue
                d = b - a
                if d == 0:
                    continue
                c = b + d
                e = c + d
                if c in s_set and e in s_set:
                    return True
        return False
    
    if has_4ap(result):
        # Fallback: return empty or a known valid set?
        # The task says "nonempty subset".
        # If verification fails, there's a bug. 
        # For safety, return a small valid set.
        return [1]
        
    return sorted(result)

# EVOLVE-BLOCK-END
