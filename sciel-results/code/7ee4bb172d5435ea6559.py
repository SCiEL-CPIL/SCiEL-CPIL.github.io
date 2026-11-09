import time
import random
import numpy as np
from math import log

def _compute_score(U):
    """Compute the score for a given set U using vectorized NumPy operations."""
    U = sorted(set(U))
    n = len(U)
    if n == 0 or U[0] != 0:
        return 0.0
    max_u = U[-1]
    if max_u == 0:
        return 1.0
    
    # Convert to numpy array for vectorized operations
    U_arr = np.array(U, dtype=np.int64)
    
    # Compute sumset cardinality
    # U_arr[:, None] + U_arr[None, :] creates a (n, n) matrix of all sums
    sums = U_arr[:, None] + U_arr[None, :]
    len_sum = len(np.unique(sums))
    
    # Compute diffset cardinality
    diffs = U_arr[:, None] - U_arr[None, :]
    len_diff = len(np.unique(diffs))
    
    if len_sum == 0:
        return 0.0
        
    ratio = len_diff / len_sum
    if ratio <= 0:
        return 0.0
        
    denominator = log(2 * max_u + 1)
    if denominator == 0:
        return 1.0
        
    return 1 + log(ratio) / denominator

def _generate_initial_sets():
    """Generate diverse initial sets for search."""
    sets = []
    
    # 1. Simple small sets
    sets.append([0, 1])
    sets.append([0, 1, 2])
    sets.append([0, 1, 3])
    sets.append([0, 1, 3, 7])
    sets.append([0, 1, 4, 9])
    sets.append([0, 1, 3, 9])
    
    # 2. Geometric progressions
    for base in [2, 3, 4]:
        s = [0]
        val = 1
        for _ in range(6):
            s.append(val)
            val *= base
            if val > 1000:
                break
        sets.append(sorted(s))
        
    # 3. Lacunary sequences (e.g., powers of 2, 3, etc.)
    for k in range(2, 5):
        s = [0]
        val = 1
        for _ in range(8):
            s.append(val)
            val *= k
        sets.append(sorted(s))
            
    # 4. Random sets
    random.seed(42)
    for _ in range(10):
        size = random.randint(3, 12)
        s = {0}
        while len(s) < size:
            s.add(random.randint(1, 100))
        sets.append(sorted(s))
        
    # 5. Sidon-like greedy (small)
    def greedy_sidon(max_val, target_size):
        U = [0]
        for x in range(1, max_val + 1):
            if len(U) >= target_size:
                break
            valid = True
            # Check if x can be added
            # Sidon: a+b = c+d implies {a,b}={c,d}
            # For new x, check if x+a = b+c for a,b,c in U?
            # Actually, simpler check for small sets:
            # Just compute score and see if it's a valid Sidon? No, Sidon is a property, not score.
            # Let's just check if adding x creates a collision in sums.
            current_sums = set()
            for i in range(len(U)):
                for j in range(len(U)):
                    current_sums.add(U[i] + U[j])
                    
            new_sums = set()
            for a in U:
                new_sums.add(a + x)
            new_sums.add(2*x)
            
            # If intersection of new_sums and (current_sums \ {sums involving x?}) is empty?
            # Actually, if a+x = b+c where a,b,c in U, collision.
            # This is covered by checking if a+x is in current_sums?
            # No, current_sums doesn't have x.
            # So if a+x is in current_sums, then a+x = b+c. Collision.
            # Also if 2x is in current_sums, 2x = b+c. Collision.
            # Also if a+x = b+x? Impossible for a!=b.
            
            if new_sums & current_sums:
                valid = False
            
            if valid:
                U.append(x)
        return U

    for size in [5, 8, 10]:
        s = greedy_sidon(50, size)
        if len(s) >= 2:
            sets.append(s)

    return sets

def _sa_search(initial_set, max_val, timeout, rng):
    """Simulated Annealing search."""
    start_time = time.time()
    
    current = sorted(set(initial_set))
    if 0 not in current:
        current.insert(0, 0)
    current_score = _compute_score(current)
    
    best = list(current)
    best_score = current_score
    
    T = 0.5
    T_min = 1e-5
    cooling = 0.9995
    
    while time.time() - start_time < timeout:
        if T < T_min:
            break
            
        # Generate neighbor
        move_type = rng.random()
        new_set = list(current)
        
        if move_type < 0.4:
            # Change an element (not 0)
            if len(new_set) <= 1:
                continue
            # Indices of non-zero elements
            non_zero_indices = [i for i, x in enumerate(new_set) if x != 0]
            if not non_zero_indices:
                continue
            idx = non_zero_indices[rng.randint(0, len(non_zero_indices) - 1)]
            old_val = new_set[idx]
            
            # Shift by small delta
            delta = rng.randint(-10, 10)
            new_val = old_val + delta
            
            # Clamp or randomize
            if new_val < 1: # Must be > 0 if it's not 0
                new_val = rng.randint(1, max_val)
            elif new_val > max_val:
                new_val = rng.randint(1, max_val)
                
            if new_val in new_set:
                # If collision, try another value or skip
                if new_val == old_val:
                    continue
                else:
                    # Find a nearby value not in set
                    found = False
                    for d in range(1, 5):
                        v1 = new_val + d
                        v2 = new_val - d
                        if v1 > 0 and v1 <= max_val and v1 not in new_set:
                            new_val = v1
                            found = True
                            break
                        if v2 > 0 and v2 <= max_val and v2 not in new_set:
                            new_val = v2
                            found = True
                            break
                    if not found:
                        continue
                        
            new_set[idx] = new_val
        elif move_type < 0.6:
            # Add an element
            if len(new_set) > 20: # Cap size
                continue
            new_elem = rng.randint(1, max_val)
            if new_elem not in new_set:
                new_set.append(new_elem)
            else:
                continue
        elif move_type < 0.8:
            # Remove an element (not 0)
            non_zero = [x for x in new_set if x != 0]
            if not non_zero:
                continue
            rem = non_zero[rng.randint(0, len(non_zero) - 1)]
            new_set.remove(rem)
        else:
            # Random restart / large jump
            if rng.random() < 0.5:
                # Keep 0 and some others, randomize rest
                keep = [0]
                if len(new_set) > 2:
                    keep.append(new_set[rng.randint(1, len(new_set)-1)])
                # Add random new elements
                target_size = rng.randint(3, 10)
                while len(keep) < target_size:
                    e = rng.randint(1, max_val)
                    if e not in keep:
                        keep.append(e)
                new_set = keep
            else:
                # Scale existing set
                factor = rng.choice([2, 3, 0.5, 0.3])
                new_set = [0] + [int(x * factor) for x in new_set if x != 0]
                # Remove duplicates and clamp
                new_set = sorted(set([x for x in new_set if 1 <= x <= max_val]))
                if 0 not in new_set:
                    new_set.insert(0, 0)
                if len(new_set) < 2:
                    new_set.append(1)
                    
        new_set = sorted(set(new_set))
        if 0 not in new_set or len(new_set) < 2:
            continue
            
        new_score = _compute_score(new_set)
        
        # Acceptance
        if new_score >= current_score:
            current = new_set
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best = list(current)
        else:
            # Boltzmann factor
            delta_s = current_score - new_score
            if delta_s < 0:
                accept_prob = 1.0
            else:
                accept_prob = np.exp(-delta_s / T)
            
            if rng.random() < accept_prob:
                current = new_set
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best = list(current)
                    
        T *= cooling
        
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145 # Safety margin
    
    best_result = [0, 1, 3, 7]
    best_score = _compute_score(best_result)
    
    rng = random.Random(42)
    
    # Generate initial sets
    initial_sets = _generate_initial_sets()
    
    # Shuffle to vary order
    rng.shuffle(initial_sets)
    
    # Run SA on each initial set
    for init in initial_sets:
        if time.time() > deadline - 5:
            break
            
        # Determine max_val for search
        # Start with a reasonable bound
        max_val = max(max(init) * 2, 100)
        
        # Time budget per run
        remaining = deadline - time.time()
        if remaining < 2.0:
            break
        sa_timeout = min(10.0, remaining * 0.2)
        
        result, score = _sa_search(init, max_val, sa_timeout, rng)
        
        if score > best_score:
            best_score = score
            best_result = result
            
    # Final refinement on the best result found
    if time.time() < deadline - 5:
        max_val_refine = max(max(best_result) * 2, 200)
        remaining = deadline - time.time()
        result, score = _sa_search(best_result, max_val_refine, min(15.0, remaining - 2), rng)
        if score > best_score:
            best_score = score
            best_result = result
            
    # Ensure validity
    final_score = _compute_score(best_result)
    if final_score < 1.0:
        best_result = [0, 1, 3, 7]
        
    return best_result
