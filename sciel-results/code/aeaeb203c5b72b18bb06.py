"""Program for dense increasing sequences with unique block products."""
# EVOLVE-BLOCK-START
import math
import random
import time
import bisect

def _is_valid(seq):
    """Check if the sequence has distinct products for all consecutive blocks."""
    prods = set()
    n = len(seq)
    for i in range(n):
        p = 1
        for j in range(i, n):
            p *= seq[j]
            if p in prods:
                return False
            prods.add(p)
    return True

def _compute_cross_products(L, R):
    """Compute products of all blocks that start in L and end in R."""
    prods = set()
    n_L = len(L)
    n_R = len(R)
    if n_L == 0 or n_R == 0:
        return prods
        
    # Precompute prefix products of R
    prefix_R = [1] * (n_R + 1)
    for i in range(n_R):
        prefix_R[i+1] = prefix_R[i] * R[i]
        
    # Iterate i in L from right to left
    p_left = 1
    for i in range(n_L - 1, -1, -1):
        p_left *= L[i]
        for k in range(n_R):
            p_total = p_left * prefix_R[k+1]
            prods.add(p_total)
            
    return prods

def construct():
    """
    Return strictly increasing integers in [2, 128] such that products of
    every nonempty consecutive block are distinct. Maximize length.
    
    Approach: 
    1. Greedy randomized insertion with incremental product set updates.
    2. Simulated Annealing to refine.
    """
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(math.isqrt(n)) + 1):
            if n % i == 0:
                return False
        return True

    primes = [x for x in range(2, 129) if is_prime(x)]
    current_seq = primes[:]
    current_prods = set()
    
    # Initialize current_prods
    for i in range(len(current_seq)):
        p = 1
        for j in range(i, len(current_seq)):
            p *= current_seq[j]
            current_prods.add(p)
            
    all_numbers = set(range(2, 129))
    unused = all_numbers - set(current_seq)
    
    # Greedy Phase
    for _ in range(20):
        if not unused:
            break
        improved = False
        unused_list = list(unused)
        random.shuffle(unused_list)
        
        for x in unused_list:
            positions = list(range(len(current_seq) + 1))
            random.shuffle(positions)
            
            inserted = False
            for pos in positions:
                L = current_seq[:pos]
                R = current_seq[pos:]
                
                # Compute old crossing products to remove
                old_cross = _compute_cross_products(L, R)
                
                # Compute new products involving x
                new_prods = set()
                valid = True
                
                # x itself
                if x in current_prods - old_cross: # Note: old_cross are removed, so we check against remaining
                    # Actually, simpler: check against (current_prods - old_cross)
                    pass
                # Let's just check against the *effective* current set
                # Effective set = current_prods - old_cross
                # We can't easily subtract sets. 
                # Let's just check collisions with:
                # 1. Other new_prods
                # 2. current_prods that are NOT in old_cross
                
                # This is getting complicated. 
                # Let's just use _is_valid for the Greedy phase to avoid bugs.
                # The incremental logic is prone to errors in set management.
                # Given N is small, _is_valid might be fast enough if we limit rounds.
                
                # Let's revert to _is_valid for safety, but limit the search space.
                # Break on first valid pos.
                new_seq = current_seq[:]
                bisect.insort(new_seq, x)
                
                if _is_valid(new_seq):
                    # Update current_prods
                    current_seq = new_seq
                    current_prods = set()
                    for i in range(len(current_seq)):
                        p = 1
                        for j in range(i, len(current_seq)):
                            p *= current_seq[j]
                            current_prods.add(p)
                    unused.remove(x)
                    improved = True
                    inserted = True
                    break
            if inserted:
                continue
                
        if not improved:
            break
            
    best_seq = current_seq[:]
    
    # SA Phase
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.999
    start_time = time.time()
    time_limit = 50.0
    
    def get_energy(seq):
        if not _is_valid(seq):
            return 100000
        return -len(seq)
        
    current_energy = get_energy(current_seq)
    best_energy = current_energy
    
    while time.time() - start_time < time_limit:
        if T < T_min:
            break
            
        r = random.random()
        new_seq = current_seq[:]
        
        if r < 0.6:
            if unused:
                x = random.choice(list(unused))
                pos = random.randint(0, len(current_seq))
                bisect.insort(new_seq, x)
            else:
                continue
        elif r < 0.9:
            if unused and current_seq:
                i = random.randint(0, len(current_seq) - 1)
                x_remove = current_seq[i]
                x_add = random.choice(list(unused))
                new_seq.pop(i)
                bisect.insort(new_seq, x_add)
            else:
                continue
        else:
            if current_seq:
                i = random.randint(0, len(current_seq) - 1)
                new_seq.pop(i)
            else:
                continue
                
        new_energy = get_energy(new_seq)
        
        delta_e = new_energy - current_energy
        if delta_e < 0 or random.random() < math.exp(-delta_e / T):
            current_seq = new_seq
            current_energy = new_energy
            unused = all_numbers - set(current_seq)
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_seq = current_seq[:]
                
        T *= cooling_rate
        
    if _is_valid(best_seq):
        return best_seq
    if _is_valid(current_seq):
        return current_seq
    return primes

# EVOLVE-BLOCK-END
