"""
Algorithm for dense increasing sequences with unique block products.

Strategy:
1. Use a beam search to explore multiple partial sequences.
2. The sequence must be strictly increasing, so we iterate candidates x from 2 to 128.
3. For each candidate x, we attempt to append it to each sequence in the beam.
4. We maintain the set of all seen block products and the list of suffix products
   for efficient collision checking.
5. We prune the beam to keep the top K sequences based on length.
"""
import time

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def construct():
    deadline = time.time() + 55.0
    
    # State: (length, last_val, seq, suffix_prods, all_prods)
    # Initial state: empty sequence
    # last_val=1 ensures any x>=2 is valid.
    beams = [(0, 1, [], [], frozenset())]
    
    K = 100 # Beam width
    
    for x in range(2, 129):
        if time.time() > deadline:
            break
            
        new_beams = []
        
        for length, last_val, seq, suffix_prods, all_prods in beams:
            # Option 1: Skip x
            new_beams.append((length, last_val, seq, suffix_prods, all_prods))
            
            # Option 2: Append x
            if x > last_val:
                # Calculate new suffix products
                new_suffix_prods = [x]
                current_prod = x
                for sp in suffix_prods:
                    current_prod *= sp
                    new_suffix_prods.append(current_prod)
                
                # Check for collisions
                collision = False
                for np in new_suffix_prods:
                    if np in all_prods:
                        collision = True
                        break
                
                if not collision:
                    new_length = length + 1
                    new_last_val = x
                    new_seq = seq + [x]
                    new_all_prods = frozenset(all_prods.union(new_suffix_prods))
                    
                    new_beams.append((new_length, new_last_val, new_seq, new_suffix_prods, new_all_prods))
        
        # Prune
        # Sort by length descending, then last_val ascending
        new_beams.sort(key=lambda state: (-state[0], state[1]))
        
        # Keep top K
        beams = new_beams[:K]
        
    # Return the best sequence
    best_seq = []
    max_len = -1
    for length, last_val, seq, suffix_prods, all_prods in beams:
        if length > max_len:
            max_len = length
            best_seq = seq
            
    return best_seq
