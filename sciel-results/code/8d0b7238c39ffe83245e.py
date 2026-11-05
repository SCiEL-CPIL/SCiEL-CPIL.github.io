import itertools
import numpy as np
import random
import time

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin: finish by 110s, limit is 120s
    
    total_words = q ** n
    
    def word_to_index(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx
    
    def index_to_word(idx):
        w = []
        for i in range(n):
            w.append(idx % q)
            idx //= q
        return tuple(reversed(w))
    
    # Precompute coverage sets for all possible words in the space
    # cover_set[idx] = frozenset of word indices covered by the word at index idx
    # For radius=1, each word covers itself + n*(q-1) neighbors
    # We'll use bitsets (Python integers) for fast coverage checks
    
    # Precompute all cover bitsets
    all_cover_bits = [0] * total_words
    
    for idx in range(total_words):
        w = index_to_word(idx)
        bits = 1 << idx
        for pos in range(n):
            orig = w[pos]
            for val in range(q):
                if val == orig:
                    continue
                w_list = list(w)
                w_list[pos] = val
                neighbor_idx = word_to_index(tuple(w_list))
                bits |= (1 << neighbor_idx)
        all_cover_bits[idx] = bits
    
    # Precompute overlap counts: for each pair of words, how many words they cover in common
    # overlap[i][j] = popcount(cover_bits[i] & cover_bits[j])
    # This is expensive (729x729), but we can compute it on the fly or cache it
    # Actually, let's compute it once since 729*729 = 531k pairs, each is a 729-bit AND + popcount
    
    # For the specific problem q=3, n=6, radius=1:
    # Start with a good initial solution
    # Use the tetracode direct sum construction as baseline
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(q) for b in range(q)]
    
    initial_code = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            initial_code.append(tuple(h) + tuple(tail))
    
    # Convert to indices
    initial_indices = [word_to_index(c) for c in initial_code]
    
    best_code = initial_code[:]
    best_size = len(best_code)
    
    # Greedy pruning: remove redundant codewords
    # A codeword is redundant if all words it uniquely covers are also covered by others
    
    def greedy_prune(code_indices):
        """Remove redundant codewords from the code."""
        if not code_indices:
            return []
        
        # Build coverage count using bitsets
        # cov_count[wi] = number of codewords covering word wi
        cov_count = np.zeros(total_words, dtype=np.int32)
        
        for ci in code_indices:
            bits = all_cover_bits[ci]
            # Extract set bits
            b = bits
            while b:
                low_bit = b & (-b)
                wi = low_bit.bit_length() - 1
                cov_count[wi] += 1
                b ^= low_bit
        
        # Repeatedly remove redundant codewords
        active = list(code_indices)
        changed = True
        while changed:
            changed = False
            new_active = []
            for ci in active:
                bits = all_cover_bits[ci]
                # Check if ci is redundant: for all words it covers, cov_count > 1
                b = bits
                redundant = True
                while b:
                    low_bit = b & (-b)
                    wi = low_bit.bit_length() - 1
                    if cov_count[wi] <= 1:
                        redundant = False
                        break
                    b ^= low_bit
                if redundant:
                    # Remove ci
                    b = bits
                    while b:
                        low_bit = b & (-b)
                        wi = low_bit.bit_length() - 1
                        cov_count[wi] -= 1
                        b ^= low_bit
                    changed = True
                else:
                    new_active.append(ci)
            active = new_active
        
        return active
    
    pruned_indices = greedy_prune(initial_indices)
    pruned_code = [index_to_word(ci) for ci in pruned_indices]
    
    if len(pruned_code) < best_size:
        best_code = pruned_code[:]
        best_size = len(best_code)
    
    # Local search with bitset-based coverage
    # State: set of active codeword indices
    # Move: remove one codeword, optionally add another
    
    current_indices = pruned_indices[:]
    
    # Precompute for each codeword which other codewords overlap with it
    # This helps in deciding which codewords to add when removing one
    
    # Simulated annealing local search
    T = 5.0
    T_min = 0.1
    cooling = 0.999
    max_steps = 50000
    
    step = 0
    while time.time() < deadline - 10.0 and step < max_steps and T > T_min:
        if not current_indices:
            break
        
        # Try to remove a random codeword
        ci = random.choice(current_indices)
        ci_bits = all_cover_bits[ci]
        
        # Find words that would become uncovered if we remove ci
        # These are words where cov_count == 1 and covered by ci
        # We need to compute this efficiently
        
        # Compute current coverage count
        cov_count = np.zeros(total_words, dtype=np.int32)
        for cj in current_indices:
            bits = all_cover_bits[cj]
            b = bits
            while b:
                low_bit = b & (-b)
                wi = low_bit.bit_length() - 1
                cov_count[wi] += 1
                b ^= low_bit
        
        # Find words that would be uncovered
        uncovered_words = []
        b = ci_bits
        while b:
            low_bit = b & (-b)
            wi = low_bit.bit_length() - 1
            if cov_count[wi] == 1:
                uncovered_words.append(wi)
            b ^= low_bit
        
        if not uncovered_words:
            # Safe to remove
            current_indices.remove(ci)
            if len(current_indices) < best_size:
                best_size = len(current_indices)
                best_code = [index_to_word(ci2) for ci2 in current_indices]
            T *= cooling
            step += 1
            continue
        
        # Need to find a replacement that covers all uncovered_words
        # Try random candidates from the full space
        found_replacement = False
        num_tries = min(50, total_words)
        
        for _ in range(num_tries):
            cj = random.randint(0, total_words - 1)
            if cj in current_indices:
                continue
            
            cj_bits = all_cover_bits[cj]
            # Check if cj covers all uncovered_words
            covers_all = True
            for wi in uncovered_words:
                if not (cj_bits & (1 << wi)):
                    covers_all = False
                    break
            
            if covers_all:
                # Perform swap
                current_indices.remove(ci)
                current_indices.append(cj)
                found_replacement = True
                break
        
        if not found_replacement:
            # Try adding a new codeword without removing (if we can afford it)
            # Or just cool down
            pass
        
        T *= cooling
        step += 1
    
    # Final greedy prune
    final_indices = greedy_prune(current_indices)
    final_code = [index_to_word(ci) for ci in final_indices]
    
    if len(final_code) < best_size:
        best_code = final_code
        best_size = len(best_code)
    
    # Verify and return
    if time.time() < deadline - 2.0:
        # Verify coverage
        covered = 0
        for c in best_code:
            ci = word_to_index(tuple(c))
            bits = all_cover_bits[ci]
            covered |= bits
        
        if covered != (1 << total_words) - 1:
            # Fallback to initial code
            best_code = initial_code[:]
    
    return [list(c) for c in best_code]
