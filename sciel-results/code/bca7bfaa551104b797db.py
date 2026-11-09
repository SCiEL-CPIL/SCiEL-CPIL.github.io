import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    candidates = list(range(2, 129))
    n = len(candidates)  # 127 candidates (2..128)
    
    # Precompute prime exponent vectors for each candidate
    # Primes up to 128
    def is_prime(num):
        if num < 2: return False
        if num < 4: return True
        if num % 2 == 0 or num % 3 == 0: return False
        i = 5
        while i * i <= num:
            if num % i == 0 or num % (i + 2) == 0:
                return False
            i += 6
        return True
    
    primes_list = [p for p in range(2, 129) if is_prime(p)]
    prime_index = {p: i for i, p in enumerate(primes_list)}
    num_primes = len(primes_list)
    
    # For each candidate value v (2..128), store its prime exponent vector as a tuple
    exp_vectors = {}
    for v in candidates:
        vec = [0] * num_primes
        temp = v
        for pi, p in enumerate(primes_list):
            while temp % p == 0:
                vec[pi] += 1
                temp //= p
        exp_vectors[v] = tuple(vec)
    
    # Key insight: A sequence is valid iff all consecutive block products are distinct.
    # Since the sequence is strictly increasing, we need to check that no two blocks have the same product.
    # Two blocks have the same product iff their prime exponent vectors are equal.
    
    # For a given sequence, block (i,j) has product with exponent vector = sum of exp_vectors[seq[k]] for k in [i,j].
    
    # We'll use a greedy + local search approach with efficient validation.
    # Start with primes (which are always valid since each prime appears once and blocks of primes have unique products).
    
    def validate_sequence(seq):
        """Check if all consecutive block products are distinct using exponent vectors."""
        n_seq = len(seq)
        if n_seq <= 1:
            return True
        # Compute all block product exponent vectors
        # block_prod[i][j] = exp vector of product seq[i]*seq[i+1]*...*seq[j]
        # We only need to check for duplicates
        seen = set()
        for i in range(n_seq):
            vec = [0] * num_primes
            for j in range(i, n_seq):
                ev = exp_vectors[seq[j]]
                for k in range(num_primes):
                    vec[k] += ev[k]
                t = tuple(vec)
                if t in seen:
                    return False
                seen.add(t)
        return True
    
    # Greedy construction: start with primes, try to insert composites
    def greedy_construct():
        # Start with all primes
        primes_in_range = sorted([x for x in candidates if is_prime(x)])
        seq = primes_in_range[:]
        used = set(seq)
        remaining = sorted([x for x in candidates if x not in used])
        
        # Try to insert each remaining number in various positions
        improved = True
        while improved:
            improved = False
            for c in remaining[:]:
                if time.time() > deadline - 3:
                    return seq
                # Try inserting c at each valid position
                best_pos = -1
                for pos in range(len(seq) + 1):
                    if pos > 0 and seq[pos-1] >= c:
                        continue
                    if pos < len(seq) and c >= seq[pos]:
                        continue
                    # Test insertion
                    new_seq = seq[:pos] + [c] + seq[pos:]
                    if validate_sequence(new_seq):
                        best_pos = pos
                        break
                if best_pos >= 0:
                    seq = seq[:best_pos] + [c] + seq[best_pos:]
                    used.add(c)
                    remaining.remove(c)
                    improved = True
        
        return seq
    
    # Local search: try to swap or replace elements to improve
    def local_search(seq, iterations=1000):
        seq = seq[:]
        current_len = len(seq)
        best_len = current_len
        
        for _ in range(iterations):
            if time.time() > deadline - 2:
                break
            
            # Try removing one element and adding another
            for i in range(len(seq)):
                if time.time() > deadline - 2:
                    break
                without_i = seq[:i] + seq[i+1:]
                # Try adding each unused element
                used = set(without_i)
                for c in candidates:
                    if c in used:
                        continue
                    # Try inserting c at each position
                    for pos in range(len(without_i) + 1):
                        if pos > 0 and without_i[pos-1] >= c:
                            continue
                        if pos < len(without_i) and c >= without_i[pos]:
                            continue
                        new_seq = without_i[:pos] + [c] + without_i[pos:]
                        if len(new_seq) > current_len and validate_sequence(new_seq):
                            seq = new_seq
                            current_len = len(seq)
                            if current_len > best_len:
                                best_len = current_len
                            break
                    if len(seq) > current_len:
                        break
                if len(seq) > current_len:
                    break
        
        return seq
    
    # Run multiple random restarts with different initial orders
    best_seq = greedy_construct()
    best_len = len(best_seq)
    
    # Try local search on the best found
    if time.time() < deadline - 5:
        improved_seq = local_search(best_seq, iterations=500)
        if len(improved_seq) > best_len:
            best_seq = improved_seq
            best_len = len(best_seq)
    
    # Try different random initializations
    for restart in range(15):
        if time.time() > deadline - 5:
            break
        
        # Random subset of primes as starting point
        all_primes = sorted([x for x in candidates if is_prime(x)])
        random.shuffle(all_primes)
        start_size = max(5, len(all_primes) - random.randint(0, 5))
        start_seq = sorted(all_primes[:start_size])
        
        if not validate_sequence(start_seq):
            continue
        
        current_seq = start_seq[:]
        used = set(current_seq)
        remaining = [x for x in candidates if x not in used]
        random.shuffle(remaining)
        
        # Greedy insertion
        progress = True
        while progress:
            progress = False
            for c in remaining[:]:
                if time.time() > deadline - 3:
                    break
                best_pos = -1
                for pos in range(len(current_seq) + 1):
                    if pos > 0 and current_seq[pos-1] >= c:
                        continue
                    if pos < len(current_seq) and c >= current_seq[pos]:
                        continue
                    new_seq = current_seq[:pos] + [c] + current_seq[pos:]
                    if validate_sequence(new_seq):
                        best_pos = pos
                        break
                if best_pos >= 0:
                    current_seq = current_seq[:best_pos] + [c] + current_seq[best_pos:]
                    used.add(c)
                    remaining.remove(c)
                    progress = True
                    break
        
        if len(current_seq) > best_len:
            best_seq = current_seq[:]
            best_len = len(best_seq)
        
        # Local search on this result
        if time.time() < deadline - 3 and len(current_seq) >= best_len - 2:
            ls_seq = local_search(current_seq, iterations=300)
            if len(ls_seq) > best_len:
                best_seq = ls_seq[:]
                best_len = len(best_seq)
    
    # Final polish: try to add more elements to best_seq
    if time.time() < deadline - 2:
        final_seq = best_seq[:]
        used = set(final_seq)
        remaining = sorted([x for x in candidates if x not in used])
        improved = True
        while improved:
            improved = False
            for c in remaining[:]:
                if time.time() > deadline - 1:
                    break
                for pos in range(len(final_seq) + 1):
                    if pos > 0 and final_seq[pos-1] >= c:
                        continue
                    if pos < len(final_seq) and c >= final_seq[pos]:
                        continue
                    new_seq = final_seq[:pos] + [c] + final_seq[pos:]
                    if validate_sequence(new_seq):
                        final_seq = new_seq
                        used.add(c)
                        remaining.remove(c)
                        improved = True
                        break
                if improved:
                    break
        if len(final_seq) > best_len:
            best_seq = final_seq
            best_len = len(best_seq)
    
    return best_seq
