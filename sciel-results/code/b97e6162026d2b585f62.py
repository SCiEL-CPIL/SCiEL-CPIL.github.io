import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 90s kill
    
    # Precompute primes up to 128
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(math.isqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    all_numbers = list(range(2, 129))
    primes = [x for x in all_numbers if is_prime(x)]
    composites = [x for x in all_numbers if not is_prime(x)]
    
    # Precompute prime exponent vectors for all numbers 2..128
    prime_list = primes
    num_primes = len(prime_list)
    prime_to_idx = {p: i for i, p in enumerate(prime_list)}
    
    # Exponent vector for each number
    exp_vecs = {}
    for n in all_numbers:
        vec = [0] * num_primes
        temp = n
        for pi, p in enumerate(prime_list):
            if p * p > temp:
                break
            while temp % p == 0:
                vec[pi] += 1
                temp //= p
        if temp > 1:
            vec[prime_to_idx[temp]] += 1
        exp_vecs[n] = tuple(vec)

    def check_sequence_valid(seq):
        """Check if all consecutive block products are distinct using prefix-sum exponent vectors."""
        if not seq:
            return True
        n = len(seq)
        # Build prefix-sum exponent vectors
        # prefix[i][j] = sum of exp_vecs[seq[k]][j] for k in 0..i-1
        # So block from i to j-1 has product vector prefix[j][k] - prefix[i][k]
        prefix = [[0] * num_primes]
        for i in range(n):
            v = exp_vecs[seq[i]]
            prev = prefix[-1]
            prefix.append([prev[k] + v[k] for k in range(num_primes)])
        
        seen = set()
        for i in range(n):
            for j in range(i+1, n+1):
                # Block from i to j-1
                vec = tuple(prefix[j][k] - prefix[i][k] for k in range(num_primes))
                if vec in seen:
                    return False
                seen.add(vec)
        return True

    def try_insert(seq, pos, val):
        """Try inserting val at position pos in seq. Returns new_seq if valid, else None."""
        if pos < 0 or pos > len(seq):
            return None
        if val in seq:
            return None
        new_seq = seq[:pos] + [val] + seq[pos:]
        if check_sequence_valid(new_seq):
            return new_seq
        return None

    def try_remove_and_reinsert(seq, old_pos, new_pos):
        """Remove element at old_pos and reinsert at new_pos. Returns new_seq if valid, else None."""
        if old_pos == new_pos:
            return seq
        val = seq[old_pos]
        without = seq[:old_pos] + seq[old_pos+1:]
        # Adjust new_pos if needed
        if new_pos > old_pos:
            new_pos -= 1
        if new_pos < 0 or new_pos > len(without):
            return None
        new_seq = without[:new_pos] + [val] + without[new_pos:]
        if check_sequence_valid(new_seq):
            return new_seq
        return None

    def local_search(seq, max_iter=2000):
        """Improve sequence via local search: insert, delete, move operations."""
        current = seq[:]
        current_len = len(current)
        no_improve = 0
        iteration = 0
        
        while no_improve < 50 and time.time() < deadline:
            iteration += 1
            improved = False
            
            # Try inserting each unused number at each position
            unused = [x for x in all_numbers if x not in current]
            random.shuffle(unused)
            for val in unused:
                if time.time() > deadline:
                    break
                for pos in range(len(current) + 1):
                    new_seq = try_insert(current, pos, val)
                    if new_seq is not None:
                        current = new_seq
                        current_len += 1
                        improved = True
                        no_improve = 0
                        break
                if improved:
                    break
            
            # Try moving elements
            if not improved:
                n = len(current)
                for _ in range(10):
                    if time.time() > deadline:
                        break
                    i = random.randint(0, n-1)
                    j = random.randint(0, n-1)
                    new_seq = try_remove_and_reinsert(current, i, j)
                    if new_seq is not None and new_seq != current:
                        current = new_seq
                        improved = True
                        no_improve = 0
                        break
            
            if not improved:
                no_improve += 1
        
        return current

    def build_initial():
        """Build initial sequence via greedy approach."""
        seq = []
        for n in all_numbers:
            new_seq = seq + [n]
            if check_sequence_valid(new_seq):
                seq = new_seq
        return seq

    # Build initial solution
    best_seq = build_initial()
    best_len = len(best_seq)
    
    # Apply local search
    improved_seq = local_search(best_seq, max_iter=5000)
    if len(improved_seq) > best_len:
        best_seq = improved_seq
        best_len = len(best_seq)
    
    # Try multiple restarts with different random seeds
    for restart in range(10):
        if time.time() > deadline - 5:
            break
        
        # Create a perturbed initial sequence
        seq = best_seq[:]
        # Randomly remove some elements
        num_remove = max(1, len(seq) // 5)
        for _ in range(num_remove):
            if seq:
                idx = random.randint(0, len(seq)-1)
                seq.pop(idx)
        
        # Rebuild from scratch with greedy, but start from perturbed state
        # Actually, let's just do local search from the perturbed state
        improved = local_search(seq, max_iter=3000)
        if len(improved) > best_len:
            best_seq = improved
            best_len = len(best_seq)
    
    # Final polish
    final_seq = local_search(best_seq, max_iter=3000)
    if len(final_seq) > best_len:
        best_seq = final_seq
        best_len = len(best_seq)
    
    return best_seq
