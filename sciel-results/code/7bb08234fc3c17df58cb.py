import time
import random
from collections import Counter
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    all_values = list(range(2, 129))
    
    # Precompute prime factorization vectors
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    num_primes = len(primes)
    vector_cache = {}
    for x in all_values:
        vec = [0] * num_primes
        temp = x
        for i, p in enumerate(primes):
            while temp % p == 0:
                vec[i] += 1
                temp //= p
        vector_cache[x] = np.array(vec, dtype=np.int32)

    def count_duplicates_fast(seq):
        """Count excess duplicates among all consecutive block products using vector sums."""
        n = len(seq)
        if n == 0:
            return 0
        seen = set()
        dups = 0
        seq_vecs = [vector_cache[x] for x in seq]
        
        for i in range(n):
            curr_vec = np.zeros(num_primes, dtype=np.int32)
            for j in range(i, n):
                curr_vec += seq_vecs[j]
                key = curr_vec.tobytes()
                if key in seen:
                    dups += 1
                else:
                    seen.add(key)
        return dups

    def energy(seq):
        """Lower is better. Penalize duplicates heavily, reward length."""
        dups = count_duplicates_fast(seq)
        return dups * 10000 - len(seq)
    
    def greedy_construct():
        """Greedy: add each value if it doesn't create duplicate products."""
        values = []
        products = set()
        for x in all_values:
            tails = [x]
            prod = x
            for a in reversed(values):
                prod *= a
                tails.append(prod)
            if not products.intersection(tails):
                values.append(x)
                products.update(tails)
        return values
    
    # Try multiple greedy variants for better initialization
    best_seq = greedy_construct()
    best_energy = energy(best_seq)
    
    # Try greedy with different starting offsets
    for skip_start in range(0, 15):
        vals = [v for v in all_values if v >= skip_start + 2]
        values = []
        products = set()
        for x in vals:
            tails = [x]
            prod = x
            for a in reversed(values):
                prod *= a
                tails.append(prod)
            if not products.intersection(tails):
                values.append(x)
                products.update(tails)
        e = energy(values)
        if e < best_energy:
            best_energy = e
            best_seq = values
    
    # Simulated annealing local search
    random.seed(42)
    seq = list(best_seq)
    current_energy = best_energy
    
    T = 300.0
    T_min = 0.05
    cooling = 0.99985
    
    while time.time() < deadline:
        if T < T_min:
            # Restart from best with perturbation
            seq = list(best_seq)
            for _ in range(random.randint(1, 4)):
                if random.random() < 0.5 and len(seq) > 1:
                    idx = random.randint(0, len(seq) - 1)
                    seq.pop(idx)
                else:
                    missing = [v for v in all_values if v not in seq]
                    if missing:
                        new_val = random.choice(missing)
                        pos = 0
                        while pos < len(seq) and seq[pos] < new_val:
                            pos += 1
                        seq.insert(pos, new_val)
            current_energy = energy(seq)
            T = 150.0
            continue
        
        # Generate a random move
        r = random.random()
        if r < 0.45:
            # Insert a missing element at sorted position
            missing = [v for v in all_values if v not in seq]
            if not missing:
                continue
            new_val = random.choice(missing)
            pos = 0
            while pos < len(seq) and seq[pos] < new_val:
                pos += 1
            new_seq = seq[:pos] + [new_val] + seq[pos:]
        elif r < 0.75:
            # Remove a random element
            if not seq:
                continue
            idx = random.randint(0, len(seq) - 1)
            new_seq = seq[:idx] + seq[idx+1:]
        else:
            # Swap: remove one, insert another
            if not seq:
                continue
            idx = random.randint(0, len(seq) - 1)
            missing = [v for v in all_values if v not in seq]
            if not missing:
                continue
            new_val = random.choice(missing)
            new_seq = seq[:idx] + seq[idx+1:]
            pos = 0
            while pos < len(new_seq) and new_seq[pos] < new_val:
                pos += 1
            new_seq = new_seq[:pos] + [new_val] + new_seq[pos:]
        
        new_energy = energy(new_seq)
        delta = new_energy - current_energy
        
        # Accept if better or with Boltzmann probability
        if delta < 0 or random.random() < pow(2.718281828, -delta / T):
            seq = new_seq
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_seq = list(seq)
        
        T *= cooling
    
    return best_seq
