"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from math import isqrt
from collections import defaultdict

# EVOLVE-BLOCK-START

def _precompute_prime_factors(max_sum):
    """Precompute prime factors for all numbers up to max_sum."""
    pf = [set() for _ in range(max_sum + 1)]
    for p in range(2, max_sum + 1):
        if not pf[p]:
            for m in range(p, max_sum + 1, p):
                pf[m].add(p)
    return pf

def _build_prime_index(primes):
    """Map each prime to its index for bitset operations."""
    prime_list = sorted(primes)
    return {p: i for i, p in enumerate(prime_list)}, prime_list

def _objective_bitset(arr, pf, prime_index, prime_list):
    """Return (bitset, count) of distinct prime factors using bitset."""
    mask = 0
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            s = arr[i] + arr[j]
            for p in pf[s]:
                if p in prime_index:
                    mask |= (1 << prime_index[p])
                else:
                    # New prime not in current index; handle dynamically
                    # For now, we'll rebuild the index if needed
                    pass
    return mask, bin(mask).count('1')

def _full_objective(arr, pf):
    """Return set of all distinct prime factors of pairwise sums."""
    primes = set()
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            s = arr[i] + arr[j]
            primes |= pf[s]
    return primes

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    n = 12
    max_val = 10000
    max_sum = 2 * max_val  # max possible sum of two elements
    rng = random.Random(42)
    
    # Precompute prime factors for all possible sums
    print(f"Precomputing prime factors up to {max_sum}...")
    pf = _precompute_prime_factors(max_sum)
    print("Precomputation done.")
    
    best_set = None
    best_primes = None
    best_count = float('inf')
    
    def evaluate_full(arr):
        """Full evaluation returning set of primes."""
        return _full_objective(arr, pf)
    
    # Generate diverse initial candidates
    candidates = []
    
    # Strategy 1: Numbers with many shared factors
    # All even numbers (sums are even, factor 2 always present)
    candidates.append(list(range(2, 2 + 2*n, 2)))
    
    # Strategy 2: Multiples of 6 (sums divisible by 2 and 3)
    candidates.append([6 * (i + 1) for i in range(n)])
    
    # Strategy 3: Multiples of 30 (sums divisible by 2, 3, 5)
    candidates.append([30 * (i + 1) for i in range(n)])
    
    # Strategy 4: Numbers ≡ 1 mod small primes
    for mod in [3, 4, 5, 6]:
        vals = [x for x in range(1, max_val + 1) if x % mod == 1]
        if len(vals) >= n:
            candidates.append(vals[:n])
    
    # Strategy 5: Powers of 2
    candidates.append([2**i for i in range(1, n + 1)])
    
    # Strategy 6: Random
    for _ in range(10):
        candidates.append(rng.sample(range(1, max_val + 1), n))
    
    # Strategy 7: Structured - arithmetic progressions with small differences
    for start in [1, 2, 3, 5, 10]:
        for diff in [1, 2, 3, 5, 10]:
            cand = [start + i * diff for i in range(n)]
            if max(cand) <= max_val:
                candidates.append(cand)
    
    # Evaluate all candidates
    for cand in candidates:
        if len(set(cand)) != n or any(x < 1 or x > max_val for x in cand):
            continue
        primes = evaluate_full(cand)
        if len(primes) < best_count:
            best_count = len(primes)
            best_primes = primes
            best_set = list(cand)
            print(f"New best: count={best_count}, primes={sorted(primes)[:10]}...")
    
    # Local search with smart moves
    def local_search(arr, max_time=None):
        nonlocal best_set, best_primes, best_count
        if max_time is None:
            max_time = deadline - 5
        
        current = list(arr)
        current_primes = evaluate_full(current)
        current_count = len(current_primes)
        no_improve = 0
        max_no_improve = 500
        
        while time.time() < max_time and no_improve < max_no_improve:
            improved = False
            no_improve += 1
            
            # Try replacing one element at a time with values that share factors
            # Focus on primes that are "expensive" (appear in few sums)
            prime_counts = defaultdict(int)
            for p in current_primes:
                count = 0
                for i in range(n):
                    for j in range(i + 1, n):
                        if current[i] + current[j] in range(2, max_sum + 1):
                            if p in pf[current[i] + current[j]]:
                                count += 1
                prime_counts[p] = count
            
            # Sort primes by count (ascending) - rare primes are more "expensive"
            sorted_primes = sorted(current_primes, key=lambda p: prime_counts.get(p, 0))
            
            # Try to eliminate the rarest primes
            for target_prime in sorted_primes[:5]:
                if target_prime not in current_primes:
                    continue
                
                # Find sums that contain this prime
                affected_pairs = []
                for i in range(n):
                    for j in range(i + 1, n):
                        s = current[i] + current[j]
                        if target_prime in pf[s]:
                            affected_pairs.append((i, j, s))
                
                if not affected_pairs:
                    continue
                
                # Try changing one element from an affected pair
                for (i, j, s) in affected_pairs[:3]:
                    # Try new values for position i
                    for _ in range(5):
                        new_val = rng.randint(1, max_val)
                        if new_val in current and new_val != current[i]:
                            continue
                        
                        new_current = list(current)
                        new_current[i] = new_val
                        
                        if len(set(new_current)) != n:
                            continue
                        
                        new_primes = evaluate_full(new_current)
                        
                        if len(new_primes) < current_count:
                            current = new_current
                            current_primes = new_primes
                            current_count = len(new_primes)
                            improved = True
                            no_improve = 0
                            if current_count < best_count:
                                best_count = current_count
                                best_primes = new_primes
                                best_set = list(current)
                                print(f"  Local improvement: count={current_count}")
                            break
                        elif len(new_primes) == current_count and rng.random() < 0.05:
                            # Accept equal with small probability to escape local minima
                            current = new_current
                            current_primes = new_primes
                            no_improve = max(0, no_improve - 10)
                            break
                    if improved:
                        break
                if improved:
                    break
            
            # If no improvement, try random restart of one element
            if not improved and no_improve % 50 == 0:
                idx = rng.randint(0, n - 1)
                for _ in range(20):
                    new_val = rng.randint(1, max_val)
                    if new_val in current and new_val != current[idx]:
                        continue
                    new_current = list(current)
                    new_current[idx] = new_val
                    if len(set(new_current)) != n:
                        continue
                    new_primes = evaluate_full(new_current)
                    if len(new_primes) <= current_count:
                        current = new_current
                        current_primes = new_primes
                        current_count = len(new_primes)
                        if current_count < best_count:
                            best_count = current_count
                            best_primes = new_primes
                            best_set = list(current)
                        break
        
        return current, current_primes, current_count
    
    # Run local search from best candidate
    print(f"Starting local search from count={best_count}...")
    current, current_primes, current_count = local_search(best_set)
    
    # Multiple restarts with local search
    num_restarts = 30
    for restart in range(num_restarts):
        if time.time() >= deadline - 10:
            break
        
        # Generate new candidate
        strategy = rng.randint(0, 4)
        if strategy == 0:
            # Random
            cand = rng.sample(range(1, max_val + 1), n)
        elif strategy == 1:
            # Multiples of random small number
            m = rng.choice([2, 3, 4, 5, 6, 10, 12, 20, 30])
            vals = [m * (i + 1) for i in range(n)]
            if max(vals) <= max_val:
                cand = vals
            else:
                cand = rng.sample(range(1, max_val + 1), n)
        elif strategy == 2:
            # Arithmetic progression
            start = rng.randint(1, 100)
            diff = rng.choice([1, 2, 3, 5, 10, 15, 20])
            cand = [start + i * diff for i in range(n)]
            if max(cand) > max_val or len(set(cand)) != n:
                cand = rng.sample(range(1, max_val + 1), n)
        elif strategy == 3:
            # Powers of 2 with variations
            base = rng.choice([2, 3, 4, 5])
            cand = [base ** i for i in range(1, n + 1)]
            if max(cand) > max_val or len(set(cand)) != n:
                cand = rng.sample(range(1, max_val + 1), n)
        else:
            # Near-consecutive with small perturbations
            base = rng.randint(1, 50)
            cand = []
            for i in range(n):
                v = base + i * rng.choice([1, 2, 3])
                v = min(v, max_val)
                if v in cand:
                    v = v + rng.randint(1, 5)
                    v = min(v, max_val)
                cand.append(v)
            if len(set(cand)) != n:
                cand = rng.sample(range(1, max_val + 1), n)
        
        # Evaluate and potentially run local search
        primes = evaluate_full(cand)
        count = len(primes)
        
        if count < best_count:
            best_count = count
            best_primes = primes
            best_set = list(cand)
            print(f"Restart {restart}: new best count={count}")
        
        # Run short local search if promising
        if count <= best_count + 2:
            cand2, p2, c2 = local_search(cand, max_time=deadline - 15)
            if c2 < best_count:
                best_count = c2
                best_primes = p2
                best_set = cand2
                print(f"  After local search: count={c2}")
    
    return best_set
# EVOLVE-BLOCK-END
