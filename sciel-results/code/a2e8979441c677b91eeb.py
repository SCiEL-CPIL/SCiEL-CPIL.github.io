"""Program for integer sum-product extremal sets with fast evaluation and factor-aware moves."""
import time
import random
import math
import numpy as np
from collections import Counter

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 55.0  # Safety margin
    n = 20
    max_val = 10000
    
    # Precompute all pair indices (i <= j)
    pairs = [(i, j) for i in range(n) for j in range(i, n)]
    
    def evaluate(A):
        """Compute max(|A+A|, |A*A|) for set A using NumPy for speed."""
        a = np.array(A, dtype=np.int64)
        # Use upper triangle indices (i <= j)
        ii, jj = np.triu_indices(n)
        sums = a[ii] + a[jj]
        prods = a[ii] * a[jj]
        n_sums = len(np.unique(sums))
        n_prods = len(np.unique(prods))
        return max(n_sums, n_prods)
    
    def evaluate_delta(A, idx, old_val, new_val):
        """Compute the change in max(|A+A|, |A*A|) when replacing A[idx] with new_val.
        Returns (new_score, changed) where changed is True if the score actually changed."""
        # Elements affected: all pairs involving idx
        # We need to remove contributions of old_val and add contributions of new_val
        # For efficiency, track which sums/prods are unique to old_val or new_val
        
        # Build the current sum/prod multisets (counts)
        sum_counts = Counter()
        prod_counts = Counter()
        for i, j in pairs:
            sum_counts[A[i] + A[j]] += 1
            prod_counts[A[i] * A[j]] += 1
        
        # Remove contributions involving idx with old_val
        for i, j in pairs:
            if i == idx or j == idx:
                s = A[i] + A[j]
                p = A[i] * A[j]
                sum_counts[s] -= 1
                prod_counts[p] -= 1
                if sum_counts[s] == 0:
                    del sum_counts[s]
                if prod_counts[p] == 0:
                    del prod_counts[p]
        
        # Add contributions involving idx with new_val
        for i, j in pairs:
            if i == idx or j == idx:
                s = (new_val if i == idx else A[i]) + (new_val if j == idx else A[j])
                p = (new_val if i == idx else A[i]) * (new_val if j == idx else A[j])
                sum_counts[s] += 1
                prod_counts[p] += 1
        
        new_score = max(len(sum_counts), len(prod_counts))
        return new_score

    best_set = None
    best_score = float('inf')
    
    # Generate high-quality initial candidates
    candidates = []
    
    # 1. Geometric Progressions
    for base in [1, 2, 3]:
        for ratio in [2, 3]:
            cand = []
            val = base
            for i in range(n):
                if val > max_val:
                    break
                cand.append(val)
                val *= ratio
            if len(cand) == n:
                candidates.append(cand)
                
    # 2. Arithmetic Progressions
    for start in [1, 2, 3, 5]:
        for step in [1, 2, 3, 4, 5, 10]:
            cand = [start + i * step for i in range(n)]
            if max(cand) <= max_val:
                candidates.append(cand)
                
    # 3. Perturbed APs (Hybrid)
    for start in [1, 2]:
        for step in [2, 3]:
            base_cand = [start + i * step for i in range(n)]
            for _ in range(5):
                cand = list(base_cand)
                for i in range(n):
                    if random.random() < 0.3:
                        offset = random.randint(-2, 2)
                        new_val = cand[i] + offset
                        if 1 <= new_val <= max_val and new_val not in cand:
                            cand[i] = new_val
                if len(set(cand)) == n:
                    candidates.append(sorted(cand))

    # 4. Random small integers
    for _ in range(50):
        cand = sorted(random.sample(range(1, 150), n))
        candidates.append(cand)
        
    # 5. Structured candidates: mix of small numbers and geometric
    for base in [1, 2, 3, 4, 5]:
        for _ in range(10):
            cand = []
            val = base
            for i in range(n):
                if val > max_val:
                    break
                cand.append(val)
                val = int(val * random.uniform(1.5, 2.5))
            if len(cand) == n:
                cand = sorted(set(cand))
                if len(cand) == n:
                    candidates.append(cand)
    
    # 6. Try sets with many collisions in sums
    for center in [10, 20, 50, 100]:
        for spread in [1, 2, 5, 10]:
            cand = []
            for i in range(n):
                offset = i - n // 2
                val = center + offset * spread
                if 1 <= val <= max_val:
                    cand.append(val)
            if len(set(cand)) == n:
                candidates.append(sorted(cand))
        
    # 7. Factor-rich sets: numbers with many divisors to create product collisions
    # Use multiples of small numbers
    for base in [2, 3, 4, 6]:
        cand = sorted([base * i for i in range(1, n + 1)])
        if max(cand) <= max_val:
            candidates.append(cand)
    
    # 8. Sets based on prime factorizations with shared factors
    # e.g., {2^a * 3^b} for various (a,b)
    cand = []
    for a in range(0, 5):
        for b in range(0, 5):
            val = (2 ** a) * (3 ** b)
            if 1 <= val <= max_val and val not in cand:
                cand.append(val)
    if len(cand) >= n:
        candidates.append(sorted(cand[:n]))
    
    # 9. Sets mixing small primes and their products
    cand = [2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 36, 40]
    if max(cand) <= max_val and len(set(cand)) == n:
        candidates.append(cand)
    
    # 10. Very dense small sets (maximize sum collisions)
    for upper in [30, 40, 50, 60, 80, 100]:
        for _ in range(20):
            cand = sorted(random.sample(range(1, upper + 1), n))
            candidates.append(cand)
    
    # 11. Sets with many even numbers (product collisions from factor 2)
    for _ in range(20):
        cand = sorted(random.sample(range(2, 200, 2), n))
        candidates.append(cand)
    
    # 12. Mixed: half small dense, half geometric
    for _ in range(20):
        small = sorted(random.sample(range(1, 20), 10))
        geo = []
        val = 20
        for i in range(10):
            geo.append(val)
            val = int(val * random.uniform(1.5, 2.5))
        cand = sorted(small + geo)
        if len(set(cand)) == n and max(cand) <= max_val:
            candidates.append(cand)
    
    # Evaluate candidates
    for cand in candidates:
        if len(set(cand)) != n or max(cand) > max_val:
            continue
        score = evaluate(cand)
        if score < best_score:
            best_score = score
            best_set = list(cand)
            
    if best_set is None:
        best_set = list(range(1, 21))
        best_score = evaluate(best_set)
        
    # Multi-start simulated annealing with factor-aware moves
    num_starts = 10
    current_set = list(best_set)
    current_score = best_score
    
    # Precompute useful factor-based replacement candidates
    # Numbers that share factors with many elements in the set
    def get_factor_candidates(A, idx, old_val):
        """Generate candidate replacements that share factors with other elements."""
        candidates = set()
        # Get factors of other elements
        other_factors = set()
        for i in range(n):
            if i == idx:
                continue
            # Get prime factors (simplified: just divisors up to sqrt)
            v = A[i]
            for d in range(2, int(math.isqrt(v)) + 1):
                if v % d == 0:
                    other_factors.add(d)
                    other_factors.add(v // d)
            other_factors.add(v)
        
        # Generate candidates that are multiples of shared factors
        for f in other_factors:
            if f <= 2:
                continue
            for mult in range(1, min(20, max_val // f + 1)):
                cand = f * mult
                if 1 <= cand <= max_val and cand != old_val and cand not in A:
                    candidates.add(cand)
        
        # Also add nearby values
        for delta in range(-10, 11):
            if delta == 0:
                continue
            cand = old_val + delta
            if 1 <= cand <= max_val and cand != old_val and cand not in A:
                candidates.add(cand)
        
        return list(candidates) if candidates else None
    
    for start_iter in range(num_starts):
        if start_iter == 0:
            current_set = list(best_set)
            current_score = best_score
        else:
            # Random restart with bias towards structured sets
            if random.random() < 0.3:
                current_set = sorted(random.sample(range(1, 200), n))
            elif random.random() < 0.6:
                # Factor-rich random set
                base = random.choice([2, 3, 4, 6])
                current_set = sorted(random.sample([base * i for i in range(1, 100) if base * i <= max_val], n))
            else:
                # Dense small set
                current_set = sorted(random.sample(range(1, 60), n))
            current_score = evaluate(current_set)
            if current_score < best_score:
                best_score = current_score
                best_set = list(current_set)
        
        # Two-phase annealing: explore then exploit
        phase1_end = time.time() + (deadline - time.time()) * 0.6
        T = 20.0
        cooling_rate = 0.997
        min_T = 0.3
        
        while time.time() < deadline:
            if T < min_T:
                T = min_T
                
            # Determine phase for move distribution
            in_exploit_phase = time.time() > phase1_end
            
            # Generate a neighbor
            move_type = random.random()
            new_set = list(current_set)
            
            if in_exploit_phase:
                # Exploit phase: more fine-tuning moves
                if move_type < 0.5:
                    # Fine nudge: change one element by small amount
                    idx = random.randint(0, n - 1)
                    old_val = new_set[idx]
                    for _ in range(8):
                        delta = random.randint(-5, 5)
                        if delta == 0:
                            continue
                        new_val = old_val + delta
                        if 1 <= new_val <= max_val and new_val not in new_set:
                            new_set[idx] = new_val
                            break
                    if new_set == current_set:
                        continue
                        
                elif move_type < 0.8:
                    # Factor-aware replace with tighter candidates
                    idx = random.randint(0, n - 1)
                    old_val = new_set[idx]
                    cands = get_factor_candidates(current_set, idx, old_val)
                    if cands:
                        new_val = random.choice(cands)
                        new_set[idx] = new_val
                    else:
                        continue
                        
                else:
                    # Swap: exchange two elements
                    i, j = random.sample(range(n), 2)
                    new_set[i], new_set[j] = new_set[j], new_set[i]
            else:
                # Explore phase: broader moves
                if move_type < 0.3:
                    # Nudge: change one element slightly
                    idx = random.randint(0, n - 1)
                    old_val = new_set[idx]
                    for _ in range(5):
                        delta = random.randint(-15, 15)
                        if delta == 0:
                            continue
                        new_val = old_val + delta
                        if 1 <= new_val <= max_val and new_val not in new_set:
                            new_set[idx] = new_val
                            break
                    if new_set == current_set:
                        continue
                        
                elif move_type < 0.5:
                    # Factor-aware replace
                    idx = random.randint(0, n - 1)
                    old_val = new_set[idx]
                    cands = get_factor_candidates(current_set, idx, old_val)
                    if cands:
                        new_val = random.choice(cands)
                        new_set[idx] = new_val
                    else:
                        continue
                        
                elif move_type < 0.7:
                    # Swap: exchange two elements
                    i, j = random.sample(range(n), 2)
                    new_set[i], new_set[j] = new_set[j], new_set[i]
                    
                else:
                    # Random Replace: change one element to a random value
                    idx = random.randint(0, n - 1)
                    old_val = new_set[idx]
                    if random.random() < 0.6:
                        new_val = random.randint(1, 80)
                    else:
                        new_val = random.randint(1, 300)
                    if new_val == old_val or new_val in new_set:
                        for _ in range(10):
                            if random.random() < 0.6:
                                new_val = random.randint(1, 80)
                            else:
                                new_val = random.randint(1, 300)
                            if new_val != old_val and new_val not in new_set:
                                break
                    if new_val == old_val or new_val in new_set:
                        continue
                    new_set[idx] = new_val

            # Evaluate neighbor using full evaluation
            new_score = evaluate(new_set)
            
            # Acceptance criterion with adaptive temperature
            delta_score = new_score - current_score
            if delta_score < 0:
                current_set = new_set
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_set = list(current_set)
            else:
                # Use higher acceptance probability in exploit phase
                accept_prob = math.exp(-delta_score / T)
                if in_exploit_phase:
                    accept_prob = min(1.0, accept_prob * 0.5)
                if random.random() < accept_prob:
                    current_set = new_set
                    current_score = new_score
                    
            T *= cooling_rate
            
        if time.time() >= deadline - 5.0:
            break
            
    # Final polish: intensive local search on best solution found
    if time.time() < deadline - 2.0:
        polish_start = time.time()
        polish_deadline = deadline - 1.0
        polish_set = list(best_set)
        polish_score = best_score
        
        # Try all single-element replacements with nearby values
        improved = True
        while improved and time.time() < polish_deadline:
            improved = False
            for idx in range(n):
                old_val = polish_set[idx]
                # Try values in a window around old_val
                for delta in range(-20, 21):
                    if delta == 0:
                        continue
                    new_val = old_val + delta
                    if not (1 <= new_val <= max_val) or new_val in polish_set:
                        continue
                    test_set = list(polish_set)
                    test_set[idx] = new_val
                    test_score = evaluate(test_set)
                    if test_score < polish_score:
                        polish_set = test_set
                        polish_score = test_score
                        improved = True
                        if polish_score < best_score:
                            best_score = polish_score
                            best_set = list(polish_set)
                        break  # Restart outer loop after improvement
                
        # Try swaps in polish phase
        if time.time() < polish_deadline:
            for i in range(n):
                if time.time() >= polish_deadline:
                    break
                for j in range(i + 1, n):
                    if time.time() >= polish_deadline:
                        break
                    test_set = list(polish_set)
                    test_set[i], test_set[j] = test_set[j], test_set[i]
                    test_score = evaluate(test_set)
                    if test_score < polish_score:
                        polish_set = test_set
                        polish_score = test_score
                        if polish_score < best_score:
                            best_score = polish_score
                            best_set = list(polish_set)
    
    return best_set
# EVOLVE-BLOCK-END
