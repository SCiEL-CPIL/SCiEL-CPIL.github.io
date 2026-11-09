"""Rewritten program with a stronger algorithm for integer sum-product extremal sets."""
import time
import random
import sys
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 75.0  # Safety margin before 90s kill
    best_set = None
    best_score = float('inf')
    
    def evaluate(A):
        """Compute max(|A+A|, |A*A|) for set A using numpy for speed."""
        A = np.array(sorted(A), dtype=np.int64)
        n = len(A)
        # Compute all pairwise sums (i <= j)
        i_idx, j_idx = np.triu_indices(n)
        sums = A[i_idx] + A[j_idx]
        sumset_size = len(np.unique(sums))
        # Compute all pairwise products (i <= j)
        prods = A[i_idx] * A[j_idx]
        prodset_size = len(np.unique(prods))
        return max(sumset_size, prodset_size)
    
    def divisors(n):
        divs = set()
        i = 1
        while i * i <= n:
            if n % i == 0:
                divs.add(i)
                divs.add(n // i)
            i += 1
        return divs
    
    random.seed(42)
    
    # Try various structured initializations
    candidates = []
    
    # Sequential: {1, 2, ..., 20}
    candidates.append(list(range(1, 21)))
    
    # Arithmetic progressions with different starts and steps
    for start in [1, 2, 3, 5, 10]:
        for step in [1, 2, 3, 4, 5, 10]:
            ap = [start + i * step for i in range(20)]
            if max(ap) <= 10000:
                candidates.append(ap)
    
    # Geometric progressions
    for ratio in [2, 3]:
        gp = [ratio ** i for i in range(20)]
        if max(gp) <= 10000:
            candidates.append(gp)
    
    # Divisor sets of highly composite numbers
    for N in [720, 1260, 2520, 5040, 10080, 25200, 50400, 100800]:
        divs = sorted([d for d in divisors(N) if d <= 10000])
        if len(divs) >= 20:
            candidates.append(divs[:20])
            # Also try different subsets
            if len(divs) > 20:
                # Take every other divisor
                candidates.append(divs[::2][:20] if len(divs[::2]) >= 20 else divs[:20])
    
    # Mixed: small numbers + some structured additions
    for base in [range(1, 11), range(1, 15), range(1, 20)]:
        base_list = list(base)
        if len(base_list) < 20:
            # Add some larger numbers
            remaining = 20 - len(base_list)
            extra = [base_list[-1] * 2 + i for i in range(remaining)]
            candidates.append(sorted(base_list + extra))
    
    # Evaluate all candidates
    for cand in candidates:
        if len(cand) != 20 or len(set(cand)) != 20 or min(cand) < 1 or max(cand) > 10000:
            continue
        score = evaluate(cand)
        if score < best_score:
            best_score = score
            best_set = cand[:]
    
    if best_set is None:
        best_set = list(range(1, 21))
        best_score = evaluate(best_set)
    
    # Local search from best candidate
    current = best_set[:]
    current_score = best_score
    current_set = set(current)
    
    # Simulated annealing style local search
    temp = 10.0
    cooling = 0.999
    
    for iteration in range(200000):
        if time.time() > deadline:
            break
        if temp < 0.1:
            temp = 1.0  # Restart with higher temperature
            # Pick a random restart point
            if random.random() < 0.5:
                current = list(range(1, 21))
            else:
                current = [random.randint(1, 1000) for _ in range(20)]
                # Ensure distinctness
                while len(set(current)) < 20:
                    i = random.randint(0, 19)
                    current[i] = random.randint(1, 1000)
            current = sorted(current)
            current_score = evaluate(current)
            if current_score < best_score:
                best_score = current_score
                best_set = current[:]
        
        # Choose a random index to modify
        i = random.randint(0, 19)
        old_val = current[i]
        
        # Generate a new value (biased toward nearby values)
        if random.random() < 0.7:
            # Local perturbation
            delta = random.randint(-50, 50)
            new_val = old_val + delta
            if new_val < 1:
                new_val = 1
            elif new_val > 10000:
                new_val = 10000
        else:
            # Global perturbation
            new_val = random.randint(1, 10000)
        
        if new_val in current_set:
            continue
        
        current[i] = new_val
        current_set.discard(old_val)
        current_set.add(new_val)
        current.sort()
        
        new_score = evaluate(current)
        delta_score = new_score - current_score
        
        # Accept if better or with probability exp(-delta/temp)
        if delta_score < 0 or (delta_score == 0 and random.random() < 0.5) or \
           (delta_score > 0 and random.random() < np.exp(-delta_score / temp)):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_set = current[:]
        else:
            # Reject: revert
            current[i] = old_val
            current_set.discard(new_val)
            current_set.add(old_val)
            current.sort()
        
        temp *= cooling
    
    return best_set
# EVOLVE-BLOCK-END
