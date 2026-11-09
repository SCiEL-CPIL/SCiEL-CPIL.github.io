import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    n = 20
    max_val = 10000
    
    # 1. Generate the search space: 7-smooth numbers up to 10000
    # A 7-smooth number is of the form 2^a * 3^b * 5^c * 7^d
    smooth_numbers = set()
    for a in range(14):  # 2^13 = 8192, 2^14 > 10000
        for b in range(9):  # 3^8 = 6561, 3^9 > 10000
            for c in range(6):  # 5^5 = 3125, 5^6 > 10000
                for d in range(4):  # 7^3 = 343, 7^4 > 10000
                    val = (2 ** a) * (3 ** b) * (5 ** c) * (7 ** d)
                    if 1 <= val <= max_val:
                        smooth_numbers.add(val)
    
    smooth_list = sorted(smooth_numbers)
    N_smooth = len(smooth_list)
    
    # Map value to index for quick lookup? Not strictly needed if we work with indices.
    # But we need to return the actual integers.
    
    def evaluate(A):
        """
        Vectorized evaluation of max(|A+A|, |A*A|) for a list of integers A.
        """
        A = np.array(A, dtype=np.int64)
        n = len(A)
        
        # Generate all pairs (i, j) with i <= j
        # i_indices, j_indices = np.triu_indices(n)
        # sums = A[i_indices] + A[j_indices]
        # prods = A[i_indices] * A[j_indices]
        
        # More efficient: use broadcasting for small n?
        # n=20, so 20x20 matrix is tiny.
        sums = A[:, None] + A[None, :]
        prods = A[:, None] * A[None, :]
        
        # We only need i <= j, but since sum/prod are symmetric and commutative,
        # the set of values is the same as the full matrix.
        # Unique count:
        sumset_size = len(np.unique(sums))
        prodset_size = len(np.unique(prods))
        
        return max(sumset_size, prodset_size)

    # 2. Initial Candidate Generation
    # We want to start with a set that has good structure.
    # Random sampling from smooth numbers is a good baseline.
    # Also try a "geometric" subset of smooth numbers.
    
    best_score = float('inf')
    best_set = None
    
    # Try several random initial sets
    for _ in range(50):
        if time.time() > deadline: break
        # Sample 20 distinct smooth numbers
        # Bias towards smaller numbers? Smaller numbers keep products smaller.
        # But we need diversity for sums?
        # Let's just sample uniformly from the first 200 smooth numbers (smaller ones)
        # and some from the rest.
        
        # Stratified sampling:
        # 10 from smallest 100, 10 from next 100
        small_part = random.sample(smooth_list[:100], 10)
        large_part = random.sample(smooth_list[100:300], 10)
        candidate = small_part + large_part
        
        score = evaluate(candidate)
        if score < best_score:
            best_score = score
            best_set = candidate[:]
            
    if best_set is None:
        # Fallback
        best_set = smooth_list[:20]
        best_score = evaluate(best_set)
        
    # 3. Local Search / Polish
    # Use vectorized evaluation to try swapping elements.
    # Since we are restricted to smooth numbers, we can try replacing an element
    # with another smooth number.
    
    current_set = best_set[:]
    current_score = best_score
    
    # To speed up, we can precompute the contribution of each element?
    # No, vectorized eval is fast enough.
    # Let's estimate: evaluate takes ~1ms? 20x20 unique is fast.
    # If 1ms per eval, we can do 50,000 evals in 50s.
    
    # Strategy:
    # Iterate through each position.
    # Try replacing it with a random smooth number.
    # If it improves, accept.
    # Repeat until no improvement or time limit.
    
    no_improvement_count = 0
    max_no_improvement = 5000
    
    while time.time() < deadline and no_improvement_count < max_no_improvement:
        improved = False
        
        # Randomize the order of indices to avoid bias
        indices = list(range(n))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline: break
            if no_improvement_count >= max_no_improvement: break
            
            old_val = current_set[idx]
            
            # Generate a few candidate replacements
            # 1. Random smooth number
            # 2. Neighbors in the smooth list (by value)
            
            candidates = []
            
            # Random
            candidates.append(random.choice(smooth_list))
            
            # Neighbors: find index of old_val in smooth_list
            # This is O(log N) if we use bisect, but let's just pick a few randoms
            # and maybe some local moves.
            
            # To be more structured, let's try swapping with another element?
            # No, replacement is better for set optimization.
            
            # Let's try 5 random replacements per step
            for _ in range(5):
                new_val = random.choice(smooth_list)
                if new_val == old_val:
                    continue
                if new_val in current_set:
                    continue
                
                # Test replacement
                current_set[idx] = new_val
                new_score = evaluate(current_set)
                
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    no_improvement_count = 0
                    if current_score < best_score:
                        best_score = current_score
                        best_set = current_set[:]
                    break # Accept first improvement, restart loop? Or continue?
                    # Restarting the index loop is a standard Hill Climbing move.
                else:
                    current_set[idx] = old_val
            
            if improved:
                # Restart the index loop to exploit the new state
                break
                
        if not improved:
            no_improvement_count += 1
            # If stuck, try a random restart or larger jump?
            # For now, just increment.
            # To escape local minima, we could occasionally accept a worse move (SA style)
            # but with vectorized eval, we can afford a simple SA.
            
            # Let's add a simple SA escape mechanism if stuck for a while
            if no_improvement_count % 1000 == 0:
                # Perturb current set significantly
                for _ in range(5):
                    idx = random.randint(0, n-1)
                    old_val = current_set[idx]
                    new_val = random.choice(smooth_list)
                    if new_val not in current_set and new_val != old_val:
                        current_set[idx] = new_val
                        new_score = evaluate(current_set)
                        if new_score < current_score:
                            current_score = new_score
                            if current_score < best_score:
                                best_score = current_score
                                best_set = current_set[:]
                        else:
                            current_set[idx] = old_val

    return sorted(best_set)
