"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    n = 128
    min_sum = 128
    max_sum = 256
    target_sums = range(min_sum, max_sum + 1)
    
    # Helper to calculate cost: peak multiplicity
    # Returns (max_count, is_valid)
    def get_cost(A):
        A = sorted(A)
        counts = np.zeros(max_sum + 1, dtype=np.int32)
        nA = len(A)
        for i in range(nA):
            ai = A[i]
            s_ii = ai + ai
            if min_sum <= s_ii <= max_sum:
                counts[s_ii] += 1
            for j in range(i + 1, nA):
                s_ij = ai + A[j]
                if min_sum <= s_ij <= max_sum:
                    counts[s_ij] += 1
        
        # Check coverage
        # We need every integer in [128, 256] to have count >= 1
        # But we only computed counts for sums formed by pairs in A.
        # If any target sum has count 0, it's invalid.
        if np.any(counts[target_sums] == 0):
            return float('inf'), False
        
        max_mult = np.max(counts[target_sums])
        return max_mult, True

    def get_full_cost(A):
        # For local search, we might want a more nuanced cost if we relax validity,
        # but here we strictly require validity.
        # Actually, to allow moves that temporarily break validity, we should use a penalty.
        A = sorted(A)
        counts = np.zeros(max_sum + 1, dtype=np.int32)
        nA = len(A)
        for i in range(nA):
            ai = A[i]
            s_ii = ai + ai
            if min_sum <= s_ii <= max_sum:
                counts[s_ii] += 1
            for j in range(i + 1, nA):
                s_ij = ai + A[j]
                if min_sum <= s_ij <= max_sum:
                    counts[s_ij] += 1
        
        # Penalty for missing sums
        missing = np.sum(counts[target_sums] == 0)
        max_mult = np.max(counts[target_sums]) if max_mult > 0 else 0
        # Cost = max_mult + large_penalty * missing
        return max_mult + 1000 * missing

    # Strategy: 
    # 1. Start with a random set of size ~40-50. 
    #    Why? |A| ~ sqrt(N) is often optimal for additive bases. 
    #    For N=128 range, sums up to 256. 
    #    If |A|=k, number of pairs is k(k+1)/2. 
    #    We need to cover 129 sums. 
    #    k=20 -> 210 pairs. Average multiplicity ~ 210/129 ~ 1.6.
    #    k=30 -> 465 pairs. Average ~ 3.6.
    #    We want to minimize peak. Smaller k is better for peak, but harder to cover.
    #    Let's try k around 25-35.
    
    best_A = None
    best_cost = float('inf')
    
    # Initial heuristic: Symmetric distribution around 64?
    # If A is symmetric around 64, i.e., x in A iff 128-x in A? No, A is in [0,128].
    # Let's just use random restarts with local search.
    
    def local_search(A, max_time=5.0):
        A = sorted(set(A))
        current_cost, is_valid = get_cost(A)
        if not is_valid:
            # Try to fix validity first? Or use penalized cost.
            # Let's stick to valid states if possible, or use penalized cost for search.
            # For simplicity and speed, let's use penalized cost for the search loop.
            current_pen_cost = get_full_cost(A)
        else:
            current_pen_cost = current_cost
            
        no_improve_count = 0
        last_improve_time = time.time()
        
        while time.time() < deadline - 2.0:
            if time.time() - last_improve_time > max_time:
                break
                
            # Generate a random move: remove one, add one
            if len(A) < 10:
                # Add
                candidates = [x for x in range(129) if x not in A]
                if not candidates: continue
                new_val = random.choice(candidates)
                new_A = A + [new_val]
            elif len(A) > 60:
                # Remove
                idx = random.randint(0, len(A)-1)
                new_A = A[:idx] + A[idx+1:]
            else:
                # Swap
                idx = random.randint(0, len(A)-1)
                old_val = A[idx]
                candidates = [x for x in range(129) if x not in A]
                if not candidates: continue
                new_val = random.choice(candidates)
                new_A = A[:idx] + [new_val] + A[idx+1:]
                
            new_cost, new_valid = get_cost(new_A)
            if not new_valid:
                new_pen_cost = get_full_cost(new_A)
            else:
                new_pen_cost = new_cost
            
            if new_pen_cost < current_pen_cost:
                A = sorted(new_A)
                current_pen_cost = new_pen_cost
                if new_valid:
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_A = A[:]
                last_improve_time = time.time()
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > 1000:
                    break
                    
        return A, best_cost, best_A

    # Run multiple restarts
    for k in [25, 30, 35, 40]:
        if time.time() > deadline - 10.0:
            break
            
        for restart in range(5):
            if time.time() > deadline - 5.0:
                break
                
            # Random initialization
            A = random.sample(range(129), k)
            # Ensure some coverage? Random might miss many sums.
            # Let's bias initialization: pick evenly spaced points?
            # Or just random.
            
            # Quick local search
            A_out, c, b_A = local_search(A, max_time=2.0)
            if b_A is not None:
                if c < best_cost:
                    best_cost = c
                    best_A = b_A[:]
                    
    # If no valid solution found, return a default valid one (like the seed)
    if best_A is None:
        # Fallback: The seed solution is valid but has high multiplicity.
        # Let's try to construct a valid one quickly if we failed.
        # Actually, range(64, 129) is valid.
        # Let's check if we can do better with a simple greedy.
        # But for safety, return the best found or the seed.
        # The seed score is 1.0. If we find nothing valid, we return seed.
        return list(range(64, 129))
        
    return best_A
# EVOLVE-BLOCK-END
