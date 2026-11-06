"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
from math import gcd
import time

def construct():
    def check_pairwise_coprime(values):
        vals = list(set(values))
        n = len(vals)
        for i in range(n):
            for j in range(i + 1, n):
                if gcd(vals[i], vals[j]) != 1:
                    return False
        return True

    def coprime_score(values):
        vals = list(set(values))
        n = len(vals)
        score = 0
        for i in range(n):
            for j in range(i + 1, n):
                if gcd(vals[i], vals[j]) == 1:
                    score += 1
        return score

    def evaluate(A, B):
        sums = [a + b for a in A for b in B]
        distinct_sums = set(sums)
        if check_pairwise_coprime(distinct_sums):
            return len(distinct_sums)
        return 0

    # Initialize with a random valid-ish start or the known baseline
    # The baseline has score 5.
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_count = evaluate(best_A, best_B)
    
    # If we want to be safe, we can try to find a 9 immediately
    # Using Simulated Annealing
    
    current_A = best_A[:]
    current_B = best_B[:]
    current_score = coprime_score([a + b for a in current_A for b in current_B])
    best_score = current_score
    
    start_time = time.time()
    timeout = 55
    
    # Simulated Annealing
    T = 1.0
    cooling_rate = 0.99
    
    while time.time() - start_time < timeout:
        if best_score == 36: # Max pairs for 9 elements
            break
            
        # Generate a neighbor
        if random.random() < 0.5:
            idx = random.randint(0, 2)
            old_val = current_A[idx]
            new_val = random.randint(1, 128)
            if new_val in current_A and new_val != old_val:
                continue
            current_A[idx] = new_val
            current_A.sort()
        else:
            idx = random.randint(0, 2)
            old_val = current_B[idx]
            new_val = random.randint(1, 128)
            if new_val in current_B and new_val != old_val:
                continue
            current_B[idx] = new_val
            current_B.sort()
            
        new_sums = [a + b for a in current_A for b in current_B]
        new_distinct = set(new_sums)
        
        # Check if valid
        if check_pairwise_coprime(new_distinct):
            new_count = len(new_distinct)
            if new_count > best_count:
                best_count = new_count
                best_A = current_A[:]
                best_B = current_B[:]
            # If we found a valid one, it's a good state, keep exploring from here
            # But we need to update best_score for the annealing logic?
            # Actually, if valid, score is effectively "infinite" or very high.
            # Let's just update best and continue.
            # To help the annealer, we can set current_score to a high value
            # but since we break on 36 pairs (which implies 9 distinct coprime), 
            # let's stick to the pair count for the gradient.
            
        new_score = coprime_score(new_sums)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta > 0:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                # Update best if valid?
                # The pair score 36 is equivalent to 9 distinct coprime sums.
                # If new_score == 36, we should have checked validity.
                # If valid, best_count is updated above.
        else:
            # Accept with probability exp(delta / T)
            # delta is negative
            p = 2.718281828 ** (delta / T)
            if random.random() < p:
                current_score = new_score
            else:
                # Revert
                if random.random() < 0.5:
                    current_A[idx] = old_val
                    current_A.sort()
                else:
                    current_B[idx] = old_val
                    current_B.sort()
                    
        T *= cooling_rate
        if T < 0.01:
            T = 1.0 # Restart temperature occasionally? Or just keep going.
            
    # Final check
    final_count = evaluate(best_A, best_B)
    if final_count > best_count:
        best_count = final_count
        
    return [best_A, best_B]
# EVOLVE-BLOCK-END
