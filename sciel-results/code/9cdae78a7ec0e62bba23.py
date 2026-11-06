import random
from fractions import Fraction
import time

def is_prime(n):
    if n < 2: return False
    for i in range(2, int(n**0.5)+1):
        if n % i == 0: return False
    return True

def construct():
    # Generate primes up to 1000
    primes = [p for p in range(2, 1001) if is_prime(p)]
    
    def get_covered_prefix(res_map, P_list, max_L=10000):
        # Efficient coverage check
        # Use a bytearray for speed
        covered = bytearray(max_L + 1)
        for p in P_list:
            if p not in res_map: continue
            r = res_map[p]
            start = r if r > 0 else p
            for n in range(start, max_L + 1, p):
                covered[n] = 1
        
        L = 0
        for n in range(1, max_L + 1):
            if covered[n]:
                L = n
            else:
                break
        return L

    def optimize_residues(P_list, max_iters=5000, temp_start=1000.0, temp_end=1.0):
        # Simulated Annealing
        # Initial residues: random
        residues = {p: random.randint(0, p-1) for p in P_list}
        
        current_L = get_covered_prefix(residues, P_list, 10000)
        best_L = current_L
        best_residues = residues.copy()
        
        temp = temp_start
        cooling = (temp_end / temp_start) ** (1.0 / max_iters)
        
        for i in range(max_iters):
            # Pick a random prime to change
            p = random.choice(P_list)
            old_r = residues[p]
            new_r = random.randint(0, p-1)
            if new_r == old_r:
                continue
            residues[p] = new_r
            
            new_L = get_covered_prefix(residues, P_list, 10000)
            
            if new_L >= current_L:
                current_L = new_L
                if new_L > best_L:
                    best_L = new_L
                    best_residues = residues.copy()
            else:
                # Metropolis acceptance
                # delta is negative
                delta = current_L - new_L
                if random.random() < pow(2.718281828, -delta / temp):
                    current_L = new_L
                else:
                    residues[p] = old_r # revert
            
            temp *= cooling
            
        return best_L, best_residues

    # Base set: primes >= 13
    base_primes = [p for p in primes if p >= 13]
    
    best_global_L = 0
    best_global_residues = {}
    
    # 1. Optimize base set
    L, res = optimize_residues(base_primes, max_iters=2000)
    if L > best_global_L:
        best_global_L = L
        best_global_residues = res
        
    # 2. Try adding small primes by removing large ones
    # Small primes: 2, 3, 5, 7, 11
    # We will try to include subsets of these.
    
    small_primes = [2, 3, 5, 7, 11]
    
    # Pre-compute sum of reciprocals for base
    # We will generate candidate sets
    
    candidate_sets = []
    
    # Candidate 1: Base + 11 (remove largest until sum <= 1)
    # Candidate 2: Base + 7 (remove largest until sum <= 1)
    # Candidate 3: Base + 5 (remove largest until sum <= 1)
    # Candidate 4: Base + 3 (remove largest until sum <= 1)
    # Candidate 5: Base + 2 (remove largest until sum <= 1)
    # Candidate 6: Base + 11 + 7 (remove largest until sum <= 1)
    
    for small_p in small_primes:
        # Start with base
        P_test = list(base_primes)
        P_test.append(small_p)
        
        # Sort by prime value descending to remove largest first (smallest cost impact? No, largest p has smallest 1/p)
        # We want to remove primes that are "least useful" or just largest to save budget?
        # Actually, to make room for 1/small_p, we need to remove sum of reciprocals >= 1/small_p - (1 - sum_base).
        # Since sum_base < 1, we have some slack.
        
        # Calculate current sum
        sum_rec = sum(Fraction(1, p) for p in P_test)
        
        # Sort P_test by p descending
        P_test_sorted = sorted(P_test, reverse=True)
        
        # Remove largest primes until sum <= 1
        # Note: We should not remove the small_p we just added if possible?
        # If we have to remove it, the candidate is invalid (same as base).
        
        i = 0
        while sum_rec > Fraction(1, 1) and i < len(P_test_sorted):
            p_to_remove = P_test_sorted[i]
            # Don't remove the small prime we are trying to add? 
            # If we remove it, we revert to a subset of base.
            # Let's allow it, but it won't be better than base usually.
            sum_rec -= Fraction(1, p_to_remove)
            P_test.remove(p_to_remove)
            i += 1
            
        if sum_rec <= Fraction(1, 1):
            candidate_sets.append(P_test)

    # Also try combinations of small primes? Might be too slow.
    # Let's just try the individual additions.
    
    # Run SA on each candidate set
    # Limit time
    start_time = time.time()
    
    for P_cand in candidate_sets:
        if time.time() - start_time > 100: # Leave 20s buffer
            break
            
        # Skip if same as base
        if set(P_cand) == set(base_primes):
            continue
            
        L, res = optimize_residues(P_cand, max_iters=1000)
        if L > best_global_L:
            best_global_L = L
            best_global_residues = res

    # Return best
    return [[p, best_global_residues[p]] for p in best_global_residues.keys()]
