import math
import time
import random
from itertools import combinations

def construct():
    n = 255
    start_time = time.time()
    deadline = start_time + 85  # 5-second safety margin

    # 255 = 3 * 5 * 17
    # A set S is valid if gcd(a,b) > 1 for all distinct a,b in S.
    # Since 255 in S, every element must share a factor with 255, i.e., be divisible by 3, 5, or 17.
    # Let M = {i in [1, 255] : gcd(i, 255) > 1}.
    # We want the largest subset of M containing 255 such that all pairs are non-coprime.
    
    # Candidates: numbers in [1, 255] divisible by 3, 5, or 17.
    candidates = []
    for i in range(1, n + 1):
        if i % 3 == 0 or i % 5 == 0 or i % 17 == 0:
            candidates.append(i)
    
    # Map candidates to indices
    idx_map = {val: idx for idx, val in enumerate(candidates)}
    num_cands = len(candidates)
    
    # Precompute coprime matrix (bitsets for speed)
    # coprime[i][j] = 1 if gcd(candidates[i], candidates[j]) == 1
    # We want to avoid coprime pairs.
    # Let's use Python integers as bitsets.
    # coprime_mask[i] is an integer where bit j is 1 if candidates[i] and candidates[j] are coprime.
    
    coprime_mask = [0] * num_cands
    for i in range(num_cands):
        for j in range(i + 1, num_cands):
            if math.gcd(candidates[i], candidates[j]) == 1:
                coprime_mask[i] |= (1 << j)
                coprime_mask[j] |= (1 << i)
    
    # 255 must be in the set.
    # Find index of 255.
    idx_255 = idx_map[255]
    
    # We can restrict candidates to those that are NOT coprime with 255.
    # But by definition of M, all are not coprime with 255?
    # Wait. 255 = 3*5*17.
    # If i is in M, gcd(i, 255) > 1.
    # So all candidates in M are compatible with 255.
    
    # We want to find a maximum clique in the graph G where vertices are M and edges connect non-coprime pairs.
    # Equivalently, a maximum independent set in the graph H where vertices are M and edges connect coprime pairs.
    # H is defined by coprime_mask.
    
    # Let's use a heuristic approach:
    # 1. Start with all multiples of 3 (a large clique).
    # 2. Try to add other numbers.
    # 3. Local search: swap elements to increase size.
    
    # Baseline: Multiples of 3 in [1, 255]
    base_set = set(range(3, n + 1, 3))
    base_set.add(255) # 255 is a multiple of 3, so already included.
    
    # Convert to bitset
    def set_to_bitset(s):
        mask = 0
        for x in s:
            mask |= (1 << idx_map[x])
        return mask
    
    def bitset_to_set(mask):
        s = set()
        for i in range(num_cands):
            if mask & (1 << i):
                s.add(candidates[i])
        return s
    
    best_mask = set_to_bitset(base_set)
    best_size = len(base_set)
    
    # Greedy expansion
    # Try adding candidates not in best_mask that are compatible with all in best_mask.
    # Compatible means: for all j in best_mask, bit j in coprime_mask[i] is 0.
    # i.e., coprime_mask[i] & best_mask == 0.
    
    current_mask = best_mask
    changed = True
    while changed:
        changed = False
        for i in range(num_cands):
            if current_mask & (1 << i):
                continue
            if (coprime_mask[i] & current_mask) == 0:
                current_mask |= (1 << i)
                changed = True
    
    if current_mask.bit_count() > best_size:
        best_mask = current_mask
        best_size = current_mask.bit_count()
    
    # Local Search: 1-swap (remove one, add one)
    # We want to increase size, so we can't just swap 1 for 1 unless we can add more later?
    # Actually, if we remove one and add one, size stays same.
    # We need to find a way to increase size.
    # Maybe remove one, add two? Or just try to add more?
    # The greedy expansion already tried to add any compatible element.
    # So we are at a local maximum for "add only".
    # Now try "remove one, add one" to reach a different local maximum, or "remove one, add two" (net +1).
    
    # Let's try "remove one, add two".
    # For each i in current_mask, remove it.
    # Then try to add two new elements j, k.
    # This is O(N^3) which might be slow, but N=140. 140^3 = 2.7M.
    # With bitsets, the check is fast.
    
    # Precompute list of indices
    indices = list(range(num_cands))
    
    # Randomized Local Search
    random.seed(42)
    
    for restart in range(500):
        if time.time() > deadline:
            break
        
        # Start from best_mask or a random perturbation?
        # Let's start from best_mask.
        curr_mask = best_mask
        
        # Try to improve
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try removing one element i
            # And adding two elements j, k
            # To keep it fast, we can just try adding one element j that becomes compatible after removing i?
            # If we remove i, maybe j was blocked only by i?
            # Then we can add j. Size stays same.
            # But maybe after adding j, we can add k?
            
            # Strategy:
            # 1. Find all i in curr_mask such that removing i allows adding some j.
            # 2. For each such (i, j), check if we can add another k.
            
            # Simpler:
            # Iterate through all i in curr_mask.
            # Let M' = curr_mask without i.
            # Find all j not in M' such that j is compatible with M'.
            # If there are at least 2 such j's, pick the best pair?
            # Or just pick any 2.
            
            # Optimization:
            # For each i in curr_mask:
            #   M_prime = curr_mask ^ (1 << i)
            #   candidates_to_add = [j for j in indices if not (M_prime >> j) & 1 and (coprime_mask[j] & M_prime) == 0]
            #   if len(candidates_to_add) >= 2:
            #       # Pick two
            #       # We need to ensure j1 and j2 are compatible with each other.
            #       # Since they are both compatible with M_prime, we just need gcd(candidates[j1], candidates[j2]) > 1.
            #       # i.e., bit j2 in coprime_mask[j1] is 0.
            
            # This loop is O(N * N^2) = O(N^3).
            # N=140. 140^3 = 2.7M.
            # Each step involves bit operations and list comprehension.
            # Might be too slow for 500 restarts * iterations.
            
            # Let's limit the number of swaps per restart.
            # Or just do a few passes.
            
            # Let's try a simpler 1-swap first to escape local minima, then greedy add.
            # But 1-swap doesn't increase size.
            
            # Let's stick to "remove 1, add 2".
            # To speed up, we can precompute for each i, the set of j blocked ONLY by i?
            # That's complex.
            
            # Let's just run the O(N^3) loop but with a time limit.
            
            found_swap = False
            for i in indices:
                if not (curr_mask >> i) & 1:
                    continue
                
                M_prime = curr_mask ^ (1 << i)
                
                # Find compatible j's
                # j must not be in M_prime
                # and coprime_mask[j] & M_prime == 0
                
                # We can iterate j
                # To find 2 such j's, we can break early.
                found_j = []
                for j in indices:
                    if (M_prime >> j) & 1:
                        continue
                    if (coprime_mask[j] & M_prime) == 0:
                        found_j.append(j)
                        if len(found_j) >= 2:
                            break
                
                if len(found_j) >= 2:
                    j1, j2 = found_j[0], found_j[1]
                    # Check if j1 and j2 are compatible
                    if (coprime_mask[j1] >> j2) & 1:
                        # Try next pair?
                        # For simplicity, just skip or try to find a compatible pair in found_j
                        # Since found_j might be small, let's check all pairs in found_j
                        # But found_j is limited to 2 in the loop above.
                        # Let's expand found_j a bit more if needed?
                        # Or just assume if the first two are bad, try others?
                        # Let's collect up to 10 candidates.
                        found_j = []
                        for j in indices:
                            if (M_prime >> j) & 1:
                                continue
                            if (coprime_mask[j] & M_prime) == 0:
                                found_j.append(j)
                                if len(found_j) >= 10:
                                    break
                        
                        # Find a compatible pair in found_j
                        pair_found = False
                        for a in range(len(found_j)):
                            for b in range(a + 1, len(found_j)):
                                if not (coprime_mask[found_j[a]] >> found_j[b]) & 1:
                                    j1, j2 = found_j[a], found_j[b]
                                    pair_found = True
                                    break
                            if pair_found:
                                break
                        
                        if not pair_found:
                            continue
                    
                    # Perform swap
                    new_mask = M_prime | (1 << j1) | (1 << j2)
                    curr_mask = new_mask
                    improved = True
                    found_swap = True
                    break
            
            if found_swap:
                # After a swap, try greedy add again
                changed = True
                while changed:
                    changed = False
                    for i in range(num_cands):
                        if curr_mask & (1 << i):
                            continue
                        if (coprime_mask[i] & curr_mask) == 0:
                            curr_mask |= (1 << i)
                            changed = True
                
                if curr_mask.bit_count() > best_size:
                    best_mask = curr_mask
                    best_size = curr_mask.bit_count()
                    # Reset improved to False? No, we want to continue searching from this new best.
                    # But the while loop condition is `improved`.
                    # We set improved=True. So it will loop again.
                    # This is good.
            else:
                break
        
        # End while improved
        
        # If we found a better solution, update best
        # (Already done inside the loop)
        
        # Optional: Random restart from a random subset of best_mask?
        # To avoid getting stuck in the same local optimum.
        # Let's try: remove 5 random elements from best_mask, then greedy add.
        if restart % 10 == 0 and restart > 0:
            random_mask = best_mask
            # Remove 5 random elements (not 255)
            to_remove = []
            for i in indices:
                if (random_mask >> i) & 1 and i != idx_255:
                    to_remove.append(i)
            random.shuffle(to_remove)
            for i in to_remove[:5]:
                random_mask &= ~(1 << i)
            
            # Greedy add
            changed = True
            while changed:
                changed = False
                for i in range(num_cands):
                    if random_mask & (1 << i):
                        continue
                    if (coprime_mask[i] & random_mask) == 0:
                        random_mask |= (1 << i)
                        changed = True
            
            # Local search on random_mask
            # (Simplified: just check if it's better)
            if random_mask.bit_count() > best_size:
                best_mask = random_mask
                best_size = random_mask.bit_count()
    
    # Final result
    result_set = bitset_to_set(best_mask)
    
    # Ensure 255 is in result
    if 255 not in result_set:
        result_set.add(255)
    
    # Verify validity
    # Just to be safe
    lst = list(result_set)
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            if math.gcd(lst[i], lst[j]) == 1:
                # Fallback to baseline
                return sorted(base_set)
    
    return sorted(result_set)

# Quick test
if __name__ == "__main__":
    res = construct()
    print(len(res))
    print(res)
