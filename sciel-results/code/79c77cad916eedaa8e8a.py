import time
import random
from fractions import Fraction
import numpy as np

# EVOLVE-BLOCK-START

def _sieve(n):
    if n < 2:
        return []
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(2, n + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    primes = _sieve(1000)
    N = 100000  # Cap for L
    
    # Precompute coverage masks as bitsets (Python integers) for fast OR operations
    # mask[p][r] has bit i set if i % p == r and 1 <= i <= N
    masks = {}
    for p in primes:
        masks[p] = []
        for r in range(p):
            m = 0
            # Start from the smallest positive integer congruent to r mod p
            start = r if r > 0 else p
            if start <= N:
                # Generate bits efficiently
                # We can build the integer by setting bits in a loop, but that's slow for large N.
                # However, N=10^5 is small enough for a few hundred primes if we do it smartly.
                # Actually, constructing 10^5 bit integers for 168 primes * avg 50 residues is ~8M bits total.
                # Python int operations are C-speed.
                # Let's construct the mask.
                # A faster way: 
                # The mask is a repeating pattern.
                # But for simplicity and correctness, we can use a helper to build it.
                # Since we only do this once, even a slightly slower construction is fine.
                val = 0
                for i in range(start, N + 1, p):
                    val |= (1 << i)
                masks[p].append(val)
            else:
                masks[p].append(0)
    
    # Helper to get L from a combined mask
    def get_L_from_mask(mask):
        # mask is a Python int. Bit i is 1 if i is covered.
        # We need the largest L such that bits 1..L are all 1.
        # This is equivalent to finding the position of the first 0 bit in 1..N.
        # If mask has 1s in 1..L and 0 at L+1, then (mask & ((1 << (L+1)) - 2)) == ((1 << L) - 2)
        # Alternatively, we can find the first zero bit.
        # Let's iterate from 1 upwards? No, that's slow if L is large.
        # Let's check chunks.
        # Or simply:
        # If we want to know if 1..L is covered, we check if (mask >> 1) & ((1 << L) - 1) == (1 << L) - 1
        # To find the max L, we can binary search or just scan.
        # Since N is 10^5, scanning is fast if we break early.
        # But we can do better.
        # Let's find the first 0 bit starting from bit 1.
        # We can use a trick: 
        # Let x = mask >> 1. We want the number of trailing 1s in x.
        # In Python, int doesn't have direct trailing ones count, but we can use:
        # ~x & (x + 1) gives the position of the first 0 bit in x (if x is not all 1s).
        # Wait, ~x is infinite 1s in Python.
        # Let's mask it to N bits.
        # Actually, simpler:
        # We know we only care up to N.
        # Let's just loop. 10^5 iterations is ~1ms in Python? Maybe 5-10ms.
        # If we do this 1000 times, that's 10s. Acceptable.
        # But we can optimize:
        # If mask is 0, return 0.
        # We can check if bit 1 is 0 -> return 0.
        # Let's try a binary search for the first gap?
        # No, simple loop is robust.
        
        # Optimization: Use a precomputed table for bytes?
        # Or just loop.
        L = 0
        # Check bit 1
        if not (mask & 2): # bit 1
            return 0
        
        # We can check in chunks of 64 bits?
        # Let's just use a while loop on the integer itself?
        # No, the gap could be anywhere.
        
        # Let's use the property:
        # If we shift mask right by 1, we get bits 0..N-1 corresponding to original 1..N.
        # We want the number of trailing 1s.
        # x = mask >> 1
        # We want the largest k such that (x & ((1 << k) - 1)) == (1 << k) - 1
        # This is equivalent to finding the position of the first 0 bit in x.
        # If x is all 1s up to N-1, then L = N.
        
        x = mask >> 1
        # Mask x to N bits (bits 0 to N-1)
        # We only care if the first N bits are 1.
        # Let's find the first 0 bit in x.
        # If x is 0, first 0 is at 0 -> L=0.
        # If x is not 0, we can look at ~x.
        # But ~x has infinite 1s.
        # Let's limit x to N bits.
        mask_limit = (1 << N) - 1
        x = x & mask_limit
        
        if x == 0:
            return 0
        
        # Find the first 0 bit in x.
        # This is the position of the least significant 0 bit.
        # ~x has 1s where x has 0s.
        # The least significant 1 in ~x corresponds to the first 0 in x.
        # Let y = ~x & mask_limit (to avoid infinite bits)
        # Then the position of the lowest set bit in y is the index of the first 0 in x.
        y = (~x) & mask_limit
        if y == 0:
            # All N bits are 1
            return N
        
        # Find position of lowest set bit in y
        # This is the index of the first 0 in x.
        # pos = (y & -y).bit_length() - 1
        pos = (y & -y).bit_length() - 1
        # pos is the 0-based index in x where the first 0 is.
        # Since x corresponds to original bits 1..N, index 0 in x is original bit 1.
        # So the first gap is at original index pos + 1.
        # The largest L covered is pos.
        return pos

    # Strategy: Backtracking Search
    # We select a subset of primes.
    # To maximize L, we should prioritize small primes.
    # Let's try a few prime sets.
    
    def select_primes(min_p=2, max_count=20):
        sel = []
        s = Fraction(0, 1)
        for p in primes:
            if p < min_p:
                continue
            if len(sel) >= max_count:
                break
            cost = Fraction(1, p)
            if s + cost <= Fraction(1, 1):
                sel.append(p)
                s += cost
        return sel

    # We will use a specific prime set that is known to be dense.
    # Primes 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37...
    # Sum of reciprocals of primes diverges, but we stop at budget 1.
    # Let's use the greedy set starting from 2.
    prime_set = select_primes(2, 30)
    
    # Order primes: Smallest first? Or largest first?
    # In CSP, we usually start with the most constrained variable.
    # Smallest primes have fewer residues (p choices) but cover more numbers (density 1/p).
    # Actually, assigning a residue for a small prime has a huge impact on the global coverage.
    # Let's order by increasing p.
    ordered_primes = sorted(prime_set)
    
    # State:
    # current_mask: int
    # current_L: int
    # idx: index in ordered_primes
    
    best_L = 0
    best_assignment = {}
    
    # Recursive backtracking
    # To avoid recursion depth issues (max 30 primes), we can use a stack or just recursion (depth 30 is fine).
    # But we need to handle the "search" aspect. Pure backtracking might get stuck in a local minimum
    # if we just pick the first valid residue.
    # We will try residues in a heuristic order:
    # 1. Residues that cover the current "frontier" (the first gap).
    # 2. Other residues.
    
    def backtrack(idx, current_mask, current_L):
        nonlocal best_L, best_assignment
        
        if time.time() > deadline - 2:
            return
        
        if idx == len(ordered_primes):
            if current_L > best_L:
                best_L = current_L
                # We don't strictly need to store assignment for score, but good for debugging
            return
        
        p = ordered_primes[idx]
        
        # Heuristic: Try residues that cover the first gap first.
        # The first gap is at current_L + 1.
        # We want r such that (current_L + 1) % p == r.
        # But we also need to check if assigning this r actually extends the coverage.
        
        # Let's generate candidates for r.
        # We want to try r values that are "promising".
        # A good heuristic: r = (current_L + 1) % p is the one that *can* cover the next number.
        # If we don't cover the next number, L stays the same or decreases (if we break a chain? No, adding a class only adds coverage).
        # Wait, adding a class NEVER decreases coverage.
        # So current_L can only increase or stay the same.
        # If we pick r such that (current_L + 1) % p != r, then current_L + 1 is NOT covered by this new class.
        # It might be covered by existing classes? No, current_L is the first gap in existing classes.
        # So if we don't pick r = (current_L + 1) % p, the gap at current_L + 1 remains.
        # Thus, L will NOT increase.
        # Therefore, to maximize L, we MUST pick r = (current_L + 1) % p?
        # Not necessarily. What if picking that r leads to a dead end later, but picking a different r allows a better configuration for subsequent primes?
        # Since adding classes only helps, "dead end" isn't really a thing in terms of losing coverage.
        # However, the *order* in which we fill gaps matters.
        # If we skip the immediate gap, we might fill a gap further out, but the immediate gap remains, so L doesn't grow.
        # So, for the *next* step to increase L, we MUST cover current_L + 1.
        # If we don't cover current_L + 1, L stays at current_L.
        # Then the next prime will face the same gap current_L + 1.
        # So, effectively, we are forced to cover the current frontier gap to make progress.
        # 
        # Is it possible that covering the frontier gap with prime p prevents a *better* coverage later?
        # No, because we are adding a class. It only adds bits to the mask.
        # So, if we cover current_L + 1, the new L will be >= current_L + 1.
        # If we don't, L stays current_L.
        # So, the ONLY choice that can improve L is r = (current_L + 1) % p.
        # 
        # Wait, is that true?
        # Suppose current_L = 10. Gap at 11.
        # Prime p=7. 11 % 7 = 4.
        # If we pick r=4, we cover 11. Maybe we also cover 12, 13... up to 15. New L=15.
        # If we pick r=5, we cover 12, 19... but NOT 11. Gap at 11 remains. L stays 10.
        # Next prime p=11. 11 % 11 = 0. If we pick r=0, we cover 11. New L might be 11 or more.
        # So, by skipping p=7's chance to cover 11, we deferred it to p=11.
        # Is it better to let p=7 cover 11, or let p=11 cover 11?
        # p=7 covers 11, 18, 25...
        # p=11 covers 11, 22, 33...
        # p=7 is denser. It might cover more subsequent numbers.
        # So, generally, it is better to let the *smallest* available prime cover the current gap.
        # 
        # This suggests a greedy algorithm:
        # For each prime in increasing order:
        #   Set r = (current_L + 1) % p
        #   Update mask and L.
        # 
        # Is this optimal?
        # Maybe not. What if p=7 covers 11, but blocks... wait, it doesn't block. It only adds.
        # So the greedy approach of "always cover the current gap with the current prime" is a very strong candidate.
        # 
        # Let's test this hypothesis.
        # If this greedy approach works, we don't need backtracking. We just need to try different prime sets or orders.
        # 
        # Let's implement the greedy approach first. It's O(P * N) roughly.
        
        # Actually, let's stick to the backtracking framework but heavily bias towards the greedy move.
        # We will try the greedy move first. If it leads to a good L, we keep it.
        # We can also try a few alternative moves if time permits, to escape local minima (though "local minima" are less of an issue with monotonic improvement, the *choice* of which prime covers which gap matters for future density).
        
        # For now, let's just do the greedy assignment for all primes in the set.
        # And then try a few variations (e.g., different prime sets).
        
        return # Disable recursive backtracking for now, use greedy

    # Greedy Solver
    def solve_greedy(prime_list):
        current_mask = 0
        current_L = 0
        
        for p in prime_list:
            # The gap is at current_L + 1
            target = current_L + 1
            if target > N:
                break
            r = target % p
            m = masks[p][r]
            new_mask = current_mask | m
            new_L = get_L_from_mask(new_mask)
            
            # Since we only add coverage, new_L >= current_L
            current_mask = new_mask
            current_L = new_L
            
            if current_L >= N:
                break
                
        return current_L, current_mask

    # Try multiple prime sets
    # Set 1: Greedy smallest primes
    ps1 = select_primes(2, 30)
    L1, mask1 = solve_greedy(ps1)
    
    # Set 2: Skip 2?
    ps2 = select_primes(3, 30)
    L2, mask2 = solve_greedy(ps2)
    
    # Set 3: Skip 2,3?
    ps3 = select_primes(5, 30)
    L3, mask3 = solve_greedy(ps3)
    
    best_L = max(L1, L2, L3)
    
    # If we have time, we can try to "repair" the solution.
    # The greedy solution might have gaps that could be filled by *reassigning* residues of primes already used?
    # No, we fixed the primes. But we could try different primes?
    # Or we could try a local search on the residues of the chosen primes.
    
    # Let's do a simple local search on the best greedy solution.
    # We take the prime list that gave the best L.
    if L1 >= L2 and L1 >= L3:
        active_primes = ps1
        current_mask = mask1
        current_L = L1
    elif L2 >= L3:
        active_primes = ps2
        current_mask = mask2
        current_L = L2
    else:
        active_primes = ps3
        current_mask = mask3
        current_L = L3
        
    # Store current residues
    current_residues = {}
    
    # Re-run to get residues
    def solve_greedy_with_residues(prime_list):
        current_mask = 0
        current_L = 0
        residues = {}
        for p in prime_list:
            target = current_L + 1
            if target > N:
                break
            r = target % p
            residues[p] = r
            m = masks[p][r]
            new_mask = current_mask | m
            new_L = get_L_from_mask(new_mask)
            current_mask = new_mask
            current_L = new_L
            if current_L >= N:
                break
        return current_L, current_mask, residues

    best_L, best_mask, best_residues = solve_greedy_with_residues(active_primes)
    
    # Local Search: Try changing residues of primes to see if we can extend L further.
    # Changing a residue might uncover some numbers but cover others.
    # Since we want to maximize L (contiguous from 1), we primarily care about the frontier.
    # If we change a residue, we might break the coverage at the frontier.
    # So we must be careful.
    
    # A better local search:
    # Identify the gap at L+1.
    # Find a prime p in our set such that we *didn't* assign r = (L+1)%p.
    # If we change p's residue to (L+1)%p, we cover the gap.
    # But we might lose coverage of other numbers.
    # If we lose coverage of a number <= L, we fail.
    # If we only lose coverage of numbers > L, we succeed in extending L.
    
    # So, for each prime p:
    #   r_new = (current_L + 1) % p
    #   if r_new == current_residues[p]: continue
    #   Check if changing p to r_new keeps 1..current_L covered.
    #   If yes, update.
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for p in active_primes:
            r_new = (current_L + 1) % p
            if r_new == best_residues[p]:
                continue
                
            # Calculate new mask if we change p to r_new
            # We need to remove old residue and add new.
            # old_mask_p = masks[p][best_residues[p]]
            # new_mask_p = masks[p][r_new]
            # We can't simply subtract because of overlaps.
            # We need to check if 1..current_L is covered by (best_mask - old_mask_p) | new_mask_p ?
            # No, (best_mask & ~old_mask_p) | new_mask_p is not correct if other primes also cover those bits.
            # We need to know if the bits 1..current_L are covered by the *other* primes OR the new residue.
            
            # Let's check:
            # Mask of other primes: other_mask = best_mask & ~masks[p][best_residues[p]]
            # This is not strictly correct if the old residue was the *only* cover for some bits, but if other primes cover them, it's fine.
            # Actually, other_mask = best_mask ^ (masks[p][best_residues[p]] & ~(best_mask & ~masks[p][best_residues[p]])) ... too complex.
            
            # Simpler:
            # We want to know if for all i in 1..current_L, i is covered by (other_primes) OR (new_residue_p).
            # Let's construct a mask for "covered by other primes".
            # This requires recalculating the mask without p.
            # Since we have the list of primes, we can just OR all masks except p.
            # This is O(P) per check. P is ~30. 30 * 30 = 900 ORs. Fast.
            
            other_mask = 0
            for q in active_primes:
                if q != p:
                    other_mask |= masks[q][best_residues[q]]
            
            new_mask_p = masks[p][r_new]
            candidate_mask = other_mask | new_mask_p
            
            # Check if 1..current_L is covered in candidate_mask
            # get_L_from_mask(candidate_mask) >= current_L
            check_L = get_L_from_mask(candidate_mask)
            
            if check_L >= current_L:
                # It's safe to make this change. It might extend L.
                new_total_L = get_L_from_mask(candidate_mask)
                if new_total_L > current_L:
                    # Accept
                    best_residues[p] = r_new
                    best_mask = candidate_mask
                    current_L = new_total_L
                    improved = True
                    # If L increased, the gap moved. We should restart the loop or continue?
                    # Continuing is fine, but the condition r_new = (L+1)%p might be stale for other primes.
                    # But we are in a loop over primes. It's okay.
                    break # Restart the prime loop with new L
    
    # Construct result
    result = [[p, best_residues[p]] for p in active_primes]
    return result

# EVOLVE-BLOCK-END
