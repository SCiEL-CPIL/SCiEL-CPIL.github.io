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
    N = 100000
    
    # Precompute bitmasks for each (p, r)
    # mask[p][r] is an integer where bit i is 1 if i % p == r and 1 <= i <= N
    # Note: Python integers are arbitrary precision, so this is efficient for bitwise ops.
    # We only care about bits 1..N. Bit 0 is ignored.
    
    masks = {}
    for p in primes:
        masks[p] = [0] * p
        for r in range(p):
            start = r if r > 0 else p
            if start <= N:
                # Create a numpy array of indices, then set bits
                # Setting bits in a Python int from a list of indices is fast enough for ~10^5/2 ~ 50k bits
                # But doing it for every (p,r) at startup might be slow.
                # Let's optimize: 
                # mask = sum(1 << i for i in range(start, N+1, p))
                # This is O(N/p) operations. Sum over p is N * sum(1/p) ~ N * log log N.
                # 10^5 * 3 ~ 300,000 operations. Very fast.
                
                # Alternative: use a loop
                m = 0
                for i in range(start, N + 1, p):
                    m |= (1 << i)
                masks[p][r] = m
            else:
                masks[p][r] = 0
                
    # Helper to get L from a coverage mask
    # coverage_mask has bits 1..L set if covered.
    # We want the largest L such that bits 1..L are all 1.
    # This is equivalent to finding the first 0 bit in 1..N.
    # If coverage_mask has 0 at bit k (k>=1), then L = k-1.
    # If all bits 1..N are 1, L = N.
    
    def get_L_from_mask(mask):
        # Check bits 1..N
        # We can iterate through set bits? No, we need the first gap.
        # A fast way:
        # Let M = mask & ((1 << (N+1)) - 2)  # Mask out bit 0 and bits > N
        # We want to find the lowest index k >= 1 such that bit k is 0.
        # This is the position of the lowest set bit in (~M) & ((1 << (N+1)) - 2).
        # Let Z = (~mask) & ((1 << (N+1)) - 2)
        # If Z == 0, then all bits 1..N are 1, so L = N.
        # Else, L = (Z & -Z).bit_length() - 1 - 1?
        # Wait.
        # If bit k is the lowest 0 in mask (within 1..N), then Z has bit k set, and no bits < k set.
        # So Z & -Z is 1 << k.
        # The index of the lowest set bit in Z is k.
        # L is the number of consecutive 1s starting from 1.
        # So if bit k is 0, then 1..k-1 are 1. So L = k - 1.
        
        # Let's test:
        # mask has bits 1,2,3 set. Bit 4 is 0.
        # Z has bit 4 set.
        # Z & -Z is 16 (1 << 4).
        # bit_length of 16 is 5.
        # Index is 4.
        # L = 4 - 1 = 3. Correct.
        
        # If mask has bits 1..N set.
        # Z = 0.
        # L = N.
        
        full_mask = (1 << (N + 1)) - 2  # Bits 1..N set
        Z = (~mask) & full_mask
        if Z == 0:
            return N
        else:
            # Find index of lowest set bit in Z
            # (Z & -Z).bit_length() - 1 gives the index k
            k = (Z & -Z).bit_length() - 1
            return k - 1

    # Strategy: Backtracking Search
    # We select a set of primes.
    # We assign residues one by one.
    # At each step, we choose the (p, r) that maximizes the immediate gain in L (or minimizes uncovered in prefix).
    # Since L is global, we approximate by checking the first few hundred uncovered numbers.
    
    def select_primes_greedy(min_p=2):
        sel = []
        s = Fraction(0, 1)
        for p in primes:
            if p < min_p:
                continue
            cost = Fraction(1, p)
            if s + cost <= Fraction(1, 1):
                sel.append(p)
                s += cost
        return sel, s

    # We will try a few prime sets.
    # Set 1: All small primes (2,3,5,7,11,13...) until budget full.
    # Set 2: Random subset of primes >= 5.
    
    prime_sets = [
        select_primes_greedy(2)[0],
        select_primes_greedy(5)[0],
        select_primes_greedy(13)[0],
    ]
    
    # Add some random sets
    for _ in range(3):
        # Randomly pick primes >= 5
        random.shuffle(primes)
        sel = []
        s = Fraction(0, 1)
        for p in primes:
            if p < 5:
                continue
            cost = Fraction(1, p)
            if s + cost <= Fraction(1, 1):
                if random.random() < 0.7: # Probabilistic selection to vary density
                    sel.append(p)
                    s += cost
        if len(sel) > 5:
            prime_sets.append(sorted(sel))
            
    unique_sets = []
    seen = set()
    for ps in prime_sets:
        key = tuple(ps)
        if key not in seen:
            seen.add(key)
            unique_sets.append(ps)

    best_L = 0
    best_pairs = []
    
    # Precompute "uncovered prefix" mask for quick check
    # We want to know the first K uncovered numbers.
    # If we have current coverage mask M, uncovered is ~M.
    # First K uncovered are the lowest K set bits in (~M) & full_mask.
    
    def get_first_K_uncovered(mask, K=200):
        # Returns list of first K uncovered indices
        full_mask = (1 << (N + 1)) - 2
        Z = (~mask) & full_mask
        uncovered = []
        for _ in range(K):
            if Z == 0:
                break
            # Get lowest set bit
            lsb = Z & -Z
            idx = lsb.bit_length() - 1
            uncovered.append(idx)
            Z ^= lsb # Clear the bit
        return uncovered

    for ps in unique_sets:
        if not ps:
            continue
        
        # Sort primes by size? Smaller primes cover more, so they are more critical.
        # Let's try sorting ascending.
        ps_sorted = sorted(ps)
        
        # Backtracking state
        # We will assign residues to primes in ps_sorted.
        # current_mask = 0
        # assigned = []
        
        def backtrack(idx, current_mask, assigned):
            nonlocal best_L, best_pairs
            
            if time.time() > deadline - 2:
                return
            
            if idx == len(ps_sorted):
                L = get_L_from_mask(current_mask)
                if L > best_L:
                    best_L = L
                    best_pairs = assigned.copy()
                return
            
            p = ps_sorted[idx]
            
            # Heuristic: Try residues that cover the most currently uncovered numbers in the prefix.
            # Get the first 200 uncovered numbers.
            uncov = get_first_K_uncovered(current_mask, 200)
            
            # For each residue r, count how many of uncov are covered by (p, r).
            # Since uncov is small (<=200), this is fast.
            scores = []
            for r in range(p):
                # Check which uncov numbers satisfy i % p == r
                count = 0
                for u in uncov:
                    if u % p == r:
                        count += 1
                # Tie-breaker: random or specific?
                # We want to explore different branches.
                scores.append((count, r))
            
            # Sort by count descending, then by r for determinism in first pass, 
            # but we want to explore. Let's shuffle ties.
            # Actually, to be fast, we can just pick the best one and backtrack if it fails?
            # But "fails" is not well-defined in this optimization context.
            # We are maximizing L. A "good" move increases L or keeps it high.
            
            # Let's sort by score descending.
            scores.sort(key=lambda x: -x[0])
            
            # To add diversity, we can randomize the order of equal scores.
            # Or just iterate through top few.
            
            # For each candidate r in sorted order:
            for count, r in scores:
                if time.time() > deadline - 2:
                    return
                
                new_mask = current_mask | masks[p][r]
                new_L = get_L_from_mask(new_mask)
                
                # Pruning: If this move doesn't improve L significantly, or if we've already
                # found a very high L, we might prune.
                # However, a move that doesn't improve L *now* might allow a better L later?
                # Unlikely if we are filling gaps.
                
                # If new_L < best_L, it's still possible that subsequent moves fix it.
                # But if new_L is very low, it's a bad branch.
                
                # Let's always recurse for the top few candidates.
                # To limit branching factor, only recurse if count > 0 or idx is small.
                if count == 0 and idx > 2:
                    continue # If it doesn't cover any current gap, and we are deep, skip.
                
                assigned.append((p, r))
                backtrack(idx + 1, new_mask, assigned)
                assigned.pop()
                
                # If we found a perfect cover or near perfect, we can stop exploring this branch?
                if best_L >= N:
                    return
                # If we found a very good L, maybe stop other branches?
                # Let's allow a few more to see if we can improve.
                if count == scores[0][0]: # Only explore the best scoring residues?
                    break # Heuristic: Only follow the single best path at each step?
                    # This makes it greedy. To allow backtracking, we should explore top 2 or 3.
                    # Let's try top 3.
            
            # To allow backtracking, we should not break after the first one if it's not perfect.
            # But iterating all p residues is too slow.
            # Let's modify the loop to only try top 3 residues.
            # The code above breaks after the first best one.
            # Let's re-implement the loop to try top 3.
            
        # The above backtrack function has a flaw: it breaks after the first best residue.
        # Let's rewrite the inner loop properly.
        
        def backtrack_v2(idx, current_mask, assigned):
            nonlocal best_L, best_pairs
            
            if time.time() > deadline - 2:
                return
            
            if idx == len(ps_sorted):
                L = get_L_from_mask(current_mask)
                if L > best_L:
                    best_L = L
                    best_pairs = assigned.copy()
                return
            
            p = ps_sorted[idx]
            
            uncov = get_first_K_uncovered(current_mask, 200)
            
            scores = []
            for r in range(p):
                count = 0
                for u in uncov:
                    if u % p == r:
                        count += 1
                scores.append((count, r))
            
            scores.sort(key=lambda x: -x[0])
            
            # Try top 3 residues
            top_k = min(3, len(scores))
            for i in range(top_k):
                if time.time() > deadline - 2:
                    return
                
                count, r = scores[i]
                # If the best residue doesn't cover any gap, and we are deep, stop.
                if count == 0 and idx > 2:
                    break
                
                new_mask = current_mask | masks[p][r]
                assigned.append((p, r))
                backtrack_v2(idx + 1, new_mask, assigned)
                assigned.pop()
                
                # If we found a very good solution, we can stop exploring other branches
                # for this node if the potential is exhausted.
                if best_L >= 9000: # Threshold for early termination of branch
                    # Check if current mask already has a gap that cannot be fixed?
                    # Hard to know.
                    pass

        # Run backtrack for this prime set
        backtrack_v2(0, 0, [])
        
        # If time is running out, stop trying new prime sets
        if time.time() > deadline - 10:
            break

    # If best_L is still 0 or very low, try a simple heuristic as fallback
    if best_L < 100:
        # Just pick random residues for a set of primes
        ps = select_primes_greedy(5)[0]
        pairs = [(p, random.randint(0, p-1)) for p in ps]
        # Calculate L
        mask = 0
        for p, r in pairs:
            mask |= masks[p][r]
        L = get_L_from_mask(mask)
        if L > best_L:
            best_L = L
            best_pairs = pairs

    return best_pairs
# EVOLVE-BLOCK-END
