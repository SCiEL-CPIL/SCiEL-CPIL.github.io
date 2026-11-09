import time
from fractions import Fraction
import math

def _primes_upto(n):
    """Sieve of Eratosthenes for primes up to n."""
    if n < 2:
        return []
    sieve = bytearray(b'\x01') * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, n + 1) if sieve[i]]

def _compute_L(cov, start, cap):
    """Find the first n >= start where cov[n] == 0. Return n-1."""
    # cov is 1-indexed effectively, but python lists are 0-indexed.
    # We will use 1-indexed logic: cov[0] is unused or dummy.
    # Actually, let's just scan.
    for n in range(start, cap + 1):
        if cov[n] == 0:
            return n - 1
    return cap

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    all_primes = _primes_upto(1000)
    prime_set = set(all_primes)
    
    # State
    # classes: list of [p, r]
    classes = []
    # used_primes: set of p
    used_primes = set()
    # budget_used: Fraction
    budget_used = Fraction(0, 1)
    # current L
    L = 0
    
    # Coverage array
    # Size 100,001 (indices 0..100000)
    MAX_L = 100000
    cov = [0] * (MAX_L + 1)
    
    def add_class(p, r):
        nonlocal L, budget_used
        for n in range(r, MAX_L + 1, p):
            if n == 0:
                continue # r can be 0, but we care about 1..L
            cov[n] += 1
        classes.append([p, r])
        used_primes.add(p)
        budget_used += Fraction(1, p)
        # Update L
        L = _compute_L(cov, L + 1, MAX_L)

    def remove_class(p, r):
        nonlocal L, budget_used
        for n in range(r, MAX_L + 1, p):
            if n == 0:
                continue
            cov[n] -= 1
        # Remove from classes
        # Find index
        for i, c in enumerate(classes):
            if c[0] == p:
                classes.pop(i)
                break
        used_primes.discard(p)
        budget_used -= Fraction(1, p)
        # Recompute L from scratch? Or from 1?
        # Since we removed coverage, L might decrease.
        # We need to find the first gap.
        # It's safer to recompute from 1.
        L = _compute_L(cov, 1, MAX_L)

    def change_residue(p, old_r, new_r):
        nonlocal L
        # Decrement old
        for n in range(old_r, MAX_L + 1, p):
            if n == 0:
                continue
            cov[n] -= 1
        # Increment new
        for n in range(new_r, MAX_L + 1, p):
            if n == 0:
                continue
            cov[n] += 1
        # Update class record
        for c in classes:
            if c[0] == p:
                c[1] = new_r
                break
        # Recompute L
        L = _compute_L(cov, 1, MAX_L)

    # Greedy Phase: Add primes in increasing order
    # We want to maximize L.
    # Try adding each prime p if it extends L.
    # Since small primes are denser, we iterate p in ascending order.
    
    # However, adding a prime might not extend L if the residue doesn't align.
    # But we can choose the residue!
    # For a new prime p, the best residue to extend current L is r = (L+1) % p.
    # Let's try this for all primes in ascending order.
    
    # We might need multiple passes if adding one prime allows another to be useful?
    # Actually, if we add p1, L increases. Then p2 might be useful.
    # So a single pass in ascending order should work, but we might miss primes
    # that become useful only after other primes are added?
    # No, if p is small, it's likely useful early.
    # But what if p=2 is not useful initially (L=0, r=1 covers 1,3,5... L=1),
    # but p=3 is useful?
    # Let's just iterate through primes and add if it helps.
    # We might need to re-iterate if adding a prime changes the "best" residue for another?
    # No, residues are fixed once added.
    
    # Let's do a greedy pass.
    # To be safe, we can do multiple passes until no improvement or time runs out.
    
    improved = True
    while improved:
        improved = False
        for p in all_primes:
            if time.time() > deadline:
                break
            if p in used_primes:
                continue
            if budget_used + Fraction(1, p) > Fraction(1, 1):
                continue
            
            # Candidate residue
            r = (L + 1) % p
            
            # Check if adding this class extends L
            # Tentatively add
            old_L = L
            for n in range(r, MAX_L + 1, p):
                if n == 0:
                    continue
                cov[n] += 1
            
            new_L = _compute_L(cov, old_L + 1, MAX_L)
            
            if new_L > old_L:
                # Accept
                classes.append([p, r])
                used_primes.add(p)
                budget_used += Fraction(1, p)
                L = new_L
                improved = True
            else:
                # Revert
                for n in range(r, MAX_L + 1, p):
                    if n == 0:
                        continue
                    cov[n] -= 1
    
    # Local Search Phase
    # 1. Try to change residues of existing primes to cover the first gap.
    # 2. Try to add new primes.
    # 3. Try to swap primes.
    
    while time.time() < deadline:
        g = L + 1
        if g > MAX_L:
            break
            
        move_made = False
        
        # 1. Residue Change
        for p in list(used_primes):
            if time.time() > deadline:
                break
            r_new = g % p
            # Find current r
            r_old = None
            for c in classes:
                if c[0] == p:
                    r_old = c[1]
                    break
            if r_old is None:
                continue
            if r_new == r_old:
                continue
                
            # Check if changing r_old to r_new breaks any k < g
            # We need to check all k < g such that k % p == r_old
            # If any such k has cov[k] == 1, it becomes 0 -> Invalid.
            valid = True
            # Iterate k in [1, g-1] with k % p == r_old
            # Start from the smallest k >= 1 with k % p == r_old
            # k = r_old + m*p
            # If r_old == 0, smallest positive is p.
            # If r_old > 0, smallest is r_old.
            start_k = r_old
            if start_k == 0:
                start_k = p
            if start_k > g - 1:
                # No such k in [1, g-1]
                valid = True
            else:
                for k in range(start_k, g, p):
                    if cov[k] == 1:
                        valid = False
                        break
            
            if not valid:
                continue
            
            # Tentatively change
            for n in range(r_old, MAX_L + 1, p):
                if n == 0:
                    continue
                cov[n] -= 1
            for n in range(r_new, MAX_L + 1, p):
                if n == 0:
                    continue
                cov[n] += 1
            
            new_L = _compute_L(cov, 1, MAX_L)
            
            if new_L > L:
                # Accept
                for c in classes:
                    if c[0] == p:
                        c[1] = r_new
                        break
                L = new_L
                move_made = True
                break # Restart loop with new g
            else:
                # Revert
                for n in range(r_new, MAX_L + 1, p):
                    if n == 0:
                        continue
                    cov[n] -= 1
                for n in range(r_old, MAX_L + 1, p):
                    if n == 0:
                        continue
                    cov[n] += 1
        
        if move_made:
            continue
        
        # 2. Add New Prime
        for p in all_primes:
            if time.time() > deadline:
                break
            if p in used_primes:
                continue
            if budget_used + Fraction(1, p) > Fraction(1, 1):
                continue
            
            r_new = g % p
            
            # Tentatively add
            for n in range(r_new, MAX_L + 1, p):
                if n == 0:
                    continue
                cov[n] += 1
            
            new_L = _compute_L(cov, g, MAX_L) # Start from g
            
            if new_L > L:
                # Accept
                classes.append([p, r_new])
                used_primes.add(p)
                budget_used += Fraction(1, p)
                L = new_L
                move_made = True
                break
            else:
                # Revert
                for n in range(r_new, MAX_L + 1, p):
                    if n == 0:
                        continue
                        cov[n] -= 1
        
        if move_made:
            continue
        
        # 3. Swap Primes (Remove one, Add another)
        # Try to remove a larger prime and add a smaller one
        # Or just remove one to free budget?
        # Let's try swapping: remove p_rm, add p_add
        # p_add should be smaller than p_rm to be beneficial?
        # Or just any p_add not in used.
        
        for p_rm in list(used_primes):
            if time.time() > deadline:
                break
            r_rm = None
            for c in classes:
                if c[0] == p_rm:
                    r_rm = c[1]
                    break
            
            # Tentatively remove p_rm
            for n in range(r_rm, MAX_L + 1, p_rm):
                if n == 0:
                    continue
                cov[n] -= 1
            
            # Check if prefix 1..L is still valid
            # If not, revert and try next p_rm
            valid_rm = True
            for k in range(1, L + 1):
                if cov[k] == 0:
                    valid_rm = False
                    break
            
            if not valid_rm:
                # Revert
                for n in range(r_rm, MAX_L + 1, p_rm):
                    if n == 0:
                        continue
                    cov[n] += 1
                continue
            
            # Find new L after removal
            L_rm = _compute_L(cov, 1, MAX_L)
            
            # Try to add p_add
            for p_add in all_primes:
                if p_add in used_primes or p_add == p_rm:
                    continue
                if budget_used - Fraction(1, p_rm) + Fraction(1, p_add) > Fraction(1, 1):
                    continue
                
                r_add = (L_rm + 1) % p_add
                
                # Tentatively add
                for n in range(r_add, MAX_L + 1, p_add):
                    if n == 0:
                        continue
                        cov[n] += 1
                
                new_L = _compute_L(cov, L_rm + 1, MAX_L)
                
                if new_L > L:
                    # Accept swap
                    # Remove p_rm from classes
                    for i, c in enumerate(classes):
                        if c[0] == p_rm:
                            classes.pop(i)
                            break
                    used_primes.discard(p_rm)
                    budget_used -= Fraction(1, p_rm)
                    
                    # Add p_add
                    classes.append([p_add, r_add])
                    used_primes.add(p_add)
                    budget_used += Fraction(1, p_add)
                    
                    L = new_L
                    move_made = True
                    break
                else:
                    # Revert add
                    for n in range(r_add, MAX_L + 1, p_add):
                        if n == 0:
                            continue
                        cov[n] -= 1
            
            if move_made:
                break
            
            # If no p_add worked, revert removal
            for n in range(r_rm, MAX_L + 1, p_rm):
                if n == 0:
                    continue
                cov[n] += 1
        
        if not move_made:
            # Stuck?
            # Maybe try random restarts or different heuristics?
            # For now, break if no move found to avoid infinite loop
            # But we might have missed a move?
            # Let's break.
            break

    return classes
