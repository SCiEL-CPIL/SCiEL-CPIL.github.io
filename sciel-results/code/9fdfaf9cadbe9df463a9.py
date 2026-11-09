"""Covering 1..L by prime residue classes with reciprocal budget 1 (primes <= 1000)."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from fractions import Fraction

MAXL = 100000
DEADLINE = 145.0  # Safety margin before the 150s kill

def _primes_upto(n):
    sieve = bytearray([1]) * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = 0
    return [i for i in range(2, n + 1) if sieve[i]]

def construct():
    t0 = time.time()
    primes = _primes_upto(1000)
    recip = {p: Fraction(1, p) for p in primes}
    
    # Precompute indices for each prime for fast NumPy updates
    # indices[p] is an array of indices in [0, MAXL] covered by residue 0 mod p
    # Actually, we just need to update ranges.
    # For a prime p and residue r, the covered indices are r, r+p, r+2p, ...
    # We can generate these on the fly or precompute masks.
    # Precomputing masks for all p and all r is too much memory (168 primes * 1000 residues * 100k bits).
    # Instead, we update cnt array directly.
    
    # cnt[x] = number of classes covering x
    # We use int16 or int32 to save memory and speed up
    cnt = np.zeros(MAXL + 1, dtype=np.int16)
    
    # State: dict mapping prime -> residue
    state = {}
    budget = Fraction(1, 1)
    
    # Helper to update cnt for a prime p and residue r
    def add_class(p, r):
        # Indices: r, r+p, r+2p, ... <= MAXL
        # If r == 0, we cover p, 2p, ... (since 0 is not in 1..L, but 1..L starts at 1)
        # Wait, the problem says 0 <= r < p.
        # If r=0, it covers multiples of p.
        # The range in numpy: np.arange(r if r!=0 else p, MAXL+1, p)
        # Actually, if r=0, the first positive multiple is p.
        # If r>0, the first is r.
        start = r if r > 0 else p
        if start <= MAXL:
            indices = np.arange(start, MAXL + 1, p)
            cnt[indices] += 1
            
    def remove_class(p, r):
        start = r if r > 0 else p
        if start <= MAXL:
            indices = np.arange(start, MAXL + 1, p)
            cnt[indices] -= 1

    def compute_L():
        # Find the first index i in 1..MAXL where cnt[i] == 0
        # We can use np.argmax on (cnt == 0) but we need to skip index 0
        # Create a boolean array of uncovered
        uncovered = (cnt[1:] == 0)
        if not np.any(uncovered):
            return MAXL
        # Find first True
        first_gap = np.argmax(uncovered)
        # first_gap is index in uncovered, which corresponds to 1+first_gap in original
        # But if all are covered, argmax returns 0, but we handled that.
        # Wait, if uncovered[0] is True, argmax is 0, so L = 0?
        # No, L is the largest i such that 1..i are covered.
        # If 1 is not covered, L=0.
        # If 1,2 covered, 3 not, L=2.
        # uncovered array is for indices 1..MAXL.
        # idx in uncovered corresponds to x = idx + 1.
        # If uncovered[idx] is True, x is not covered.
        # So L = idx (since 1..idx are covered, idx+1 is not).
        return int(first_gap)

    # Initial Greedy Construction
    # We want to maximize L.
    # Start with empty state.
    # Greedy: pick (p,r) that extends L the most.
    
    # To make greedy faster, we can just pick primes in increasing order of 1/p? No, small p cover more.
    # But small p have large 1/p.
    # Let's try a randomized greedy with restarts.
    
    best_state = {}
    best_budget = Fraction(1, 1)
    best_L = 0
    
    rng = random.Random(42)
    
    for seed in range(20):
        if time.time() > DEADLINE - 60:
            break
        rng = random.Random(seed)
        state = {}
        budget = Fraction(1, 1)
        cnt[:] = 0
        L = 0
        remaining_primes = primes[:]
        rng.shuffle(remaining_primes)
        
        # Greedy Step
        for p in remaining_primes:
            if time.time() > DEADLINE - 50:
                break
            if budget < recip[p]:
                continue
            
            # Try to find a residue r that maximizes new L
            # The most promising r is one that covers L+1.
            # r = (L+1) % p
            # But maybe another r covers more?
            # Let's try r = (L+1) % p first.
            r = (L + 1) % p
            
            # Check if this r is better than some random ones?
            # For speed, just use this one.
            # However, if (L+1) % p is 0, it covers p, 2p...
            # Let's evaluate.
            
            add_class(p, r)
            new_L = compute_L()
            
            if new_L > L:
                state[p] = r
                budget -= recip[p]
                L = new_L
            else:
                remove_class(p, r)
                # If it didn't extend L, maybe skip? Or try another residue?
                # Let's try a few random residues if it didn't work.
                found = False
                for _ in range(5):
                    r_rand = rng.randrange(p)
                    if r_rand == r: continue
                    add_class(p, r_rand)
                    new_L_rand = compute_L()
                    if new_L_rand > L:
                        state[p] = r_rand
                        budget -= recip[p]
                        L = new_L_rand
                        found = True
                        break
                    else:
                        remove_class(p, r_rand)
                if not found:
                    # Don't add this prime if it doesn't help extend L immediately
                    # This keeps the set small and budget high
                    pass
        
        if L > best_L:
            best_L = L
            best_state = dict(state)
            best_budget = budget
            # Store a copy of cnt for SA initialization? No, we'll rebuild.

    # Initialize SA with best greedy state
    state = dict(best_state)
    budget = best_budget
    cnt[:] = 0
    for p, r in state.items():
        add_class(p, r)
    L = compute_L()
    best_L_sa = L
    best_state_sa = dict(state)
    best_budget_sa = budget
    
    # Simulated Annealing
    rng = random.Random(123)
    T = 1.0
    T_min = 1e-4
    alpha = 0.9999
    
    # Precompute list of available primes for adding
    all_primes = primes[:]
    
    while time.time() < DEADLINE:
        T = max(T_min, T * alpha)
        
        # Determine move type
        # 1. Change residue of an existing prime (most common)
        # 2. Add a new prime
        # 3. Remove a prime
        
        move_type = rng.random()
        
        if move_type < 0.7:
            # Change residue
            if not state:
                continue
            p = rng.choice(list(state.keys()))
            old_r = state[p]
            
            # Target the current gap L+1
            # But maybe a random gap?
            # Let's focus on L+1 to push the boundary.
            target = L + 1
            if target > MAXL:
                target = rng.randint(1, MAXL)
            
            new_r = target % p
            
            if new_r == old_r:
                continue
            
            # Apply move
            remove_class(p, old_r)
            add_class(p, new_r)
            state[p] = new_r
            
            new_L = compute_L()
            
            delta = new_L - L
            if delta < 0:
                # Accept with probability exp(delta/T) ? 
                # Since L is integer, delta is negative integer.
                # If T is small, reject.
                if rng.random() > np.exp(delta / T):
                    # Revert
                    remove_class(p, new_r)
                    add_class(p, old_r)
                    state[p] = old_r
                    new_L = L
            else:
                # Always accept if delta >= 0
                pass
            
            L = new_L
            
            if L > best_L_sa:
                best_L_sa = L
                best_state_sa = dict(state)
                best_budget_sa = budget

        elif move_type < 0.9:
            # Add a prime
            avail = [p for p in all_primes if p not in state and budget >= recip[p]]
            if not avail:
                continue
            p = rng.choice(avail)
            
            # Choose residue to cover L+1
            r = (L + 1) % p
            
            add_class(p, r)
            state[p] = r
            budget -= recip[p]
            
            new_L = compute_L()
            
            delta = new_L - L
            if delta < 0:
                if rng.random() > np.exp(delta / T):
                    # Revert
                    remove_class(p, r)
                    del state[p]
                    budget += recip[p]
                    new_L = L
            else:
                pass
            
            L = new_L
            
            if L > best_L_sa:
                best_L_sa = L
                best_state_sa = dict(state)
                best_budget_sa = budget

        else:
            # Remove a prime
            if len(state) <= 2:
                continue
            # Remove a random prime, or one with high 1/p?
            # Removing high 1/p frees more budget.
            # Let's pick random to explore.
            p = rng.choice(list(state.keys()))
            r = state[p]
            
            remove_class(p, r)
            del state[p]
            budget += recip[p]
            
            new_L = compute_L()
            
            delta = new_L - L
            if delta < 0:
                if rng.random() > np.exp(delta / T):
                    # Revert
                    add_class(p, r)
                    state[p] = r
                    budget -= recip[p]
                    new_L = L
            else:
                pass
            
            L = new_L
            
            if L > best_L_sa:
                best_L_sa = L
                best_state_sa = dict(state)
                best_budget_sa = budget

    # Final verification and cleanup
    # Ensure budget <= 1
    total = Fraction(0, 1)
    for p in best_state_sa:
        total += recip[p]
    
    if total > 1:
        # This shouldn't happen if we tracked budget correctly, but just in case
        # Remove primes with largest 1/p until valid
        keys = sorted(best_state_sa.keys(), key=lambda x: 1.0/x, reverse=True)
        for p in keys:
            if total <= 1:
                break
            del best_state_sa[p]
            total -= recip[p]
            
    return [[p, r] for p, r in best_state_sa.items()]
# EVOLVE-BLOCK-END
