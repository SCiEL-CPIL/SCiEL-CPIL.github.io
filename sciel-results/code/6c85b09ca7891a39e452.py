"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import sys

def construct():
    # Deadline guard: 150s limit, use 145s for safety
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Generate primes up to 1000
    primes = [p for p in range(2, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
    n_primes = len(primes)
    
    # We want to find residues r_p for each prime p such that
    # every integer x in [1, L] satisfies at least 2 congruences x == r_p (mod p).
    
    # Strategy: Greedy Frontier Extension
    # We maintain a set of "active" primes (those whose residue hasn't been finalized for the current frontier).
    # Actually, since each prime has only ONE residue for the whole interval, we can't "finalize" and move on easily
    # unless we use backtracking or a constraint satisfaction approach.
    
    # However, the "Greedy Set Cover" inspiration suggests:
    # 1. Start with L=0.
    # 2. To extend to L+1, we need coverage(L+1) >= 2.
    # 3. We check current coverage of L+1 based on current residues.
    # 4. If coverage < 2, we need to assign residues to some primes to cover L+1.
    # 5. Which primes? The ones that currently cover the most other numbers in [1, L+1] might be good candidates to "steal" from?
    #    Or simply pick a prime that doesn't already cover L+1, set its residue to (L+1) % p.
    #    This might break coverage of other numbers.
    
    # A more robust approach inspired by "constraint-driven greedy":
    # We process integers x = 1, 2, ... in order.
    # For each x, we ensure it has coverage >= 2.
    # We maintain a list of residues `r`.
    # We maintain a count of coverage for all x up to some limit?
    # No, we only care about the current frontier x.
    # But changing r_p affects future x's too.
    
    # Let's try a different angle:
    # The problem is equivalent to finding a vector r in product of Z_p such that
    # for all x in 1..L, sum_p [x % p == r_p] >= 2.
    
    # Greedy Algorithm:
    # 1. Initialize r_p = 0 for all p.
    # 2. L = 0.
    # 3. While time < deadline:
    #    a. Check if L+1 is covered by >= 2 primes.
    #    b. If yes, L += 1.
    #    c. If no, we need to increase coverage of L+1.
    #       - Find all primes p such that (L+1) % p != r_p. (These are candidates to change).
    #       - For each candidate p, if we set r_p = (L+1) % p, how many existing covered numbers (1..L) lose coverage?
    #         Specifically, we look for numbers y in 1..L that were covered by p (i.e., y % p == r_p_old) and are now at risk.
    #         Actually, we just need to ensure we don't drop any y in 1..L below coverage 2.
    #       - This is expensive to check for all y in 1..L for each candidate p.
    
    # Optimized Greedy with Local Search:
    # Let's combine the frontier extension with a local repair step.
    
    # State:
    residues = [0] * n_primes
    # We need to track coverage for the current frontier efficiently.
    # Since L can grow, we might need to recalculate coverage for 1..L.
    # But L is likely small initially.
    
    # Let's use a coverage array for 1..MAX_CHECK.
    # If L grows beyond MAX_CHECK, we extend it.
    MAX_CHECK = 5000
    coverage = [0] * (MAX_CHECK + 1)
    
    def recalc_coverage():
        for i in range(MAX_CHECK + 1):
            coverage[i] = 0
        for idx, p in enumerate(primes):
            r = residues[idx]
            if r == 0:
                start = p
            else:
                start = r
            x = start
            while x <= MAX_CHECK:
                coverage[x] += 1
                x += p
                
    recalc_coverage()
    
    best_L = 0
    current_L = 0
    
    # Calculate initial L
    for x in range(1, MAX_CHECK + 1):
        if coverage[x] >= 2:
            current_L = x
        else:
            break
    best_L = current_L
    
    # If current_L is 0, we start from scratch.
    
    # Greedy Loop
    while time.time() < deadline:
        # Try to extend current_L by 1
        next_x = current_L + 1
        
        if next_x > MAX_CHECK:
            # Extend coverage array
            old_max = MAX_CHECK
            MAX_CHECK *= 2
            coverage.extend([0] * (MAX_CHECK - old_max))
            # Recalculate everything? That's expensive.
            # Just recalculate the new part? No, residues are global.
            # For simplicity, if we hit this, just break or do a full recalc.
            recalc_coverage()
        
        # Check coverage of next_x
        if coverage[next_x] >= 2:
            current_L = next_x
            if current_L > best_L:
                best_L = current_L
            continue
        
        # Coverage is insufficient (< 2).
        # We need to pick a prime p and change its residue to cover next_x.
        # Candidates: primes where residues[idx] != next_x % p.
        
        # Heuristic: Pick a prime that minimizes damage to previous coverage.
        # Damage = number of y in [1, current_L] such that:
        #   1. y % p == residues[idx] (currently covered by p)
        #   2. coverage[y] == 2 (so removing p drops it to 1, which is bad)
        
        # To speed this up, we can precompute for each prime p, the list of y in [1, current_L] covered by p.
        # But that's O(L * n_primes) memory/time if done naively.
        
        # Alternative: Just try a few random candidates or the one with largest p (covers fewer numbers in [1,L] relative to density? No, larger p covers fewer numbers in [1,L] if L < p).
        # If p > L, then p covers at most 1 number in [1, L] (which is p itself, but p>L so 0 numbers? No, x % p = x. So if r_p = x, it covers x. But x < p. So if we set r_p = next_x % p, and p > next_x, then r_p = next_x. It covers next_x. Does it cover anything in [1, L]? Only if next_x is in [1, L], which it isn't. So changing a prime p > L only affects next_x and future numbers. It does NOT damage [1, L].
        
        # This is a KEY INSIGHT:
        # If we pick a prime p > current_L, changing its residue will NOT affect the coverage of any number in [1, current_L].
        # Why? Because for any y <= current_L < p, y % p = y.
        # The residue r_p is a single value.
        # If we change r_p, it only affects numbers x where x % p == r_p.
        # Since x <= current_L < p, x % p = x.
        # So p covers x in [1, current_L] iff x == r_p.
        # Previously, r_p was some value. It covered at most one number in [1, current_L] (if r_p <= current_L).
        # Now we set r_p = next_x % p = next_x (since next_x < p if we pick p > next_x).
        # So it will cover next_x.
        # Does it still cover the old number? Only if old_r == next_x. But next_x > current_L >= old_r (if old_r was in range).
        # Wait, if old_r was in [1, current_L], and we change it to next_x (which is > current_L), then we LOSE coverage of old_r.
        # So it DOES damage [1, current_L] if the old residue was in that range.
        
        # However, if we pick a prime p such that its current residue r_p is NOT in [1, current_L], then changing it to next_x (assuming p > next_x) will not damage [1, current_L].
        # When is r_p not in [1, current_L]?
        # If r_p > current_L.
        
        # So, strategy:
        # 1. Identify primes p > next_x.
        # 2. Among those, find ones where residues[idx] > current_L.
        # 3. If such a prime exists, change its residue to next_x % p (which is next_x).
        #    This adds coverage to next_x and damages nothing in [1, current_L].
        # 4. If no such prime exists, we might need to accept damage or use a smaller prime.
        
        # Let's implement this.
        
        candidates = []
        for idx, p in enumerate(primes):
            if p > next_x:
                # Check if changing this residue damages [1, current_L]
                old_r = residues[idx]
                # It covers y in [1, current_L] iff y == old_r (since y < p).
                # So it damages [1, current_L] iff 1 <= old_r <= current_L.
                if old_r > current_L or old_r == 0: 
                    # old_r == 0 covers p, 2p... none in [1, current_L] since p > current_L.
                    # So no damage.
                    candidates.append(idx)
        
        if candidates:
            # Pick one randomly or deterministically
            idx = random.choice(candidates)
            p = primes[idx]
            new_r = next_x % p
            residues[idx] = new_r
            # Update coverage
            # Remove old contributions, add new
            # Since p > next_x, old_r and new_r are both > current_L (or 0).
            # Old contribution in [1, MAX_CHECK]:
            #   x = old_r, old_r+p, ...
            #   Since old_r > current_L, the first hit is old_r.
            #   We need to subtract from coverage[old_r], coverage[old_r+p], etc.
            # New contribution:
            #   x = new_r, new_r+p, ...
            #   new_r = next_x.
            #   We add to coverage[next_x], coverage[next_x+p], etc.
            
            # Efficient update:
            if old_r == 0:
                start_old = p
            else:
                start_old = old_r
            
            x = start_old
            while x <= MAX_CHECK:
                coverage[x] -= 1
                x += p
                
            if new_r == 0:
                start_new = p
            else:
                start_new = new_r
                
            x = start_new
            while x <= MAX_CHECK:
                coverage[x] += 1
                x += p
                
            # Check if next_x is now covered >= 2
            if coverage[next_x] >= 2:
                current_L = next_x
                if current_L > best_L:
                    best_L = current_L
            else:
                # It wasn't enough? We added 1. It was 0 or 1 before.
                # If it was 1, now it is 2. Should be fine.
                # If it was 0, now it is 1. Not enough.
                # We need another prime.
                pass
        
        else:
            # No "safe" large prime.
            # Fallback: Pick a random prime and change it, hoping it works or use local search.
            # Or, pick the prime with the smallest "damage".
            # For now, let's try a random prime > next_x and accept damage, then repair?
            # Or just break if we can't extend safely?
            
            # Let's try to find ANY prime that can cover next_x.
            # If we change a prime p, coverage[next_x] increases by 1 (if it wasn't covered before).
            # We need coverage[next_x] to become >= 2.
            
            # If we are stuck, we might need to backtrack or use a different strategy.
            # Given the time limit, let's just pick a random prime and hope for the best,
            # or better, stop if we can't make progress efficiently.
            
            # Let's try a targeted repair:
            # Find the prime p that minimizes the number of y in [1, current_L] with coverage[y]==2 and y%p==residues[idx].
            # This is expensive.
            
            # Alternative: Just pick a random prime and change it.
            idx = random.randint(0, n_primes - 1)
            p = primes[idx]
            old_r = residues[idx]
            new_r = next_x % p
            residues[idx] = new_r
            
            # Update coverage
            if old_r == 0:
                start_old = p
            else:
                start_old = old_r
            x = start_old
            while x <= MAX_CHECK:
                coverage[x] -= 1
                x += p
                
            if new_r == 0:
                start_new = p
            else:
                start_new = new_r
            x = start_new
            while x <= MAX_CHECK:
                coverage[x] += 1
                x += p
            
            if coverage[next_x] >= 2:
                current_L = next_x
                if current_L > best_L:
                    best_L = current_L
            else:
                # Still not covered? Revert?
                # If we revert, we are stuck.
                # Let's revert and break to avoid wasting time.
                residues[idx] = old_r
                if new_r == 0:
                    start_new = p
                else:
                    start_new = new_r
                x = start_new
                while x <= MAX_CHECK:
                    coverage[x] -= 1
                    x += p
                    
                if old_r == 0:
                    start_old = p
                else:
                    start_old = old_r
                x = start_old
                while x <= MAX_CHECK:
                    coverage[x] += 1
                    x += p
                break

    return residues
# EVOLVE-BLOCK-END
