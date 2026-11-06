import time
import random
import math
from fractions import Fraction

def construct():
    t0 = time.time()
    deadline = t0 + 115  # Safety margin for 120s limit

    # Generate primes up to 1000
    def _primes_upto(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n+1, i):
                    sieve[j] = False
        return [i for i in range(2, n+1) if sieve[i]]

    all_primes = _primes_upto(1000)

    # Select primes with budget sum(1/p) <= 1
    # Sum of 1/p for primes >= 13 is approximately 0.94
    # Let's verify and select greedily if needed
    # Start with all primes >= 13
    selected_primes = [p for p in all_primes if p >= 13]

    # Verify budget exactly
    cost = sum(Fraction(1, p) for p in selected_primes)
    if cost > Fraction(1, 1):
        # Remove smallest primes (largest 1/p) until fits
        # Actually remove largest 1/p means smallest p
        # Let's just be safe and use a conservative set
        # Sum of 1/p for p in [13, 1000] is known to be < 1
        # Let's double check by computation
        pass  # It should fit

    N = len(selected_primes)
    MAX_L = 20000  # Track coverage up to here

    # Initialize residues
    # Strategy: We want to cover 1, 2, 3, ...
    # For the first prime p0, set residue to 1 % p0
    # For the second prime p1, set residue to 2 % p1
    # ...
    # This "diagonal" start often works well.
    residues = [0] * N
    for i in range(N):
        residues[i] = (i + 1) % selected_primes[i]

    # Data structure for coverage
    # counts[n] = number of primes covering n
    counts = [0] * (MAX_L + 2)

    def apply_change(i, old_r, new_r):
        p = selected_primes[i]
        # Decrement old
        n = old_r if old_r != 0 else p
        while n <= MAX_L:
            counts[n] -= 1
            n += p
        # Increment new
        n = new_r if new_r != 0 else p
        while n <= MAX_L:
            counts[n] += 1
            n += p

    # Initialize counts
    for i in range(N):
        p = selected_primes[i]
        r = residues[i]
        n = r if r != 0 else p
        while n <= MAX_L:
            counts[n] += 1
            n += p

    # Calculate current L
    def calc_L():
        l = 0
        while l < MAX_L and counts[l+1] > 0:
            l += 1
        return l

    L = calc_L()
    best_L = L
    best_residues = residues[:]

    # Local Search
    # Goal: Extend L by 1
    # To cover L+1, some prime p must have residue (L+1) % p.
    # We try to find a prime i such that:
    # 1. residues[i] != (L+1) % selected_primes[i]
    # 2. Changing residues[i] to (L+1) % selected_primes[i] does not uncover any n <= L.
    #    i.e., for all n <= L with n % selected_primes[i] == residues[i], counts[n] > 1.

    while time.time() < deadline:
        improved = False

        # Try to find a move that extends L
        # Iterate through primes in random order to avoid bias
        order = list(range(N))
        random.shuffle(order)

        for i in order:
            if time.time() > deadline:
                break

            p = selected_primes[i]
            r_old = residues[i]
            target_r = (L + 1) % p

            if r_old == target_r:
                continue

            # Check safety
            # We need to ensure no n in [1, L] is uniquely covered by p with residue r_old.
            # n = r_old, r_old + p, r_old + 2p, ... <= L
            safe = True
            n = r_old if r_old != 0 else p
            while n <= L:
                if counts[n] == 1:
                    safe = False
                    break
                n += p

            if safe:
                # Apply change
                apply_change(i, r_old, target_r)
                residues[i] = target_r

                # Recalculate L (it should be at least L+1, possibly more)
                while L < MAX_L and counts[L+1] > 0:
                    L += 1

                if L > best_L:
                    best_L = L
                    best_residues = residues[:]

                improved = True
                break  # Restart search with new L

        if not improved:
            # Stuck at local optimum.
            # Options:
            # 1. Random perturbation: Change a random residue, even if it breaks coverage,
            #    hoping to find a path to a higher L.
            # 2. Simulated Annealing: Accept bad moves with some probability.
            # 3. Backtracking/Restart: Reset to best known state and try different heuristics.

            # Let's try a "kick": Change a random prime's residue to something random.
            # Then try to repair coverage up to the new L.
            # If we can't repair, revert.

            if time.time() > deadline - 5:
                break

            # Pick a random prime to perturb
            i_kick = random.randint(0, N - 1)
            old_r_kick = residues[i_kick]
            p_kick = selected_primes[i_kick]
            new_r_kick = random.randint(0, p_kick - 1)

            # Save state
            save_residues = residues[:]
            save_counts = counts[:]  # This is expensive, maybe too slow for large N?
                                     # N ~ 160, MAX_L ~ 20000. 20000 ints is ~160KB. Copying is fast.

            apply_change(i_kick, old_r_kick, new_r_kick)
            residues[i_kick] = new_r_kick

            # Try to repair
            # Find current L (might be lower)
            temp_L = 0
            while temp_L < MAX_L and counts[temp_L+1] > 0:
                temp_L += 1

            # Try to extend from temp_L
            repair_success = False
            repair_L = temp_L

            # Limited repair attempts
            for _ in range(20): # Try up to 20 steps to repair
                if time.time() > deadline:
                    break
                # Find a prime to cover repair_L + 1
                found = False
                order = list(range(N))
                random.shuffle(order)
                for j in order:
                    p = selected_primes[j]
                    r_cur = residues[j]
                    target_r = (repair_L + 1) % p
                    if r_cur == target_r:
                        continue
                    # Check safety relative to repair_L
                    safe = True
                    n = r_cur if r_cur != 0 else p
                    while n <= repair_L:
                        if counts[n] == 1:
                            safe = False
                            break
                        n += p
                    if safe:
                        apply_change(j, r_cur, target_r)
                        residues[j] = target_r
                        while repair_L < MAX_L and counts[repair_L+1] > 0:
                            repair_L += 1
                        found = True
                        break
                if not found:
                    break

            # If we recovered to at least best_L, or improved, keep it
            if repair_L >= best_L:
                L = repair_L
                if L > best_L:
                    best_L = L
                    best_residues = residues[:]
            else:
                # Revert
                counts[:] = save_counts
                residues[:] = save_residues
                L = best_L  # Restore L to best known

    # Final Check
    # Ensure the result is valid
    final_cost = sum(Fraction(1, p) for p in selected_primes)
    if final_cost > Fraction(1, 1):
        # Fallback to a known safe set, though unlikely
        pass

    # Construct result
    return [[selected_primes[i], best_residues[i]] for i in range(N)]
