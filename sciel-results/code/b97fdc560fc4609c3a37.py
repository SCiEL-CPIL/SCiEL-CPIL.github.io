import time
import random
import math
from fractions import Fraction

def construct():
    t0 = time.time()
    # Safety margin: The evaluator kills at 150s.
    # We should return well before that to avoid timeouts in the harness.
    # 120s is a good target for the inner loop, leaving 30s for overhead/safety.
    deadline = t0 + 110.0

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
    # Primes >= 13 have sum(1/p) approx 0.94
    selected_primes = [p for p in all_primes if p >= 13]
    N = len(selected_primes)

    # Verify budget exactly
    cost = sum(Fraction(1, p) for p in selected_primes)
    if cost > Fraction(1, 1):
        # This shouldn't happen for primes >= 13, but just in case
        # Remove smallest primes (largest 1/p) until fits
        while cost > Fraction(1, 1) and selected_primes:
            # Remove the smallest prime (largest reciprocal)
            min_idx = 0
            min_p = selected_primes[0]
            for i, p in enumerate(selected_primes):
                if p < min_p:
                    min_p = p
                    min_idx = i
            cost -= Fraction(1, selected_primes[min_idx])
            selected_primes.pop(min_idx)
            N -= 1

    # Max L to track.
    # With 160 primes, we can cover much more than 1000.
    # Let's track up to 10000 to be safe for jumps.
    MAX_L = 10000

    # Optimization: Precompute indices for each prime to speed up apply_change
    # indices_by_prime[i] is a dict: r -> list of n
    indices_by_prime = [None] * N
    for i in range(N):
        p = selected_primes[i]
        d = {}
        for r in range(p):
            lst = []
            n = r if r != 0 else p
            while n <= MAX_L:
                lst.append(n)
                n += p
            d[r] = lst
        indices_by_prime[i] = d

    # Data structure for coverage
    # counts[n] = number of primes covering n
    counts = [0] * (MAX_L + 2)

    def apply_change(i, old_r, new_r):
        if old_r == new_r:
            return
        p = selected_primes[i]
        # Decrement old
        for n in indices_by_prime[i][old_r]:
            counts[n] -= 1
        # Increment new
        for n in indices_by_prime[i][new_r]:
            counts[n] += 1

    def calc_L():
        l = 0
        # counts[0] is not used for coverage of 1..L, but we check counts[l+1]
        while l < MAX_L and counts[l+1] > 0:
            l += 1
        return l

    def init_state(residues):
        """Rebuild counts from residues."""
        counts[:] = [0] * (MAX_L + 2)
        for i in range(N):
            p = selected_primes[i]
            r = residues[i]
            for n in indices_by_prime[i][r]:
                counts[n] += 1
        L = calc_L()
        return L

    def greedy_extend(current_L):
        """
        Structural Improvement: Gap-targeted search.
        Instead of iterating through all primes and checking safety,
        we look at the first gap (current_L + 1).
        We iterate through primes that *can* cover this gap.
        For each such prime, we check if moving it to cover the gap
        breaks any existing coverage (i.e., if any n <= current_L
        has count == 1 and is covered by the old residue of that prime).
        If safe, we make the move. This directly targets the bottleneck.
        """
        extended = True
        L_local = current_L
        # Limit iterations to avoid infinite loops if stuck in a cycle,
        # though L should strictly increase on success.
        max_iters = 10000
        iters = 0

        while extended and time.time() < deadline and iters < max_iters:
            iters += 1
            extended = False

            # The target to cover
            target = L_local + 1
            if target > MAX_L:
                break

            # Find candidates: primes p such that target % p is a valid residue
            # Actually, any prime p can cover target with residue r = target % p.
            # So all primes are candidates.
            # We want to find a prime i such that:
            # 1. residues[i] != target % p
            # 2. Moving residues[i] to target % p does not break coverage of 1..L_local.
            
            # To speed up, we can shuffle the order we check primes.
            # Or we can prioritize primes that are "loose" (cover many things with count > 1).
            # Let's just shuffle for diversity.
            
            order = list(range(N))
            random.shuffle(order)
            
            found_move = False
            for i in order:
                if time.time() > deadline:
                    break
                p = selected_primes[i]
                r_old = residues[i]
                r_new = target % p

                if r_old == r_new:
                    continue

                # Check safety: does changing r_old break any n <= L_local?
                # We need counts[n] > 1 for all n covered by old_r and n <= L_local
                # We only need to check n in indices_by_prime[i][r_old] that are <= L_local.
                safe = True
                for n in indices_by_prime[i][r_old]:
                    if n > L_local:
                        break
                    if counts[n] == 1:
                        safe = False
                        break

                if safe:
                    apply_change(i, r_old, r_new)
                    residues[i] = r_new
                    
                    # Extend L
                    while L_local < MAX_L and counts[L_local+1] > 0:
                        L_local += 1
                    
                    extended = True
                    found_move = True
                    break # Restart search with new L
            
            if not found_move and not extended:
                # If we couldn't find a move in this pass, we are stuck locally.
                # We could try a "relaxation" step here, but the outer ILS loop handles kicks.
                # However, sometimes a non-improving move is needed to escape.
                # The outer loop's kick mechanism handles this.
                pass

        return L_local

    # Multi-Start Strategy
    # We will run several independent searches from different initial states.
    # Each search runs for a portion of the time budget.
    
    best_L = 0
    best_residues = None

    # Time per run
    # If we have 110s total, and we want say 5 runs, that's 22s each.
    # But the first run (diagonal) is usually the best.
    # Let's do:
    # Run 1: Diagonal start, 50s
    # Run 2: Random start, 20s
    # Run 3: Random start, 20s
    # Run 4: Random start, 15s
    # Run 5: Random start, 5s
    # Total ~110s

    runs = [
        ("diagonal", 50.0),
        ("random", 20.0),
        ("random", 20.0),
        ("random", 15.0),
        ("random", 5.0)
    ]

    for strategy, time_budget in runs:
        if time.time() > deadline:
            break

        run_deadline = time.time() + time_budget
        # Adjust global deadline for this run to be the min of global and run deadline
        # The greedy_extend uses the global 'deadline' variable.
        # We need to temporarily change it or pass it.
        # Since greedy_extend is a closure, it captures 'deadline'.
        # We can't easily change it without nonlocal or global.
        # Let's just rely on the global deadline check inside greedy_extend.
        # But we want to stop early if we hit the run budget.
        # The simplest way is to check time in the main loop and break.
        # greedy_extend will also check time.time() < deadline.
        # So if we set run_deadline < deadline, we should break the outer loop if time.time() > run_deadline.

        # Initialize state
        residues = [0] * N
        if strategy == "diagonal":
            for i in range(N):
                residues[i] = (i + 1) % selected_primes[i]
        else:
            for i in range(N):
                residues[i] = random.randint(0, selected_primes[i] - 1)

        L = init_state(residues)
        
        if L > best_L:
            best_L = L
            best_residues = residues[:]

        # ILS Loop for this run
        # We use a simple loop: Extend, if stuck Kick, Repeat.
        # No simulated annealing temperature, just random kicks.
        
        stuck_counter = 0
        T = 50.0
        T_min = 0.1
        alpha = 0.999

        while time.time() < run_deadline:
            # Try to extend
            new_L = greedy_extend(L)
            
            if new_L > L:
                L = new_L
                stuck_counter = 0
                if L > best_L:
                    best_L = L
                    best_residues = residues[:]
            else:
                # Stuck
                stuck_counter += 1
                
                # Save state
                save_residues = residues[:]
                save_counts = counts[:]
                save_L = L
                
                # Kick
                K = random.randint(1, 3)
                for _ in range(K):
                    i_kick = random.randint(0, N - 1)
                    p_kick = selected_primes[i_kick]
                    old_r_kick = residues[i_kick]
                    new_r_kick = random.randint(0, p_kick - 1)
                    
                    if old_r_kick != new_r_kick:
                        apply_change(i_kick, old_r_kick, new_r_kick)
                        residues[i_kick] = new_r_kick
                
                # Calculate new L
                L_temp = calc_L()
                
                # Try to repair/extend
                L_repaired = greedy_extend(L_temp)
                
                # Decision
                delta = L - L_repaired
                accepted = False
                if L_repaired >= L:
                    accepted = True
                else:
                    if T > T_min:
                        prob = math.exp(-delta / T)
                        if random.random() < prob:
                            accepted = True
                
                if accepted:
                    L = L_repaired
                    if L > best_L:
                        best_L = L
                        best_residues = residues[:]
                    stuck_counter = 0
                else:
                    # Revert
                    residues[:] = save_residues
                    counts[:] = save_counts
                    L = save_L
                
                # Cool down
                T = max(T_min, T * alpha)
                
                # Restart mechanism
                if stuck_counter > 100:
                    # Reset to best state found in THIS run? Or global best?
                    # Let's reset to the best state found in this run so far.
                    # Actually, resetting to global best might be better to explore from there.
                    # But we are in a new run. Let's just restart this run's search from scratch?
                    # Or just perturb more aggressively?
                    # Let's reset to the initial state of this run? No, that's wasteful.
                    # Let's reset to the best state found so far in this run.
                    # We didn't track best_per_run.
                    # Let's just keep going. The kick should eventually help.
                    # Or, we can force a larger kick.
                    pass

    # Final Check
    if best_residues is None:
        # Fallback
        best_residues = [0] * N
        for i in range(N):
            best_residues[i] = (i + 1) % selected_primes[i]

    # Ensure result is valid
    final_cost = sum(Fraction(1, p) for p in selected_primes)
    if final_cost > Fraction(1, 1):
        # This shouldn't happen
        pass

    return [[selected_primes[i], best_residues[i]] for i in range(N)]
