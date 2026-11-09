import math
import time
import random

def construct():
    t0 = time.time()
    DEADLINE = 85.0

    def now():
        return time.time()

    # Moduli from 2 to 128
    moduli = list(range(2, 129))
    n = len(moduli)

    # Precompute gcd matrix
    gcds = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            g = math.gcd(moduli[i], moduli[j])
            gcds[i][j] = g
            gcds[j][i] = g

    # For each modulus, compute its prime factors (distinct)
    def get_prime_factors(m):
        factors = set()
        d = 2
        temp = m
        while d * d <= temp:
            while temp % d == 0:
                factors.add(d)
                temp //= d
            d += 1
        if temp > 1:
            factors.add(temp)
        return factors

    prime_factors = [get_prime_factors(m) for m in moduli]

    # Two residue classes (r1, m1) and (r2, m2) are disjoint iff
    # for every prime p dividing gcd(m1, m2), r1 % p != r2 % p.
    # Equivalently, for every shared prime p, the residues must differ mod p.

    def can_assign(indices):
        """
        Check if we can assign residues to the given set of moduli indices
        such that all pairs are disjoint.
        This is a constraint satisfaction problem.
        We use a simple backtracking with forward checking.
        """
        if not indices:
            return []

        # Order by modulus (fewer residues to try first)
        sorted_indices = sorted(indices, key=lambda i: moduli[i])

        assignment = {}

        def backtrack(pos):
            if pos == len(sorted_indices):
                return True

            idx = sorted_indices[pos]
            m = moduli[idx]

            # Try each residue
            for r in range(m):
                valid = True
                for prev_idx, prev_r in assignment.items():
                    g = gcds[idx][prev_idx]
                    if g > 1:
                        # Check all shared prime factors
                        for p in prime_factors[idx]:
                            if p in prime_factors[prev_idx]:
                                if r % p == prev_r % p:
                                    valid = False
                                    break
                    if not valid:
                        break

                if valid:
                    assignment[idx] = r
                    if backtrack(pos + 1):
                        return True
                    del assignment[idx]

            return False

        if backtrack(0):
            return [[assignment[idx], moduli[idx]] for idx in indices]
        return None

    # Greedy construction: pick moduli one by one, ensuring disjointness is feasible
    # We use a simple approach: start with all moduli, remove those that cause conflicts.

    def greedy_construct():
        # Start with all moduli
        selected = list(range(n))
        # Sort by number of prime factors (fewer is better - more flexible)
        selected.sort(key=lambda i: len(prime_factors[i]))

        result = []
        used_residues = {}  # For each prime p, track which residues mod p are used by selected moduli divisible by p

        for idx in selected:
            m = moduli[idx]
            pfs = prime_factors[idx]

            # Check if we can assign a residue to this modulus
            # For each prime p in pfs, we need a residue r such that for all other primes q in pfs,
            # r % q != used_residues[q] (if q divides m)
            # Actually, we need r such that for each p in pfs, r % p is not in used_residues[p]

            # This is a constraint satisfaction on the residue r.
            # Let's just try to find one via backtracking on the small number of constraints.

            # Collect constraints: for each p in pfs, forbidden residues mod p
            forbidden = {}
            for p in pfs:
                forbidden[p] = used_residues.get(p, set())

            # Find a residue r in [0, m) that satisfies: for all p in pfs, r % p not in forbidden[p]
            # Use CRT-like search or just iterate
            found = False
            for r in range(m):
                valid = True
                for p in pfs:
                    if r % p in forbidden[p]:
                        valid = False
                        break
                if valid:
                    # Assign this residue
                    for p in pfs:
                        if p not in used_residues:
                            used_residues[p] = set()
                        used_residues[p].add(r % p)
                    result.append([r, m])
                    found = True
                    break

            if found:
                pass
            else:
                # Cannot assign, skip this modulus
                pass

        return result

    # Better approach: use the insight that for each prime p, at most p moduli divisible by p can be selected
    # because we need distinct residues mod p.
    # So we have a capacity constraint: for each prime p, count of selected moduli divisible by p <= p.

    def capacity_greedy():
        # For each modulus, list of small primes (<=128) that divide it
        # Actually, we only care about primes p where the capacity constraint matters.
        # For large primes p > sqrt(128) ~ 11.3, p can divide at most one modulus in [2,128]
        # (since 2p > 128 for p > 64, and for p in (11, 64], 2p is in range but 3p > 128 for p > 42)
        # Wait, let's just handle all primes properly.

        # For each prime p, the number of moduli in [2,128] divisible by p is floor(128/p).
        # Capacity is p. So if floor(128/p) <= p, no constraint.
        # floor(128/p) <= p  =>  p >= sqrt(128) ~ 11.3
        # So for p >= 11, no capacity constraint (we can select all moduli divisible by p).
        # Only p in {2, 3, 5, 7} have constraints.
        # p=2: floor(128/2)=64, capacity=2. Can select at most 2 even moduli.
        # p=3: floor(128/3)=42, capacity=3. Can select at most 3 multiples of 3.
        # p=5: floor(128/5)=25, capacity=5. Can select at most 5 multiples of 5.
        # p=7: floor(128/7)=18, capacity=7. Can select at most 7 multiples of 7.

        constrained_primes = {2: 2, 3: 3, 5: 5, 7: 7}

        # For each modulus, which constrained primes divide it?
        def get_constrained_primes(m):
            result = []
            for p in constrained_primes:
                if m % p == 0:
                    result.append(p)
            return result

        mod_constrained = [get_constrained_primes(m) for m in moduli]

        # Separate moduli into two groups:
        # A: moduli not divisible by any constrained prime (always selectable)
        # B: moduli divisible by at least one constrained prime (capacity constrained)

        group_A = [i for i in range(n) if not mod_constrained[i]]
        group_B = [i for i in range(n) if mod_constrained[i]]

        # Sort group_B by number of constrained primes (fewer first, more flexible)
        group_B.sort(key=lambda i: len(mod_constrained[i]))

        selected = set(group_A)
        counts = {p: 0 for p in constrained_primes}

        for idx in group_B:
            pfs = mod_constrained[idx]
            # Check if adding this modulus would violate any capacity
            can_add = True
            for p in pfs:
                if counts[p] + 1 > constrained_primes[p]:
                    can_add = False
                    break
            if can_add:
                selected.add(idx)
                for p in pfs:
                    counts[p] += 1

        return list(selected)

    def assign_residues_crt(indices):
        """
        Assign residues using the capacity-based approach.
        For each constrained prime p, assign distinct residues mod p to selected moduli divisible by p.
        For unconstrained moduli, residue is 0.
        Use CRT to combine constraints for each modulus.
        """
        if not indices:
            return []

        # Group selected moduli by constrained prime
        from collections import defaultdict
        prime_to_mods = defaultdict(list)
        for idx in indices:
            pfs = mod_constrained[idx]
            for p in pfs:
                prime_to_mods[p].append(idx)

        # Assign residues mod p for each prime
        residue_mod_p = {}  # (idx, p) -> r % p
        for p, mods in prime_to_mods.items():
            # Assign distinct residues 0, 1, ..., len(mods)-1 mod p
            # Since len(mods) <= p, this is possible
            random.shuffle(mods)
            for k, idx in enumerate(mods):
                residue_mod_p[(idx, p)] = k % p

        # For each selected modulus, use CRT to find r
        result = []
        for idx in indices:
            m = moduli[idx]
            pfs = mod_constrained[idx]

            if not pfs:
                result.append([0, m])
                continue

            # Solve r ≡ residue_mod_p[(idx, p)] (mod p) for each p in pfs
            # Use CRT
            # Since pfs are distinct primes, we can use iterative CRT
            r = 0
            M = 1
            for p in pfs:
                rp = residue_mod_p[(idx, p)]
                # Find r such that r ≡ current_r (mod M) and r ≡ rp (mod p)
                # r = current_r + M * t
                # current_r + M * t ≡ rp (mod p)
                # M * t ≡ rp - current_r (mod p)
                # t ≡ (rp - current_r) * M^{-1} (mod p)
                inv_M = pow(M, -1, p)  # Python 3.8+
                t = ((rp - r) * inv_M) % p
                r = r + M * t
                M *= p

            r = r % m  # Ensure canonical
            result.append([r, m])

        return result

    # Verify solution
    def verify(solution):
        if not solution:
            return False
        moduli_set = set()
        for r, m in solution:
            if m < 2 or m > 128:
                return False
            if r < 0 or r >= m:
                return False
            if m in moduli_set:
                return False
            moduli_set.add(m)

        for i in range(len(solution)):
            for j in range(i + 1, len(solution)):
                r1, m1 = solution[i]
                r2, m2 = solution[j]
                g = math.gcd(m1, m2)
                if g > 1:
                    # Check all shared prime factors
                    for p in get_prime_factors(g):
                        if r1 % p == r2 % p:
                            return False
        return True

    # Main algorithm
    best_solution = None

    # Step 1: Get selected moduli via capacity greedy
    selected_indices = capacity_greedy()

    # Step 2: Assign residues
    solution = assign_residues_crt(selected_indices)

    # Step 3: Verify
    if solution and verify(solution):
        best_solution = solution

    # If the capacity greedy didn't give a valid solution, try local search or other methods
    if best_solution is None or len(best_solution) < 10:
        # Try randomized local search to improve
        best_clique = set(i for _, m in best_solution) if best_solution else set()

        # Try to add more moduli
        for _ in range(1000):
            if now() - t0 > DEADLINE:
                break

            # Try to add a random unselected modulus
            unselected = [i for i in range(n) if i not in best_clique]
            if not unselected:
                break

            idx = random.choice(unselected)
            new_clique = best_clique | {idx}
            sol = assign_residues_crt(list(new_clique))
            if sol and verify(sol) and len(sol) > len(best_solution or []):
                best_solution = sol
                best_clique = new_clique

    # Fallback if still no solution
    if best_solution is None:
        # Simple valid pair
        best_solution = [[0, 2], [1, 3]]

    return best_solution
