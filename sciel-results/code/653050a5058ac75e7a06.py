import time
import random
from fractions import Fraction

def get_primes(limit):
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = 0
    return [i for i in range(2, limit + 1) if sieve[i]]

class CoverageTracker:
    """Incremental coverage tracker for fast L evaluation."""
    def __init__(self, primes, residues, L_limit=100000):
        self.primes = primes
        self.residues = residues[:]
        self.L_limit = L_limit
        self.cover_count = [0] * (L_limit + 1)
        self._rebuild()
    
    def _rebuild(self):
        self.cover_count = [0] * (self.L_limit + 1)
        for i, p in enumerate(self.primes):
            r = self.residues[i]
            start = r if r > 0 else p
            if start <= self.L_limit:
                for x in range(start, self.L_limit + 1, p):
                    self.cover_count[x] += 1
    
    def change_residue(self, idx, new_r):
        """Change residue of prime at idx, updating cover_count incrementally."""
        p = self.primes[idx]
        old_r = self.residues[idx]
        if new_r == old_r:
            return
        # Remove old
        old_start = old_r if old_r > 0 else p
        if old_start <= self.L_limit:
            for x in range(old_start, self.L_limit + 1, p):
                self.cover_count[x] -= 1
        # Add new
        new_start = new_r if new_r > 0 else p
        if new_start <= self.L_limit:
            for x in range(new_start, self.L_limit + 1, p):
                self.cover_count[x] += 1
        self.residues[idx] = new_r
    
    def get_L(self):
        """Find largest L such that 1..L are all covered."""
        for x in range(1, self.L_limit + 1):
            if self.cover_count[x] == 0:
                return x - 1
        return self.L_limit
    
    def get_residues(self):
        return self.residues[:]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    primes = get_primes(1000)
    prime_set = set(primes)

    # Initial set: primes >= 13
    initial_primes = [p for p in primes if p >= 13]

    def sum_recip_frac(prime_list):
        return sum(Fraction(1, p) for p in prime_list)

    # Fast coverage evaluation using bytearray
    def eval_L(res_list, p_set, L_limit=100000):
        """Find largest L such that 1..L are all covered."""
        covered = bytearray(L_limit + 1)
        for i, p in enumerate(p_set):
            r = res_list[i]
            start = r if r > 0 else p
            if start <= L_limit:
                for x in range(start, L_limit + 1, p):
                    covered[x] = 1
        for x in range(1, L_limit + 1):
            if not covered[x]:
                return x - 1
        return L_limit

    # Greedy: for each uncovered x, pick best (p, r) to cover it
    def greedy_can_cover(L, prime_list, time_limit=5.0):
        t0 = time.time()
        used_primes = set()
        residues = {}
        covered = bytearray(L + 1)
        covered[0] = 1  # dummy

        for x in range(1, L + 1):
            if time.time() - t0 > time_limit:
                return None
            if covered[x]:
                continue

            best_p = None
            best_gain = -1
            best_r = None

            for p in prime_list:
                if p in used_primes:
                    continue

                r = x % p
                # Count how many new positions this covers
                gain = 0
                for k in range(0, (L - x) // p + 1):
                    y = x + k * p
                    if not covered[y]:
                        gain += 1

                if gain > best_gain:
                    best_gain = gain
                    best_p = p
                    best_r = r

            if best_p is None or best_gain <= 0:
                return None

            used_primes.add(best_p)
            residues[best_p] = best_r

            for k in range(0, (L - best_p) // best_p + 1):
                y = best_r if best_r > 0 else best_p
                for k2 in range(0, (L - y) // best_p + 1):
                    yy = y + k2 * best_p
                    if yy <= L:
                        covered[yy] = 1

        return residues

    # CSP-style backtracking with forward checking for residue assignment
    def csp_solve(L, prime_list, time_budget=5.0):
        t0 = time.time()
        n_primes = len(prime_list)

        # Precompute masks for each prime and residue
        prime_masks = []
        for p in prime_list:
            masks = []
            for r in range(p):
                m = 0
                first_x = r if r > 0 else p
                if first_x > L:
                    masks.append(0)
                    continue
                for x in range(first_x, L + 1, p):
                    m |= (1 << (x - 1))
                masks.append(m)
            prime_masks.append(masks)

        # Order primes: smaller first (more constraining)
        order = list(range(n_primes))
        order.sort(key=lambda i: prime_list[i])

        best_solution = None
        solution_count = 0
        max_backtracks = 50000

        def backtrack(idx, uncovered, assigned):
            nonlocal best_solution, solution_count, max_backtracks

            if time.time() - t0 > time_budget:
                return False
            if solution_count > max_backtracks:
                return False

            if uncovered == 0:
                if best_solution is None:
                    best_solution = assigned.copy()
                solution_count += 1
                return True

            if idx == len(order):
                return False

            p_idx = order[idx]
            p = prime_list[p_idx]

            candidates = []
            for r in range(p):
                mask = prime_masks[p_idx][r]
                new_uncovered = uncovered & ~mask
                newly_covered = bin(uncovered & mask).count('1')
                if newly_covered > 0:
                    candidates.append((newly_covered, r, new_uncovered))

            candidates.sort(key=lambda x: -x[0])

            for _, r, new_uncovered in candidates:
                assigned[p_idx] = r
                if backtrack(idx + 1, new_uncovered, assigned):
                    return True
                if best_solution is not None:
                    return True

            return False

        initial_uncovered = (1 << L) - 1
        assigned = [None] * n_primes
        if backtrack(0, initial_uncovered, assigned):
            result = {}
            for i, p_idx in enumerate(order):
                if assigned[p_idx] is not None:
                    result[prime_list[p_idx]] = assigned[p_idx]
            for p in prime_list:
                if p not in result:
                    result[p] = 0
            return result
        return None

    # Start with seed configuration
    current_primes = initial_primes[:]
    current_L = 163

    # Verify seed with greedy
    seed_assign = greedy_can_cover(current_L, current_primes, time_limit=10.0)
    if seed_assign is None:
        seed_assign = csp_solve(current_L, current_primes, time_budget=10.0)
    if seed_assign is None:
        return [[p, (i + 1) % p] for i, p in enumerate(initial_primes)]

    # Track best known solution
    best_primes = current_primes[:]
    best_L = current_L
    best_assignment = seed_assign.copy()

    # Build initial res_list for local search
    def primes_to_reslist(p_list, assign_dict):
        return [assign_dict.get(p, 0) for p in p_list]

    current_res_list = primes_to_reslist(current_primes, seed_assign)

    # Phase 1: Extend L incrementally
    candidates_to_add = [p for p in primes if p < 13]
    candidates_to_add.sort(reverse=True)  # 11, 7, 5, 3, 2

    while time.time() < deadline - 30:
        next_L = current_L + 1

        # Try current set with greedy
        assign = greedy_can_cover(next_L, current_primes, time_limit=3.0)
        if assign is not None:
            current_L = next_L
            best_L = max(best_L, current_L)
            best_assignment = assign.copy()
            best_primes = current_primes[:]
            current_res_list = primes_to_reslist(current_primes, assign)
            continue

        # Try CSP with current set
        assign = csp_solve(next_L, current_primes, time_budget=3.0)
        if assign is not None:
            current_L = next_L
            best_L = max(best_L, current_L)
            best_assignment = assign.copy()
            best_primes = current_primes[:]
            current_res_list = primes_to_reslist(current_primes, assign)
            continue

        # Try adding a new prime
        added = False
        for p_add in candidates_to_add:
            if p_add in current_primes:
                continue

            new_primes = current_primes + [p_add]
            if sum_recip_frac(new_primes) > 1:
                continue

            assign = greedy_can_cover(next_L, new_primes, time_limit=3.0)
            if assign is None:
                assign = csp_solve(next_L, new_primes, time_budget=3.0)

            if assign is not None:
                current_primes = new_primes
                candidates_to_add.remove(p_add)
                current_L = next_L
                best_L = max(best_L, current_L)
                best_assignment = assign.copy()
                best_primes = current_primes[:]
                current_res_list = primes_to_reslist(current_primes, assign)
                added = True
                break

        if not added:
            break

    # Phase 2: Simulated annealing with incremental coverage tracking
    if time.time() < deadline - 5:
        p_set = best_primes[:]
        k = len(p_set)
        res_list = primes_to_reslist(p_set, best_assignment)

        # Use incremental coverage tracker for speed
        tracker = CoverageTracker(p_set, res_list, L_limit=100000)
        current_L_check = tracker.get_L()
        best_L_sa = current_L_check

        # Simulated annealing with incremental updates
        temp = 30.0
        cooling = 0.98
        no_improve = 0
        max_no_improve = 20000
        steps_since_best = 0
        max_steps_since_best = 5000

        while time.time() < deadline - 3 and no_improve < max_no_improve and steps_since_best < max_steps_since_best:
            # Pick a random prime index, biased toward small primes (more impact)
            i = random.randint(0, k - 1)
            p = p_set[i]
            old_r = res_list[i]

            # Try a random new residue
            new_r = random.randint(0, p - 1)

            if new_r == old_r:
                continue

            # Incremental update
            tracker.change_residue(i, new_r)
            new_L = tracker.get_L()
            res_list[i] = new_r

            delta = new_L - current_L_check

            if delta > 0:
                # Accept
                current_L_check = new_L
                no_improve = 0
                steps_since_best = 0
                if new_L > best_L_sa:
                    best_L_sa = new_L
                    best_assignment = {p_set[j]: res_list[j] for j in range(k)}
                    best_primes = p_set[:]
            else:
                # Metropolis criterion
                if random.random() < pow(2.718281828, delta / max(temp, 0.1)):
                    current_L_check = new_L
                    no_improve += 1
                    steps_since_best += 1
                else:
                    # Reject - revert
                    tracker.change_residue(i, old_r)
                    res_list[i] = old_r
                    no_improve += 1
                    steps_since_best += 1

            temp *= cooling

        # Phase 3: Local refinement - try all residues for small primes, then random for large
        if time.time() < deadline - 3:
            # First pass: exhaustive for small primes
            for i in range(k):
                if time.time() > deadline - 3:
                    break
                p = p_set[i]
                old_r = res_list[i]
                best_r_for_p = old_r
                best_L_for_p = tracker.get_L()
                
                if p <= 50:
                    candidates = list(range(p))
                else:
                    # For large primes, try a few strategic residues
                    gap = best_L_sa + 1
                    candidates = [gap % p, (gap + 1) % p, (gap - 1) % p, 0, p // 2, random.randint(0, p-1)]
                    candidates = list(set(candidates))

                for new_r in candidates:
                    if new_r == old_r:
                        continue
                    tracker.change_residue(i, new_r)
                    res_list[i] = new_r
                    new_L = tracker.get_L()
                    if new_L > best_L_for_p:
                        best_L_for_p = new_L
                        best_r_for_p = new_r
                    else:
                        tracker.change_residue(i, old_r)
                        res_list[i] = old_r
                
                # Apply best
                if best_r_for_p != old_r:
                    tracker.change_residue(i, best_r_for_p)
                    res_list[i] = best_r_for_p
                    current_L_check = best_L_for_p
                    if current_L_check > best_L_sa:
                        best_L_sa = current_L_check
                        best_assignment = {p_set[j]: res_list[j] for j in range(k)}
                        best_primes = p_set[:]

            # Second pass: iterate refinement
            for _ in range(3):
                if time.time() > deadline - 3:
                    break
                improved = False
                for i in range(k):
                    if time.time() > deadline - 3:
                        break
                    p = p_set[i]
                    old_r = res_list[i]
                    best_r_for_p = old_r
                    best_L_for_p = tracker.get_L()
                    
                    if p <= 20:
                        candidates = list(range(p))
                    else:
                        gap = best_L_sa + 1
                        candidates = [gap % p, (gap + 1) % p, (gap - 1) % p, 0, random.randint(0, p-1)]
                        candidates = list(set(candidates))

                    for new_r in candidates:
                        if new_r == old_r:
                            continue
                        tracker.change_residue(i, new_r)
                        res_list[i] = new_r
                        new_L = tracker.get_L()
                        if new_L > best_L_for_p:
                            best_L_for_p = new_L
                            best_r_for_p = new_r
                        else:
                            tracker.change_residue(i, old_r)
                            res_list[i] = old_r
                    
                    if best_r_for_p != old_r:
                        tracker.change_residue(i, best_r_for_p)
                        res_list[i] = best_r_for_p
                        current_L_check = best_L_for_p
                        improved = True
                        if current_L_check > best_L_sa:
                            best_L_sa = current_L_check
                            best_assignment = {p_set[j]: res_list[j] for j in range(k)}
                            best_primes = p_set[:]
                
                if not improved:
                    break

    # Final: try to add small primes if budget allows and it helps
    if time.time() < deadline - 3:
        for p_add in [2, 3, 5, 7, 11]:
            if p_add in best_primes:
                continue
            new_primes = best_primes + [p_add]
            if sum_recip_frac(new_primes) > 1:
                continue

            # Try to find good residue for new prime
            gap = best_L_sa + 1
            r_guess = gap % p_add

            # Try a few residues
            for r_try in [r_guess, (r_guess + 1) % p_add, (r_guess - 1) % p_add, 0, p_add // 2]:
                res_list_test = primes_to_reslist(best_primes, best_assignment) + [r_try]
                new_L = eval_L(res_list_test, new_primes)
                if new_L > best_L_sa:
                    best_primes = new_primes
                    best_L_sa = new_L
                    best_assignment = {best_primes[j]: res_list_test[j] for j in range(len(best_primes))}
                    break

    # Final verification
    if not best_assignment:
        return [[p, (i + 1) % p] for i, p in enumerate(initial_primes)]

    for p in best_primes:
        if p not in best_assignment:
            best_assignment[p] = 0

    # Verify budget
    if sum_recip_frac(best_primes) > 1:
        # Fallback to valid solution
        return [[p, (i + 1) % p] for i, p in enumerate(initial_primes)]

    return [[p, best_assignment[p]] for p in best_primes]
