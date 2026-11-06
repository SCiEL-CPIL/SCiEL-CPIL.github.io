"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin within 150s limit

    # Generate all primes up to 1000
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    primes = [p for p in range(2, 1001) if is_prime(p)]
    prime_set = set(primes)
    n_primes = len(primes)

    # For each prime, precompute which integers in 1..1000 it covers for each residue
    # coverage[p_idx][r] = list of integers n in 1..1000 where n % p == r
    max_L_target = 1000  # We aim to cover at least up to 999
    # Actually, let's think about what L we're trying to achieve.
    # The score is the largest L such that every integer 1..L is covered.
    # We want to maximize L. Let's try to cover as much as possible.

    # Greedy approach with local search:
    # Start with a good initial solution, then iteratively improve.

    # Key insight: We need to select a subset of primes and assign residues to them
    # such that sum(1/p) <= 1 and the union of covered integers is maximized (contiguous from 1).

    # Strategy:
    # 1. Start with small primes (they cover more integers per unit cost)
    # 2. Use a greedy set-cover-like approach
    # 3. Then do local search to improve

    # Let's think about this differently.
    # For a given set of primes, we want to assign residues to maximize the contiguous coverage from 1.
    # This is still hard, but we can use a different approach.

    # Alternative approach: Dynamic programming or greedy with backtracking on a limited set.

    # Let's try a different strategy:
    # 1. Sort primes by size (smaller primes first - they have lower 1/p cost and cover more)
    # 2. Greedily add primes, choosing the residue that maximizes new contiguous coverage
    # 3. Then do local search: try swapping primes or changing residues

    # First, let's compute for each prime p, for each residue r, which numbers in 1..N are covered
    # where N is some upper bound. Let's use N = 2000 to be safe.

    N_bound = 2000  # Upper bound for checking coverage

    # coverage_map[p][r] = set of numbers in 1..N_bound covered by (p, r)
    # But storing sets is expensive. Let's use bitmasks or just check on the fly.

    # For efficiency, let's precompute for each prime p and residue r,
    # the list of numbers in 1..N_bound that are covered.
    # But that's a lot of data. Let's be smarter.

    # For a given prime p and residue r, the numbers covered are: r, r+p, r+2p, ... up to N_bound
    # (with r=0 meaning multiples of p, but we need 0 <= r < p, and numbers start from 1)
    # Actually, if r == 0, covered numbers are p, 2p, 3p, ... (since we start from 1, 0 is not included)
    # If r > 0, covered numbers are r, r+p, r+2p, ...

    def get_covered(p, r, limit):
        """Return the set of numbers in 1..limit covered by residue class (p, r)."""
        if r == 0:
            # Multiples of p starting from p
            if p > limit:
                return set()
            return set(range(p, limit + 1, p))
        else:
            # r, r+p, r+2p, ...
            if r > limit:
                return set()
            return set(range(r, limit + 1, p))

    # Greedy construction:
    # We'll maintain:
    # - selected: list of (p, r) pairs
    # - covered: set of integers covered
    # - budget_used: Fraction, sum of 1/p for selected primes
    # - remaining_budget: 1 - budget_used

    # At each step, try adding a new prime p with the best residue r that maximizes
    # the contiguous coverage from 1, subject to budget constraints.

    # To evaluate "best residue", we need to compute the new contiguous coverage length.
    # This is expensive if done naively. Let's optimize.

    # Let's define a function to compute the contiguous coverage length from 1 given a set of (p, r) pairs.
    def compute_contiguous_coverage(pairs, limit):
        """Compute the largest L such that all integers 1..L are covered by the given pairs."""
        if not pairs:
            return 0
        # Create a boolean array for 1..limit
        covered_arr = [False] * (limit + 1)
        for p, r in pairs:
            if r == 0:
                start = p
            else:
                start = r
            if start <= limit:
                for n in range(start, limit + 1, p):
                    covered_arr[n] = True
        # Find the largest L such that 1..L are all covered
        L = 0
        for i in range(1, limit + 1):
            if covered_arr[i]:
                L = i
            else:
                break
        return L

    # This is O(limit * len(pairs)) per call, which could be slow.
    # Let's use a different approach for the greedy step.

    # For the greedy step, when considering adding (p, r), we need to know:
    # 1. What's the current contiguous coverage?
    # 2. What's the new contiguous coverage after adding (p, r)?
    # 3. Does the budget allow it?

    # Let's maintain the covered array incrementally.

    # Start with empty selection
    selected = []
    budget_used = Fraction(0, 1)
    max_budget = Fraction(1, 1)

    # Sort primes: smaller primes first (lower cost, more coverage)
    sorted_primes = sorted(primes, key=lambda p: p)

    # For the greedy step, we'll iterate through primes and try to add them.
    # For each prime, try all residues and pick the one that gives the best improvement.

    # To make this efficient, let's precompute for each prime p, for each residue r,
    # the list of numbers covered (as a sorted list or set).

    # Actually, let's just do it directly but with a time check.

    current_L = 0  # Current contiguous coverage

    # We'll do multiple passes. In each pass, try to add a new prime.
    # Stop when we can't add any more or time is running out.

    # For efficiency, let's maintain a covered_set as a set of integers.
    covered_set = set()

    def update_covered(p, r, limit):
        """Add (p, r) to covered_set."""
        if r == 0:
            start = p
        else:
            start = r
        if start <= limit:
            for n in range(start, limit + 1, p):
                covered_set.add(n)

    def compute_L_from_set(covered_set, limit):
        """Compute contiguous coverage from 1 given a set of covered integers."""
        L = 0
        for i in range(1, limit + 1):
            if i in covered_set:
                L = i
            else:
                break
        return L

    # Greedy: try to add primes one by one
    # For each prime not yet selected, try all residues and pick the best one.

    selected_primes = set()

    # We'll do this in a loop, trying to improve the solution.
    # First pass: greedy addition
    improved = True
    while improved and time.time() < deadline - 20:
        improved = False
        best_gain = 0
        best_prime = None
        best_residue = None
        best_new_L = current_L

        for p in sorted_primes:
            if p in selected_primes:
                continue
            cost = Fraction(1, p)
            if budget_used + cost > max_budget:
                continue

            # Try all residues for this prime
            for r in range(p):
                # Compute what new integers would be covered
                if r == 0:
                    start = p
                else:
                    start = r
                if start > N_bound:
                    continue

                # Compute the new contiguous coverage
                # We need to check: with current covered_set plus the new numbers from (p, r),
                # what's the new contiguous L?

                # This is expensive. Let's approximate:
                # The new L is at least current_L. It extends if the numbers current_L+1, current_L+2, ...
                # are covered.

                # Let's compute it properly but efficiently.
                # First, check if current_L + 1 is covered by the new (p, r) or already in covered_set.

                # Actually, let's just compute the full new L for now, with a time guard.
                if time.time() > deadline - 30:
                    break

                new_covered = set(covered_set)
                for n in range(start, N_bound + 1, p):
                    new_covered.add(n)

                new_L = compute_L_from_set(new_covered, N_bound)

                if new_L > best_new_L:
                    best_new_L = new_L
                    best_gain = new_L - current_L
                    best_prime = p
                    best_residue = r

            if time.time() > deadline - 30:
                break

        if best_prime is not None and best_gain > 0:
            selected.append((best_prime, best_residue))
            selected_primes.add(best_prime)
            budget_used += Fraction(1, best_prime)
            update_covered(best_prime, best_residue, N_bound)
            current_L = best_new_L
            improved = True

    # Now do local search: try to improve by changing residues or swapping primes
    # This is where we can potentially do much better.

    # Local search 1: For each selected prime, try changing its residue to see if it improves L
    for _ in range(50):  # Up to 50 iterations of local search
        if time.time() > deadline - 10:
            break

        improved_ls = False
        for idx, (p, r) in enumerate(selected):
            if time.time() > deadline - 10:
                break

            old_covered = set(covered_set)
            # Remove the contribution of (p, r)
            # We need to rebuild covered_set without (p, r)
            # This is expensive. Let's use a different approach.

            # Actually, let's just try all residues for this prime and see which gives the best L.
            best_r = r
            best_L_ls = current_L

            for r_new in range(p):
                if r_new == r:
                    continue
                # Compute new L with (p, r_new) instead of (p, r)
                # This requires rebuilding covered_set, which is expensive.
                # Let's skip this for now and do a simpler local search.
                pass

        # Skip detailed local search for now due to time constraints
        break

    # Final computation of L
    final_L = compute_L_from_set(covered_set, 100000)  # Cap at 10^5 as per task

    # Return the result
    return [[p, r] for p, r in selected]
# EVOLVE-BLOCK-END
