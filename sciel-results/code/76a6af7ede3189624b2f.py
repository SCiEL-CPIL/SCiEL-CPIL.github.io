import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    N = 512
    candidates = list(range(1, N))
    weights = {a: 1.0 / (N - a) for a in candidates}

    # Sieve primes up to 511
    is_prime = [True] * N
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, N, i):
                is_prime[j] = False
    primes = [p for p in range(2, N) if is_prime[p]]

    # For each candidate, set of prime factors
    prime_factors = {}
    for a in range(1, N):
        x = a
        pf = set()
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                pf.add(p)
                while x % p == 0:
                    x //= p
        if x > 1:
            pf.add(x)
        prime_factors[a] = pf

    # For each prime, list of multiples (candidates divisible by p)
    prime_multiples = {}
    for p in primes:
        multiples = [a for a in candidates if a % p == 0]
        prime_multiples[p] = multiples

    # Conflict: two candidates conflict if they share a prime factor
    # Build adjacency
    adj = {a: set() for a in candidates}
    for a in candidates:
        for b in candidates:
            if a >= b:
                continue
            if prime_factors[a] & prime_factors[b]:
                adj[a].add(b)
                adj[b].add(a)

    def greedy_select(order_key):
        """Greedy: repeatedly pick best available node by order_key, remove conflicts."""
        avail = set(candidates)
        selected = []
        score = 0.0
        while avail:
            best = max(avail, key=order_key)
            selected.append(best)
            score += weights[best]
            avail.discard(best)
            for nb in adj[best]:
                avail.discard(nb)
        return score, selected

    # Multiple greedy strategies
    best_score = 0.0
    best_set = set()

    # Strategy 1: max weight
    s, sel = greedy_select(lambda x: weights[x])
    if s > best_score:
        best_score, best_set = s, set(sel)

    # Strategy 2: weight / (degree+1)
    s, sel = greedy_select(lambda x: weights[x] / (len(adj[x]) + 1))
    if s > best_score:
        best_score, best_set = s, set(sel)

    # Strategy 3: weight / number of prime factors (fewer factors = less conflict)
    s, sel = greedy_select(lambda x: weights[x] / (len(prime_factors[x]) + 1))
    if s > best_score:
        best_score, best_set = s, set(sel)

    # Strategy 4: weight * (number of primes not dividing it) -- coprime-friendly
    def coprime_count(a):
        return sum(1 for p in primes if a % p != 0)
    s, sel = greedy_select(lambda x: weights[x] * coprime_count(x))
    if s > best_score:
        best_score, best_set = s, set(sel)

    # Strategy 5: pure weight but with tie-break by fewer conflicts
    s, sel = greedy_select(lambda x: (weights[x], -len(adj[x])))
    if s > best_score:
        best_score, best_set = s, set(sel)

    # Local search: try to improve by swapping
    def local_search(current_set):
        current_score = sum(weights[a] for a in current_set)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            cur = set(current_set)
            # Try adding a node
            for a in candidates:
                if a in cur:
                    continue
                if any(a % p == 0 and b % p == 0 for p in primes for b in cur):
                    continue
                # Check pairwise coprime
                ok = True
                for b in cur:
                    if prime_factors[a] & prime_factors[b]:
                        ok = False
                        break
                if ok:
                    new_set = cur | {a}
                    new_score = current_score + weights[a]
                    if new_score > current_score + 1e-15:
                        current_set = new_set
                        current_score = new_score
                        improved = True
                        break
            # Try removing a node to allow better additions
            for a in list(current_set):
                trial = current_set - {a}
                trial_score = current_score - weights[a]
                # Try to add multiple nodes
                for b in candidates:
                    if b in trial:
                        continue
                    ok = True
                    for c in trial:
                        if prime_factors[b] & prime_factors[c]:
                            ok = False
                            break
                    if ok:
                        new_score = trial_score + weights[b]
                        if new_score > current_score + 1e-15:
                            current_set = trial | {b}
                            current_score = new_score
                            improved = True
                            break
                    if improved:
                        break
        return current_score, current_set

    best_score, best_set = local_search(best_set)

    # Branch and bound with prime-based upper bound
    # Upper bound: for each prime p, at most one multiple of p can be in the set.
    # So sum over primes of max weight among available multiples of p is an upper bound.
    # But this overcounts numbers that are coprime to all primes (only 1) and double-counts.
    # Actually the correct bound: partition candidates by their "primary" prime (smallest prime factor).
    # Each number has a smallest prime factor (except 1). Numbers with the same smallest
    # prime factor are NOT necessarily conflicting... wait, they ARE because they share that prime.
    # So grouping by smallest prime factor gives valid groups where at most one per group.
    # Plus 1 is its own group.
    # This is a valid upper bound!

    # Group by smallest prime factor
    def smallest_prime_factor(a):
        if a == 1:
            return 1  # special
        for p in primes:
            if a % p == 0:
                return p
        return a  # a is prime

    group_by_spf = {}
    for a in candidates:
        spf = smallest_prime_factor(a)
        if spf not in group_by_spf:
            group_by_spf[spf] = []
        group_by_spf[spf].append(a)

    # Precompute group weights sorted
    group_weights = {}
    for g, members in group_by_spf.items():
        group_weights[g] = max(weights[a] for a in members)

    def upper_bound(avail):
        """Upper bound: for each group (by smallest prime factor), at most one member can be picked."""
        # For each group, find the max weight among available members
        bound = 0.0
        for g, members in group_by_spf.items():
            max_w = 0.0
            for a in members:
                if a in avail:
                    if weights[a] > max_w:
                        max_w = weights[a]
            bound += max_w
        return bound

    # Branch and bound
    def branch_and_bound(avail, current_score, current_set):
        nonlocal best_score, best_set

        if time.time() > deadline:
            return

        if not avail:
            if current_score > best_score:
                best_score = current_score
                best_set = set(current_set)
            return

        # Upper bound
        ub = current_score + upper_bound(avail)
        if ub <= best_score + 1e-15:
            return

        # Pick a node to branch on: highest weight
        v = max(avail, key=lambda x: weights[x])

        # Option 1: Include v
        new_avail_1 = set(avail)
        new_avail_1.discard(v)
        for nb in adj[v]:
            new_avail_1.discard(nb)
        branch_and_bound(new_avail_1, current_score + weights[v], current_set | {v})

        # Option 2: Exclude v
        new_avail_2 = set(avail)
        new_avail_2.discard(v)
        branch_and_bound(new_avail_2, current_score, current_set)

    # Run branch and bound with the current best as initial bound
    initial_avail = set(candidates)
    branch_and_bound(initial_avail, 0.0, set())

    # Final local search polish
    final_score, final_set = local_search(best_set)
    if final_score > best_score:
        best_score = final_score
        best_set = final_set

    return list(best_set)

if __name__ == "__main__":
    res = construct()
    print(f"Selected {len(res)} numbers")
    score = sum(1.0 / (512 - a) for a in res)
    print(f"Score: {score}")
    # Verify
    valid = True
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            if math.gcd(res[i], res[j]) != 1:
                print(f"ERROR: {res[i]} and {res[j]} not coprime")
                valid = False
    print("Valid" if valid else "INVALID")
