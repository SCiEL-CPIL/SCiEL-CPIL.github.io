"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 40, 'maximum_size': 18}

CANDIDATES = list(range(2, P['n'] + 1))
N_CANDS = len(CANDIDATES)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    # Precompute primes (indices) for initial seed
    def is_prime(x):
        if x < 2:
            return False
        for d in range(2, int(x**0.5) + 1):
            if x % d == 0:
                return False
        return True
    prime_indices = [i for i, x in enumerate(CANDIDATES) if is_prime(x)]

    best_set = set(prime_indices)
    best_size = len(best_set)

    def greedy_fill(S, products, deadline):
        """Greedily add elements to S (set of indices) maintaining products set."""
        added_any = True
        while added_any and time.time() < deadline:
            added_any = False
            candidates = [c for c in range(N_CANDS) if c not in S]
            random.shuffle(candidates)
            for c in candidates:
                val = CANDIDATES[c]
                # Check if adding val creates a collision
                # New products = products ∪ {p * val for p in products}
                # Collision if any p*val already in products, or if p*val == q*val for p≠q (impossible if val>0)
                # Actually collision iff ∃ p in products such that p*val in products
                # i.e., ∃ p,q in products with p*val = q  =>  q/val = p  =>  q is a multiple of val and q/val in products
                # Simpler: compute new_prods = {p*val for p in products}, check overlap with products
                new_prods = {p * val for p in products}
                if not new_prods & products:
                    S.add(c)
                    products |= new_prods
                    added_any = True
                    break
        return S, products

    for restart in range(200):
        if time.time() > deadline:
            break

        # Start from primes (or a random subset for diversity)
        if restart % 3 == 0:
            S = set(prime_indices)
        else:
            # Random subset of primes, size 6-10
            k = random.randint(6, min(10, len(prime_indices)))
            S = set(random.sample(prime_indices, k))

        # Build products set
        products = {1}
        for idx in S:
            val = CANDIDATES[idx]
            products = products | {p * val for p in products}

        # Greedy fill
        S, products = greedy_fill(S, products, deadline)

        if len(S) > best_size:
            best_set = S
            best_size = len(S)
            if best_size == P['maximum_size']:
                break

        # Swap phase: try to improve by removing one and adding one or more
        if len(S) < P['maximum_size'] and time.time() < deadline - 3:
            for _ in range(30):
                if time.time() > deadline:
                    break
                if not S:
                    break
                to_remove = random.choice(tuple(S))
                S.remove(to_remove)

                # Rebuild products after removal
                products = {1}
                for idx in S:
                    val = CANDIDATES[idx]
                    products = products | {p * val for p in products}

                # Try to add one candidate
                candidates = [c for c in range(N_CANDS) if c not in S]
                random.shuffle(candidates)
                found = False
                for c in candidates:
                    val = CANDIDATES[c]
                    new_prods = {p * val for p in products}
                    if not new_prods & products:
                        S.add(c)
                        products |= new_prods
                        found = True
                        break

                if found:
                    # Greedy fill again
                    S, products = greedy_fill(S, products, deadline)
                    if len(S) > best_size:
                        best_set = S
                        best_size = len(S)
                        if best_size == P['maximum_size']:
                            break
                else:
                    # Restore
                    S.add(to_remove)
                    products = {1}
                    for idx in S:
                        val = CANDIDATES[idx]
                        products = products | {p * val for p in products}

        if best_size == P['maximum_size']:
            break

    return [CANDIDATES[i] for i in best_set]

# EVOLVE-BLOCK-END
