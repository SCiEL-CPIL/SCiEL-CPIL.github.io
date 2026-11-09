"""Search-based construction of dense increasing sequences with unique block products."""
import time
import random
import itertools
from collections import defaultdict

# Precompute prime factorization for numbers 2..128
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
NUM_PRIMES = len(PRIMES)
PRIME_IDX = {p: i for i, p in enumerate(PRIMES)}

# Factorization as tuple of exponents
def factorize(n):
    exps = [0] * NUM_PRIMES
    for p in PRIMES:
        while n % p == 0:
            exps[PRIME_IDX[p]] += 1
            n //= p
    return tuple(exps)

# Precompute factorizations
FACT = {}
for n in range(2, 129):
    FACT[n] = factorize(n)

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    best_seq = []
    best_len = 0

    def get_prefix_exp_vectors(seq):
        """Compute prefix exponent vectors: prefix[i] = product of seq[0..i-1] as exponent tuple.
        prefix[0] = (0,0,...,0), prefix[i+1] = prefix[i] + FACT[seq[i]]
        """
        n = len(seq)
        prefix = [tuple([0] * NUM_PRIMES)] * (n + 1)
        cur = [0] * NUM_PRIMES
        for i in range(n):
            f = FACT[seq[i]]
            for j in range(NUM_PRIMES):
                cur[j] += f[j]
            prefix[i + 1] = tuple(cur)
        return prefix

    def get_all_prods_as_tuples(seq):
        """Get set of all consecutive block products as exponent tuples."""
        prefix = get_prefix_exp_vectors(seq)
        n = len(seq)
        prods = set()
        for i in range(n):
            for j in range(i + 1, n + 1):
                # product of seq[i..j-1] = prefix[j] - prefix[i]
                pi = prefix[i]
                pj = prefix[j]
                vec = tuple(pj[k] - pi[k] for k in range(NUM_PRIMES))
                prods.add(vec)
        return prods

    def can_append_fast(seq, prods, new_val):
        """Check if new_val can be appended to seq. prods is the set of block-product exponent tuples for seq.
        Returns (ok, new_prods) where new_prods is the updated set if ok.
        """
        if new_val <= seq[-1] if seq else False:
            return False, prods
        if new_val > 128:
            return False, prods
        
        n = len(seq)
        prefix = get_prefix_exp_vectors(seq)
        f_new = FACT[new_val]
        
        # New blocks are suffixes: seq[i..n-1] + [new_val] for i in 0..n-1, plus [new_val]
        # Product of seq[i..n-1] = prefix[n] - prefix[i]
        # Product of new block = (prefix[n] - prefix[i]) + f_new
        new_prods = set()
        
        # The singleton [new_val]
        new_prods.add(f_new)
        
        for i in range(n):
            pi = prefix[i]
            pn = prefix[n]
            # suffix product vector
            sv = tuple(pn[k] - pi[k] for k in range(NUM_PRIMES))
            # new block product = sv + f_new
            bp = tuple(sv[k] + f_new[k] for k in range(NUM_PRIMES))
            if bp in prods or bp in new_prods:
                return False, prods
            new_prods.add(bp)
        
        return True, prods | new_prods

    def can_insert_fast(seq, prods, pos, x):
        """Check if inserting x at position pos in seq keeps all block products distinct.
        Returns (ok, new_prods) if ok.
        """
        if pos > 0 and x <= seq[pos - 1]:
            return False, prods
        if pos < len(seq) and x >= seq[pos]:
            return False, prods
        if x < 2 or x > 128:
            return False, prods
        
        n = len(seq)
        # Build test sequence
        test_seq = seq[:pos] + [x] + seq[pos:]
        m = len(test_seq)
        
        # We need to check all blocks in test_seq that include position pos (the new x)
        # and verify they don't conflict with existing prods or each other
        prefix = get_prefix_exp_vectors(test_seq)
        
        new_prods = set()
        f_x = FACT[x]
        
        for i in range(pos + 1):  # i from 0 to pos (start of block, must be <= pos)
            for j in range(pos, m):  # j from pos to m-1 (end of block, must be >= pos)
                # block is test_seq[i..j]
                pi = prefix[i]
                pj = prefix[j + 1]
                vec = tuple(pj[k] - pi[k] for k in range(NUM_PRIMES))
                if vec in prods or vec in new_prods:
                    return False, prods
                new_prods.add(vec)
        
        return True, prods | new_prods

    # Strategy 1: Greedy with exponent-vector fast checks
    baseline = []
    baseline_prods = set()
    for x in range(2, 129):
        if not baseline:
            baseline.append(x)
            baseline_prods.add(FACT[x])
            continue
        ok, new_prods = can_append_fast(baseline, baseline_prods, x)
        if ok:
            baseline.append(x)
            baseline_prods = new_prods

    if len(baseline) > best_len:
        best_len = len(baseline)
        best_seq = baseline[:]

    # Strategy 2: Randomized greedy with fast checks
    random.seed(42)

    for attempt in range(500):
        if time.time() > deadline:
            break

        candidates = list(range(2, 129))
        random.shuffle(candidates)

        seq = []
        seq_prods = set()
        for x in candidates:
            if time.time() > deadline:
                break
            if not seq:
                seq.append(x)
                seq_prods.add(FACT[x])
                continue
            if x <= seq[-1]:
                continue
            ok, new_prods = can_append_fast(seq, seq_prods, x)
            if ok:
                seq.append(x)
                seq_prods = new_prods

        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq[:]

    # Strategy 3: Local search / hill climbing with fast insert checks
    if best_seq:
        best_prods = get_all_prods_as_tuples(best_seq)
        in_seq = set(best_seq)
        missing = [x for x in range(2, 129) if x not in in_seq]

        improved = True
        while improved and time.time() < deadline:
            improved = False
            for x in missing[:]:
                if time.time() > deadline:
                    break
                for pos in range(len(best_seq) + 1):
                    if time.time() > deadline:
                        break
                    ok, new_prods = can_insert_fast(best_seq, best_prods, pos, x)
                    if ok:
                        best_seq = best_seq[:pos] + [x] + best_seq[pos:]
                        best_prods = new_prods
                        best_len = len(best_seq)
                        if x in missing:
                            missing.remove(x)
                        improved = True
                        break
                if improved:
                    break

    # Strategy 4: Structured candidate sets
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    prime_powers = []
    for p in primes:
        v = p
        while v <= 128:
            prime_powers.append(v)
            v *= p
    prime_powers = sorted(set(prime_powers))

    for variant in range(50):
        if time.time() > deadline:
            break

        if variant == 0:
            candidates = primes
        elif variant == 1:
            candidates = prime_powers
        elif variant == 2:
            candidates = random.sample(range(2, 129), 40)
            candidates.sort()
        else:
            candidates = list(range(2, 129))
            random.shuffle(candidates)

        seq = []
        seq_prods = set()
        for x in candidates:
            if time.time() > deadline:
                break
            if not seq:
                seq.append(x)
                seq_prods.add(FACT[x])
                continue
            if x <= seq[-1]:
                continue
            ok, new_prods = can_append_fast(seq, seq_prods, x)
            if ok:
                seq.append(x)
                seq_prods = new_prods

        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq[:]

    # Strategy 5: Beam search with fast checks
    if time.time() < deadline - 5:
        beam_width = 10
        beam = [(baseline, get_all_prods_as_tuples(baseline))]

        for round_num in range(15):
            if time.time() > deadline:
                break
            new_beam = []
            for seq, prods in beam:
                if time.time() > deadline:
                    break
                last = seq[-1] if seq else 1
                for x in range(last + 1, 129):
                    if time.time() > deadline:
                        break
                    ok, new_prods = can_append_fast(seq, prods, x)
                    if ok:
                        new_beam.append((seq + [x], new_prods))

            if not new_beam:
                break

            new_beam.sort(key=lambda x: (-len(x[0]), x[0]))
            beam = new_beam[:beam_width]

            if beam[0][0] and len(beam[0][0]) > best_len:
                best_len = len(beam[0][0])
                best_seq = beam[0][0][:]

    return best_seq
