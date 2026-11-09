"""Combined program: fast incremental construction + iterated local search."""
# EVOLVE-BLOCK-START
import time
import random
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    # Precompute prime factorization vectors for 2..128
    # Each number maps to a tuple of prime exponents (fixed prime order)
    PRIMES = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113]
    P2 = {p: i for i, p in enumerate(PRIMES)}
    def factor_vec(n):
        v = [0] * len(PRIMES)
        for i, p in enumerate(PRIMES):
            if p * p > n:
                break
            while n % p == 0:
                v[i] += 1
                n //= p
        if n > 1:
            if n in P2:
                v[P2[n]] += 1
        return tuple(v)

    FVEC = {}
    for x in range(2, 129):
        FVEC[x] = factor_vec(x)

    def vec_add(a, b):
        return tuple(x + y for x, y in zip(a, b))

    def vec_sub(a, b):
        return tuple(x - y for x, y in zip(a, b))

    # Fast incremental validity check:
    # Given a sorted sequence `seq`, compute all block products (as exponent vectors)
    # and return (is_valid, set_of_bad_indices, prods_dict)
    def check_valid(seq):
        n = len(seq)
        if n == 0:
            return True, set(), {}
        prods = {}
        bad = set()
        for i in range(n):
            v = (0,) * len(PRIMES)
            for j in range(i, n):
                v = vec_add(v, FVEC[seq[j]])
                if v in prods:
                    pi, pj = prods[v]
                    for k in range(pi, pj + 1):
                        bad.add(k)
                    for k in range(i, j + 1):
                        bad.add(k)
                else:
                    prods[v] = (i, j)
        return len(bad) == 0, bad, prods

    # Greedy initialization: include x if it doesn't create collisions
    def greedy_init():
        seq = []
        prods = set()
        for x in range(2, 129):
            new_prods = []
            valid = True
            v = FVEC[x]
            new_prods.append(v)
            if v in prods:
                valid = False
            else:
                for i in range(len(seq) - 1, -1, -1):
                    v = vec_add(v, FVEC[seq[i]])
                    new_prods.append(v)
                    if v in prods:
                        valid = False
                        break
            if valid:
                seq.append(x)
                prods.update(new_prods)
        return seq

    best_seq = greedy_init()
    best_len = len(best_seq)

    # Iterated Local Search
    # Move types:
    # 1. Remove an element from seq
    # 2. Insert an element from removed_set into seq
    # 3. Swap: remove one, insert another
    def local_search(start_seq, time_limit):
        current_seq = start_seq[:]
        current_len = len(current_seq)
        removed = [x for x in range(2, 129) if x not in set(current_seq)]

        best_local = current_seq[:]
        best_local_len = current_len

        # Check initial validity
        is_val, _, _ = check_valid(current_seq)

        iter_count = 0
        while time.time() < time_limit:
            iter_count += 1
            is_val, bad, _ = check_valid(current_seq)

            if is_val:
                # Try to insert elements from removed
                improved = False
                # Try each removed element (sorted by value for determinism, but randomize order)
                candidates = removed[:]
                random.shuffle(candidates)
                for val in candidates:
                    if time.time() > time_limit:
                        break
                    pos = bisect.bisect_left(current_seq, val)
                    candidate = current_seq[:pos] + [val] + current_seq[pos:]
                    cand_val, _, _ = check_valid(candidate)
                    if cand_val:
                        current_seq = candidate
                        current_len += 1
                        removed.remove(val)
                        improved = True
                        if current_len > best_local_len:
                            best_local_len = current_len
                            best_local = current_seq[:]
                        break  # restart insertion search
                if not removed:
                    break
            else:
                # Remove an element involved in a collision
                if bad:
                    # Pick a random bad index
                    idx = random.choice(list(bad))
                    val = current_seq[idx]
                    current_seq.pop(idx)
                    current_len -= 1
                    removed.append(val)
                    removed.sort()
                else:
                    # Shouldn't happen, but break to be safe
                    break

        if current_len > best_local_len:
            best_local = current_seq[:]
            best_local_len = current_len
        return best_local, best_local_len

    # Iterated Local Search with kicks
    while time.time() < deadline:
        # Run local search
        ls_seq, ls_len = local_search(best_seq, time.time() + 30.0)
        if ls_len > best_len:
            best_seq = ls_seq
            best_len = ls_len

        # If we still have time, do a kick: perturb and re-search
        if time.time() >= deadline - 5.0:
            break

        # Kick: remove a random element, then re-run local search
        if len(best_seq) > 2:
            kick_seq = best_seq[:]
            kick_idx = random.randrange(len(kick_seq))
            kick_val = kick_seq.pop(kick_idx)
            kick_seq.sort()
            # Try to re-insert kick_val or others
            kick_seq2, kick_len2 = local_search(kick_seq, time.time() + 15.0)
            if kick_len2 > best_len:
                best_seq = kick_seq2
                best_len = kick_len2
            # Also try a different kick: swap two elements
            if time.time() < deadline - 10.0 and len(best_seq) > 2:
                swap_seq = best_seq[:]
                i1 = random.randrange(len(swap_seq))
                i2 = random.randrange(len(swap_seq) - 1)
                if i2 >= i1:
                    i2 += 1
                swap_seq[i1], swap_seq[i2] = swap_seq[i2], swap_seq[i1]
                swap_seq.sort()
                swap_seq2, swap_len2 = local_search(swap_seq, time.time() + 15.0)
                if swap_len2 > best_len:
                    best_seq = swap_seq2
                    best_len = swap_len2

    # Final: try greedy repair on best_seq to ensure validity
    # (best_seq should already be valid, but double-check)
    is_val, _, _ = check_valid(best_seq)
    if not is_val:
        # Fallback to greedy
        best_seq = greedy_init()
        best_len = len(best_seq)

    return best_seq
# EVOLVE-BLOCK-END
