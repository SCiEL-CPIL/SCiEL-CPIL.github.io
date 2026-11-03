"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import numpy as np
from collections import defaultdict, deque
import time

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius radius.
    Uses structured linear initialization + greedy fill + local search.
    """
    if n == 0:
        return [[]]

    total_words = q ** n

    def int_to_word(val, n, q):
        word = []
        for _ in range(n):
            word.append(val % q)
            val //= q
        return word

    def word_to_int(word, q):
        val = 0
        for i in range(len(word) - 1, -1, -1):
            val = val * q + word[i]
        return val

    # Precompute neighbors as sets. Fast path for radius=1.
    neighbors = [None] * total_words
    if radius == 1:
        for w in range(total_words):
            neigh = [w]
            word = int_to_word(w, n, q)
            for pos in range(n):
                cur = word[pos]
                for val in range(q):
                    if val == cur:
                        continue
                    word[pos] = val
                    neigh.append(word_to_int(word, q))
                word[pos] = cur
            neighbors[w] = set(neigh)
    else:
        # Generic neighbor generation for other radii (not optimized for this specific task but safe)
        for w in range(total_words):
            neigh = [w]
            word = int_to_word(w, n, q)
            if radius >= 1:
                for pos in range(n):
                    cur = word[pos]
                    for val in range(q):
                        if val == cur:
                            continue
                        word[pos] = val
                        neigh.append(word_to_int(word, q))
                    word[pos] = cur
            if radius >= 2:
                word = int_to_word(w, n, q)
                for pos1 in range(n):
                    c1 = word[pos1]
                    for val1 in range(q):
                        if val1 == c1:
                            continue
                        word[pos1] = val1
                        for pos2 in range(pos1 + 1, n):
                            c2 = word[pos2]
                            for val2 in range(q):
                                if val2 == c2:
                                    continue
                                word[pos2] = val2
                                neigh.append(word_to_int(word, q))
                            word[pos2] = c2
                    word[pos1] = c1
            neighbors[w] = set(neigh)

    # Precompute neighbor lists for fast iteration (list form)
    neigh_lists = [list(s) for s in neighbors]

    def linear_init(dim):
        """
        Initialize with a random linear code of given dimension.
        """
        # Generate random basis
        basis = []
        for _ in range(dim):
            while True:
                v = [random.randint(0, q-1) for _ in range(n)]
                # Check independence (simple Gaussian elimination logic or just check span for small dims)
                # For small n and q, simple check is fine
                independent = True
                # Convert to numpy for easy linear algebra if needed, but manual is safer for small ints
                # We'll just check if v is in the span of existing basis
                # Span check: try to subtract multiples of basis vectors
                test_v = v[:]
                for b in basis:
                    # Find first non-zero in b
                    first_non_zero = -1
                    for i in range(n):
                        if b[i] != 0:
                            first_non_zero = i
                            break
                    if first_non_zero == -1:
                        continue
                    if test_v[first_non_zero] != 0:
                        factor = test_v[first_non_zero] * pow(b[first_non_zero], q-2, q) # Inverse in F_q
                        for i in range(n):
                            test_v[i] = (test_v[i] - factor * b[i]) % q
                if all(x == 0 for x in test_v):
                    independent = False
                if independent:
                    basis.append(v)
                    break
        
        # Generate all codewords in the span
        codewords = set()
        # Recursive generation or iterative
        # Since dim is small (e.g., 5), 3^5 = 243 is small
        def gen(prefix, remaining_basis):
            if not remaining_basis:
                codewords.add(tuple(prefix))
                return
            b = remaining_basis[0]
            rest = remaining_basis[1:]
            for coeff in range(q):
                new_prefix = [(p + coeff * bi) % q for p, bi in zip(prefix, b)]
                gen(new_prefix, rest)
        
        gen([0]*n, basis)
        return [word_to_int(list(cw), q) for cw in codewords]

    def greedy_fill(codewords):
        """
        Greedily add codewords to cover all uncovered words.
        """
        covered = [False] * total_words
        for cw in codewords:
            for u in neigh_lists[cw]:
                covered[u] = True
        
        uncovered = set(range(total_words))
        for u in range(total_words):
            if covered[u]:
                uncovered.discard(u)
        
        while uncovered:
            # Pick a random uncovered word to focus on
            u = random.choice(list(uncovered))
            
            # Candidates are neighbors of u
            candidates = neighbors[u]
            
            best_cw = -1
            best_new_cov = -1
            for cw in candidates:
                new_cov = 0
                for v in neigh_lists[cw]:
                    if not covered[v]:
                        new_cov += 1
                if new_cov > best_new_cov:
                    best_new_cov = new_cov
                    best_cw = cw
                    if new_cov == len(uncovered):
                        break
            
            if best_cw == -1:
                best_cw = u
            
            codewords.append(best_cw)
            for v in neigh_lists[best_cw]:
                if not covered[v]:
                    covered[v] = True
                    uncovered.discard(v)
        
        return codewords

    def build_coverage(codewords):
        coverage_count = [0] * total_words
        for cw in codewords:
            for u in neigh_lists[cw]:
                coverage_count[u] += 1
        return coverage_count

    def remove_redundant(codewords, coverage_count, cw_set):
        removed_any = False
        changed = True
        while changed:
            changed = False
            indices = list(range(len(codewords)))
            random.shuffle(indices)
            for i in indices:
                cw = codewords[i]
                can_remove = True
                for u in neigh_lists[cw]:
                    if coverage_count[u] == 1:
                        can_remove = False
                        break
                if can_remove:
                    for u in neigh_lists[cw]:
                        coverage_count[u] -= 1
                    codewords.pop(i)
                    cw_set.discard(cw)
                    changed = True
                    removed_any = True
                    break
        return removed_any

    def try_best_swap(codewords, coverage_count, cw_set):
        """Try to swap one codeword for a better one (maximizes redundant coverage)."""
        if not codewords:
            return False
        idx = random.randrange(len(codewords))
        old_cw = codewords[idx]
        would_uncover = [u for u in neigh_lists[old_cw] if coverage_count[u] == 1]

        if not would_uncover:
            for u in neigh_lists[old_cw]:
                coverage_count[u] -= 1
            codewords.pop(idx)
            cw_set.discard(old_cw)
            return True

        # Candidate pool: intersection of neighbors of all would_uncover words
        candidates = None
        for u in would_uncover:
            if candidates is None:
                candidates = set(neighbors[u])
            else:
                candidates.intersection_update(neighbors[u])
                if not candidates:
                    break
        if not candidates:
            return False
        candidates.discard(old_cw)
        if not candidates:
            return False

        # Pick candidate maximizing redundant coverage (words with coverage>=2 it will cover)
        best_cw = None
        best_redundancy = -1
        cand_list = list(candidates)
        # If too many, sample
        if len(cand_list) > 40:
            cand_list = random.sample(cand_list, 40)
        for new_cw in cand_list:
            redundancy = 0
            for u in neigh_lists[new_cw]:
                if coverage_count[u] >= 2:
                    redundancy += 1
            if redundancy > best_redundancy:
                best_redundancy = redundancy
                best_cw = new_cw

        if best_cw is None:
            return False

        for u in neigh_lists[old_cw]:
            coverage_count[u] -= 1
        for u in neigh_lists[best_cw]:
            coverage_count[u] += 1
        codewords[idx] = best_cw
        cw_set.discard(old_cw)
        cw_set.add(best_cw)
        return True

    def optimize(codewords, time_limit):
        coverage_count = build_coverage(codewords)
        cw_set = set(codewords)
        remove_redundant(codewords, coverage_count, cw_set)

        start_time = time.time()
        no_improve = 0

        while time.time() - start_time < time_limit:
            # Remove redundant
            if remove_redundant(codewords, coverage_count, cw_set):
                no_improve = 0
                continue

            # Try a best-swap
            if try_best_swap(codewords, coverage_count, cw_set):
                no_improve = 0
                # After a swap, try to remove redundant again
                remove_redundant(codewords, coverage_count, cw_set)
                continue

            # No simple improvement: perturb via multi-swap (ILS) to escape local minima
            no_improve += 1
            if no_improve >= 3 and codewords:
                # Perturb: do k random valid swaps
                k = random.randint(2, 4)
                for _ in range(k):
                    if not codewords:
                        break
                    idx = random.randrange(len(codewords))
                    old_cw = codewords[idx]
                    would_uncover = [u for u in neigh_lists[old_cw] if coverage_count[u] == 1]
                    if not would_uncover:
                        for u in neigh_lists[old_cw]:
                            coverage_count[u] -= 1
                        codewords.pop(idx)
                        cw_set.discard(old_cw)
                        continue
                    candidates = None
                    for u in would_uncover:
                        if candidates is None:
                            candidates = set(neighbors[u])
                        else:
                            candidates.intersection_update(neighbors[u])
                            if not candidates:
                                break
                    if not candidates:
                        continue
                    candidates.discard(old_cw)
                    if not candidates:
                        continue
                    new_cw = random.choice(list(candidates))
                    for u in neigh_lists[old_cw]:
                        coverage_count[u] -= 1
                    for u in neigh_lists[new_cw]:
                        coverage_count[u] += 1
                    codewords[idx] = new_cw
                    cw_set.discard(old_cw)
                    cw_set.add(new_cw)
                # After perturbation, run a short local search to recover
                for _ in range(50):
                    if time.time() - start_time > time_limit:
                        break
                    if remove_redundant(codewords, coverage_count, cw_set):
                        continue
                    if try_best_swap(codewords, coverage_count, cw_set):
                        remove_redundant(codewords, coverage_count, cw_set)
                        continue
                    break
                no_improve = 0

            if time.time() - start_time > time_limit - 1:
                break

        remove_redundant(codewords, coverage_count, cw_set)
        return codewords

    # Multiple restarts
    start_time = time.time()
    time_limit = 115
    best_codewords = []

    # Determine dimension for linear init
    # For q=3, n=8, target ~486. 3^5=243. 3^6=729.
    # Start with dim 5 (243 words) and fill.
    dim = 5
    
    while time.time() - start_time < time_limit:
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed
        if remaining < 3:
            break
        
        # Allocate time: Init is fast, fill is fast, optimize takes rest
        opt_time = max(1.0, remaining - 2)
        
        # 1. Structured Init
        cw = linear_init(dim)
        
        # 2. Greedy Fill
        cw = greedy_fill(cw)
        
        # 3. Optimize
        cw = optimize(cw, opt_time)

        if len(cw) < len(best_codewords) or not best_codewords:
            best_codewords = cw

        if time.time() - start_time > time_limit - 5:
            break

    # Fallback: if linear init failed or was too slow, try random greedy
    if not best_codewords:
        cw = []
        cw = greedy_fill(cw)
        cw = optimize(cw, 10)
        best_codewords = cw

    result = [int_to_word(cw, n, q) for cw in best_codewords]
    return result
# EVOLVE-BLOCK-END
