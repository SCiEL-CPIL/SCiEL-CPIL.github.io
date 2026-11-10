import time
import math

def construct(n):
    start_time = time.time()
    deadline = 140.0  # Safety margin

    word = generate_classic(n)
    target_count = math.factorial(n)

    # Factorials for Lehmer-code ranking
    fact = [1] * (n + 1)
    for i in range(1, n + 1):
        fact[i] = fact[i - 1] * i

    # Precompute rank for all permutations if n is small enough
    # For n=7, 5040 permutations; we can precompute a lookup dict
    rank_cache = {}
    
    def perm_rank(p):
        key = tuple(p)
        r = rank_cache.get(key)
        if r is not None:
            return r
        rank = 0
        used = [False] * (n + 1)
        for i in range(n):
            count = 0
            for j in range(1, p[i]):
                if not used[j]:
                    count += 1
            rank += count * fact[n - 1 - i]
            used[p[i]] = True
        rank_cache[key] = rank
        return rank

    def build_state(w):
        Lw = len(w)
        nr = Lw - n + 1
        rk = [0] * nr
        cov = [0] * target_count
        for i in range(nr):
            r = perm_rank(w[i:i + n])
            rk[i] = r
            cov[r] += 1
        return rk, cov

    ranks, cov = build_state(word)
    best_len = len(word)

    # Track critical permutations (cov == 1) for faster pruning
    def try_deletion(w, i, rk, cov):
        Lw = len(w)
        nr = Lw - n + 1
        if i < 0 or i >= Lw:
            return None
        
        lo = max(0, i - (n - 1))
        hi = i - 1
        old_lo = max(0, i - (n - 1))
        old_hi = i
        
        old_ranks = [rk[k] for k in range(old_lo, old_hi + 1)]
        
        # Quick check: if any old rank has cov == 1 and it's not in the new complex windows,
        # we need to verify carefully. But let's just do the full check.
        
        # Decrement old ranks
        for r in old_ranks:
            cov[r] -= 1
            if cov[r] < 0:
                for r2 in old_ranks:
                    cov[r2] += 1
                return None

        # Compute new ranks for complex windows
        new_ranks_complex = []
        for j in range(lo, hi + 1):
            part1 = w[j:i]
            part2 = w[i+1:j+n+1]
            virtual_window = part1 + part2
            r = perm_rank(virtual_window)
            new_ranks_complex.append(r)
            cov[r] += 1

        # Check if any old rank became 0
        for r in old_ranks:
            if cov[r] == 0:
                for r2 in new_ranks_complex:
                    cov[r2] -= 1
                for r3 in old_ranks:
                    cov[r3] += 1
                return None

        # Build new rk array
        new_rk = [0] * (nr - 1)
        for j in range(lo):
            new_rk[j] = rk[j]
        for idx, r in enumerate(new_ranks_complex):
            new_rk[lo + idx] = r
        for j in range(i, nr - 1):
            new_rk[j] = rk[j + 1]

        new_w = w[:i] + w[i+1:]
        return new_w, new_rk, cov

    def try_block_deletion(w, start, length, rk, cov):
        """Try deleting w[start:start+length]. Returns (new_w, new_rk, cov) or None."""
        Lw = len(w)
        if start < 0 or start + length > Lw:
            return None
        if Lw - length < n:  # Can't be shorter than n
            return None
        
        end = start + length - 1  # last index to delete
        nr = Lw - n + 1
        new_nr = Lw - length - n + 1
        
        # Old windows that are affected: those that contain any of the deleted positions
        # A window k (old) is affected if k <= end and k+n-1 >= start
        # i.e., k in [max(0, end-n+1), min(start, nr-1)]
        old_lo = max(0, end - n + 1)
        old_hi = min(start, nr - 1)
        
        old_ranks = [rk[k] for k in range(old_lo, old_hi + 1)]
        
        for r in old_ranks:
            cov[r] -= 1
            if cov[r] < 0:
                for r2 in old_ranks:
                    cov[r2] += 1
                return None
        
        # New windows that are "complex" (cross the deletion boundary):
        # New window j is complex if it includes positions from both sides of the gap.
        # In the new word, the gap is at position start (0-indexed in new word).
        # New window j starts at j in new word.
        # It crosses the gap if j < start and j + n - 1 >= start (in new word coords)
        # But we need to map back to old word.
        # New window j corresponds to old positions:
        #   For old pos < start: same
        #   For old pos >= start: shifted by length
        # So new window j covers old positions:
        #   j..min(j+n-1, start-1) and (start+length)..(j+n-1+length) if j+n-1 >= start
        # Complex if j < start and j+n-1 >= start.
        # j in [max(0, start-n+1), start-1]
        
        new_lo = max(0, start - n + 1)
        new_hi = start - 1
        
        new_ranks_complex = []
        for j in range(new_lo, new_hi + 1):
            # Build virtual window from old w
            # Part before gap: w[j : start] (length start - j)
            # Part after gap: w[start+length : j+n+length] (length j+n - start)
            # Total: (start - j) + (j + n - start) = n
            part1 = w[j:start]
            part2 = w[start+length : j+n+length]
            virtual_window = part1 + part2
            r = perm_rank(virtual_window)
            new_ranks_complex.append(r)
            cov[r] += 1
        
        for r in old_ranks:
            if cov[r] == 0:
                for r2 in new_ranks_complex:
                    cov[r2] -= 1
                for r3 in old_ranks:
                    cov[r3] += 1
                return None
        
        # Build new rk
        new_rk = [0] * new_nr
        
        # Unaffected prefix: j < new_lo
        for j in range(new_lo):
            new_rk[j] = rk[j]
        
        # Complex middle
        for idx, r in enumerate(new_ranks_complex):
            new_rk[new_lo + idx] = r
        
        # Unaffected suffix: j >= start in new word
        # New window j >= start corresponds to old window j + length
        for j in range(start, new_nr):
            new_rk[j] = rk[j + length]
        
        new_w = w[:start] + w[start+length:]
        return new_w, new_rk, cov

    pass_no = 0
    while time.time() < deadline:
        pass_no += 1
        L = len(word)
        best_len = min(best_len, L)
        improved = False

        # Phase 1: Try single deletions, find the best one
        best_res = None
        for i in range(L):
            if time.time() > deadline:
                break
            res = try_deletion(word, i, ranks, cov)
            if res is not None:
                # All valid deletions give same length (L-1), so first is fine
                best_res = res
                break

        if best_res is not None:
            word, ranks, cov = best_res
            improved = True
            continue

        # Phase 2: Try block deletions of size 2 and 3
        block_improved = False
        for blen in (2, 3):
            if time.time() > deadline:
                break
            for start in range(L - blen + 1):
                if time.time() > deadline:
                    break
                res = try_block_deletion(word, start, blen, ranks, cov)
                if res is not None:
                    word, ranks, cov = res
                    improved = True
                    block_improved = True
                    break
            if block_improved:
                break

        if not improved:
            break

    return "".join(map(str, word))

def generate_classic(n):
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1)
                          if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
    return word
