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

    def perm_rank(p):
        rank = 0
        used = [False] * (n + 1)
        for i in range(n):
            count = 0
            for j in range(1, p[i]):
                if not used[j]:
                    count += 1
            rank += count * fact[n - 1 - i]
            used[p[i]] = True
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

    # Optimized try_deletion: Does not create new_w initially. 
    # Computes new ranks for affected windows directly from w.
    def try_deletion(w, i, rk, cov):
        Lw = len(w)
        nr = Lw - n + 1
        if i < 0 or i >= Lw:
            return None
        
        # Affected windows in new word:
        # Complex windows are j in [lo, hi] where:
        lo = max(0, i - (n - 1))
        hi = i - 1
        
        # Old ranks to decrement: k in [old_lo, old_hi]
        old_lo = max(0, i - (n - 1))
        old_hi = i
        
        old_ranks = [rk[k] for k in range(old_lo, old_hi + 1)]
        
        # Decrement coverage
        valid = True
        for r in old_ranks:
            cov[r] -= 1
            if cov[r] < 0:
                valid = False
                break
        
        if not valid:
            # Revert
            for r in old_ranks:
                cov[r] += 1
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

        # Check if any old rank became 0 (and wasn't replenished by new complex windows)
        # We only need to check the ranks we decremented.
        # If a rank was decremented and is now 0, it means the new complex windows
        # did not cover it, OR it was only covered by the deleted/modified windows.
        # Wait, if cov[r] becomes 0, it means no window in the new word has rank r.
        # So we must revert.
        for r in old_ranks:
            if cov[r] == 0:
                # Revert
                for r2 in new_ranks_complex:
                    cov[r2] -= 1
                for r3 in old_ranks:
                    cov[r3] += 1
                return None

        # Update rk array
        new_rk = [0] * (nr - 1)
        
        # Copy unaffected parts
        for j in range(lo):
            new_rk[j] = rk[j]
            
        for idx, r in enumerate(new_ranks_complex):
            new_rk[lo + idx] = r
            
        for j in range(i, nr - 1):
            new_rk[j] = rk[j + 1]

        # Commit
        new_w = w[:i] + w[i+1:]
        
        return new_w, new_rk, cov

    pass_no = 0
    while time.time() < deadline:
        pass_no += 1
        L = len(word)
        best_len = min(best_len, L)
        improved = False

        # Try single deletions
        # We scan from left to right. If we find a deletion, we commit and break.
        # This is a greedy hill-climbing approach.
        for i in range(L):
            if time.time() > deadline:
                break
            res = try_deletion(word, i, ranks, cov)
            if res is not None:
                word, ranks, cov = res
                improved = True
                break

        if not improved:
            # Try 2-opt swaps? Or just break?
            # Given the time budget, single deletion is the primary driver.
            # If no deletion works, we are likely at a local minimum for 1-deletion.
            # We could try swaps to escape local minima, but for now, break.
            # A more advanced strategy might be to try random deletions or swaps
            # if stuck, but that requires more careful state management.
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
