"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START

def _valid(A, n=20, maximum=10000):
    if len(A) != n:
        return False
    if len(set(A)) != n:
        return False
    if any(x <= 0 or x > maximum for x in A):
        return False
    return True

def _score(A):
    """Return max(|A+A|, |A*A|) for sorted list A of length 20."""
    n = len(A)
    ss = set()
    ps = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            ss.add(ai + A[j])
            ps.add(ai * A[j])
    return max(len(ss), len(ps))

def _score_with_bound(A, bound):
    """Return max(|A+A|, |A*A|) but stop early if either exceeds bound."""
    n = len(A)
    ss = set()
    ps = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            s = ai + A[j]
            ss.add(s)
            if len(ss) > bound:
                return len(ss)
            p = ai * A[j]
            ps.add(p)
            if len(ps) > bound:
                return len(ps)
    return max(len(ss), len(ps))

class _State:
    """Maintains sumset and productset with per-element counts for O(n) swap scoring."""
    __slots__ = ('A', 'ss', 'ps', 'ss_cnt', 'ps_cnt')
    def __init__(self, A):
        self.A = list(A)
        self.ss = {}
        self.ps = {}
        self.ss_cnt = [0] * len(A)
        self.ps_cnt = [0] * len(A)
        self._rebuild()

    def _rebuild(self):
        A = self.A
        n = len(A)
        self.ss = {}
        self.ps = {}
        self.ss_cnt = [0] * n
        self.ps_cnt = [0] * n
        for i in range(n):
            ai = A[i]
            for j in range(i, n):
                s = ai + A[j]
                self.ss[s] = self.ss.get(s, 0) + 1
                self.ss_cnt[i] += 1
                self.ss_cnt[j] += 1
                p = ai * A[j]
                self.ps[p] = self.ps.get(p, 0) + 1
                self.ps_cnt[i] += 1
                self.ps_cnt[j] += 1

    def score(self):
        return max(len(self.ss), len(self.ps))

    def score_with_bound(self, bound):
        # Quick check current sizes
        if len(self.ss) > bound or len(self.ps) > bound:
            return max(len(self.ss), len(self.ps))
        return self.score()

    def swap(self, i, new_val):
        """Replace A[i] with new_val. Returns (new_score, new_ss_size, new_ps_size)."""
        A = self.A
        n = len(A)
        old_val = A[i]
        if new_val == old_val:
            return self.score(), len(self.ss), len(self.ps)

        # Remove old_val's contributions
        # Pairs involving i: (i, i) and (i, j) for j != i
        # Pair (i,i): sum=2*old_val, prod=old_val^2
        s_ii = 2 * old_val
        p_ii = old_val * old_val
        self.ss[s_ii] -= 1
        if self.ss[s_ii] == 0:
            del self.ss[s_ii]
        self.ss_cnt[i] -= 2  # (i,i) contributes 2 to count of i
        self.ps[p_ii] -= 1
        if self.ps[p_ii] == 0:
            del self.ps[p_ii]
        self.ps_cnt[i] -= 2

        # Pairs (i, j) for j != i
        for j in range(n):
            if j == i:
                continue
            aj = A[j]
            s = old_val + aj
            p = old_val * aj
            self.ss[s] -= 1
            if self.ss[s] == 0:
                del self.ss[s]
            self.ss_cnt[i] -= 1
            self.ss_cnt[j] -= 1
            self.ps[p] -= 1
            if self.ps[p] == 0:
                del self.ps[p]
            self.ps_cnt[i] -= 1
            self.ps_cnt[j] -= 1

        # Set new value
        A[i] = new_val

        # Add new_val's contributions
        s_ii = 2 * new_val
        p_ii = new_val * new_val
        self.ss[s_ii] = self.ss.get(s_ii, 0) + 1
        self.ss_cnt[i] += 2
        self.ps[p_ii] = self.ps.get(p_ii, 0) + 1
        self.ps_cnt[i] += 2

        for j in range(n):
            if j == i:
                continue
            aj = A[j]
            s = new_val + aj
            p = new_val * aj
            self.ss[s] = self.ss.get(s, 0) + 1
            self.ss_cnt[i] += 1
            self.ss_cnt[j] += 1
            self.ps[p] = self.ps.get(p, 0) + 1
            self.ps_cnt[i] += 1
            self.ps_cnt[j] += 1

        return self.score(), len(self.ss), len(self.ps)

    def undo_swap(self, i, new_val):
        """Undo a swap: replace A[i] (currently new_val) back to old_val."""
        # We need old_val; let's just rebuild to be safe, but that's slow.
        # Actually, let's store old_val. For now, this method is not used directly;
        # we use swap with the original value to undo.
        pass

def _gen_geometric_perturbed(b, exponents, k_list, maximum):
    """Generate A = {b^e_i * k_i} for given base, exponents, and small multipliers."""
    A = set()
    for e, k in zip(exponents, k_list):
        val = (b ** e) * k
        if val <= maximum and val > 0:
            A.add(val)
    return sorted(A)

def _try_geometric_base(b, n, maximum):
    """Try geometric progression with base b and multiplicative perturbations."""
    best_A = None
    best_score = float('inf')
    
    # Generate exponents that keep values <= maximum
    max_exp = 0
    if b > 1:
        val = 1
        while val * b <= maximum:
            val *= b
            max_exp += 1
    else:
        max_exp = 0
    
    # Try different exponent patterns
    for max_e in range(max(1, max_exp - 2), max_exp + 1):
        if max_e < n:
            continue
        
        exponents = list(range(n))
        if max(exponents) >= max_exp:
            continue
        
        small_mults = [1] * n
        # Try perturbing one or two elements
        for perturb_idx in range(n):
            for k in [2, 3, 5, 7]:
                test_mults = list(small_mults)
                test_mults[perturb_idx] = k
                A = _gen_geometric_perturbed(b, exponents, test_mults, maximum)
                if len(A) == n and all(x > 0 for x in A) and _valid(A, n, maximum):
                    sc = _score_with_bound(A, best_score)
                    if sc < best_score:
                        best_score = sc
                        best_A = A
                for perturb_idx2 in range(perturb_idx + 1, n):
                    for k2 in [2, 3, 5]:
                        test_mults = list(small_mults)
                        test_mults[perturb_idx] = k
                        test_mults[perturb_idx2] = k2
                        A = _gen_geometric_perturbed(b, exponents, test_mults, maximum)
                        if len(A) == n and all(x > 0 for x in A) and _valid(A, n, maximum):
                            sc = _score_with_bound(A, best_score)
                            if sc < best_score:
                                best_score = sc
                                best_A = A
    
    return best_A, best_score

def _local_search(A, best_score, candidate_pool, deadline, rng):
    """Local search using incremental state for O(n) per swap."""
    n = len(A)
    state = _State(A)
    current_score = state.score()
    if current_score < best_score:
        best_score = current_score
        best_A = list(A)
    
    improved = True
    no_improve_rounds = 0
    while improved and time.time() < deadline:
        improved = False
        # Try replacing each element with candidates from pool
        for i in range(n):
            if time.time() >= deadline:
                break
            old_val = A[i]
            # Try candidates
            for new_val in candidate_pool:
                if new_val == old_val or new_val in A:
                    continue
                sc, ss_sz, ps_sz = state.swap(i, new_val)
                if sc < best_score:
                    best_score = sc
                    best_A = list(A)
                    improved = True
                    no_improve_rounds = 0
                    break
                else:
                    # Undo: swap back to old_val
                    state.swap(i, old_val)
            if improved:
                break
        if not improved:
            # Try random perturbations
            for _ in range(50):
                if time.time() >= deadline:
                    break
                i = rng.randint(0, n - 1)
                old_val = A[i]
                new_val = rng.choice(candidate_pool)
                if new_val == old_val or new_val in A:
                    continue
                sc, ss_sz, ps_sz = state.swap(i, new_val)
                if sc < best_score:
                    best_score = sc
                    best_A = list(A)
                    improved = True
                    no_improve_rounds = 0
                    break
                else:
                    state.swap(i, old_val)
            if not improved:
                no_improve_rounds += 1
                if no_improve_rounds >= 5:
                    break
                # Random restart within local search: try a few random single swaps even if worse
                for _ in range(20):
                    if time.time() >= deadline:
                        break
                    i = rng.randint(0, n - 1)
                    old_val = A[i]
                    new_val = rng.choice(candidate_pool)
                    if new_val == old_val or new_val in A:
                        continue
                    sc, ss_sz, ps_sz = state.swap(i, new_val)
                    if sc < best_score:
                        best_score = sc
                        best_A = list(A)
                        improved = True
                        no_improve_rounds = 0
                        break
                    else:
                        state.swap(i, old_val)
                if not improved:
                    break
    return best_A, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    n = 20
    maximum = 10000
    
    # Strategy: Geometric progression with multiplicative perturbations
    best_A = None
    best_score = float('inf')
    
    # Try base 2
    A2, s2 = _try_geometric_base(2, n, maximum)
    if A2 is not None:
        best_A = A2
        best_score = s2
    
    # Try base 3
    A3, s3 = _try_geometric_base(3, n, maximum)
    if A3 is not None and s3 < best_score:
        best_A = A3
        best_score = s3
    
    # Try base 4
    A4, s4 = _try_geometric_base(4, n, maximum)
    if A4 is not None and s4 < best_score:
        best_A = A4
        best_score = s4
    
    # Smooth number based initial
    smooth = set()
    for a in range(1, maximum + 1):
        x = a
        while x % 2 == 0:
            x //= 2
        while x % 3 == 0:
            x //= 3
        while x % 5 == 0:
            x //= 5
        if x == 1:
            smooth.add(a)
    smooth_list = sorted([x for x in smooth if x <= maximum])
    
    if len(smooth_list) >= n:
        A_smooth = sorted(smooth_list[:n])
        sc = _score_with_bound(A_smooth, best_score)
        if sc < best_score:
            best_score = sc
            best_A = A_smooth
    
    if best_A is None:
        best_A = list(range(1, n + 1))
        best_score = _score(best_A)
    
    # Build candidate pool for local search
    rng = random.Random(12345)
    candidate_pool = set()
    for x in best_A:
        candidate_pool.add(x)
    # Add smooth numbers
    for x in smooth_list[:500]:
        candidate_pool.add(x)
    # Add geometric-like numbers
    for b in [2, 3, 4, 5]:
        val = 1
        while val <= maximum:
            candidate_pool.add(val)
            val *= b
    # Add some random
    for _ in range(200):
        candidate_pool.add(rng.randint(1, maximum))
    candidate_pool = sorted(candidate_pool)
    
    # Limit pool size
    if len(candidate_pool) > 1500:
        candidate_pool = candidate_pool[:750] + candidate_pool[-750:]
        candidate_pool = sorted(set(candidate_pool))
    
    # Local search
    best_A, best_score = _local_search(best_A, best_score, candidate_pool, deadline, rng)
    
    # Final validation
    if not _valid(best_A, n, maximum):
        best_A = list(range(1, n + 1))
    
    return best_A
# EVOLVE-BLOCK-END
