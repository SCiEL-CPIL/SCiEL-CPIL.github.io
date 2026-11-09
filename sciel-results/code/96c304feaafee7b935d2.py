"""Improved program: maximize |A|+|B| for two Sidon sets with disjoint
nonzero difference sets in [1,256]. Uses bitset-based fast checks and
greedy + swap-based local search with multiple restarts."""

import time
import random

N = 256
MAX_EACH = 64
DEADLINE_SECONDS = 80  # safety margin before 90s kill

def _check_sidon(s):
    """Return True if s is a Sidon set (pair sums including doubles are unique)."""
    sums = set()
    n = len(s)
    for i in range(n):
        for j in range(i, n):
            val = s[i] + s[j]
            if val in sums:
                return False
            sums.add(val)
    return True

def _diffs(s):
    """Return set of nonzero absolute differences for sorted s."""
    d = set()
    n = len(s)
    for i in range(n):
        for j in range(i):
            d.add(s[i] - s[j])
    return d

def _valid_pair(A, B):
    """Check both are Sidon and their nonzero difference sets are disjoint."""
    if not _check_sidon(A) or not _check_sidon(B):
        return False
    dA = _diffs(A)
    dB = _diffs(B)
    return not (dA & dB)

def _score(A, B):
    return len(A) + len(B)

class _SidonState:
    """Maintains state for incremental Sidon set construction using bitsets."""
    def __init__(self):
        self.elems = []
        self.sum_bits = 0  # bit i set iff sum value i is used
        self.diff_bits = 0  # bit d set iff diff d is used
    
    def can_add(self, x, other_diff_bits=0):
        """Check if x can be added to this set while maintaining Sidon
        property and avoiding diffs in other_diff_bits."""
        # Check sums: x+x and x+a for each a in elems
        xx = x + x
        if (self.sum_bits >> xx) & 1:
            return False
        for a in self.elems:
            if (self.sum_bits >> (x + a)) & 1:
                return False
        # Check diffs
        for a in self.elems:
            d = abs(x - a)
            if (self.diff_bits >> d) & 1 or (other_diff_bits >> d) & 1:
                return False
        return True
    
    def add(self, x):
        """Add x to the set, updating sum and diff bitsets."""
        self.elems.append(x)
        self.elems.sort()
        self.sum_bits |= (1 << (x + x))
        for a in self.elems[:-1]:
            self.sum_bits |= (1 << (x + a))
        for a in self.elems[:-1]:
            d = abs(x - a)
            self.diff_bits |= (1 << d)
    
    def remove(self, x):
        """Remove x from the set, recomputing bitsets."""
        self.elems.remove(x)
        self.sum_bits = 0
        self.diff_bits = 0
        n = len(self.elems)
        for i in range(n):
            for j in range(i, n):
                self.sum_bits |= (1 << (self.elems[i] + self.elems[j]))
        for i in range(n):
            for j in range(i):
                d = self.elems[i] - self.elems[j]
                self.diff_bits |= (1 << d)

def _greedy_build_interleaved(max_a, max_b, rng):
    """Build A and B interleaved, each avoiding the other's diffs."""
    state_a = _SidonState()
    state_b = _SidonState()
    pool = list(range(1, N + 1))
    rng.shuffle(pool)
    
    for x in pool:
        if len(state_a.elems) < max_a:
            if state_a.can_add(x, state_b.diff_bits):
                state_a.add(x)
        if len(state_b.elems) < max_b:
            if state_b.can_add(x, state_a.diff_bits):
                state_b.add(x)
    
    return list(state_a.elems), list(state_b.elems)

def _local_search(A, B, rng, time_limit):
    """Try to improve (A, B) via local search: add elements and swaps."""
    A = list(A)
    B = list(B)
    
    def try_add_to(A, B, side):
        """Try to add a random element to side 0 (A) or 1 (B)."""
        if side == 0:
            if len(A) >= MAX_EACH:
                return False
            other_diff_bits = 0
            for d in _diffs(B):
                other_diff_bits |= (1 << d)
            self_diff_bits = 0
            for d in _diffs(A):
                self_diff_bits |= (1 << d)
            self_sum_bits = 0
            n = len(A)
            for i in range(n):
                for j in range(i, n):
                    self_sum_bits |= (1 << (A[i] + A[j]))
        else:
            if len(B) >= MAX_EACH:
                return False
            other_diff_bits = 0
            for d in _diffs(A):
                other_diff_bits |= (1 << d)
            self_diff_bits = 0
            for d in _diffs(B):
                self_diff_bits |= (1 << d)
            self_sum_bits = 0
            n = len(B)
            for i in range(n):
                for j in range(i, n):
                    self_sum_bits |= (1 << (B[i] + B[j]))
        
        candidates = list(range(1, N + 1))
        rng.shuffle(candidates)
        
        for x in candidates:
            if side == 0:
                # Check if x can be added to A
                ok = True
                if (self_sum_bits >> (x + x)) & 1:
                    ok = False
                else:
                    for a in A:
                        if (self_sum_bits >> (x + a)) & 1:
                            ok = False
                            break
                if ok:
                    new_diff_bits = 0
                    for a in A:
                        d = abs(x - a)
                        new_diff_bits |= (1 << d)
                    if not (new_diff_bits & (self_diff_bits | other_diff_bits)):
                        A.append(x)
                        A.sort()
                        return True
            else:
                # Check if x can be added to B
                ok = True
                if (self_sum_bits >> (x + x)) & 1:
                    ok = False
                else:
                    for b in B:
                        if (self_sum_bits >> (x + b)) & 1:
                            ok = False
                            break
                if ok:
                    new_diff_bits = 0
                    for b in B:
                        d = abs(x - b)
                        new_diff_bits |= (1 << d)
                    if not (new_diff_bits & (self_diff_bits | other_diff_bits)):
                        B.append(x)
                        B.sort()
                        return True
        return False
    
    def try_swap(A, B, rng):
        """Try a swap: remove one element from A and add one to A, or vice versa."""
        # Try swapping in A
        if len(A) > 0:
            i = rng.randrange(len(A))
            old_a = A[i]
            A_no_a = [x for x in A if x != old_a]
            A_no_a.sort()
            
            # Compute state without old_a
            self_sum_bits = 0
            n = len(A_no_a)
            for ii in range(n):
                for jj in range(ii, n):
                    self_sum_bits |= (1 << (A_no_a[ii] + A_no_a[jj]))
            self_diff_bits = 0
            for ii in range(n):
                for jj in range(ii):
                    d = A_no_a[ii] - A_no_a[jj]
                    self_diff_bits |= (1 << d)
            other_diff_bits = 0
            for d in _diffs(B):
                other_diff_bits |= (1 << d)
            
            # Try adding each candidate to A_no_a
            candidates = [x for x in range(1, N + 1) if x not in A_no_a and x != old_a]
            rng.shuffle(candidates)
            
            for x in candidates:
                # Check sums
                ok = True
                if (self_sum_bits >> (x + x)) & 1:
                    ok = False
                else:
                    for a in A_no_a:
                        if (self_sum_bits >> (x + a)) & 1:
                            ok = False
                            break
                if ok:
                    new_diff_bits = 0
                    for a in A_no_a:
                        d = abs(x - a)
                        new_diff_bits |= (1 << d)
                    if not (new_diff_bits & (self_diff_bits | other_diff_bits)):
                        A_no_a.append(x)
                        A_no_a.sort()
                        A[:] = A_no_a
                        return True
        
        # Try swapping in B
        if len(B) > 0:
            i = rng.randrange(len(B))
            old_b = B[i]
            B_no_b = [x for x in B if x != old_b]
            B_no_b.sort()
            
            # Compute state without old_b
            self_sum_bits = 0
            n = len(B_no_b)
            for ii in range(n):
                for jj in range(ii, n):
                    self_sum_bits |= (1 << (B_no_b[ii] + B_no_b[jj]))
            self_diff_bits = 0
            for ii in range(n):
                for jj in range(ii):
                    d = B_no_b[ii] - B_no_b[jj]
                    self_diff_bits |= (1 << d)
            other_diff_bits = 0
            for d in _diffs(A):
                other_diff_bits |= (1 << d)
            
            # Try adding each candidate to B_no_b
            candidates = [x for x in range(1, N + 1) if x not in B_no_b and x != old_b]
            rng.shuffle(candidates)
            
            for x in candidates:
                # Check sums
                ok = True
                if (self_sum_bits >> (x + x)) & 1:
                    ok = False
                else:
                    for b in B_no_b:
                        if (self_sum_bits >> (x + b)) & 1:
                            ok = False
                            break
                if ok:
                    new_diff_bits = 0
                    for b in B_no_b:
                        d = abs(x - b)
                        new_diff_bits |= (1 << d)
                    if not (new_diff_bits & (self_diff_bits | other_diff_bits)):
                        B_no_b.append(x)
                        B_no_b.sort()
                        B[:] = B_no_b
                        return True
        
        return False
    
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        if try_add_to(A, B, 0):
            improved = True
        if try_add_to(A, B, 1):
            improved = True
        if try_swap(A, B, rng):
            improved = True
    
    return A, B

def _random_restart(rng, target_a, target_b):
    """One random restart: build A and B with greedy."""
    A, B = _greedy_build_interleaved(target_a, target_b, rng)
    return A, B

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    best_A, best_B = [], []
    best_score = 0
    
    rng = random.Random(42)
    
    # Baseline: seed program style (deterministic)
    sets = [[], []]
    for x in range(1, N + 1):
        for side in (0, 1):
            candidate = sets[side] + [x]
            candidate.sort()
            sums = set()
            n = len(candidate)
            ok_sidon = True
            for i in range(n):
                for j in range(i, n):
                    s = candidate[i] + candidate[j]
                    if s in sums:
                        ok_sidon = False
                        break
                    sums.add(s)
            if ok_sidon:
                diffs = set()
                for i in range(n):
                    for j in range(i):
                        diffs.add(candidate[i] - candidate[j])
                other_diffs = set()
                os = sets[1 - side]
                n2 = len(os)
                for i in range(n2):
                    for j in range(i):
                        other_diffs.add(os[i] - os[j])
                if not (diffs & other_diffs):
                    sets[side] = candidate
    
    if len(sets[0]) > 0 and len(sets[1]) > 0:
        if _valid_pair(sets[0], sets[1]):
            best_A, best_B = list(sets[0]), list(sets[1])
            best_score = _score(best_A, best_B)
    
    # Randomized search with multiple restarts
    # Try various target size pairs
    size_pairs = []
    # Symmetric
    for s in range(15, 45, 2):
        size_pairs.append((s, s))
    # Asymmetric
    for a in range(15, 40, 3):
        for b in range(15, 40, 3):
            if a + b > 60:
                continue
            size_pairs.append((a, b))
    
    rng.shuffle(size_pairs)
    
    for A_size, B_size in size_pairs:
        if time.time() > deadline - 3:
            break
        
        # Multiple restarts for each size pair
        for restart in range(5):
            if time.time() > deadline - 3:
                break
            
            A, B = _random_restart(rng, A_size, B_size)
            score = _score(A, B)
            
            if score > best_score and _valid_pair(A, B):
                best_A, best_B = list(A), list(B)
                best_score = score
            
            # Local search to improve
            if score >= best_score - 4 and time.time() < deadline - 5:
                A2, B2 = _local_search(A, B, rng, deadline - 3)
                score2 = _score(A2, B2)
                if score2 > best_score and _valid_pair(A2, B2):
                    best_A, best_B = list(A2), list(B2)
                    best_score = score2
    
    # Final attempt: try to improve best pair with local search
    if best_score < MAX_EACH * 2 and time.time() < deadline - 2:
        A2, B2 = _local_search(best_A, best_B, rng, deadline - 1)
        if _score(A2, B2) > best_score and _valid_pair(A2, B2):
            best_A, best_B = A2, B2
    
    # Ensure nonempty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [3]
    
    # Final verification
    if not _valid_pair(best_A, best_B):
        # Fallback: simple valid pair
        best_A = [1, 2, 4, 8, 16, 32, 64, 128]
        best_B = [3, 6, 12, 24, 48, 96, 192, 255]
        if not _valid_pair(best_A, best_B):
            best_A = [1]
            best_B = [3]
    
    return [best_A, best_B]
