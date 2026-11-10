"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

N = 12
MAX_M = 1000000
MAX_L = 144  # proven upper bound: any 144 consecutive ints contain a full matching
LCM_12 = 27720  # lcm(1,2,...,12)


def _has_matching_hall(m, L):
    """Return True iff (m, m+L] contains distinct b_i with i | b_i for i=1..N.
    Uses Hall's theorem: for every subset S of {1..N}, |N(S)| >= |S|.
    N(S) = set of values in (m, m+L] divisible by at least one member of S."""
    # For each value v in (m, m+L], compute which i in 1..N divide v.
    # Then for each subset S, count how many v have at least one i in S dividing v.
    # 
    # Optimization: precompute for each v its divisor mask (bit i set if (i+1)|v).
    # Then for subset S with bitmask s, v is in N(S) iff (divisor_mask[v] & s) != 0.
    
    # Build divisor masks for all v in (m, m+L]
    # div_mask[j] for j=0..L-1 corresponds to v = m+1+j
    div_masks = [0] * L
    for i in range(1, N + 1):
        bit = 1 << (i - 1)
        # first multiple of i in (m, m+L]
        start = m + 1
        q = (start + i - 1) // i
        v = q * i
        while v <= m + L:
            j = v - m - 1
            div_masks[j] |= bit
            v += i
    
    # Now check Hall's condition for all non-empty subsets S of {1..N}
    # For subset with bitmask s, |N(S)| = count of j where div_masks[j] & s != 0
    # We need |N(S)| >= popcount(s) for all s.
    
    # Precompute popcount for all 2^12 masks
    # Actually we can just iterate all 4095 non-empty subsets
    for s in range(1, 1 << N):
        # count values in N(S)
        cnt = 0
        for j in range(L):
            if div_masks[j] & s:
                cnt += 1
        # check Hall: cnt >= popcount(s)
        pc = bin(s).count('1')
        if cnt < pc:
            return False
    return True


def _has_matching_hall_fast(m, L):
    """Faster Hall's check using precomputed divisor masks and early termination."""
    # Build divisor masks
    div_masks = [0] * L
    for i in range(1, N + 1):
        bit = 1 << (i - 1)
        start = m + 1
        q = (start + i - 1) // i
        v = q * i
        while v <= m + L:
            j = v - m - 1
            div_masks[j] |= bit
            v += i
    
    # For each subset S, we need count of j with (div_masks[j] & S) != 0 >= |S|
    # Key insight: the worst case is small subsets. If a single element {i} has
    # fewer than 1 multiple, fail. If {i,j} has fewer than 2 multiples, fail. Etc.
    # 
    # Actually, let's think about this differently. The Hall condition is equivalent
    # to: for every subset S, the number of elements covered by S is >= |S|.
    # 
    # We can use inclusion-exclusion or just iterate. With L <= 144 and 4096 subsets,
    # that's at most 144*4096 = 589,824 operations per check. That's fast enough.
    
    # But we can be smarter: only check "minimal" violating subsets.
    # Actually, let's just do the direct check but with early termination.
    
    # Precompute for each possible divisor mask value (0 to 2^12-1) how many times it appears
    mask_counts = [0] * (1 << N)
    for j in range(L):
        mask_counts[div_masks[j]] += 1
    
    # For subset S (bitmask s), N(S) = union of all j where div_masks[j] & s != 0
    # |N(S)| = total - count of j where div_masks[j] & s == 0
    # = L - count of j where div_masks[j] is subset of complement of s
    # i.e., div_masks[j] & s == 0 means all bits in div_masks[j] are in ~s
    
    # For each s, count = sum of mask_counts[dm] for dm where (dm & s) == 0
    # This is a zeta transform! Let f[dm] = mask_counts[dm].
    # g[s] = sum of f[dm] for dm where (dm & s) == 0
    #       = sum of f[dm] for dm subset of (~s & ((1<<N)-1))
    # This is the "subset sum" or "zeta transform over subsets".
    
    # Precompute g[s] = sum_{dm: dm & s == 0} mask_counts[dm]
    # Equivalently, let h[t] = sum_{dm: dm subset of t} mask_counts[dm]
    # Then g[s] = h[(1<<N)-1 - s]  ... wait, that's not quite right.
    # dm & s == 0 means dm is a subset of the complement of s.
    # So g[s] = sum_{dm subset of comp(s)} mask_counts[dm] where comp(s) = ((1<<N)-1) ^ s
    
    # Compute h[t] = sum_{dm subset of t} mask_counts[dm] via zeta transform
    h = mask_counts[:]
    for i in range(N):
        for t in range(1 << N):
            if t & (1 << i):
                h[t] += h[t ^ (1 << i)]
    
    # Now g[s] = h[((1<<N)-1) ^ s]
    full_mask = (1 << N) - 1
    for s in range(1, 1 << N):
        pc = bin(s).count('1')
        # |N(S)| = L - h[full_mask ^ s]
        nS = L - h[full_mask ^ s]
        if nS < pc:
            return False
    return True


def _compute_f_hall(m):
    """Compute f(N, m): least positive L with a matching in (m, m+L]."""
    lo, hi = 1, MAX_L
    while lo < hi:
        mid = (lo + hi) // 2
        if _has_matching_hall_fast(m, mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def _find_best(deadline):
    """Search for m maximizing f(N,m). Returns (best_m, best_f)."""
    t0 = time.time()
    remaining = deadline - (time.time() - t0)
    
    # Exploit periodicity: f(12, m) depends only on m mod 27720
    # because the divisibility pattern repeats every LCM(1..12) = 27720.
    # So we only need to check m in [0, 27719] and then map back.
    
    best_m = 0
    best_f = 0
    
    # Phase 1: Exhaustive scan over one period [0, 27720)
    # With Hall's theorem + zeta transform, each _compute_f_hall call is fast.
    # Binary search over L (7-8 iterations) × 4096 subsets × zeta (12*4096) ≈ manageable.
    
    # Let's estimate: _has_matching_hall_fast does:
    # - Build div_masks: L iterations × N/i work ≈ L * (1 + 1/2 + ... ) ≈ L * log(N) ≈ 144*4 = 576
    # - Zeta transform: 12 * 4096 = 49152
    # - Check all subsets: 4096
    # Total per L check: ~50k ops
    # Binary search: ~8 checks
    # Per m: ~400k ops
    # For 27720 m values: ~11 billion ops... too slow in Python.
    
    # Need to optimize further. Let me think...
    # Actually, the zeta transform is the bottleneck at 49k per call.
    # But we do ~8 calls per m, so ~400k per m, times 27720 = 11 billion. Too slow.
    
    # Alternative: don't use zeta transform. Just do direct counting with early termination.
    # For small L, many subsets will fail quickly.
    
    # Let me try a different approach: direct Hall check with early termination on small subsets first.
    
    def _has_matching_direct(m, L):
        """Direct Hall check with early termination."""
        # Build divisor masks
        div_masks = [0] * L
        for i in range(1, N + 1):
            bit = 1 << (i - 1)
            start = m + 1
            q = (start + i - 1) // i
            v = q * i
            while v <= m + L:
                j = v - m - 1
                div_masks[j] |= bit
                v += i
        
        # Check Hall: for each subset S, |N(S)| >= |S|
        # Start with singletons (easiest to violate)
        for i in range(N):
            bit = 1 << i
            cnt = sum(1 for j in range(L) if div_masks[j] & bit)
            if cnt < 1:
                return False
        
        # Check pairs
        for i in range(N):
            for j in range(i + 1, N):
                mask = (1 << i) | (1 << j)
                cnt = sum(1 for k in range(L) if div_masks[k] & mask)
                if cnt < 2:
                    return False
        
        # Check triples
        for i in range(N):
            for j in range(i + 1, N):
                for k in range(j + 1, N):
                    mask = (1 << i) | (1 << j) | (1 << k)
                    cnt = sum(1 for x in range(L) if div_masks[x] & mask)
                    if cnt < 3:
                        return False
        
        # For larger subsets, it's less likely to fail, but we should check
        # Actually, by Hall's theorem, if all subsets of size <= k satisfy the condition
        # and the total is sufficient, we might not need to check all. But to be safe,
        # let's check all. However, for L >= 12, the total number of elements is L >= 12,
        # and the union of all N(S) for S={1..N} is at most L, so we need L >= 12.
        # For larger subsets, the constraint is weaker.
        
        # Actually, let's just check all subsets but with the direct count.
        # With early termination on small subsets, most failures happen early.
        # For successful matches (which is what we care about when L is large enough),
        # we do all 4095 checks. Each check is O(L) = O(144). So 4095 * 144 ≈ 590k.
        # That's about the same as before.
        
        # Let's just do all subsets directly:
        for s in range(1, 1 << N):
            cnt = 0
            for j in range(L):
                if div_masks[j] & s:
                    cnt += 1
            pc = bin(s).count('1')
            if cnt < pc:
                return False
        return True
    
    # Given the complexity, let's do a smart search:
    # 1. First, do a coarse scan over the period to find promising regions
    # 2. Then refine
    
    # Phase 1: Scan m from 0 to 27719 with step 1, but use a fast pre-check
    # Actually, let's just scan all 27720 values. With ~400k ops each, that's 11 billion.
    # In Python, that's too slow (maybe 100+ seconds).
    
    # Let me optimize: use numpy for the divisor mask building and counting.
    import numpy as np
    
    def _has_matching_np(m, L):
        """Hall check using numpy for speed."""
        # Build divisor masks using numpy
        div_masks = np.zeros(L, dtype=np.uint32)
        for i in range(1, N + 1):
            bit = 1 << (i - 1)
            start = m + 1
            q = (start + i - 1) // i
            v = q * i
            while v <= m + L:
                j = v - m - 1
                div_masks[j] |= bit
                v += i
        
        # For each subset s, count = sum((div_masks & s) != 0)
        # Vectorize: create a 2D array? No, L is small.
        # Just loop over subsets:
        for s in range(1, 1 << N):
            cnt = np.count_nonzero(div_masks & s)
            pc = bin(s).count('1')
            if cnt < pc:
                return False
        return True
    
    # Let's test timing on a few values first
    t_test = time.time()
    for test_m in range(0, 100, 10):
        _has_matching_np(test_m, 50)
    t_test_elapsed = time.time() - t_test
    ops_per_check = t_test_elapsed / 10
    print(f"Est. time per Hall check: {ops_per_check*1000:.2f} ms")
    print(f"Est. time for 27720 checks: {ops_per_check * 27720:.1f} s")
    
    remaining = deadline - (time.time() - t0)
    
    if ops_per_check * 27720 < remaining * 0.7:
        # Can afford full scan of one period
        for m in range(0, LCM_12):
            if (time.time() - t0) > deadline - 5:
                break
            f = _compute_f_hall(m)
            if f > best_f:
                best_f = f
                best_m = m
        # Map best_m to range [0, MAX_M]
        # best_m is already in [0, 27719], which is within [0, 1000000]
    else:
        # Can't afford full scan; do coarse scan
        step = max(1, int(27720 / max(1, remaining / (ops_per_check * 8))))
        for m in range(0, LCM_12, step):
            if (time.time() - t0) > deadline - 5:
                break
            f = _compute_f_hall(m)
            if f > best_f:
                best_f = f
                best_m = m
        # Refine around best
        if best_f > 0:
            for dm in range(1, min(step, 50)):
                for m2 in (best_m - dm, best_m + dm):
                    if m2 < 0 or m2 >= LCM_12:
                        continue
                    if (time.time() - t0) > deadline - 5:
                        break
                    f = _compute_f_hall(m2)
                    if f > best_f:
                        best_f = f
                        best_m = m2
    
    # best_m is in [0, 27719], which is valid since MAX_M = 1000000
    return best_m, best_f


def construct():
    deadline = 85.0  # safety margin before 90s kill
    t0 = time.time()
    best_m, best_f = _find_best(deadline)
    # Ensure type is Python int
    return int(best_m)
# EVOLVE-BLOCK-END
