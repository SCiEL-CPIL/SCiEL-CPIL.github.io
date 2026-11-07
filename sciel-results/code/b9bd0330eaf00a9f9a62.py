# EVOLVE-BLOCK-START
import time
from math import isqrt

LO = 210
MAXV = 10210
MAX_SIZE = 100
TARGET = 210  # 2*3*5*7


def _is_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n


def _sieve_sfp(limit):
    """Squarefree part for 1..limit via a smallest-prime-factor sieve."""
    spf = list(range(limit + 1))
    for i in range(2, int(limit ** 0.5) + 1):
        if spf[i] == i:
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    sfp = [0] * (limit + 1)
    sfp[1] = 1
    for x in range(2, limit + 1):
        v = x
        r = 1
        while v > 1:
            p = spf[v]
            cnt = 0
            while v % p == 0:
                v //= p
                cnt ^= 1
            if cnt:
                r *= p
        sfp[x] = r
    return sfp


def _sfp_of(n):
    r = 1
    p = 2
    while p * p <= n:
        cnt = 0
        while n % p == 0:
            n //= p
            cnt ^= 1
        if cnt:
            r *= p
        p += 1
    if n > 1:
        r *= n
    return r


def _verify(vals):
    if not vals:
        return False
    if len(set(vals)) != len(vals):
        return False
    if len(vals) > MAX_SIZE:
        return False
    for v in vals:
        if not isinstance(v, int) or isinstance(v, bool):
            return False
        if v <= LO or v > MAXV:
            return False
    prod = LO
    for v in vals:
        prod *= v
    return _is_square(prod)


def construct():
    t0 = time.time()
    deadline = t0 + 85.0

    sfp = _sieve_sfp(MAXV)
    sfp_target = sfp[TARGET]  # 210

    # Precompute squarefree parts for all candidates
    # We only care about numbers in (LO, MAXV]
    candidates = list(range(LO + 1, MAXV + 1))
    c_sfps = [sfp[x] for x in candidates]
    
    # Map sfp value to list of indices in candidates that have this sfp
    # This allows O(1) lookup for "is there a k with sfp[k] == req?"
    sfp_to_indices = {}
    for idx, val in enumerate(candidates):
        sf = c_sfps[idx]
        if sf not in sfp_to_indices:
            sfp_to_indices[sf] = []
        sfp_to_indices[sf].append(idx)

    # ---- Fast direct search: single, then pairs, then triples ----
    # Single
    for idx, x in enumerate(candidates):
        if c_sfps[idx] == sfp_target:
            return [x]

    # Pairs: sfp(sfp[x]*sfp[y]) == target
    # Condition: sfp[a*b] == target
    # We can iterate i and check if there exists j > i such that sfp[c_sfps[i]*c_sfps[j]] == target
    # To speed up, we can group by sfp.
    # But N is ~10000, N^2 is 10^8 which might be tight in Python.
    # Let's optimize: 
    # For each i, we need sfp[j] such that sfp(c_sfps[i] * c_sfp[j]) == target.
    # Let A = c_sfps[i]. We need sfp(A * B) == T.
    # This implies A*B = T * k^2 for some integer k.
    # So B = T * k^2 / A.
    # B must be an integer and B must be in the set of available sfps.
    # Also B must be the sfp of some number in candidates.
    
    # Let's try a smarter pair search using the sfp_to_indices map.
    # Iterate i.
    # Let A = c_sfps[i].
    # We need to find B such that sfp(A*B) == T.
    # This is equivalent to: the squarefree part of A*B is T.
    # Let A = T * a_sq * rem_a? No.
    # Let's just iterate j? 10^8 operations is too slow for 90s in Python if not optimized.
    # However, we can prune.
    # If we iterate i, and for each i, we check all j>i, that's ~50M iterations.
    # In Python, 50M simple ops take ~5-10s. This is feasible.
    
    for i in range(len(candidates)):
        if time.time() > deadline - 2:
            break
        A = c_sfps[i]
        for j in range(i + 1, len(candidates)):
            B = c_sfps[j]
            # Check if sfp(A*B) == T
            # We can compute sfp(A*B) quickly if we factor A and B?
            # A and B are squarefree.
            # sfp(A*B) is the product of primes in A xor B.
            # Since A and B are squarefree, A*B is squarefree iff gcd(A,B)=1.
            # If gcd(A,B) > 1, let g = gcd(A,B). Then A*B = g^2 * (A/g) * (B/g).
            # sfp(A*B) = sfp((A/g)*(B/g)).
            # Since A/g and B/g are coprime, sfp((A/g)*(B/g)) = (A/g)*(B/g).
            # So sfp(A*B) = (A/g)*(B/g) = A*B/g^2.
            # We need A*B/g^2 == T.
            # So A*B == T * g^2.
            
            # Calculate g = gcd(A, B)
            g = A
            b = B
            while b:
                A, b = b, A % b
            g = A # This is wrong, A is modified. Need to be careful.
            
            # Let's use a proper gcd function or math.gcd
            # Import math.gcd for speed? It's C-implemented.
            import math
            g = math.gcd(c_sfps[i], c_sfps[j])
            if c_sfps[i] * c_sfps[j] == sfp_target * g * g:
                # Verify distinctness (i != j is guaranteed by loop)
                # Verify values are distinct (guaranteed by indices)
                # Verify range (guaranteed by candidates)
                # Verify square condition (guaranteed by sfp logic)
                return [candidates[i], candidates[j]]

    # Triples: sfp(sfp[x]*sfp[y]*sfp[z]) == target
    # Iterate i, j.
    # Let A = c_sfps[i], B = c_sfps[j].
    # Let P = sfp(A*B).
    # We need sfp(P * C) == T, where C = c_sfps[k].
    # This implies P * C = T * g^2 for some g.
    # So C = T * g^2 / P.
    # C must be an integer.
    # Also C must be in sfp_to_indices.
    # And the index k must be distinct from i and j.
    
    for i in range(len(candidates)):
        if time.time() > deadline - 5:
            break
        A = c_sfps[i]
        for j in range(i + 1, len(candidates)):
            B = c_sfps[j]
            # Compute P = sfp(A*B)
            g = math.gcd(A, B)
            P = (A * B) // (g * g)
            
            # We need to find C such that sfp(P*C) == T.
            # Let g2 = gcd(P, C).
            # sfp(P*C) = (P*C) / g2^2 = T.
            # So P*C = T * g2^2.
            # C = T * g2^2 / P.
            
            # C must be an integer.
            # This requires P to divide T * g2^2.
            # Since we don't know g2, we can iterate possible g2?
            # Or simply: C must be a multiple of P / gcd(P, T*g2^2)?
            
            # Alternative: Iterate possible C values from sfp_to_indices keys?
            # There are at most ~10000 distinct sfp values.
            # For each i, j, iterating all C is 10000 * 10000 * 10000 = 10^12. Too slow.
            
            # Let's reverse:
            # For each i, j, we have P.
            # We need to find C in the set of available sfps such that sfp(P*C) == T.
            # Let's precompute for each P (which is a squarefree number <= MAXV)
            # which C values satisfy the condition?
            # No, P depends on i, j.
            
            # Let's try a different angle for Triples:
            # We want sfp(A*B*C) == T.
            # This is equivalent to: the parity vector of A xor B xor C == parity vector of T.
            # Let v_T be the bit vector of T.
            # We need v_A xor v_B xor v_C = v_T.
            # So v_C = v_T xor v_A xor v_B.
            # Let target_c_vec = v_T xor v_A xor v_B.
            # We need to find a squarefree number C (from our available set) whose bit vector is target_c_vec.
            # Since C is squarefree, its value is determined by its prime factors, i.e., its bit vector.
            # So we can map bit vector -> value for all squarefree numbers in range.
            
            # Let's build this map once.
            # We need to represent the bit vector.
            # The primes involved are those <= MAXV.
            # There are ~1200 primes <= 10000.
            # Python integers can handle this as bitmasks.
            
            # Re-initialize prime mapping
            primes = []
            prime_seen = set()
            # We only need primes that appear in any candidate or target
            for x in candidates:
                v = x
                p = 2
                while p * p <= v:
                    cnt = 0
                    while v % p == 0:
                        v //= p
                        cnt ^= 1
                    if cnt and p not in prime_seen:
                        prime_seen.add(p)
                        primes.append(p)
                    p += 1
                if v > 1 and v not in prime_seen:
                    prime_seen.add(v)
                    primes.append(v)
            for p in (2, 3, 5, 7):
                if p not in prime_seen:
                    prime_seen.add(p)
                    primes.append(p)
            
            prime_index = {p: i for i, p in enumerate(primes)}
            D = len(primes)
            
            # Map sfp value to its bitmask
            sfp_to_mask = {}
            for x in candidates:
                sf = c_sfps[candidates.index(x)] # This is O(N) lookup, bad.
                # We already have c_sfps list.
                pass
            
            # Let's do this properly in the main loop setup
            # We already have c_sfps.
            # Let's compute masks for all candidates.
            masks = [0] * len(candidates)
            for idx, x in enumerate(candidates):
                m = 0
                v = x
                p = 2
                while p * p <= v:
                    cnt = 0
                    while v % p == 0:
                        v //= p
                        cnt ^= 1
                    if cnt:
                        m |= (1 << prime_index[p])
                    p += 1
                if v > 1:
                    m |= (1 << prime_index[v])
                masks[idx] = m
                
            # Mask for target
            target_mask = 0
            for p in (2, 3, 5, 7):
                target_mask |= (1 << prime_index[p])
                
            # Map mask to list of indices having that mask
            mask_to_indices = {}
            for idx in range(len(candidates)):
                m = masks[idx]
                if m not in mask_to_indices:
                    mask_to_indices[m] = []
                mask_to_indices[m].append(idx)

            # Now, for triples:
            # Iterate i, j.
            # Calculate target_mask_c = target_mask ^ masks[i] ^ masks[j]
            # Check if target_mask_c is in mask_to_indices.
            # If yes, pick any index k from that list such that k != i and k != j.
            
            for i in range(len(candidates)):
                if time.time() > deadline - 5:
                    break
                m_i = masks[i]
                for j in range(i + 1, len(candidates)):
                    m_j = masks[j]
                    m_c = target_mask ^ m_i ^ m_j
                    if m_c in mask_to_indices:
                        # Find a valid k
                        # The list might be small.
                        for k in mask_to_indices[m_c]:
                            if k != i and k != j:
                                return [candidates[i], candidates[j], candidates[k]]

    # ---- GF(2) Gaussian-elimination sweep for guaranteed minimal max ----
    # This is the fallback if no single, pair, or triple works.
    # We sweep M from LO+1 upward.
    # Maintain a basis of parity vectors.
    # Each basis row stores: (mask, set_of_indices)
    
    # basis[pivot_bit] = (mask, set_of_indices)
    basis = {}
    
    def add_to_basis(mask, idx):
        # Eliminate mask using existing basis
        m = mask
        # We need to find the highest set bit to determine pivot
        # Or just iterate through existing pivots?
        # Standard Gaussian elimination:
        # For each pivot in basis (sorted by pivot bit descending):
        #   if m has that bit set, xor with basis row
        # If m becomes 0, it's dependent.
        # Else, find new pivot (highest set bit), store.
        
        # To make it fast, we can just try to reduce m.
        # Since D is ~1200, and basis size is <= D.
        
        # We need to know which bits are pivots.
        # Let's keep a list of pivot bits.
        
        # Actually, simpler:
        # Store basis as a dictionary pivot_bit -> (mask, set)
        # When adding new mask:
        # 1. For each existing pivot bit (in descending order):
        #    if (m >> pivot) & 1:
        #       m ^= basis[pivot][0]
        #       m_set = m_set XOR basis[pivot][1]  (symmetric difference)
        # 2. If m == 0: return None (dependent)
        # 3. Else: find highest set bit in m.
        #    pivot = m.bit_length() - 1
        #    basis[pivot] = (m, m_set)
        #    return pivot
        
        m_set = {idx}
        # Get sorted pivots
        # To avoid sorting every time, we can maintain a list of pivots?
        # Or just iterate over keys. D is small (~1000).
        for p in sorted(basis.keys(), reverse=True):
            if (m >> p) & 1:
                m ^= basis[p][0]
                # Symmetric difference of sets
                m_set = m_set ^ basis[p][1]
        
        if m == 0:
            return None
            
        pivot = m.bit_length() - 1
        basis[pivot] = (m, m_set)
        return pivot

    def get_solution(target_m):
        # Reduce target_m using basis
        m = target_m
        m_set = set()
        for p in sorted(basis.keys(), reverse=True):
            if (m >> p) & 1:
                m ^= basis[p][0]
                m_set = m_set ^ basis[p][1]
        if m == 0:
            return m_set
        return None

    # Sweep M
    for M in range(LO + 1, MAXV + 1):
        if time.time() > deadline:
            break
        
        # Find index of M in candidates
        # Since candidates are sorted, we can use bisect or just map
        # We have candidates list.
        # idx = M - (LO + 1)
        idx = M - (LO + 1)
        
        # Add to basis
        add_to_basis(masks[idx], idx)
        
        # Check if target is in span
        sol = get_solution(target_mask)
        if sol is not None:
            # sol is a set of indices
            vals = [candidates[i] for i in sol]
            # Check size
            if len(vals) <= MAX_SIZE:
                return vals
            else:
                # Try to reduce?
                # For now, if too large, we might have to continue sweeping?
                # But smaller M means smaller max value.
                # If the first solution found is too large, we might miss a smaller max value solution
                # that requires more elements but smaller max?
                # No, we are sweeping M increasing.
                # The first time the target is in the span, the max element is M.
                # Any other solution with max element <= M would have been found earlier?
                # Not necessarily, because the basis might not have captured all combinations.
                # But the span is complete for the set {1..M}.
                # So if target is in span, there IS a solution with max <= M.
                # The specific solution we found might have size > 100.
                # We need a solution with size <= 100.
                # If this one is too big, we might need to find another one in the span.
                # This is a "Shortest Vector" problem in GF(2) with constraints, which is hard.
                # However, for small M, the number of elements is small.
                # If M is large, the size might be large.
                # Let's assume for now that the first solution is valid.
                # If not, we return it and hope _verify fails? No, _verify checks size.
                # If _verify fails, we should probably continue?
                # But we can't easily find another solution.
                # Let's just return it. If it's invalid, score is 0.
                # To be safe, if len > 100, we can try to remove elements?
                # Removing an element changes the parity.
                # We need to find a subset of sol with size <= 100 that still sums to target_mask.
                # This is equivalent to finding a linear dependence in sol that allows reduction.
                # This is complex.
                # Given the constraints, it's unlikely we need > 100 elements for small M.
                pass

    # Fallback
    return [840]

# EVOLVE-BLOCK-END
