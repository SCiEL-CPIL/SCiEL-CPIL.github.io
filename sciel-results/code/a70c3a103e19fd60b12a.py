"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
from collections import defaultdict

# EVOLVE-BLOCK-START
N = 210
MAX_VAL = 10210
MAX_SIZE = 100
DEADLINE = 85  # seconds, safety margin before 90s kill

def _factorize(n):
    """Return dict of prime -> exponent for n."""
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1 if d == 2 else 2
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

def _parity_vector(factors):
    """Return frozenset of primes with odd exponent."""
    return frozenset(p for p, e in factors.items() if e % 2 == 1)

def _is_perfect_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def construct():
    start_time = time.time()
    
    # Factorize N = 210 = 2 * 3 * 5 * 7
    n_factors = _factorize(N)
    target = _parity_vector(n_factors)  # frozenset({2, 3, 5, 7})
    
    # We need a subset S of {211, ..., M} such that:
    # XOR of parity vectors of elements in S == target
    # Minimize M - 210, subject to |S| <= 100, all distinct, > 210, <= 10210
    
    # Pre-compute parity vectors for all candidates
    candidates = list(range(N + 1, MAX_VAL + 1))
    parity_map = {}  # value -> frozenset
    for v in candidates:
        parity_map[v] = _parity_vector(_factorize(v))
    
    # Gaussian elimination over GF(2) using bitsets
    # Each vector is a frozenset of primes. We need to represent as bits.
    # Collect all primes that appear in any candidate or in target
    all_primes = set(target)
    for v in candidates:
        all_primes.update(parity_map[v])
    all_primes = sorted(all_primes)
    prime_to_idx = {p: i for i, p in enumerate(all_primes)}
    
    # Convert to integer bitmasks
    def to_bitmask(fset):
        mask = 0
        for p in fset:
            mask |= (1 << prime_to_idx[p])
        return mask
    
    target_mask = to_bitmask(target)
    
    # We'll do incremental Gaussian elimination as we add candidates from 211 upward.
    # For each M (from 211 upward), we check if target_mask is in the span of {parity_map[v] for v in 211..M}.
    # If yes, we can extract a subset. Then we minimize M.
    
    # Build the row-reduced echelon form incrementally
    # Basis: list of (pivot_bit, mask) pairs, kept in reduced row echelon form
    basis = []  # list of (pivot_index, mask)
    
    def reduce_mask(mask):
        """Reduce mask using basis. Return reduced mask (should be 0 if in span)."""
        for pivot, bmask in basis:
            if (mask >> pivot) & 1:
                mask ^= bmask
        return mask
    
    def add_to_basis(mask):
        """Add mask to basis if independent. Return True if added (was independent)."""
        reduced = mask
        for pivot, bmask in basis:
            if (reduced >> pivot) & 1:
                reduced ^= bmask
        if reduced == 0:
            return False
        # Find pivot (highest set bit)
        pivot = reduced.bit_length() - 1
        # Eliminate this pivot from existing basis vectors
        new_basis = []
        for p, bm in basis:
            if (bm >> pivot) & 1:
                bm ^= reduced
            new_basis.append((p, bm))
        new_basis.append((pivot, reduced))
        # Sort by pivot descending for consistent reduction
        new_basis.sort(key=lambda x: -x[0])
        # Re-reduce to maintain RREF
        for i in range(len(new_basis)):
            for j in range(i + 1, len(new_basis)):
                if (new_basis[j][1] >> new_basis[i][0]) & 1:
                    new_basis[j] = (new_basis[j][0], new_basis[j][1] ^ new_basis[i][1])
            for j in range(i - 1, -1, -1):
                if (new_basis[i][1] >> new_basis[j][0]) & 1:
                    new_basis[i] = (new_basis[i][0], new_basis[i][1] ^ new_basis[j][1])
        basis.clear()
        basis.extend(new_basis)
        return True
    
    # Also need to track which vectors contribute to each basis vector for extraction
    # For extraction, we'll use a different approach: keep track of original vector indices
    
    # Let's use a different strategy: for each M, do a fresh check using a maintained basis
    # with tracking of which original vectors are used.
    
    # Actually, let's use a simpler approach: maintain a basis with parent tracking.
    # basis[i] = (mask, set_of_original_indices)
    
    basis_with_track = []  # list of (pivot, mask, frozenset_of_indices)
    
    def reduce_with_track(mask, idx_set):
        """Reduce mask, tracking which indices are used. Return (reduced_mask, reduced_idx_set)."""
        current_mask = mask
        current_set = set(idx_set)
        for pivot, bmask, bset in basis_with_track:
            if (current_mask >> pivot) & 1:
                current_mask ^= bmask
                # Symmetric difference of index sets
                current_set = current_set ^ bset
        return current_mask, current_set
    
    def add_to_basis_tracked(mask, idx):
        """Add vector with index idx to basis. Return (success, reduced_mask, reduced_set)."""
        current_mask = mask
        current_set = {idx}
        for pivot, bmask, bset in basis_with_track:
            if (current_mask >> pivot) & 1:
                current_mask ^= bmask
                current_set = current_set ^ bset
        if current_mask == 0:
            return False, 0, current_set
        pivot = current_mask.bit_length() - 1
        # Insert into basis
        new_basis = []
        inserted = False
        for p, bm, bs in basis_with_track:
            if p == pivot:
                # Should not happen if we maintain properly
                new_basis.append((p, bm ^ current_mask, bs ^ current_set))
                inserted = True
            else:
                new_basis.append((p, bm, bs))
        if not inserted:
            new_basis.append((pivot, current_mask, current_set))
        # Sort by pivot descending
        new_basis.sort(key=lambda x: -x[0])
        # Re-reduce
        for i in range(len(new_basis)):
            for j in range(i + 1, len(new_basis)):
                if (new_basis[j][1] >> new_basis[i][0]) & 1:
                    new_basis[j] = (new_basis[j][0], new_basis[j][1] ^ new_basis[i][1], new_basis[j][2] ^ new_basis[i][2])
            for j in range(i - 1, -1, -1):
                if (new_basis[i][1] >> new_basis[j][0]) & 1:
                    new_basis[i] = (new_basis[i][0], new_basis[i][1] ^ new_basis[j][1], new_basis[i][2] ^ new_basis[j][2])
        basis_with_track.clear()
        basis_with_track.extend(new_basis)
        return True, current_mask, current_set
    
    # Incrementally add candidates and check if target is in span
    # We need to also track: when we add a vector, does it help represent target?
    
    # Alternative: for each M from 211 upward, check if target_mask reduces to 0.
    # But we need the subset. Let's track it.
    
    # Let's maintain: for the current basis, what is the representation of target_mask?
    # We'll recompute this each time we add a new vector.
    
    # Simpler: just check each M, and when found, extract the subset.
    
    best_result = None
    best_M = MAX_VAL + 1
    
    for M in range(N + 1, MAX_VAL + 1):
        if time.time() - start_time > DEADLINE:
            break
        
        v = M
        mask = parity_map[v]
        
        # Check if target is in span of {parity_map[211..M]}
        # We maintain basis incrementally. Let's just reduce target against current basis.
        # But first, add the new vector to basis.
        
        # Actually, let's check BEFORE adding: is target in span of previous vectors?
        # No, we want to include v=M as well.
        
        # Let's add v to basis, then check if target reduces to 0.
        success, red_mask, red_set = add_to_basis_tracked(mask, v)
        
        # Now check if target is in the span
        t_mask = target_mask
        t_set = set()
        for pivot, bmask, bset in basis_with_track:
            if (t_mask >> pivot) & 1:
                t_mask ^= bmask
                t_set = t_set ^ bset
        
        if t_mask == 0 and len(t_set) <= MAX_SIZE and len(t_set) >= 1:
            # We found a subset!
            selected = sorted(t_set)
            # Verify
            prod = N
            for s in selected:
                prod *= s
            if _is_perfect_square(prod):
                best_result = selected
                best_M = M
                break  # We want minimal M, and we're going in order
    
    if best_result is not None:
        # Convert to list of Python ints
        result = [int(x) for x in best_result]
        return result
    else:
        # Fallback: return a valid solution (the seed one)
        return [4 * N]
# EVOLVE-BLOCK-END
