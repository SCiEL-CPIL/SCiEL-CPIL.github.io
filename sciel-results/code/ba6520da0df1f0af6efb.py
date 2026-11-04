import time
from math import log, floor, ceil
import itertools
import numpy as np

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
        
    U_arr = np.array(U, dtype=np.int64)
    sums_arr = U_arr[:, None] + U_arr[None, :]
    unique_sums = np.unique(sums_arr)
    diffs_arr = U_arr[:, None] - U_arr[None, :]
    unique_diffs = np.unique(diffs_arr)
    
    ratio = len(unique_diffs) / len(unique_sums)
    if ratio <= 1:
        return -1
        
    max_u = U_arr[-1]
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def score_set_fast(U):
    """Faster scoring using numpy vectorized operations."""
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    
    U_arr = np.array(U, dtype=np.int64)
    
    # Use numpy for all sizes - vectorized operations are fast
    sums_arr = U_arr[:, None] + U_arr[None, :]
    unique_sums_count = np.unique(sums_arr).shape[0]
    diffs_arr = U_arr[:, None] - U_arr[None, :]
    unique_diffs_count = np.unique(diffs_arr).shape[0]
    
    ratio = unique_diffs_count / unique_sums_count
    if ratio <= 1:
        return -1
        
    max_u = U_arr[-1]
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def score_set_vectorized(U):
    """Ultra-fast scoring using only numpy unique counts."""
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    
    U_arr = np.array(U, dtype=np.int64)
    
    sums_arr = U_arr[:, None] + U_arr[None, :]
    unique_sums_count = len(np.unique(sums_arr))
    diffs_arr = U_arr[:, None] - U_arr[None, :]
    unique_diffs_count = len(np.unique(diffs_arr))
    
    ratio = unique_diffs_count / unique_sums_count
    if ratio <= 1:
        return -1
        
    max_u = U_arr[-1]
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def hamming_code_set(n_bits, rate=2):
    """
    Generate a set inspired by Hamming code structure.
    Uses positions where binary representation has controlled weight patterns.
    """
    if n_bits < 2:
        return [0]
    
    # Generate all 2^n_bits combinations, select those with specific weight patterns
    # For a rate-2 Hamming code, we want elements where the number of 1-bits
    # follows a specific pattern to minimize sum collisions
    
    elements = [0]
    
    # Strategy: select elements based on bit-weight classes
    # Elements with even weight create structured sum collisions
    # Elements with odd weight create more unique differences
    
    for mask in range(1, 2**n_bits):
        weight = bin(mask).count('1')
        # Select based on a pattern that maximizes difference/sum ratio
        # Keep elements where weight is in a specific set
        if weight % rate == 0 or weight == 1:
            elements.append(mask)
    
    return sorted(set(elements))

def binary_structure_set(n_bits, pattern='sparse'):
    """
    Generate sets with binary structure to exploit unique difference representations.
    """
    if n_bits < 2:
        return [0]
    
    elements = [0]
    
    if pattern == 'powers':
        # Powers of 2 - each difference is unique
        for k in range(n_bits):
            elements.append(1 << k)
    elif pattern == 'sparse_binary':
        # Select binary numbers with limited 1-bits
        for mask in range(1, 2**n_bits):
            weight = bin(mask).count('1')
            if weight <= 2:
                elements.append(mask)
    elif pattern == 'alternating':
        # Alternating bit patterns
        for mask in range(1, 2**n_bits):
            # Keep if bits alternate or have specific structure
            bits = bin(mask)[2:].zfill(n_bits)
            if all(bits[i] != bits[i+1] for i in range(len(bits)-1) if i+1 < len(bits)):
                elements.append(mask)
            elif mask & (mask >> 1) == 0:  # No adjacent 1-bits
                elements.append(mask)
    elif pattern == 'ldpc_like':
        # LDPC-inspired: sparse rows in binary representation
        for mask in range(1, 2**n_bits):
            weight = bin(mask).count('1')
            # Sparse: at most 3 ones, and no two adjacent
            if weight <= 3:
                bits = bin(mask)[2:].zfill(n_bits)
                if '11' not in bits:
                    elements.append(mask)
    
    return sorted(set(elements))

def geometric_set(base, n_terms):
    """Generate geometric progression set."""
    U = [0]
    for k in range(1, n_terms + 1):
        val = floor(base ** k)
        if val > 0 and val not in U:
            U.append(val)
    return sorted(set(U))

def mixed_binary_geometric(n_bits, base, n_terms):
    """Mix binary structure with geometric progression."""
    binary_elems = set()
    for k in range(n_bits):
        binary_elems.add(1 << k)
    
    geom_elems = set()
    for k in range(1, n_terms + 1):
        val = floor(base ** k)
        if val > 0:
            geom_elems.add(val)
    
    U = [0] + sorted(binary_elems | geom_elems)
    return U

def construct():
    deadline = time.time() + 145  # Safety margin for 150s limit
    
    best_score = -1
    best_set = [0, 1, 4, 9, 11]
    
    s = score_set_fast(best_set)
    if s > best_score:
        best_score = s
    
    # Phase 1: Binary structure constructions (key insight from observations)
    if time.time() < deadline - 30:
        # Powers of 2 - unique differences
        for n_bits in range(3, 20):
            if time.time() > deadline - 30:
                break
            U = binary_structure_set(n_bits, 'powers')
            s = score_set_fast(U)
            if s > best_score:
                best_score = s
                best_set = list(U)
        
        # Sparse binary (weight <= 2)
        for n_bits in range(4, 16):
            if time.time() > deadline - 30:
                break
            U = binary_structure_set(n_bits, 'sparse_binary')
            s = score_set_fast(U)
            if s > best_score:
                best_score = s
                best_set = list(U)
        
        # LDPC-like (sparse, no adjacent 1-bits)
        for n_bits in range(4, 18):
            if time.time() > deadline - 30:
                break
            U = binary_structure_set(n_bits, 'ldpc_like')
            s = score_set_fast(U)
            if s > best_score:
                best_score = s
                best_set = list(U)
        
        # Hamming code inspired
        for n_bits in range(3, 14):
            if time.time() > deadline - 30:
                break
            for rate in [2, 3]:
                U = hamming_code_set(n_bits, rate)
                if len(U) >= 2:
                    s = score_set_fast(U)
                    if s > best_score:
                        best_score = s
                        best_set = list(U)
    
    # Phase 2: Geometric progressions
    if time.time() < deadline - 25:
        for m in range(3, 15):
            if time.time() > deadline - 25:
                break
            # Coarse grid search for base
            best_b = None
            best_s_coarse = -1
            for i in range(50):
                b = 1.01 + (15.0 - 1.01) * i / 49.0
                U = geometric_set(b, m)
                if len(U) < 2:
                    continue
                s = score_set_fast(U)
                if s > best_s_coarse:
                    best_s_coarse = s
                    best_b = b
            
            if best_b is not None and time.time() < deadline - 25:
                # Fine search around best
                lo = max(1.001, best_b * 0.9)
                hi = min(15.0, best_b * 1.1)
                for i in range(30):
                    b = lo + (hi - lo) * i / 29.0
                    U = geometric_set(b, m)
                    if len(U) < 2:
                        continue
                    s = score_set_fast(U)
                    if s > best_score:
                        best_score = s
                        best_set = list(U)
    
    # Phase 3: Mixed binary + geometric
    if time.time() < deadline - 20:
        for n_bits in range(3, 12):
            if time.time() > deadline - 20:
                break
            for base in [1.5, 2.0, 2.5, 3.0, 4.0]:
                for n_terms in range(3, 10):
                    if time.time() > deadline - 20:
                        break
                    U = mixed_binary_geometric(n_bits, base, n_terms)
                    if len(U) >= 2:
                        s = score_set_fast(U)
                        if s > best_score:
                            best_score = s
                            best_set = list(U)
    
    # Phase 4: Exhaustive search for small sets
    if time.time() < deadline - 15:
        max_val = 30
        for n in range(3, 6):
            if time.time() > deadline - 15:
                break
            for combo in itertools.combinations(range(1, max_val + 1), n - 1):
                if time.time() > deadline - 15:
                    break
                U = (0,) + combo
                s = score_set_fast(U)
                if s > best_score:
                    best_score = s
                    best_set = list(U)
    
    # Phase 5: Local search / perturbation
    if time.time() < deadline - 5:
        current_set = set(best_set)
        
        # Try adding elements
        for new_elem in range(1, 200):
            if time.time() > deadline - 5:
                break
            if new_elem in current_set:
                continue
            new_set = current_set | {new_elem}
            s = score_set_fast(list(new_set))
            if s > best_score:
                best_score = s
                best_set = list(new_set)
                current_set = new_set
        
        # Try removing elements
        if len(best_set) > 2:
            for elem in list(best_set):
                if time.time() > deadline - 5:
                    break
                if elem == 0:
                    continue
                new_set = set(best_set) - {elem}
                if len(new_set) < 2:
                    continue
                s = score_set_fast(list(new_set))
                if s > best_score:
                    best_score = s
                    best_set = list(new_set)
        
        # Try shifting elements
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            improved = False
            for i, elem in enumerate(list(best_set)):
                if time.time() > deadline - 5:
                    break
                if elem == 0:
                    continue
                for delta in range(-5, 6):
                    if delta == 0:
                        continue
                    new_elem = elem + delta
                    if new_elem < 0 or new_elem in best_set:
                        continue
                    new_set = list(best_set)
                    new_set[i] = new_elem
                    new_set = sorted(set(new_set))
                    if 0 not in new_set or len(new_set) < 2:
                        continue
                    s = score_set_fast(new_set)
                    if s > best_score:
                        best_score = s
                        best_set = new_set
                        improved = True
                        break
            if not improved:
                break
        
        # Scale the set
        for scale in [2, 3, 5]:
            if time.time() > deadline - 5:
                break
            scaled = [0] + [x * scale for x in best_set if x > 0]
            s = score_set_fast(scaled)
            if s > best_score:
                best_score = s
                best_set = scaled
    
    # Phase 6: Final polish - try binary structures around best max value
    if time.time() < deadline - 3:
        max_u = max(best_set)
        n_bits_needed = max(3, max_u.bit_length() + 1)
        
        for nb in range(n_bits_needed, n_bits_needed + 4):
            if time.time() > deadline - 3:
                break
            for pattern in ['powers', 'sparse_binary', 'ldpc_like']:
                U = binary_structure_set(nb, pattern)
                if len(U) >= 2:
                    s = score_set_fast(U)
                    if s > best_score:
                        best_score = s
                        best_set = list(U)
    
    return best_set
