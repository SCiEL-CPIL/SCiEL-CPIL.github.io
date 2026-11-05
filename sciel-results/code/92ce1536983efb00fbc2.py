import time
import random
import numpy as np

N = 18
MIN_DISTANCE = 6

def _min_dist_to_subspace(cand, basis, k):
    """Compute min distance from cand to subspace spanned by basis[0:k] using Gray code."""
    min_d = N + 1
    u = 0
    for i in range(1 << k):
        d = (cand ^ u).bit_count()
        if d < min_d:
            min_d = d
            if min_d < MIN_DISTANCE:
                return min_d
        # Gray code step
        if i < (1 << k) - 1:
            gc = i ^ (i + 1)
            changed_bit = gc.bit_length() - 1
            u ^= basis[changed_bit]
    return min_d

def _greedy_linear_extend(basis, target_dim, deadline, rng):
    """Greedily extend linear basis to target_dim using Gray-code accelerated distance check."""
    k = len(basis)
    while k < target_dim:
        if time.time() > deadline:
            break
        found = False
        # Try many random candidates
        max_attempts = 2000000
        for _ in range(max_attempts):
            if time.time() > deadline:
                break
            cand = rng.randint(0, (1 << N) - 1)
            md = _min_dist_to_subspace(cand, basis, k)
            if md >= MIN_DISTANCE:
                basis.append(cand)
                k += 1
                found = True
                break
        if not found:
            break
    return basis

def _greedy_nonlinear_extend(code_set, target_size, deadline, rng):
    """Greedily extend a nonlinear code set to target_size."""
    code_list = list(code_set)
    while len(code_list) < target_size:
        if time.time() > deadline:
            break
        cand = rng.randint(0, (1 << N) - 1)
        if cand in code_set:
            continue
        ok = True
        for w in code_list:
            if (cand ^ w).bit_count() < MIN_DISTANCE:
                ok = False
                break
        if ok:
            code_list.append(cand)
            code_set.add(cand)
    return code_list

def _verify_code(code_words):
    """Verify all pairs have distance >= MIN_DISTANCE."""
    n = len(code_words)
    for i in range(n):
        for j in range(i + 1, n):
            if (code_words[i] ^ code_words[j]).bit_count() < MIN_DISTANCE:
                return False
    return True

def _generate_linear_code(basis):
    """Generate all codewords from a linear basis."""
    dim = len(basis)
    words = []
    for mask in range(1 << dim):
        w = 0
        for i in range(dim):
            if (mask >> i) & 1:
                w ^= basis[i]
        words.append(w)
    return words

def construct():
    """
    Constructs the largest possible binary code of length 18 and distance 6.
    Strategy: Search for a 9-dimensional linear code basis (targeting the [18,9,6] QR code)
    starting from the Reed-Muller RM(1,4) subcode, with fallback to nonlinear greedy.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin (90s limit)
    
    # 1. Generate the Reed-Muller RM(1,4) basis (dimension 5)
    rm_basis = []
    b0 = (1 << 16) - 1  # constant 1, weight 16
    rm_basis.append(b0)
    for j in range(4):
        bj = 0
        for i in range(16):
            if (i >> j) & 1:
                bj |= 1 << i
        rm_basis.append(bj)
    
    # 2. Try multiple seeds to find a 9-dim linear code
    best_code = None
    best_size = 0
    
    for seed in [42, 123, 777, 2024, 31415]:
        if time.time() > deadline:
            break
        rng = random.Random(seed)
        basis = rm_basis.copy()
        
        # Try to extend to dimension 9
        basis = _greedy_linear_extend(basis, 9, deadline, rng)
        
        if len(basis) >= 6:
            code_words = _generate_linear_code(basis)
            if _verify_code(code_words):
                if len(code_words) > best_size:
                    best_size = len(code_words)
                    best_code = code_words
                if len(basis) == 9:
                    # Found the target! Return immediately
                    return code_words
    
    # 3. If we didn't get 512 words, try nonlinear greedy extension
    if best_code is not None and best_size < 512:
        if time.time() < deadline - 10.0:
            code_set = set(best_code)
            for seed in [42, 123, 777]:
                if time.time() > deadline:
                    break
                rng = random.Random(seed)
                extended = _greedy_nonlinear_extend(code_set, 512, deadline, rng)
                if len(extended) > best_size:
                    if _verify_code(extended):
                        best_size = len(extended)
                        best_code = extended
                    if len(extended) == 512:
                        return extended
    
    # 4. Fallback: just the RM(1,4) code (32 words)
    if best_code is None:
        best_code = _generate_linear_code(rm_basis)
    
    return best_code
