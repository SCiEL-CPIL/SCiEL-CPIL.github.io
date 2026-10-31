import time
import random
import math
import numpy as np

_START = time.time()
_DEADLINE = _START + 115.0  # Safety margin vs 150 s kill

def _score_from_set(U):
    """
    Compute the score for a set U.
    Score = 1 + log(|U-U|/|U+U|) / log(2*max(U) + 1)
    """
    if not U:
        return 0.0
    max_val = max(U)
    if max_val < 1:
        return 0.0
    
    # FFT based counting
    # We need linear convolution/correlation.
    # Pad to next power of 2 >= 2*max_val + 1
    L = max_val + 1
    fft_len = 1
    while fft_len < 2 * L - 1:
        fft_len *= 2
        
    vec = np.zeros(fft_len, dtype=np.float64)
    for u in U:
        vec[u] = 1.0
        
    f = np.fft.fft(vec)
    
    # Sumset |U+U|
    # Coeffs of (sum v[i] x^i)^2
    conv_sum = np.fft.ifft(f * f).real
    sums_count = np.sum(conv_sum > 0.5)
    
    # Difference set |U-U|
    # Cross-correlation
    conv_diff = np.fft.ifft(np.conj(f) * f).real
    diffs_count = np.sum(np.abs(conv_diff) > 0.5)
    
    if sums_count == 0:
        return 0.0
        
    ratio = diffs_count / float(sums_count)
    if ratio <= 0:
        return 0.0
        
    return 1.0 + math.log(ratio) / math.log(2 * max_val + 1)

def _digits_to_set(digits, base):
    """
    Convert a list of digits (representing a single number in base 'base') 
    to a set? No, the strategy says U is the image of a linear recurrence 
    or we treat the elements of U as having base-B representations.
    
    Actually, the prompt says: "Construct sets U as the image of a linear recurrence sequence 
    (e.g., powers of a large integer base B) ... then apply a bit-level genetic algorithm 
    that mutates individual digits of the base-B representation to optimize the score."
    
    This implies we represent each element u in U by its base-B digits.
    However, mutating "individual digits" of the elements in U is a high-dimensional search.
    
    Alternative interpretation:
    We define U by a set of "codewords" or indices.
    Let's stick to the most direct interpretation: 
    We have a set U. We represent each u in U in base B.
    We mutate the digits of these representations.
    
    But how do we initialize?
    The prompt suggests "powers of a large integer base B".
    If U = {B^0, B^1, ..., B^{k-1}}, then |U+U| = k(k+1)/2 (all sums distinct if B is large enough? 
    No, B^i + B^j = B^k + B^l implies {i,j}={k,l}. So |U+U| = k(k+1)/2.
    |U-U|: B^i - B^j. Distinct pairs give distinct diffs? 
    B^i - B^j = B^k - B^l. 
    If i>j, k>l. 
    B^i(1-B^{j-i}) = B^k(1-B^{l-k}).
    This implies i=k and j=l or some symmetry. 
    Generally, for large B, the differences are distinct. 
    |U-U| = k^2 (including 0 and +/-).
    Ratio R = k^2 / (k(k+1)/2) = 2k/(k+1) -> 2 as k -> inf.
    Score = 1 + log(2) / log(2 B^{k-1} + 1) ~ 1 + log(2)/((k-1)log B).
    This score is very close to 1 for large k.
    
    To get a better score, we need R close to 2 (max for Sidon) and small M.
    Or R > 2? Max R is 2 for Sidon. 
    Wait, |U-U| <= 2|U|^2 - |U| (if 0 in U). |U+U| >= |U|(|U|+1)/2.
    Max ratio is ~ 2|U|^2 / (|U|^2/2) = 4? No.
    For Sidon: |U+U| = |U|(|U|+1)/2. |U-U| = 2|U|(|U|-1)+1.
    Ratio ~ 4.
    Wait, let's re-evaluate.
    If U is Sidon, sums are unique (mostly). |U+U| ~ n^2/2.
    Diffs: u_i - u_j. For i!=j, u_i - u_j != u_k - u_l unless {i,j}={k,l} (symmetry).
    So |U-U| ~ 2n^2.
    Ratio ~ 4.
    Score = 1 + log(4) / log(2M+1).
    
    So we want a Sidon set with small M.
    Greedy Golomb rulers are Sidon sets.
    The current program uses random local search.
    The new strategy: Digit-level GA on base-B representation.
    
    Let's define the representation:
    We fix a base B (e.g., 3 or 4).
    We represent the set U as a list of integers.
    Each integer is represented by its digits in base B.
    We mutate digits.
    
    To make this efficient, we should probably fix the number of elements n and the "length" of the digits (i.e., max value M).
    Then we have a matrix of digits.
    
    Let's try:
    1. Choose n (size of U) and L (number of digits in base B).
    2. Initialize U with a greedy Sidon set or random.
    3. Represent each u in U as a vector of digits d_{u, 0..L-1}.
    4. Mutation: Pick a random element u, pick a random digit position, change the digit.
    5. Evaluate score.
    
    This is still local search, but the move space is structured.
    Changing a digit changes the value by a multiple of B^k.
    This allows "jumps" in value space that are aligned with the base.
    
    Let's implement this.
    """
    pass # Placeholder

def _generate_initial_sidon(n, base, max_val):
    """
    Generate a simple Sidon set (Golomb ruler) of size n.
    """
    U = [0]
    diffs = {0}
    current_max = 0
    for _ in range(n - 1):
        candidate = current_max + 1
        while True:
            valid = True
            for u in U:
                d = candidate - u
                if d in diffs:
                    valid = False
                    break
            if valid:
                U.append(candidate)
                for u in U[:-1]:
                    diffs.add(candidate - u)
                current_max = candidate
                break
            else:
                candidate += 1
            if candidate > max_val:
                # Fallback to random if greedy fails to find within bound
                # But greedy Golomb grows exponentially, so max_val should be large enough
                # For small n, it's fine.
                pass
    return U

def construct():
    global _DEADLINE
    
    best_score = 0.0
    best_set = [0, 1]
    
    # Strategy:
    # 1. Try different bases B (2, 3, 4, 5).
    # 2. Try different sizes n (5 to 30).
    # 3. Initialize with greedy Golomb rulers (which are Sidon).
    # 4. Perform digit-level mutations.
    
    random.seed(42)
    
    bases = [2, 3, 4, 5]
    sizes = [5, 10, 15, 20, 25]
    
    # Pre-compute powers of base for fast digit extraction/modification
    # We will work with the set U directly, but mutations are defined in base B.
    
    def _get_digits(val, base, num_digits):
        """Return list of digits from least significant."""
        digits = []
        for _ in range(num_digits):
            digits.append(val % base)
            val //= base
        return digits

    def _set_digit(val, base, pos, new_digit):
        """Set the digit at position pos to new_digit."""
        # val + (new_digit - old_digit) * base^pos
        old_digit = (val // (base ** pos)) % base
        return val + (new_digit - old_digit) * (base ** pos)

    def _digit_local_search(U_init, base, time_limit):
        current_U = sorted(U_init)
        current_score = _score_from_set(current_U)
        best_U = current_U[:]
        best_score_local = current_score
        
        start_time = time.time()
        no_improvement = 0
        max_no_improvement = 2000 # Allow more exploration in structured space
        
        # Determine max span
        max_val = max(current_U)
        # Estimate max digits needed
        # If max_val is M, digits ~ log_B(M)
        # We will allow mutations to increase max_val slightly.
        # Cap max_val to keep FFT fast.
        max_allowed_val = 5000
        
        while time.time() < start_time + time_limit:
            if no_improvement > max_no_improvement:
                break
                
            # Pick a random element
            idx = random.randint(0, len(current_U) - 1)
            if idx == 0 and len(current_U) > 1:
                # Keep 0 fixed to ensure 0 in U? 
                # The problem requires 0 in U. 
                # If we move 0, we must ensure 0 is still in U or re-add it.
                # Easier: never move 0.
                continue
                
            u = current_U[idx]
            
            # Determine number of digits to consider
            # We consider digits up to the highest non-zero digit, plus 1 for expansion.
            if u == 0:
                num_digits = 1
            else:
                num_digits = int(math.log(u, base)) + 2 if base > 1 else 1
            
            # Pick a random digit position
            pos = random.randint(0, num_digits - 1)
            
            # Pick a new digit
            old_digit = (u // (base ** pos)) % base
            new_digit = random.randint(0, base - 1)
            
            if new_digit == old_digit:
                continue
                
            new_val = _set_digit(u, base, pos, new_digit)
            
            # Check bounds
            if new_val > max_allowed_val:
                continue
                
            # Check uniqueness
            if new_val in current_U:
                continue
                
            # Apply move
            new_U = current_U[:]
            new_U[idx] = new_val
            new_U.sort()
            
            # Evaluate
            new_score = _score_from_set(new_U)
            
            if new_score > current_score:
                current_U = new_U
                current_score = new_score
                no_improvement = 0
                if current_score > best_score_local:
                    best_score_local = current_score
                    best_U = current_U[:]
            else:
                no_improvement += 1
                
        return best_U, best_score_local

    # Main Loop
    for base in bases:
        if time.time() > _DEADLINE - 10.0:
            break
            
        for n in sizes:
            if time.time() > _DEADLINE - 10.0:
                break
            
            # Initialize with Golomb ruler
            try:
                U_init = _generate_initial_sidon(n, base, 5000)
            except:
                continue
                
            # Allocate time
            remaining = _DEADLINE - time.time()
            # Distribute time
            # 10 configs * 4 bases = 40 configs.
            # ~2s each.
            time_per_config = max(1.0, min(2.0, remaining / 40))
            
            U_opt, score_opt = _digit_local_search(U_init, base, time_per_config)
            
            if score_opt > best_score:
                best_score = score_opt
                best_set = U_opt[:]
                
    # Final polish
    if time.time() < _DEADLINE - 5.0 and len(best_set) > 1:
        remaining = _DEADLINE - time.time() - 5.0
        # Pick a base that worked well or just 3
        U_polished, score_polished = _digit_local_search(best_set, 3, remaining)
        if score_polished > best_score:
            best_score = score_polished
            best_set = U_polished[:]

    # Sanitize
    best_set = sorted(set(int(x) for x in best_set))
    if 0 not in best_set:
        best_set.append(0)
    if len(best_set) < 2:
        best_set = [0, 1]
        
    return best_set

# EVOLVE-BLOCK-END
