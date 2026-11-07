"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from collections import defaultdict

P = {'n': 210, 'maximum_window': 10210, 'maximum_size': 100}

def _is_perfect_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _squarefree_signature(n):
    """Return frozenset of primes with odd exponent in n."""
    sig = set()
    d = 2
    while d * d <= n:
        cnt = 0
        while n % d == 0:
            n //= d
            cnt ^= 1
        if cnt:
            sig.add(d)
        d += 1
    if n > 1:
        sig.add(n)
    return frozenset(sig)

def construct():
    deadline = time.time() + 80  # 80s safety margin within 90s limit
    n = P['n']  # 210
    max_val = P['maximum_window']
    max_size = P['maximum_size']
    
    # 210 = 2 * 3 * 5 * 7, all to power 1
    sig_210 = frozenset({2, 3, 5, 7})
    
    # Precompute signatures for all numbers in (210, max_val]
    sigs = [None] * (max_val + 1)
    for i in range(211, max_val + 1):
        sigs[i] = _squarefree_signature(i)
    
    # Build lookup: signature -> list of numbers with that signature
    sig_to_nums = defaultdict(list)
    for i in range(211, max_val + 1):
        sig_to_nums[sigs[i]].append(i)
    
    best_result = None
    best_max = max_val + 1
    
    # Search for solutions incrementally by max value
    # For each M (as the maximum), check if there's a subset with max = M
    for M in range(211, max_val + 1):
        if time.time() > deadline:
            break
        if M >= best_max:
            break
        
        # Single element: sig(M) == sig_210
        if sigs[M] == sig_210:
            if M < best_max:
                best_max = M
                best_result = [M]
            continue
        
        # Pair: need a with sigs[a] = sigs[M] XOR sig_210, 210 < a < M
        target_sig = sigs[M] ^ sig_210
        if target_sig in sig_to_nums:
            for a in sig_to_nums[target_sig]:
                if a < M:
                    if M < best_max:
                        best_max = M
                        best_result = [a, M]
                    break
        
        # Triple: need a < b < M with sigs[a] XOR sigs[b] = sig_210 XOR sigs[M]
        if M < best_max:
            target = sig_210 ^ sigs[M]
            seen = {}
            for a in range(211, M):
                if time.time() > deadline:
                    break
                needed = sigs[a] ^ target
                if needed in seen:
                    b = seen[needed]
                    if b != a:
                        if M < best_max:
                            best_max = M
                            best_result = [a, b, M]
                        break
                else:
                    seen[sigs[a]] = a
                if best_result is not None and best_max == M:
                    break
    
    if best_result is not None:
        return [int(x) for x in best_result]
    
    # Fallback: [4*n] = [840], since 840*210 = 4*210*210 = 4*210^2 = (2*210)^2
    return [4 * n]

# EVOLVE-BLOCK-END
