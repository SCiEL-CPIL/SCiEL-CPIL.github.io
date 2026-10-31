import time
import math

_start_time = None
_DEADLINE = 330 * 0.95  # 313.5s

# Precompute base-3 powers up to a reasonable n (say 64) for fast ternary conversion
_MAX_N = 64
_POW3 = [1] * (_MAX_N + 1)
for i in range(1, _MAX_N + 1):
    _POW3[i] = _POW3[i-1] * 3

# Fixed mask for XOR-based chaotic interleaving
_MASK = 0x5A5A5A5A5A5A5A5A  # 64-bit pattern

def priority(vector, n):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    if time.time() - _start_time > _DEADLINE:
        return 0.0
    
    # Count occurrences of each symbol
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    
    # Distinct count: strongly prefer vectors using all 3 symbols
    distinct = (1 if c0 > 0 else 0) + (1 if c1 > 0 else 0) + (1 if c2 > 0 else 0)
    # Primary structural signal: distinct count dominates
    distinct_score = distinct * 10000.0
    
    # Balance: L1 deviation from uniform distribution
    ideal = n / 3.0
    dev = abs(c0 - ideal) + abs(c1 - ideal) + abs(c2 - ideal)
    balance_score = -dev * 10.0
    
    # Bitwise parity-based chaotic interleaving:
    # Convert vector to ternary integer, XOR with mask, count bits
    # This creates a deterministic but chaotic ordering that breaks
    # correlation between local symbol balance and global line formation
    ternary_val = 0
    for i in range(n):
        if i < _MAX_N:
            ternary_val += vector[i] * _POW3[i]
    
    # XOR with fixed mask to create chaotic interleaving
    xor_val = ternary_val ^ _MASK
    # Count set bits (popcount)
    bit_count = bin(xor_val).count('1')
    
    # Normalize bit count to [0, 1)
    max_bits = min(n * 2, 64)  # rough max bits
    norm_bits = bit_count / max_bits if max_bits > 0 else 0.0
    
    # Secondary hash for fine tie-breaking
    h = 0
    for i, val in enumerate(vector):
        h = (h * 31 + val + i) & 0x7FFFFFFF
    hash_score = h / 0x7FFFFFFF
    
    # Combine: distinct dominates, then balance, then chaotic bit pattern, then hash
    # The bit pattern provides a structured but non-obvious ordering that
    # should reduce correlation with line-forming patterns
    score = distinct_score + balance_score + norm_bits * 5.0 + hash_score * 0.01
    
    return float(score)
