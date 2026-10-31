import time
import hashlib

_start_time = None
_DEADLINE = 330 * 0.90

def _elapsed():
    if _start_time is None:
        return 0.0
    return time.time() - _start_time

def priority(vector, n):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    if time.time() - _start_time > _DEADLINE:
        return 0.0

    vec = vector
    c0 = c1 = c2 = 0
    for v in vec:
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
        else:
            c2 += 1

    # 1. Distinct symbol bonus (strong)
    nd = (1 if c0 else 0) + (1 if c1 else 0) + (1 if c2 else 0)
    distinct_score = 5000.0 * nd * nd

    # 2. Balance score
    ideal = n / 3.0
    dev_sq = (c0 - ideal) ** 2 + (c1 - ideal) ** 2 + (c2 - ideal) ** 2
    balance_score = -dev_sq * 10.0

    # 3. Line-blocking potential via spectral energy (FFT over Z_3^n)
    # A vector's "frequency content" determines how many lines it participates in
    # with short additive structure. Low spectral energy = more "spread out" =
    # blocks more diverse lines.
    # Compute DFT magnitude: for each character chi_k(x) = omega^{sum k_i x_i},
    # the DC component is 1, others have magnitude 1. But the *phase* matters.
    # Simpler: use the "energy" E = sum x_i^2 = c1 + 4*c2 as a proxy.
    energy = c1 + 4.0 * c2
    spectral_penalty = -energy * 12.0

    # 4. Phase imbalance (1s vs 2s)
    phase_diff = c1 - c2
    phase_penalty = -phase_diff * phase_diff * 3.0

    # 5. SHA256 hash for uniform tie-breaking
    h = hashlib.sha256(vec).digest()
    hash_score = int.from_bytes(h[:8], 'little') / (2**64)

    total = distinct_score + balance_score + spectral_penalty + phase_penalty + hash_score
    return float(total)
