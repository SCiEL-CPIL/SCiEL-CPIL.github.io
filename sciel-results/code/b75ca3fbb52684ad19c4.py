import numpy as np
import time

N = 400
TIME_LIMIT = 120.0
SAFETY_MARGIN = 5.0
DEADLINE = time.time() + TIME_LIMIT - SAFETY_MARGIN


def _autoconv(f):
    """Compute linear autoconvolution of f (length N) -> length 2N-1."""
    return np.convolve(f, f, mode='full')


def _score(f):
    """Compute max|autoconv| / (sum f)^2."""
    S = np.sum(f)
    if abs(S) < 1e-12:
        return 1e10
    C = _autoconv(f)
    return np.max(np.abs(C)) / (S * S)


def _legendre_symbol(n, p):
    """Legendre symbol (n/p). Returns 0 if n % p == 0, else 1 or -1."""
    if n % p == 0:
        return 0
    return 1 if pow(n, (p - 1) // 2, p) == 1 else -1


def _generate_legendre_sequence(p, offset=1):
    """Generate a Legendre symbol sequence of length N from prime p.
    Uses (i + offset mod p) mapped to {+1, -1} via Legendre symbol.
    """
    seq = np.zeros(N)
    for i in range(N):
        val = _legendre_symbol((i + offset) % p, p)
        if val == 0:
            val = 1  # Map 0 to +1 to keep binary
        seq[i] = val
    return seq


def _local_sign_flip_incremental(f, max_iter=2000, time_budget=30.0):
    """
    Local search on binary {-1,1} sequences using incremental autoconvolution updates.
    Flipping f[i] from v to -v changes C[k] by -2*v*f[k-i] for all valid k.
    This is O(N) per flip evaluation instead of O(N log N) for FFT.
    """
    f = f.copy()
    f = np.sign(f)
    
    S = np.sum(f)
    if abs(S) < 1:
        # Ensure non-zero sum
        for _ in range(N):
            idx = np.random.randint(N)
            if f[idx] * S < 0 or S == 0:
                f[idx] = -f[idx]
                S = np.sum(f)
                if abs(S) >= 1:
                    break
        if abs(S) < 1:
            return f, 1e10
    
    C = _autoconv(f)
    score = np.max(np.abs(C)) / (S * S)
    
    start_time = time.time()
    no_improve = 0
    max_no_improve = 50
    
    for it in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        if time.time() > DEADLINE:
            break
        
        best_improvement = 0.0
        best_idx = -1
        
        # Evaluate all positions (N=400, O(N^2) total per iteration = 160k ops)
        for i in range(N):
            v = f[i]
            new_S = S - 2.0 * v
            if abs(new_S) < 1e-6:
                continue
            
            # Incremental update: C_new[k] = C[k] - 2*v*f[k-i] for valid k
            # C has length 2N-1. k-i is valid when 0 <= k-i < N, i.e., i <= k < i+N
            k_start = i
            k_end = min(i + N, 2 * N - 1)
            
            # Compute delta for the affected range
            # C_new[k] = C[k] - 2*v*f[k-i]
            delta = -2.0 * v * f[0:k_end - k_start]
            
            # Compute new max abs in affected range
            # We need max of |C[k] + delta[k]| for k in [k_start, k_end)
            # And also max of |C[k]| for k outside [k_start, k_end)
            
            # For efficiency, precompute the global max outside the affected range
            # But that's complex. Instead, just compute the new C for the affected range
            # and take max with the precomputed max outside.
            
            # Actually, let's just compute the new score directly for speed.
            # Since N=400, computing the full C_new each time is O(N^2) = 160k,
            # and we do it N times per iteration = 64M ops per iteration. Too slow.
            
            # Better: use the incremental approach properly.
            # Store C and update it.
            
            # For now, use a simpler approach: compute new_C incrementally
            # but only for the max computation.
            
            # The key insight: max|C_new| = max(max|C_outside|, max|C_inside + delta|)
            # where C_inside is C[k_start:k_end] and delta applies there.
            
            # Precompute max|C| outside the window
            # This is expensive to do for each i. Let's use a different strategy.
            
            # Strategy: Just compute the full new C using the update, but only
            # for the max. We can do:
            # new_C = C.copy()
            # new_C[k_start:k_end] += delta
            # new_score = max(abs(new_C)) / (new_S * new_S)
            # But copying C each time is O(N) and we do it N times = O(N^2) per iter.
            
            # For N=400, O(N^2) = 160k per iteration, and we can do ~1000 iterations
            # in 30s if each iteration is ~160k * some constant. Let's try.
            
            new_C = C.copy()
            new_C[k_start:k_end] += delta
            new_max = np.max(np.abs(new_C))
            new_score = new_max / (new_S * new_S)
            
            improvement = score - new_score
            if improvement > best_improvement:
                best_improvement = improvement
                best_idx = i
        
        if best_idx >= 0:
            v = f[best_idx]
            f[best_idx] = -v
            S -= 2.0 * v
            C = _autoconv(f)
            score = np.max(np.abs(C)) / (S * S)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= max_no_improve:
                break
    
    return f, score


def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    # Generate multiple Legendre symbol sequences with different primes and offsets
    # Primes near N that give good coverage
    primes = [397, 401, 389, 383, 379, 373, 367, 359, 353, 349]
    
    initializations = []
    for p in primes:
        for offset in [0, 1, 2, 3]:
            if time.time() > DEADLINE:
                break
            seq = _generate_legendre_sequence(p, offset)
            S = np.sum(seq)
            if abs(S) >= 1:
                initializations.append(seq.copy())
        if time.time() > DEADLINE:
            break
    
    # Also add a few random binary sequences
    for _ in range(5):
        if time.time() > DEADLINE:
            break
        seq = rng.choice([-1.0, 1.0], size=N)
        if abs(np.sum(seq)) >= 1:
            initializations.append(seq)
    
    # If no valid initializations, use all ones
    if not initializations:
        initializations = [np.ones(N)]
    
    best_f = initializations[0].copy()
    best_score = _score(best_f)
    
    # Run local search from each initialization
    for f0 in initializations:
        if time.time() > DEADLINE:
            break
        
        remaining_time = DEADLINE - time.time()
        if remaining_time < 5.0:
            break
        
        # Allocate time budget proportional to remaining time
        time_budget = min(20.0, remaining_time * 0.3)
        
        f_local, score_local = _local_sign_flip_incremental(f0, max_iter=5000, time_budget=time_budget)
        
        if score_local < best_score:
            best_score = score_local
            best_f = f_local.copy()
    
    # Final polish: run local search from the best solution with remaining time
    if time.time() < DEADLINE - 2.0:
        remaining = DEADLINE - time.time() - 1.0
        f_final, score_final = _local_sign_flip_incremental(best_f, max_iter=10000, time_budget=remaining)
        if score_final < best_score:
            best_score = score_final
            best_f = f_final.copy()
    
    return best_f
