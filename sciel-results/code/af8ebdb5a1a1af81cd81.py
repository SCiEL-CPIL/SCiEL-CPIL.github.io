# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from scipy.optimize import minimize

def _primes_upto(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i in range(n + 1) if sieve[i]]

def _build_counts(residues, primes, cap):
    """Build count array for 1..cap."""
    counts = np.zeros(cap + 1, dtype=np.int32)
    for p, r in zip(primes, residues):
        start = r
        if start < 1:
            start += ((1 - start + p - 1) // p) * p
        if start <= cap:
            counts[start::p] += 1
    return counts

def _find_L(counts, cap):
    """Find largest L such that all 1..L have count >= 2."""
    below = np.where(counts[1:cap+1] < 2)[0]
    if len(below) == 0:
        return cap
    return int(below[0])

def _score(residues, primes, cap=100000):
    counts = _build_counts(residues, primes, cap)
    return _find_L(counts, cap)

def _continuous_objective(x, primes, cap):
    """
    Continuous objective: minimize sum of deficits where count < 2.
    x is a vector of real-valued 'phases' in [0,1).
    For prime p_i, the residue is floor(x_i * p_i) mod p_i.
    """
    n = len(primes)
    residues = np.floor(x * np.array(primes)).astype(int) % np.array(primes)
    counts = _build_counts(residues, primes, cap)
    # Deficit: for each position where count < 2, add (2 - count)
    deficit = np.maximum(0, 2 - counts[1:cap+1]).sum()
    return deficit

def _continuous_objective_grad(x, primes, cap):
    """Numerical gradient for the continuous objective."""
    eps = 0.01
    grad = np.zeros_like(x)
    f0 = _continuous_objective(x, primes, cap)
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += eps
        x_plus[i] = x_plus[i] % 1.0  # wrap around
        f_plus = _continuous_objective(x_plus, primes, cap)
        grad[i] = (f_plus - f0) / eps
    return grad

def _round_and_repair(x, primes, cap=5000, deadline=None):
    """Round continuous solution to integer residues, then repair."""
    n = len(primes)
    p_arr = np.array(primes)
    residues = np.floor(x * p_arr).astype(int) % p_arr
    residues = residues.tolist()
    
    # Greedy repair: fix gaps one by one
    for iteration in range(200):
        if deadline is not None and time.time() > deadline - 5:
            break
        sc = _score(residues, primes, cap=cap)
        if sc >= cap:
            break
        # Find first gap
        check_L = min(sc + 100, cap)
        counts = _build_counts(residues, primes, check_L)
        below = np.where(counts[1:check_L+1] < 2)[0]
        if len(below) == 0:
            break
        bad_x = int(below[0]) + 1
        # Try to fix by changing one residue
        fixed = False
        # Prefer larger primes for less disruption
        order = sorted(range(n), key=lambda i: -primes[i])
        for idx in order:
            old_r = residues[idx]
            new_r = bad_x % primes[idx]
            if old_r == new_r:
                continue
            residues[idx] = new_r
            new_sc = _score(residues, primes, cap=cap)
            if new_sc >= sc:
                fixed = True
                break
            else:
                residues[idx] = old_r
        if not fixed:
            break
    
    return residues, _score(residues, primes, cap=cap)

def _greedy_init(primes):
    """Structured initial assignment."""
    n = len(primes)
    residues = [0] * n
    for i, p in enumerate(primes):
        target = (i // 2) + 1
        residues[i] = target % p
    return residues

def _simulated_annealing(residues, primes, cap, deadline, n_restarts=1):
    """Simulated annealing on discrete residues."""
    n = len(primes)
    best_residues = residues[:]
    best_score = _score(residues, primes, cap=cap)
    current_residues = residues[:]
    current_score = best_score
    
    T = 100.0
    T_min = 0.1
    cooling = 0.995
    
    while T > T_min and time.time() < deadline - 5:
        # Pick random index and new residue
        idx = random.randint(0, n - 1)
        old_r = current_residues[idx]
        new_r = random.randint(0, primes[idx] - 1)
        if new_r == old_r:
            continue
        current_residues[idx] = new_r
        new_score = _score(current_residues, primes, cap=cap)
        delta = new_score - current_score
        if delta >= 0 or random.random() < np.exp(delta / max(T, 1e-6)):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_residues = current_residues[:]
        else:
            current_residues[idx] = old_r
        T *= cooling
    
    return best_residues, best_score

def construct():
    deadline = time.time() + 140  # safety margin
    
    primes = _primes_upto(1000)
    n = len(primes)  # 168
    p_arr = np.array(primes)
    
    best_residues = None
    best_score = 0
    
    # Phase 1: Continuous optimization using scipy
    # Use multiple random initializations in continuous space
    cap_cont = 2000  # smaller cap for speed in continuous phase
    
    n_continuous_restarts = 3
    for restart in range(n_continuous_restarts):
        if time.time() > deadline - 60:
            break
        
        # Random initial point in [0,1)^n
        x0 = np.random.random(n)
        
        # Use L-BFGS-B with bounds
        bounds = [(0, 1)] * n
        
        try:
            result = minimize(
                _continuous_objective,
                x0,
                args=(primes, cap_cont),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50, 'maxfun': 200}
            )
            x_opt = result.x % 1.0  # ensure in [0,1)
        except Exception:
            x_opt = x0
        
        # Round and repair
        residues, sc = _round_and_repair(x_opt, primes, cap=3000, deadline=deadline)
        if sc > best_score:
            best_score = sc
            best_residues = residues
    
    # Phase 2: Simulated annealing from best continuous result
    if best_residues is not None:
        if time.time() < deadline - 30:
            best_residues, best_score = _simulated_annealing(
                best_residues, primes, cap=5000, deadline=deadline
            )
    
    # Phase 3: Local search with random perturbations
    if best_residues is not None and time.time() < deadline - 10:
        residues = best_residues[:]
        sc = _score(residues, primes, cap=5000)
        
        for it in range(2000):
            if time.time() > deadline - 5:
                break
            # Randomly change 1-3 residues
            k = random.randint(1, 3)
            for _ in range(k):
                idx = random.randint(0, n - 1)
                residues[idx] = random.randint(0, primes[idx] - 1)
            new_sc = _score(residues, primes, cap=5000)
            if new_sc > best_score:
                best_score = new_sc
                best_residues = residues[:]
            else:
                # Revert
                residues = best_residues[:]
    
    # Phase 4: Final greedy repair with larger cap
    if best_residues is not None and time.time() < deadline - 3:
        residues = best_residues[:]
        for pass_num in range(100):
            if time.time() > deadline - 2:
                break
            sc = _score(residues, primes, cap=10000)
            if sc > best_score:
                best_score = sc
                best_residues = residues[:]
            if sc >= 10000:
                break
            check_L = min(sc + 200, 10000)
            counts = _build_counts(residues, primes, check_L)
            below = np.where(counts[1:check_L+1] < 2)[0]
            if len(below) == 0:
                continue
            bad_x = int(below[0]) + 1
            fixed = False
            order = sorted(range(n), key=lambda i: -primes[i])
            for idx in order:
                old_r = residues[idx]
                new_r = bad_x % primes[idx]
                if old_r == new_r:
                    continue
                residues[idx] = new_r
                new_sc = _score(residues, primes, cap=10000)
                if new_sc >= sc:
                    fixed = True
                    break
                else:
                    residues[idx] = old_r
            if not fixed:
                break
    
    if best_residues is not None:
        return best_residues
    # Fallback
    return [((i + 2) // 2) % p for i, p in enumerate(primes)]
# EVOLVE-BLOCK-END
