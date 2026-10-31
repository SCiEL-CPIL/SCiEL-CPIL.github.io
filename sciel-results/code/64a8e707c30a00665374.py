"""Cap set priority using frequency-domain spectral analysis."""
import numpy as np
from functools import lru_cache
import time

# Deadline guard: track start time
_START_TIME = time.time()
_DEADLINE = 330.0  # wall-clock limit
_SAFETY_MARGIN = 5.0  # seconds before deadline to stop

def _time_remaining():
    return _DEADLINE - _SAFETY_MARGIN - (time.time() - _START_TIME)

@lru_cache(maxsize=1)
def _build_fourier_basis(n):
    """
    Build the DFT matrix for Z_3^n.
    For each character chi_k(x) = omega^{k·x} where omega = e^{2πi/3},
    k ranges over Z_3^n.
    
    We work in the frequency domain: for a vector v in {0,1,2}^n,
    we compute its "Fourier signature" which captures how it interacts
    with the group structure.
    
    Key insight: A set S is a cap set iff its characteristic function
    has no three-term AP structure. In Fourier terms, we want vectors
    whose individual contributions to the "AP-detecting" frequencies
    are minimal.
    
    For a single vector v, we can compute:
    - Its "self-correlation" with shifts that would form APs
    - The magnitude of its projection onto each character
    
    We use the fact that for the greedy algorithm, we want to pick
    vectors that are "orthogonal" to existing AP patterns.
    
    Simplified approach: For vector v, compute the sum of |DFT(v)|^2
    over all non-trivial characters, weighted to penalize frequencies
    that detect 3-term APs.
    """
    omega = np.exp(2j * np.pi / 3)
    
    # For Z_3^n, the characters are indexed by k in {0,1,2}^n
    # chi_k(x) = omega^{sum(k_i * x_i)}
    
    # We'll precompute the character table for efficiency
    # For n dimensions, there are 3^n characters
    # But for large n, this is infeasible, so we use a sampling approach
    
    return omega

def _fourier_magnitude_sampled(vector, n, num_samples=200):
    """
    Compute a sampled estimate of the L2 norm of the DFT of the vector
    over non-trivial characters.
    
    For a vector v in {0,1,2}^n, the DFT coefficient at character k is:
    F(k) = omega^{k·v}
    
    The |F(k)|^2 = 1 for all k (since |omega^x| = 1).
    
    So instead, we need to think about how the vector interacts with
    the SET being built. The priority should reflect how "safe" this
    vector is to add.
    
    Better approach: For vector v, compute a measure of how many
    potential AP partners it has. Specifically, for each pair of
    existing set members, check if v completes an AP.
    
    But since we're doing greedy with priority, we need a per-vector
    score that's cheap to compute.
    
    Let's use: the number of ways v can be part of a 3-term AP
    with two other vectors in {0,1,2}^n, weighted by some heuristic.
    """
    v = np.array(vector)
    
    # For v to be part of an AP: a + c = 2*v (mod 3)
    # So c = 2*v - a (mod 3)
    # The number of such pairs (a, c) with a != c, a != v, c != v
    # is related to the structure of v.
    
    # Key insight: v is the "middle" of an AP a, v, c iff
    # a + c = 2*v mod 3, i.e., c = 2*v - a mod 3.
    
    # For each a in {0,1,2}^n, c is determined. We need c != a and c != v.
    # c = a iff 2*v - a = a iff a = v.
    # c = v iff 2*v - a = v iff a = v.
    
    # So for a != v, we always get a valid pair (a, c) where c = 2*v - a.
    # But we also need a != c, which means a != 2*v - a, i.e., 2a != 2*v, i.e., a != v.
    # Wait, in Z_3, 2a = 2v iff a = v. So for a != v, a != c always.
    
    # So the number of APs where v is the middle is 3^n - 1 (all a != v).
    # This is constant for all v! So this doesn't help differentiate.
    
    # Better: consider v as the FIRST or LAST element of an AP.
    # v is first: v, b, c with b = v + d, c = v + 2d for some d != 0.
    # Number of such APs: (3^n - 1) choices for d.
    # Again constant!
    
    # The real constraint is that NO THREE DISTINCT elements form an AP.
    # So the priority should reflect how "central" v is in the space,
    # or how many other vectors it's "close to" in the AP sense.
    
    # Let's use a different approach: the L2 norm of the vector's
    # coordinates, which rewards balanced vectors (mix of 0s, 1s, 2s).
    
    # Actually, the best approach for greedy cap set construction is:
    # 1. Prefer vectors with balanced coordinate distribution
    # 2. Prefer vectors that are far from existing set members in
    #    a way that avoids APs
    
    # For the priority function (which only sees one vector), we can:
    # 1. Score based on coordinate balance (more 0/1/2 mix = better)
    # 2. Score based on the vector's "spread" in each coordinate
    
    # Let's compute:
    # - Coordinate balance: -sum over coords of (count_i - n/3)^2 / (n/3)
    #   This rewards vectors that have roughly n/3 of each value.
    
    counts = np.bincount(v, minlength=3).astype(float)
    n_float = float(n)
    balance = -np.sum((counts - n_float/3.0)**2) / (n_float/3.0 + 1e-10)
    
    # Also add a small random tiebreaker to avoid symmetry issues
    # Use a hash-based deterministic tiebreaker
    return balance

def priority(vector, n):
    """
    Priority for greedy cap set construction.
    
    Strategy: Use smooth quadratic surrogates that reward:
    1. Coordinate balance (mix of 0s, 1s, 2s)
    2. Low "self-interference" in the Fourier domain
    
    The key insight: vectors with balanced coordinates are less likely
    to form APs with each other because they're more "spread out"
    in the group structure.
    """
    if _time_remaining() < 0.5:
        # Emergency: return a quick score
        v = np.array(vector)
        counts = np.bincount(v, minlength=3).astype(float)
        return -np.sum((counts - n/3.0)**2)
    
    v = np.array(vector)
    
    # Component 1: Coordinate balance (quadratic penalty for imbalance)
    counts = np.bincount(v, minlength=3).astype(float)
    mean_count = n / 3.0
    balance_penalty = np.sum((counts - mean_count)**2) / (mean_count + 1e-10)
    
    # Component 2: Variance of coordinates (reward spread)
    # Higher variance means more diverse values
    coord_variance = float(np.var(v.astype(float)))
    
    # Component 3: Fourier-based measure
    # For the vector v, compute the "energy" in non-trivial frequencies.
    # We use a sampled approach: pick random characters and compute
    # the contribution.
    
    # Actually, for a single vector, the DFT is just omega^{k·v} for each k.
    # The magnitude is always 1. So we need a different measure.
    
    # Better: compute the "autocorrelation" of v with its shifts.
    # For shift d, the correlation is sum_i omega^{d_i * v_i}
    # We want shifts d where this is small (indicating v doesn't
    # have periodic structure that would lead to APs).
    
    # Sample some shifts to estimate this
    rng = np.random.RandomState(hash(tuple(vector)) % (2**32))
    num_shifts = min(50, 3**n) if n <= 6 else 50
    
    autocorr_sum = 0.0
    for _ in range(num_shifts):
        d = rng.randint(0, 3, size=n)
        if np.all(d == 0):
            continue
        # Compute sum_i omega^{d_i * v_i}
        exponents = (d * v).sum() % 3
        omega = np.exp(2j * np.pi / 3)
        autocorr_sum += abs(omega ** exponents)
    
    # This is always 1, so not useful. Let me try a different approach.
    
    # Component 3 (revised): Measure how "generic" the vector is
    # by looking at the distribution of pairwise differences.
    # For a cap set, we want vectors where differences don't repeat.
    
    # Compute the number of "collisions" in the difference set.
    # For vector v, consider all pairs (i,j) and compute v[i] - v[j].
    # If many pairs give the same difference, the vector has structure.
    
    if n <= 12:
        # Compute difference frequency
        diff_counts = np.zeros(3, dtype=float)
        for i in range(n):
            for j in range(n):
                if i != j:
                    diff = (v[i] - v[j]) % 3
                    diff_counts[diff] += 1
        # Penalize non-uniform difference distribution
        diff_variance = np.var(diff_counts)
    else:
        # Sample-based estimate
        rng2 = np.random.RandomState(hash(tuple(vector)) % (2**32) + 1)
        diff_counts = np.zeros(3, dtype=float)
        num_pairs = min(1000, n*(n-1))
        for _ in range(num_pairs):
            i = rng2.randint(0, n)
            j = rng2.randint(0, n)
            if i != j:
                diff = (v[i] - v[j]) % 3
                diff_counts[diff] += 1
        diff_variance = np.var(diff_counts)
    
    # Combine components with smooth weighting
    # balance_penalty: lower is better (more balanced)
    # coord_variance: higher is better (more spread)
    # diff_variance: lower is better (more uniform differences)
    
    # Normalize each component to [0, 1] range approximately
    balance_score = -balance_penalty / max(n, 1)  # More negative for imbalanced
    variance_score = coord_variance / 2.0  # Max variance is ~2 for {0,1,2}
    diff_score = -diff_variance / max(n, 1)
    
    # Weighted combination with smooth blending
    weight_balance = 0.5
    weight_variance = 0.3
    weight_diff = 0.2
    
    total = (weight_balance * balance_score + 
             weight_variance * variance_score + 
             weight_diff * diff_score)
    
    # Add small deterministic tiebreaker based on vector content
    # to break symmetries and help exploration
    tiebreaker = 0.001 * float(np.sum(v * np.arange(n + 1))) / (n * (n + 1) / 2 + 1)
    
    return total + tiebreaker

