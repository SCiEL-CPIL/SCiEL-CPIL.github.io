"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    """
    Minimize max_s ∫ h(x)(1-h(x+s)) dx over shifts s.
    For binary h (0/1), this is minimizing max over shifts of the
    measure where h=1 at x and h=0 at x+s.
    
    Strategy: Use a periodic binary pattern with good autocorrelation properties.
    Try patterns based on quadratic residues mod p for prime p.
    """
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin
    
    best_score = 0.5  # Constant 0.5 baseline
    best_result = np.full(N, 0.5)
    
    def compute_max_overlap(h):
        """Compute max over shifts of ∫ h(x)(1-h(x+s)) dx."""
        n = len(h)
        # h is defined on [0,2] with n steps of width 2/n
        # For shift s in [0,2), compute overlap
        # Use discrete approximation: shift by k steps
        max_ov = 0.0
        for k in range(n):
            # h_shifted[i] = h[(i+k) % n]
            h_shifted = np.roll(h, -k)
            overlap = np.sum(h * (1 - h_shifted)) / n
            max_ov = max(max_ov, overlap)
        return max_ov
    
    # Try binary patterns with N/2 ones and N/2 zeros
    # Pattern 1: Alternating 0,1,0,1,...
    if N % 2 == 0:
        alt = np.zeros(N)
        alt[1::2] = 1.0
        score = compute_max_overlap(alt)
        if score < best_score:
            best_score = score
            best_result = alt.copy()
    
    # Try quadratic residue patterns for various primes
    def quadratic_residue_pattern(p):
        """Generate a ±1 pattern from quadratic residues mod p, then map to 0/1."""
        if p <= 1:
            return None
        qr = np.zeros(p)
        for a in range(1, p):
            if pow(a, (p-1)//2, p) == 1:
                qr[a] = 1
            else:
                qr[a] = -1
        qr[0] = 0  # Convention
        # Convert to 0/1: map -1->0, 0->0.5, 1->1? No, need exactly half 1s.
        # Better: use the Legendre symbol pattern directly as ±1, then threshold
        # Actually, let's create a 0/1 sequence from the QR set
        qr_set = set()
        for a in range(1, p):
            if pow(a, (p-1)//2, p) == 1:
                qr_set.add(a)
        # Create sequence of length p: 1 if i in qr_set, 0 otherwise
        seq = np.zeros(p)
        for i in range(p):
            if i in qr_set:
                seq[i] = 1.0
        return seq
    
    # Try various primes and tile/truncate to get length N
    primes = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    for p in primes:
        if time.time() > deadline - 5.0:
            break
        seq = quadratic_residue_pattern(p)
        if seq is None:
            continue
        # Tile to reach length >= N, then truncate
        repeats = (N + p - 1) // p
        tiled = np.tile(seq, repeats)[:N]
        
        # Adjust to have exactly N/2 ones
        num_ones = np.sum(tiled)
        if num_ones > N / 2:
            # Remove excess ones: set some 1s to 0s
            ones_idx = np.where(tiled == 1.0)[0]
            excess = int(num_ones - N / 2)
            # Randomly pick which to flip (deterministic for reproducibility)
            np.random.seed(42)
            to_flip = np.random.choice(ones_idx, excess, replace=False)
            tiled[to_flip] = 0.0
        elif num_ones < N / 2:
            # Add ones: set some 0s to 1s
            zeros_idx = np.where(tiled == 0.0)[0]
            deficit = int(N / 2 - num_ones)
            np.random.seed(42)
            to_flip = np.random.choice(zeros_idx, deficit, replace=False)
            tiled[to_flip] = 1.0
        
        score = compute_max_overlap(tiled)
        if score < best_score:
            best_score = score
            best_result = tiled.copy()
    
    # Try a local optimization: start from best binary pattern and flip individual bits
    if time.time() < deadline - 10.0:
        current = best_result.copy()
        if np.all((current == 0) | (current == 1)):  # Only optimize binary
            improved = True
            while improved and time.time() < deadline - 10.0:
                improved = False
                current_score = compute_max_overlap(current)
                for i in range(N):
                    if time.time() > deadline - 10.0:
                        break
                    # Try flipping bit i
                    old_val = current[i]
                    current[i] = 1.0 - old_val
                    # Check if sum is still N/2
                    if np.sum(current) == N / 2:
                        new_score = compute_max_overlap(current)
                        if new_score < current_score - 1e-12:
                            current_score = new_score
                            improved = True
                        else:
                            current[i] = old_val
                    else:
                        current[i] = old_val
                
                if current_score < best_score:
                    best_score = current_score
                    best_result = current.copy()
    
    # Final check: ensure sum is exactly N/2
    final = best_result.copy()
    if not np.isclose(np.sum(final), N / 2, atol=1e-10):
        # Adjust
        diff = N / 2 - np.sum(final)
        if diff > 0:
            zeros_idx = np.where(final < 0.5)[0]
            for idx in zeros_idx[:int(diff)]:
                final[idx] = 1.0 - final[idx]
        else:
            ones_idx = np.where(final > 0.5)[0]
            for idx in ones_idx[:int(-diff)]:
                final[idx] = 1.0 - final[idx]
    
    return final
# EVOLVE-BLOCK-END
