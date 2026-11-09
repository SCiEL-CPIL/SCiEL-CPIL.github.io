"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 400

# EVOLVE-BLOCK-START

def _lfsr_m_sequence(n, poly, initial_state):
    """Generate n bits from an LFSR with given polynomial and initial state."""
    bits = []
    state = initial_state
    degree = poly.bit_length() - 1
    mask = (1 << degree) - 1
    
    for _ in range(n):
        bits.append(state & 1)
        feedback = 0
        for i in range(degree):
            if (poly >> i) & 1:
                feedback ^= (state >> i) & 1
        state = (state >> 1) | (feedback << (degree - 1))
        state &= mask
    
    return bits


def _compute_c3_bound(signal):
    """Compute the C3 upper bound for a given ±1 signal."""
    n = len(signal)
    # Compute autocorrelation
    ac = np.correlate(signal, signal, mode='full')
    # ac has length 2*n-1, centered at index n-1
    # The peak is at lag 0 (index n-1)
    peak = ac[n-1]
    if peak == 0:
        return float('inf')
    
    # Normalize
    ac_norm = ac / peak
    
    # C3 bound involves the maximum sidelobe relative to peak
    # For the third autocorrelation inequality, we consider
    # the sum of absolute values of off-center autocorrelation
    # or the max sidelobe
    
    # Try multiple metrics and return the one that gives a good bound
    # The C3 bound is typically related to:
    # C3 <= 1 + (max sidelobe) / peak or similar
    
    # Let's compute several candidates:
    sidelobes = np.concatenate([ac_norm[:n-1], ac_norm[n:]])
    
    # Max sidelobe magnitude
    max_side = np.max(np.abs(sidelobes))
    
    # L2 norm of sidelobes
    l2_side = np.sqrt(np.sum(sidelobes**2))
    
    # L1 norm of sidelobes  
    l1_side = np.sum(np.abs(sidelobes))
    
    # The C3 bound for the third autocorrelation inequality
    # likely involves: C3 = 1 + max_k |R(k)|/R(0) for k != 0
    # or a variant involving integrals
    
    # Return the max sidelobe as a conservative bound
    return max_side


def _flip_bit(signal, idx):
    """Flip the sign at index idx."""
    new_signal = signal.copy()
    new_signal[idx] = -new_signal[idx]
    return new_signal


def _local_search(signal, max_iter=10000, time_limit=100):
    """Improve signal via local search (flip single bits)."""
    current = signal.copy()
    current_bound = _compute_c3_bound(current)
    best = current.copy()
    best_bound = current_bound
    
    start_time = time.time()
    no_improve = 0
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Try flipping each position
        best_flip = -1
        best_flip_bound = current_bound
        
        for i in range(len(current)):
            candidate = current.copy()
            candidate[i] = -candidate[i]
            bound = _compute_c3_bound(candidate)
            if bound < best_flip_bound:
                best_flip_bound = bound
                best_flip = i
        
        if best_flip >= 0:
            current = _flip_bit(current, best_flip)
            current_bound = best_flip_bound
            if current_bound < best_bound:
                best = current.copy()
                best_bound = current_bound
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
        
        if no_improve > 500:
            break
    
    return best, best_bound


def construct():
    """Construct a signed step function that minimizes the C3 autoconvolution bound."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Generate m-sequence using LFSR
    # For N=400, we need a sequence of length 400
    # Use a primitive polynomial for LFSR
    
    # Common primitive polynomials for LFSR
    # Degree 9: x^9 + x^4 + 1 -> poly = 0b100010001 = 273
    # Degree 10: x^10 + x^7 + 1 -> poly = 0b10000100001 = 2689
    
    # Let's try several LFSR configurations and pick the best
    candidates = []
    
    # Try different primitive polynomials
    polys = [
        (9, 0b100010001),   # x^9 + x^4 + 1
        (10, 0b10000100001), # x^10 + x^7 + 1
        (11, 0b1000000101001), # x^11 + x^8 + x^6 + x^5 + 1
        (12, 0b1000000000101), # x^12 + x^6 + 1
    ]
    
    for degree, poly in polys:
        if time.time() > deadline:
            break
        
        # Generate a sequence longer than N, then take first N
        # m-sequence has period 2^degree - 1
        period = (1 << degree) - 1
        
        # Try several initial states
        for init_state in range(1, min(20, period)):
            if time.time() > deadline:
                break
            
            bits = _lfsr_m_sequence(period, poly, init_state)
            
            # Take first N bits, map 0->1, 1->-1 (or 0->-1, 1->1)
            # Try both mappings
            for mapping in [lambda b: 1 if b == 0 else -1, 
                           lambda b: -1 if b == 0 else 1]:
                if time.time() > deadline:
                    break
                
                signal = np.array([mapping(b) for b in bits[:N]], dtype=np.float64)
                bound = _compute_c3_bound(signal)
                candidates.append((bound, signal))
    
    if not candidates:
        # Fallback: return all ones
        return np.ones(N)
    
    # Sort by bound and take the best
    candidates.sort(key=lambda x: x[0])
    best_bound, best_signal = candidates[0]
    
    # Try local search to improve if time permits
    if time.time() < deadline - 10:
        time_remaining = deadline - 10 - time.time()
        best_signal, best_bound = _local_search(
            best_signal, 
            max_iter=5000, 
            time_limit=max(1, time_remaining)
        )
    
    return best_signal

# EVOLVE-BLOCK-END
