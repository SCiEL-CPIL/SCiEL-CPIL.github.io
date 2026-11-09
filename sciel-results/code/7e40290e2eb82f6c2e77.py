"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def construct():
    # Deadline: 145 seconds (safety margin for 150s limit)
    start_time = time.time()
    time_limit = 145.0
    
    def compute_score(a):
        fa = np.fft.fft(a)
        power = fa * np.conj(fa)
        R = np.fft.ifft(power).real
        return np.sum(R[1:]**2)
    
    # Strategy: Use m-sequence based initialization for near-ideal autocorrelation properties,
    # then apply targeted boundary optimization.
    
    # Generate m-sequences for various prime lengths
    # An m-sequence of length 2^m - 1 has near-perfect autocorrelation properties.
    # We'll generate several candidates and pick the best one.
    
    def generate_m_sequence(m, offset=0):
        """Generate an m-sequence of length 2^m - 1 using a primitive polynomial."""
        # Use a simple LFSR with a known primitive polynomial
        # For m=8, use polynomial x^8 + x^4 + x^3 + x^2 + 1 (0x1D in hex, but we need the taps)
        # Actually, let's use a simpler approach: generate using numpy
        
        # Common primitive polynomials for small m:
        # m=8: x^8 + x^4 + x^3 + x^2 + 1
        # Taps for LFSR: [0, 3, 4, 6, 7] (0-indexed from LSB)
        
        if m == 8:
            # Primitive polynomial: x^8 + x^4 + x^3 + x^2 + 1
            # In binary: 1 0000 1111 (from MSB to LSB for x^8 to x^0)
            # Actually, let's use a known good one
            # For simplicity, let's use the Thue-Morse as fallback if m-seq generation is complex
            
            # Generate using a simple LFSR
            # State: 8-bit register
            state = 0x01  # Non-zero initial state
            length = 2**m - 1
            seq = []
            
            # Taps for x^8 + x^4 + x^3 + x^2 + 1
            # Feedback = bit 0 XOR bit 3 XOR bit 4 XOR bit 6 XOR bit 7
            for _ in range(length):
                seq.append(1.0 if (state & 1) == 0 else -1.0)
                # Compute feedback
                fb = ((state >> 0) ^ (state >> 3) ^ (state >> 4) ^ (state >> 6) ^ (state >> 7)) & 1
                # Shift right and insert feedback at MSB
                state = (state >> 1) | (fb << (m - 1))
            
            return np.array(seq)
        
        # Fallback: Thue-Morse
        return np.array([1.0 if bin(i).count('1') % 2 == 0 else -1.0 for i in range(2**m - 1)])
    
    # Generate several m-sequence candidates and truncate to N=400
    candidates = []
    
    # m=8 gives length 255, too short. m=9 gives 511, which is > 400.
    # Let's try m=9 with different initial states
    for m in [9, 10]:
        length = 2**m - 1
        if length >= N:
            # Generate with different initial states
            for init_state in range(1, min(16, 2**(m-1))):
                # We need to modify generate_m_sequence to accept initial state
                # Let's create a more flexible generator
                pass
    
    # Simpler approach: Use the fact that random sequences with local optimization work well
    # But let's try a few structured initializations
    
    # Candidate 1: Thue-Morse
    tm = np.array([1.0 if bin(i).count('1') % 2 == 0 else -1.0 for i in range(N)])
    candidates.append(tm)
    
    # Candidate 2: Random with fixed seed
    rng = np.random.default_rng(42)
    rand = rng.choice([-1.0, 1.0], size=N)
    candidates.append(rand)
    
    # Candidate 3: Alternating
    alt = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)])
    candidates.append(alt)
    
    # Candidate 4: Block pattern
    block = np.zeros(N)
    for i in range(N):
        block[i] = 1.0 if (i // 10) % 2 == 0 else -1.0
    candidates.append(block)
    
    # Evaluate all candidates and pick the best
    best_a = None
    best_score = float('inf')
    for cand in candidates:
        score = compute_score(cand)
        if score < best_score:
            best_score = score
            best_a = cand.copy()
    
    current_a = best_a.copy()
    current_score = best_score
    
    # Simulated Annealing
    T_start = 20.0
    T_end = 0.05
    num_iterations = 1000000
    
    for i in range(num_iterations):
        if time.time() - start_time > time_limit:
            break
        
        T = T_start * (T_end / T_start) ** (i / num_iterations)
        
        # Flip one bit
        idx = rng.integers(0, N)
        candidate_a = current_a.copy()
        candidate_a[idx] *= -1.0
        
        candidate_score = compute_score(candidate_a)
        delta = candidate_score - current_score
        
        # Accept if better, or with probability exp(-delta/T) if worse
        if delta < 0 or (T > 0 and rng.random() < np.exp(-delta / T)):
            current_a = candidate_a
            current_score = candidate_score
            
            if current_score < best_score:
                best_score = current_score
                best_a = current_a.copy()
    
    # Final greedy polish
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for idx in range(N):
            if (time.time() - start_time) > time_limit:
                break
            candidate_a = current_a.copy()
            candidate_a[idx] *= -1.0
            candidate_score = compute_score(candidate_a)
            if candidate_score < current_score:
                current_a = candidate_a
                current_score = candidate_score
                improved = True
                if current_score < best_score:
                    best_score = current_score
                    best_a = current_a.copy()
    
    return best_a
# EVOLVE-BLOCK-END
