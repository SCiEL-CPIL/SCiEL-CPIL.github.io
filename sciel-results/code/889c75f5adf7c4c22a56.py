"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

DEADLINE = 150.0
SAFETY_MARGIN = 15.0

def _autoconv_linear(s):
    """Compute linear autoconvolution (not circular).
    acf[k] = sum_{i=0}^{N-1-k} s[i]*s[i+k] for k=0..N-1
    """
    n_fft = 1 << (2 * N - 1).bit_length()
    S = np.fft.rfft(s, n=n_fft)
    acf = np.fft.irfft(S * np.conj(S), n=n_fft)[:N]
    return acf

def _score_from_acf(acf):
    """Max |acf[k]| for k >= 1."""
    return np.max(np.abs(acf[1:]))

def _eval_batch(s, moves, n_fft):
    """Evaluate multiple moves in parallel.
    moves: list of (start, length) tuples
    Returns list of scores for each move applied to s.
    """
    n_moves = len(moves)
    # Create batch of modified signals
    batch = np.tile(s, (n_moves, 1)).copy()
    for i, (start, length) in enumerate(moves):
        batch[i, start:start+length] *= -1.0
    # Compute autocorrelation for each
    S = np.fft.rfft(batch, n=n_fft, axis=1)
    acf = np.fft.irfft(S * np.conj(S), n=n_fft, axis=1)[:, :N]
    # Score: max |acf[:, 1:]|
    scores = np.max(np.abs(acf[:, 1:]), axis=1)
    return scores, batch, acf

def construct():
    start_time = time.time()
    n_fft = 1 << (2 * N - 1).bit_length()
    
    # Multiple random initializations
    best_s = None
    best_score = np.inf
    for seed in range(20):
        rng = np.random.RandomState(seed)
        s = rng.choice([-1.0, 1.0], size=N)
        acf = _autoconv_linear(s)
        sc = _score_from_acf(acf)
        if sc < best_score:
            best_score = sc
            best_s = s.copy()
    
    current_s = best_s.copy()
    current_score = best_score
    
    # Phase 1: Parallel batch local search with simulated annealing
    batch_size = 64
    block_lengths = [1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192]
    
    T_start = 2.0
    T_end = 0.005
    max_iters = 200000
    iter_count = 0
    
    while time.time() - start_time < DEADLINE - SAFETY_MARGIN:
        # Generate batch of random moves
        moves = []
        for _ in range(batch_size):
            blen = block_lengths[np.random.randint(len(block_lengths))]
            if blen >= N:
                blen = N - 1
            start = np.random.randint(0, N - blen + 1)
            moves.append((start, blen))
        
        # Evaluate all moves in parallel
        scores, batch, acfs = _eval_batch(current_s, moves, n_fft)
        
        # Find best move in batch
        best_idx = np.argmin(scores)
        best_move_score = scores[best_idx]
        best_move = moves[best_idx]
        
        # Annealing temperature
        T = T_start * (T_end / T_start) ** (iter_count / max_iters)
        
        # Accept best move if better or with probability
        delta = best_move_score - current_score
        if delta < 0 or (np.random.random() < np.exp(-delta / max(T, 1e-10))):
            start, blen = best_move
            current_s[start:start+blen] *= -1.0
            current_score = best_move_score
            if current_score < best_score:
                best_score = current_score
                best_s = current_s.copy()
        
        iter_count += 1
        
        # Time check
        if iter_count % 500 == 0:
            if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                break
    
    # Phase 2: Greedy coordinate descent polish on best solution
    if time.time() - start_time < DEADLINE - SAFETY_MARGIN:
        polish_s = best_s.copy()
        polish_score = best_score
        
        improved = True
        while improved and time.time() - start_time < DEADLINE - SAFETY_MARGIN - 5.0:
            improved = False
            # Try flipping each position
            for i in range(N):
                if time.time() - start_time > DEADLINE - SAFETY_MARGIN - 5.0:
                    break
                polish_s[i] *= -1.0
                acf = _autoconv_linear(polish_s)
                new_score = _score_from_acf(acf)
                if new_score < polish_score:
                    polish_score = new_score
                    improved = True
                else:
                    polish_s[i] *= -1.0  # undo
        
        if polish_score < best_score:
            best_s = polish_s
            best_score = polish_score
    
    # Phase 3: Final small-block polish
    if time.time() - start_time < DEADLINE - SAFETY_MARGIN:
        final_s = best_s.copy()
        final_score = best_score
        small_lengths = [1, 2, 3, 4, 5]
        
        for blen in small_lengths:
            if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                break
            improved = True
            while improved and time.time() - start_time < DEADLINE - SAFETY_MARGIN:
                improved = False
                for i in range(0, N - blen + 1):
                    if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                        break
                    final_s[i:i+blen] *= -1.0
                    acf = _autoconv_linear(final_s)
                    new_score = _score_from_acf(acf)
                    if new_score < final_score:
                        final_score = new_score
                        improved = True
                    else:
                        final_s[i:i+blen] *= -1.0  # undo
        
        if final_score < best_score:
            best_s = final_s
            best_score = final_score
    
    return best_s
# EVOLVE-BLOCK-END
