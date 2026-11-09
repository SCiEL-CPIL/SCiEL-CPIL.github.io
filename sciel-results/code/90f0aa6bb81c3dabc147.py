"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 400

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 100  # 100s deadline with 20s safety margin
    
    def compute_C3(seq):
        """Compute max normalized autocorrelation (excluding lag 0)."""
        n_fft = 1 << (2 * N - 1).bit_length()
        F = np.fft.rfft(seq, n=n_fft)
        power = F * np.conj(F)
        acf = np.fft.irfft(power, n=n_fft)[:N]
        norms = np.arange(N, 0, -1, dtype=np.float64)
        normalized = np.abs(acf[1:]) / norms[1:]
        return float(np.max(normalized)) if len(normalized) > 0 else 0.0
    
    def seq_from_blocks(block_lengths, start_sign=1):
        """Build ±1 sequence from alternating block lengths."""
        seq = np.empty(N, dtype=np.float64)
        pos = 0
        sign = start_sign
        for blen in block_lengths:
            end = min(pos + blen, N)
            seq[pos:end] = sign
            sign *= -1
            pos = end
            if pos >= N:
                break
        if pos < N:
            seq[pos:] = sign
        return seq
    
    def blocks_from_seq(seq):
        """Extract block lengths from a ±1 sequence."""
        blocks = []
        cur = seq[0]
        count = 1
        for i in range(1, N):
            if seq[i] == cur:
                count += 1
            else:
                blocks.append(count)
                cur = seq[i]
                count = 1
        blocks.append(count)
        return blocks
    
    rng = np.random.RandomState(42)
    
    # Generate initial candidates using various structured sequences
    candidates = []
    
    # Thue-Morse
    def thue_morse(n):
        tm = np.zeros(n, dtype=np.float64)
        for i in range(n):
            c = bin(i).count('1')
            tm[i] = 1.0 if c % 2 == 0 else -1.0
        return tm
    candidates.append(thue_morse(N))
    
    # Quadratic residue
    p = 401
    qr = np.zeros(N, dtype=np.float64)
    for i in range(N):
        val = (i * i) % p
        if val == 0:
            qr[i] = 1.0
        else:
            qr[i] = 1.0 if pow(val, (p-1)//2, p) == 1 else -1.0
    candidates.append(qr)
    
    # M-sequence
    def m_sequence(n, poly_bits):
        seq = np.zeros(n, dtype=np.float64)
        state = 1
        for i in range(n):
            seq[i] = 1.0 if (state & 1) == 0 else -1.0
            feedback = 0
            for b in poly_bits:
                feedback ^= (state >> b) & 1
            state = (state >> 1) | (feedback << (len(poly_bits) - 1))
        return seq
    candidates.append(m_sequence(N, [8, 0]))
    
    # Random
    for _ in range(5):
        candidates.append(rng.choice([-1, 1], size=N).astype(np.float64))
    
    # Pick best initial
    best_seq = None
    best_score = float('inf')
    for cand in candidates:
        s = compute_C3(cand)
        if s < best_score:
            best_score = s
            best_seq = cand.copy()
    
    # Structural optimization: work in block-length space
    current_blocks = blocks_from_seq(best_seq)
    current_score = best_score
    
    # Simulated annealing on block lengths
    T = 0.5
    T_min = 0.01
    cooling = 0.999
    
    while time.time() < deadline - 5:
        T *= cooling
        if T < T_min:
            T = T_min
        
        # Generate a new candidate by perturbing the block structure
        new_seq = None
        move_type = rng.randint(4)
        
        if move_type == 0 and len(current_blocks) > 1:
            # Move 1: Shift boundary between two adjacent blocks
            i = rng.randint(len(current_blocks) - 1)
            j = i + 1
            # Move some length from block i to block j (or vice versa)
            amt = rng.randint(1, min(current_blocks[i], current_blocks[j]) + 1)
            if amt <= current_blocks[i] and amt <= current_blocks[j]:
                new_blocks = current_blocks.copy()
                new_blocks[i] -= amt
                new_blocks[j] += amt
                if new_blocks[i] == 0:
                    new_blocks.pop(i)
                new_seq = seq_from_blocks(new_blocks)
        
        elif move_type == 1 and len(current_blocks) > 1:
            # Move 2: Merge two adjacent blocks
            i = rng.randint(len(current_blocks) - 1)
            new_blocks = current_blocks.copy()
            new_blocks[i] += new_blocks[i+1]
            new_blocks.pop(i+1)
            new_seq = seq_from_blocks(new_blocks)
        
        elif move_type == 2:
            # Move 3: Split a block
            i = rng.randint(len(current_blocks))
            if current_blocks[i] >= 2:
                split_at = rng.randint(1, current_blocks[i])
                new_blocks = current_blocks.copy()
                new_blocks[i] = split_at
                new_blocks.insert(i + 1, current_blocks[i] - split_at)
                new_seq = seq_from_blocks(new_blocks)
        
        elif move_type == 3:
            # Move 4: Random single-bit flip as fallback
            new_seq = best_seq.copy()
            idx = rng.randint(N)
            new_seq[idx] *= -1
        
        if new_seq is None:
            # Fallback: random bit flip
            new_seq = best_seq.copy()
            idx = rng.randint(N)
            new_seq[idx] *= -1
        
        new_score = compute_C3(new_seq)
        
        delta = new_score - current_score
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
            current_blocks = blocks_from_seq(new_seq)
            current_score = new_score
            best_seq = new_seq.copy()
            if new_score < best_score:
                best_score = new_score
    
    return best_seq
# EVOLVE-BLOCK-END
