"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _autoconv_fft(sig):
    """Compute autoconvolution of a signal using FFT."""
    n_fft = 1
    while n_fft < 2 * N - 1:
        n_fft *= 2
    f = np.fft.rfft(sig, n_fft)
    ac = np.fft.irfft(f * np.conj(f), n_fft)[:2*N-1]
    return ac


def _exact_score(sig):
    """Compute C3' = max|autoconv| / (integral)^2."""
    integral = np.sum(sig)
    if abs(integral) < 1e-12:
        return float('inf')
    ac = _autoconv_fft(sig)
    max_abs = np.max(np.abs(ac))
    return max_abs / (integral * integral)


def _score_blocks(blocks):
    """Compute C3' for a block-constant signal given as list of (height, width)."""
    sig = np.zeros(N)
    pos = 0
    for h, w in blocks:
        sig[pos:pos+w] = h
        pos += w
    return _exact_score(sig)


def _blocks_to_sig(blocks):
    sig = np.zeros(N)
    pos = 0
    for h, w in blocks:
        sig[pos:pos+w] = h
        pos += w
    return sig


def _sig_to_blocks(sig):
    blocks = []
    i = 0
    while i < N:
        h = sig[i]
        j = i
        while j < N and sig[j] == h:
            j += 1
        blocks.append((h, j - i))
        i = j
    return blocks


def _merge_adjacent(blocks):
    """Merge adjacent blocks with the same sign."""
    if not blocks:
        return blocks
    merged = [blocks[0]]
    for h, w in blocks[1:]:
        if merged[-1][0] == h:
            merged[-1] = (h, merged[-1][1] + w)
        else:
            merged.append((h, w))
    return merged


def _local_search_blocks(blocks, deadline):
    """Local search on block representation: flip blocks, split blocks, re-merge."""
    best_blocks = list(blocks)
    best_score = _score_blocks(best_blocks)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try flipping each block
        for i in range(len(best_blocks)):
            if time.time() > deadline:
                return best_blocks, best_score
            new_blocks = list(best_blocks)
            h, w = new_blocks[i]
            new_blocks[i] = (-h, w)
            new_blocks = _merge_adjacent(new_blocks)
            s = _score_blocks(new_blocks)
            if s < best_score - 1e-12:
                best_score = s
                best_blocks = new_blocks
                improved = True
        
        # Try splitting a block and flipping the second part
        for i in range(len(best_blocks)):
            if time.time() > deadline:
                return best_blocks, best_score
            h, w = best_blocks[i]
            if w < 4:
                continue
            for frac in [0.25, 0.33, 0.5, 0.67, 0.75]:
                if time.time() > deadline:
                    return best_blocks, best_score
                w1 = max(1, int(w * frac))
                w2 = w - w1
                if w2 < 1:
                    continue
                new_blocks = list(best_blocks)
                new_blocks[i] = (h, w1)
                new_blocks.insert(i + 1, (-h, w2))
                new_blocks = _merge_adjacent(new_blocks)
                s = _score_blocks(new_blocks)
                if s < best_score - 1e-12:
                    best_score = s
                    best_blocks = new_blocks
                    improved = True
    
    return best_blocks, best_score


def _construct_continuous():
    """Use L-BFGS-B to optimize the continuous signal."""
    start_time = time.time()
    deadline = start_time + 60.0  # Safety margin
    
    def objective(sig):
        if time.time() > deadline:
            raise Exception("timeout")
        integral = np.sum(sig)
        if abs(integral) < 1e-10:
            return 1e10
        ac = _autoconv_fft(sig)
        ac_sq = ac * ac
        max_val = np.max(ac_sq)
        if max_val < 1e-15:
            return 0.0
        beta = 50.0
        shifted = ac_sq - max_val
        exp_shifted = np.exp(beta * shifted)
        logsumexp_val = np.log(np.sum(exp_shifted)) / beta + max_val / beta
        surrogate = np.sqrt(logsumexp_val)
        return surrogate / (integral * integral)
    
    best_sig = None
    best_score = float('inf')
    
    initializations = []
    
    # 1. Two-block: +1 first half, -1 second half
    sig1 = np.zeros(N)
    sig1[:N//2] = 1.0
    sig1[N//2:] = -1.0
    initializations.append(sig1.copy())
    
    # 2. Three-block: +1, -1, +1
    sig2 = np.zeros(N)
    sig2[:N//3] = 1.0
    sig2[N//3:2*N//3] = -1.0
    sig2[2*N//3:] = 1.0
    initializations.append(sig2.copy())
    
    # 3. Random signed
    np.random.seed(42)
    sig3 = np.random.choice([-1.0, 1.0], size=N)
    initializations.append(sig3.copy())
    
    # 4. +1 then -1 with different split
    sig4 = np.zeros(N)
    sig4[:N//4] = 1.0
    sig4[N//4:] = -1.0
    initializations.append(sig4.copy())
    
    # 5. Four-block: +1, -1, +1, -1
    sig5 = np.zeros(N)
    sig5[:N//4] = 1.0
    sig5[N//4:N//2] = -1.0
    sig5[N//2:3*N//4] = 1.0
    sig5[3*N//4:] = -1.0
    initializations.append(sig5.copy())
    
    for init_sig in initializations:
        if time.time() > deadline:
            break
        
        integral = np.sum(init_sig)
        if abs(integral) > 1e-10:
            init_sig = init_sig / integral
        
        try:
            result = minimize(
                objective,
                init_sig,
                method='L-BFGS-B',
                options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-12, 'maxls': 50}
            )
            
            if time.time() > deadline:
                break
            
            opt_sig = result.x
            rounded = np.sign(opt_sig)
            rounded[rounded == 0] = 1.0
            
            score = _exact_score(rounded)
            if score < best_score:
                best_score = score
                best_sig = rounded.copy()
                
        except Exception:
            continue
    
    return best_sig, best_score


def _elementwise_local_search(sig, deadline):
    """Local search: flip individual elements."""
    best_score = _exact_score(sig)
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            new_sig = sig.copy()
            new_sig[i] = -new_sig[i]
            s = _exact_score(new_sig)
            if s < best_score - 1e-12:
                best_score = s
                sig = new_sig
                improved = True
    
    return sig, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin
    
    # Phase 1: Continuous optimization with L-BFGS-B
    best_sig, best_score = _construct_continuous()
    
    if best_sig is None:
        # Fallback: simple two-block
        best_sig = np.zeros(N)
        best_sig[:N//2] = 1.0
        best_sig[N//2:] = -1.0
        best_score = _exact_score(best_sig)
    
    if time.time() > deadline:
        return best_sig
    
    # Phase 2: Block-based local search (structural refinement)
    blocks = _sig_to_blocks(best_sig)
    blocks, block_score = _local_search_blocks(blocks, deadline)
    if block_score < best_score:
        best_score = block_score
        best_sig = _blocks_to_sig(blocks)
    
    if time.time() > deadline:
        return best_sig
    
    # Phase 3: Element-wise local search (fine-grained refinement)
    refined_sig, refined_score = _elementwise_local_search(best_sig, deadline)
    if refined_score < best_score:
        best_sig = refined_sig
        best_score = refined_score
    
    # Phase 4: Final block-based pass on the refined signal
    if time.time() < deadline - 5.0:
        blocks = _sig_to_blocks(best_sig)
        blocks, block_score = _local_search_blocks(blocks, deadline)
        if block_score < best_score:
            best_score = block_score
            best_sig = _blocks_to_sig(blocks)
    
    return best_sig
# EVOLVE-BLOCK-END
