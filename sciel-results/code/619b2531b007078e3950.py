"""Rewritten program for the suite task; optimized local search with incremental autoconv update."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.signal import fftconvolve


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill

    def initial_autoconv(x):
        """Compute full autoconvolution and its max absolute value."""
        if np.all(x == 0):
            return np.zeros(2*N-1), 0.0
        ac = fftconvolve(x, x, mode='full')
        return ac, np.max(np.abs(ac))

    def score_from_state(x, ac, max_abs):
        """Compute score from precomputed state."""
        s = np.sum(x)
        if abs(s) < 1e-12:
            return float('inf')
        return max_abs / (s * s)

    def try_flip(x, ac, idx):
        """
        Attempt to flip x[idx]. Returns new_x, new_ac, new_max_abs if accepted, else None.
        Updates ac incrementally.
        """
        v = x[idx]
        # Change in x is -2*v at position idx
        # ac_new[j] = ac_old[j] - 2*v * x[(j-idx) % N] - 2*v * x[(idx-j) % N] ? 
        # Careful with indexing. 
        # x is length N. ac is length 2N-1.
        # ac[k] = sum_i x[i] x[i+k-N+1] ? No, standard fftconvolve indexing:
        # ac[k] = sum_{i=0}^{2N-2-k} x[i] x[k-i] for k=0..2N-2?
        # Actually, scipy.signal.fftconvolve(x, x, mode='full') returns c where
        # c[k] = sum_i x[i] x[k-i] with appropriate bounds.
        # Flipping x[idx] by -2v:
        # Contribution to c[k] from x[idx] is x[idx]*x[k-idx].
        # Change is -2v * x[k-idx] (for the first term) and x[k-idx]*x[idx] (for the second, symmetric).
        # So total change at k is -4v * x[k-idx] if k-idx is valid index in x.
        # Wait, x is not periodic.
        # Let's just recompute the delta.
        # delta_x = -2v * e_idx
        # new_ac = ac + delta_x * x + x * delta_x
        # (delta_x * x)[k] = sum_i delta_x[i] x[k-i] = delta_x[idx] x[k-idx] = -2v x[k-idx]
        # (x * delta_x)[k] = sum_i x[i] delta_x[k-i] = x[k-idx] delta_x[idx] = -2v x[k-idx]
        # So new_ac[k] = ac[k] - 4v x[k-idx]
        # Valid for k such that k-idx is a valid index in [0, N-1].
        
        # Indices in ac are 0 to 2N-2.
        # k-idx must be in [0, N-1].
        # So k in [idx, idx+N-1].
        
        start_k = idx
        end_k = idx + N - 1
        
        # x indices involved: k-idx ranges from 0 to N-1.
        # x_vals = x[0:N]
        
        if v != 0:
            ac[start_k:end_k+1] -= 4 * v * x
            
        new_max = np.max(np.abs(ac))
        # We modified ac in place. If we reject, we must revert.
        # For simplicity in the search loop, we will copy ac if we want to be safe, 
        # or just revert the change if rejected.
        
        return new_max

    # Start with all ones (constant) as baseline
    best_x = np.ones(N)
    best_ac, best_max_abs = initial_autoconv(best_x)
    best_val = score_from_state(best_x, best_ac, best_max_abs)

    rng = np.random.default_rng(42)

    # Van der Corput sequence (base-b) for low-discrepancy initialization
    def van_der_corput(n, base=2):
        """Generate first n values of Van der Corput sequence in [0,1)."""
        seq = np.zeros(n)
        for i in range(n):
            f = 1.0 / base
            num = i
            val = 0.0
            while num > 0:
                val += f * (num % base)
                num //= base
                f /= base
            seq[i] = val
        return seq

    # Kronecker sequence
    def kronecker(n, alpha):
        """Generate Kronecker sequence {n*alpha mod 1} for n=0..N-1."""
        return np.mod(np.arange(n) * alpha, 1.0)

    # Thue-Morse sequence (first N bits)
    def thue_morse(n):
        """Generate Thue-Morse sequence."""
        t = np.zeros(n)
        for i in range(1, n):
            # t[i] = t[i//2] XOR (i%2)
            t[i] = t[i//2] ^ (i % 2)
        # Map 0->1, 1->-1
        return np.where(t == 0, 1.0, -1.0)

    # Initialization strategies
    def get_initialization(r, rng):
        if r % 7 == 0:
            # Random ±1
            return rng.choice([-1.0, 1.0], size=N)
        elif r % 7 == 1:
            # Van der Corput base 2
            vdc = van_der_corput(N, base=2)
            return np.where(vdc < 0.5, 1.0, -1.0)
        elif r % 7 == 2:
            # Van der Corput base 3
            vdc = van_der_corput(N, base=3)
            return np.where(vdc < 0.5, 1.0, -1.0)
        elif r % 7 == 3:
            # Kronecker sequence with golden ratio
            alpha = (1 + np.sqrt(5)) / 2
            kron = kronecker(N, alpha)
            return np.where(kron < 0.5, 1.0, -1.0)
        elif r % 7 == 4:
            # Kronecker sequence with sqrt(2)-1
            alpha = np.sqrt(2) - 1
            kron = kronecker(N, alpha)
            return np.where(kron < 0.5, 1.0, -1.0)
        elif r % 7 == 5:
            # Thue-Morse
            return thue_morse(N)
        else:
            # Block pattern
            x = np.ones(N)
            pos = 0
            while pos < N:
                block_len = rng.integers(2, max(3, N // 20 + 1))
                sign = rng.choice([-1.0, 1.0])
                x[pos:pos + min(pos + block_len, N)] = sign
                pos += block_len
            return x

    # Try diverse initialization strategies with local search
    n_restarts = 100
    for r in range(n_restarts):
        if time.time() > deadline:
            break

        x = get_initialization(r, rng)
        
        # Ensure sum is not too small; if sum is 0, it's invalid (inf score)
        # We want large |sum| to minimize denominator? No, score = max_abs / sum^2.
        # We want small max_abs and large sum^2.
        # Usually signed random walks have sum ~ sqrt(N).
        # If sum is very small, score blows up.
        s = np.sum(x)
        if abs(s) < N * 0.05:
             # Bias towards 1
             x = np.sign(x + 0.5)
             x[x == 0] = 1.0

        ac, max_abs = initial_autoconv(x)
        val = score_from_state(x, ac, max_abs)
        
        if val < best_val:
            best_val = val
            best_x = x.copy()
            best_ac = ac.copy()
            best_max_abs = max_abs

        # Local search using incremental updates
        # We work on copies of x and ac to allow backtracking
        curr_x = x.copy()
        curr_ac = ac.copy()
        curr_max_abs = max_abs
        
        improved = True
        max_iters = 200 # More iterations possible due to speed
        for it in range(max_iters):
            if time.time() > deadline:
                break
            if not improved:
                break
            improved = False

            # Try random subset of positions
            # Sort indices by potential impact? Random is fine for now.
            trial_indices = rng.choice(N, size=min(50, N), replace=False)
            for idx in trial_indices:
                if time.time() > deadline:
                    break
                
                orig_val = curr_x[idx]
                
                # Perform flip
                curr_x[idx] = -orig_val
                new_max_abs = try_flip(curr_x, curr_ac, idx)
                
                # Calculate new score
                new_sum = np.sum(curr_x)
                if abs(new_sum) < 1e-12:
                    new_score = float('inf')
                else:
                    new_score = new_max_abs / (new_sum * new_sum)
                
                if new_score < best_val - 1e-12:
                    best_val = new_score
                    best_x = curr_x.copy()
                    best_ac = curr_ac.copy()
                    best_max_abs = new_max_abs
                    improved = True
                else:
                    # Revert flip
                    curr_x[idx] = orig_val
                    # Revert ac change
                    # try_flip modified curr_ac in place.
                    # We need to undo: ac[start:end] -= 4*v*x  ->  ac[start:end] += 4*v*x
                    # v was orig_val
                    v = orig_val
                    if v != 0:
                        start_k = idx
                        end_k = idx + N - 1
                        curr_ac[start_k:end_k+1] += 4 * v * curr_x # curr_x is back to original state here? 
                        # Wait, curr_x[idx] was reverted. The other x values are unchanged.
                        # The update formula used x. Since we only flipped one, x is the same.
                        # So adding 4*v*x reverts it.
                        # However, try_flip used the NEW x value (-v) to compute the change?
                        # No, try_flip logic:
                        # v = x[idx] (which was -orig_val when called? No, I passed idx, and inside it reads x[idx]).
                        # In try_flip: v = x[idx]. If I flip x[idx] to -orig, then v becomes -orig.
                        # Update: ac -= 4*(-orig)*x = ac += 4*orig*x.
                        # To revert: ac -= 4*orig*x.
                        # My revert code above does ac += 4*orig*x. This is WRONG.
                        
                        # Let's re-verify try_flip logic.
                        # Inside try_flip:
                        # v = x[idx]. (This is the NEW value, -orig_val)
                        # ac[start:end] -= 4 * v * x
                        # So ac_new = ac_old - 4*(-orig_val)*x = ac_old + 4*orig_val*x.
                        # To revert: ac_old = ac_new - 4*orig_val*x.
                        
                        # So revert should be: curr_ac[start_k:end_k+1] -= 4 * v_original * curr_x
                        # Where v_original is the value BEFORE flip, which is orig_val.
                        
                        pass # Logic handled below

                    # Correct Revert Logic:
                    # The change applied was: delta_ac = -4 * (new_val) * x
                    # new_val = -orig_val
                    # delta_ac = -4 * (-orig_val) * x = 4 * orig_val * x
                    # To revert: curr_ac -= delta_ac = curr_ac - 4 * orig_val * x
                    # Since curr_x is now back to original, we can use curr_x.
                    if orig_val != 0:
                        curr_ac[start_k:end_k+1] -= 4 * orig_val * curr_x

                    # Restore max_abs? We don't track it separately, we just compute score on the fly.
                    # But we need to know if we should continue. The loop breaks if no improvement.

    # Final polish
    if time.time() < deadline - 5:
        x = best_x.copy()
        ac = best_ac.copy()
        for trial in range(500):
            if time.time() > deadline - 2:
                break
            idx = rng.integers(N)
            orig_val = x[idx]
            x[idx] = -orig_val
            new_max_abs = try_flip(x, ac, idx)
            new_sum = np.sum(x)
            if abs(new_sum) < 1e-12:
                new_score = float('inf')
            else:
                new_score = new_max_abs / (new_sum * new_sum)
            
            if new_score < best_val - 1e-12:
                best_val = new_score
                best_x = x.copy()
                best_ac = ac.copy()
                best_max_abs = new_max_abs
            else:
                x[idx] = orig_val
                if orig_val != 0:
                    start_k = idx
                    end_k = idx + N - 1
                    ac[start_k:end_k+1] -= 4 * orig_val * x

    return best_x
# EVOLVE-BLOCK-END
