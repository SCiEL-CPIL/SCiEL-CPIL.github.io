"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

DEADLINE = 150.0
TIME_MARGIN = 15.0  # safety margin in seconds

def _compute_c3_slow(x):
    """
    Compute the upper bound C3 for a signed step function x of length N using FFT.
    Used for initialization and verification.
    """
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(x, n=n_fft)
    power = F * np.conj(F)
    R = np.fft.irfft(power, n=n_fft)[:2 * N - 1]
    R0 = R[0]
    if R0 == 0:
        return 1.0
    sidelobes = np.concatenate([R[1:N], R[N:2*N-1]])
    max_sidelobe = np.max(np.abs(sidelobes))
    return max_sidelobe / R0

class _FastAutocorr:
    def __init__(self, x):
        self.x = x.copy()
        self.n_fft = 1 << (2 * N - 1).bit_length()
        self.X = np.fft.rfft(x, n=self.n_fft)
        self.R = np.fft.irfft(self.X * np.conj(self.X), n=self.n_fft)[:N]
        self.current_c3 = self._calc_c3()
        # Precompute the exponential basis for updates
        # We need e^{-i 2 pi k m / n_fft} for m=0..n_fft/2
        # Actually, we can just compute the vector for the specific index flipped each time,
        # or precompute a matrix if memory allows. N=400, n_fft=1024 or 2048.
        # Precomputing a matrix of shape (N, n_fft/2+1) is ~ 400 * 1024 * 16 bytes = 6MB. Acceptable.
        self.basis = np.exp(-2j * np.pi * np.outer(np.arange(N), np.arange(self.n_fft // 2 + 1)) / self.n_fft)

    def _calc_c3(self):
        if self.R[0] == 0:
            return 1.0
        max_sidelobe = np.max(np.abs(self.R[1:N]))
        return max_sidelobe / self.R[0]

    def flip(self, idx):
        """
        Flip x[idx] and update R incrementally using FFT properties.
        Returns the change in C3 (new_c3 - old_c3).
        """
        old_c3 = self.current_c3
        old_x_val = self.x[idx]
        
        # Update X: X_new = X_old - 2 * old_x_val * basis[idx]
        # basis[idx] contains exp(-i 2 pi idx m / n_fft)
        update_vec = -2.0 * old_x_val * self.basis[idx]
        self.X += update_vec
        
        # Update R: R = IFFT(|X|^2)
        power = self.X * np.conj(self.X)
        self.R = np.fft.irfft(power, n=self.n_fft)[:N]
        
        # Flip the bit in x
        self.x[idx] = -old_x_val
        
        new_c3 = self._calc_c3()
        return new_c3 - old_c3, new_c3

    def get_best_x(self):
        return self.x.copy()

def _fast_local_search(x_init, max_iter=200000, time_limit=None):
    start_time = time.time()
    fa = _FastAutocorr(x_init)
    best_x = fa.get_best_x()
    best_c3 = fa.current_c3
    no_improve_count = 0
    max_no_improve = 500 # Higher because steps are faster
    
    for iteration in range(max_iter):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        
        idx = random.randint(0, N - 1)
        delta, new_c3 = fa.flip(idx)
        
        if delta < -1e-12:
            no_improve_count = 0
            if new_c3 < best_c3:
                best_x = fa.get_best_x()
                best_c3 = new_c3
        else:
            # Revert
            fa.flip(idx) # Flip back to restore state
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break
    
    return best_x, best_c3

def _fast_simulated_annealing(x_init, max_iter=300000, T_start=0.5, T_end=0.01, time_limit=None):
    start_time = time.time()
    fa = _FastAutocorr(x_init)
    best_x = fa.get_best_x()
    best_c3 = fa.current_c3
    current_c3 = fa.current_c3
    T = T_start
    
    for iteration in range(max_iter):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        
        T = T_start * (T_end / T_start) ** (iteration / max_iter)
        
        idx = random.randint(0, N - 1)
        delta, new_c3 = fa.flip(idx)
        
        if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
            current_c3 = new_c3
            if current_c3 < best_c3:
                best_x = fa.get_best_x()
                best_c3 = current_c3
        else:
            # Revert
            fa.flip(idx)
    
    return best_x, best_c3

def _logistic_sequence(n, r=3.99, x0=0.5):
    x = np.zeros(n)
    val = x0
    for i in range(n):
        val = r * val * (1.0 - val)
        x[i] = 1.0 if val > 0.5 else -1.0
    return x

def _construct():
    start_time = time.time()
    random.seed(42)
    np.random.seed(42)
    
    best_x = np.ones(N)
    best_c3 = _compute_c3_slow(best_x)
    
    # Strategy 1: Random initializations with fast local search
    for trial in range(20):
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        x_init = np.random.choice([-1, 1], size=N)
        time_limit = max(2.0, 15.0 - trial * 0.5)
        x_trial, c3_trial = _fast_local_search(x_init, max_iter=100000, time_limit=time_limit)
        if c3_trial < best_c3:
            best_x = x_trial
            best_c3 = c3_trial
    
    # Strategy 2: Simulated Annealing from best
    if time.time() < DEADLINE - TIME_MARGIN:
        remaining = DEADLINE - TIME_MARGIN - time.time()
        x3, c3_3 = _fast_simulated_annealing(best_x, max_iter=500000, T_start=0.5, T_end=0.01,
                                             time_limit=remaining * 0.6)
        if c3_3 < best_c3:
            best_x = x3
            best_c3 = c3_3

    # Strategy 3: Final Polish
    if time.time() < DEADLINE - TIME_MARGIN:
        remaining = DEADLINE - TIME_MARGIN - time.time()
        x4, c3_4 = _fast_local_search(best_x, max_iter=500000, time_limit=remaining)
        if c3_4 < best_c3:
            best_x = x4
            best_c3 = c3_4

    return best_x

def construct():
    x = _construct()
    return np.sign(x).astype(int)
# EVOLVE-BLOCK-END
