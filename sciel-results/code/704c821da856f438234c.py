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

def _compute_c3(x):
    """
    Compute the upper bound C3 for a signed step function x of length N using FFT.
    C3 = max(|R[m]| for m in 1..N-1) / R[0]
    where R is the autocorrelation.
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
        self.R = np.zeros(N)
        self._update_R_full()
        self.current_c3 = self._calc_c3()

    def _update_R_full(self):
        # Compute R[0..N-1] using FFT for speed
        n_fft = 1 << (2 * N - 1).bit_length()
        F = np.fft.rfft(self.x, n=n_fft)
        power = F * np.conj(F)
        R_full = np.fft.irfft(power, n=n_fft)[:2 * N - 1]
        # R_full[k] for k=0..N-1 gives R[0], R[1], ..., R[N-1]
        self.R[:N] = R_full[:N]
        self.current_c3 = self._calc_c3()

    def _calc_c3(self):
        if self.R[0] == 0:
            return 1.0
        # Max absolute sidelobe
        max_sidelobe = np.max(np.abs(self.R[1:N]))
        return max_sidelobe / self.R[0]

    def flip(self, idx):
        """
        Flip x[idx] and update R incrementally.
        Returns the change in C3 (new_c3 - old_c3).
        """
        old_c3 = self.current_c3
        old_x_val = self.x[idx]
        
        # Calculate deltas for R[m]
        # Delta R[m] = -2 * x[idx] * (x[idx+m] if valid + x[idx-m] if valid)
        # Note: For m=0, R[0] is constant N, so delta is 0.
        
        # We update R in place
        for m in range(1, N):
            delta = 0.0
            # Term 1: i=idx, so x[idx]*x[idx+m]. Valid if idx + m < N
            if idx + m < N:
                delta -= 2.0 * old_x_val * self.x[idx + m]
            # Term 2: i+m=idx, so x[idx-m]*x[idx]. Valid if idx - m >= 0
            if idx - m >= 0:
                delta -= 2.0 * self.x[idx - m] * old_x_val
            
            if delta != 0.0:
                self.R[m] += delta
        
        # Flip the bit
        self.x[idx] = -old_x_val
        
        new_c3 = self._calc_c3()
        return new_c3 - old_c3, new_c3

    def get_best_x(self):
        return self.x.copy()

def _fast_local_search(x_init, max_iter=50000, time_limit=None):
    start_time = time.time()
    fa = _FastAutocorr(x_init)
    best_x = fa.get_best_x()
    best_c3 = fa.current_c3
    no_improve_count = 0
    max_no_improve = 200 # Higher because steps are faster
    
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

def _fast_simulated_annealing(x_init, max_iter=100000, T_start=0.5, T_end=0.01, time_limit=None):
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

def _van_der_corput(n):
    """Generate van der Corput sequence and convert to signs."""
    x = np.zeros(n)
    for i in range(1, n + 1):
        # Van der Corput sequence in base 2
        vdc = 0.0
        denom = 1.0
        val = i
        while val > 0:
            denom *= 2.0
            vdc += (val % 2) / denom
            val //= 2
        x[i-1] = 1.0 if vdc > 0.5 else -1.0
    return x

def _chaos_initializations(n):
    """Generate diverse initializations using chaos theory."""
    inits = []
    
    # Logistic map with different parameters
    for r in [3.5, 3.7, 3.9, 3.99]:
        for x0 in [0.2, 0.3, 0.7, 0.8]:
            x = np.zeros(n)
            val = x0
            for i in range(n):
                val = r * val * (1.0 - val)
                x[i] = 1.0 if val > 0.5 else -1.0
            inits.append(x)
    
    # Van der Corput
    inits.append(_van_der_corput(n))
    
    # Alternating blocks of different lengths
    for block_len in [2, 3, 5, 7, 10]:
        x = np.zeros(n)
        sign = 1
        pos = 0
        while pos < n:
            end = min(pos + block_len, n)
            x[pos:end] = sign
            sign = -sign
            pos = end
        inits.append(x)
    
    # Random with bias
    for bias in [0.3, 0.4, 0.6, 0.7]:
        x = np.random.choice([-1, 1], size=n, p=[bias, 1-bias])
        inits.append(x)
    
    return inits

def _construct():
    start_time = time.time()
    random.seed(42)
    np.random.seed(42)
    
    best_x = np.ones(N)
    best_c3 = _compute_c3(best_x)
    
    # Generate diverse initializations
    inits = _chaos_initializations(N)
    # Add random initializations
    for _ in range(5):
        inits.append(np.random.choice([-1, 1], size=N))
    
    # Strategy 1: Try each initialization with fast local search
    for trial, x_init in enumerate(inits):
        if time.time() > DEADLINE - TIME_MARGIN:
            break
        time_limit = max(3.0, 25.0 - trial * 2.0)
        x_trial, c3_trial = _fast_local_search(x_init, max_iter=15000, time_limit=time_limit)
        if c3_trial < best_c3:
            best_x = x_trial
            best_c3 = c3_trial
    
    # Strategy 2: Simulated Annealing from best
    if time.time() < DEADLINE - TIME_MARGIN:
        remaining = DEADLINE - TIME_MARGIN - time.time()
        x3, c3_3 = _fast_simulated_annealing(best_x, max_iter=40000, T_start=0.3, T_end=0.005,
                                             time_limit=remaining * 0.5)
        if c3_3 < best_c3:
            best_x = x3
            best_c3 = c3_3

    # Strategy 3: Final Polish with more iterations
    if time.time() < DEADLINE - TIME_MARGIN:
        remaining = DEADLINE - TIME_MARGIN - time.time()
        x4, c3_4 = _fast_local_search(best_x, max_iter=30000, time_limit=remaining)
        if c3_4 < best_c3:
            best_x = x4
            best_c3 = c3_4

    return best_x

def construct():
    x = _construct()
    return np.sign(x).astype(int)
# EVOLVE-BLOCK-END
