import numpy as np
import time

N = 400

def _calc_acf(s):
    """Compute full autocorrelation array A[0..N-1]"""
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
    s_f = s.astype(np.float64)
    F = np.fft.rfft(s_f, n=n_fft)
    P = F * np.conj(F)
    acf = np.fft.irfft(P, n=n_fft)
    return np.real(acf[:N])

def _max_abs_acf_from_array(A):
    return float(np.max(np.abs(A[1:])))

def _flip_delta(s, A, i):
    """
    Compute the change in ACF if we flip s[i] -> -s[i].
    Returns delta array where A_new = A + delta.
    """
    delta = np.zeros(N)
    old_si = s[i]
    factor = -2.0 * old_si
    
    if i < N:
        delta[:N-i] += factor * s[i:N]
    
    if i >= 0:
        delta[:i+1] += factor * s[i::-1]
    
    return delta

def _thue_morse_init(N):
    """Generate Thue-Morse sequence mapped to signs."""
    # Thue-Morse: t(n) = parity of number of 1s in binary representation of n
    s = np.zeros(N)
    for n in range(N):
        # Count bits
        bits = bin(n).count('1')
        s[n] = 1.0 if bits % 2 == 0 else -1.0
    return s

def _legendre_init(N):
    """Generate Legendre symbol sequence for a prime modulus."""
    # Use a prime close to N
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    # Find a prime >= N
    p = N
    while not is_prime(p):
        p += 1
    
    s = np.zeros(N)
    for n in range(N):
        # Legendre symbol (n|p) = n^((p-1)/2) mod p
        if n == 0:
            s[n] = 1.0
        else:
            # Compute n^((p-1)/2) mod p
            result = pow(n, (p-1)//2, p)
            if result == 1:
                s[n] = 1.0
            elif result == p-1:  # -1 mod p
                s[n] = -1.0
            else:
                s[n] = 1.0  # Should not happen for prime p
    return s

def _ca_init(N, rule=30, steps=10):
    """Generate pattern from 1D cellular automaton."""
    # Start with a random seed pattern
    tape = np.random.randint(0, 2, size=N)
    
    for _ in range(steps):
        new_tape = np.zeros(N, dtype=int)
        for i in range(N):
            left = tape[(i-1) % N]
            center = tape[i]
            right = tape[(i+1) % N]
            # Rule 30: new state depends on (left, center, right)
            idx = left * 4 + center * 2 + right
            # Rule 30 in binary: 00011110
            rule_bits = [0, 0, 0, 1, 1, 1, 1, 0]
            new_tape[i] = rule_bits[idx]
        tape = new_tape
    
    s = 2.0 * tape - 1.0  # Map 0/1 to -1/1
    return s

def _turing_machine_init(N, steps=50):
    """Generate pattern from a simple 1D Turing machine."""
    # Simple Turing machine: tape of 0/1, head position, state
    tape = np.zeros(N, dtype=int)
    head = N // 2
    state = 0
    
    # Transition table: (state, read) -> (write, direction, new_state)
    # This is a simplified version
    transitions = {
        (0, 0): (1, 1, 1),   # Write 1, move right, go to state 1
        (0, 1): (0, -1, 2),  # Write 0, move left, go to state 2
        (1, 0): (1, -1, 0),  # Write 1, move left, go to state 0
        (1, 1): (0, 1, 3),   # Write 0, move right, go to state 3
        (2, 0): (1, 1, 1),   # Write 1, move right, go to state 1
        (2, 1): (0, -1, 2),  # Write 0, move left, go to state 2
        (3, 0): (1, -1, 0),  # Write 1, move left, go to state 0
        (3, 1): (0, 1, 3),   # Write 0, move right, go to state 3
    }
    
    for _ in range(steps):
        read = tape[head]
        write, direction, new_state = transitions[(state, read)]
        tape[head] = write
        head = (head + direction) % N
        state = new_state
    
    s = 2.0 * tape.astype(float) - 1.0
    return s

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    best_val = float('inf')
    best_seq = np.ones(N)
    
    # Try different initializations
    init_methods = [
        _thue_morse_init,
        _legendre_init,
        _ca_init,
        _turing_machine_init,
        lambda N: np.random.randint(0, 2, size=N) * 2 - 1,
    ]
    
    # Shuffle initialization order
    np.random.shuffle(init_methods)
    
    for init_func in init_methods:
        if time.time() > deadline:
            break
        
        try:
            s = init_func(N).astype(np.float64)
        except:
            s = np.random.randint(0, 2, size=N) * 2 - 1
        
        A = _calc_acf(s)
        current_val = _max_abs_acf_from_array(A)
        
        # Greedy local search with multiple passes
        improved = True
        passes = 0
        max_passes = 100
        patience = 0
        max_patience = 2000
        
        while improved and passes < max_passes and patience < max_patience and time.time() < deadline:
            improved = False
            passes += 1
            patience = 0
            
            # Try different orderings
            if passes % 3 == 0:
                indices = np.random.permutation(N)
            elif passes % 3 == 1:
                indices = np.arange(N)
            else:
                indices = np.arange(N-1, -1, -1)
            
            for i in indices:
                if time.time() > deadline:
                    break
                
                # Compute delta
                delta = _flip_delta(s, A, i)
                
                # New value would be max(|A[k] + delta[k]|) for k=1..N-1
                A_new = A + delta
                val_new = _max_abs_acf_from_array(A_new)
                
                if val_new < current_val:
                    # Accept the flip
                    s[i] = -s[i]
                    A = A_new
                    current_val = val_new
                    improved = True
                    patience = 0
                else:
                    patience += 1
        
        # Simulated annealing polish
        if time.time() < deadline:
            T = 0.05
            T_min = 0.001
            cooling = 0.998
            sa_start = time.time()
            sa_deadline = min(deadline, sa_start + 8.0)
            
            while T > T_min and time.time() < sa_deadline:
                i = np.random.randint(N)
                delta = _flip_delta(s, A, i)
                A_new = A + delta
                val_new = _max_abs_acf_from_array(A_new)
                
                if val_new < current_val or np.random.random() < np.exp(-(val_new - current_val) / T):
                    s[i] = -s[i]
                    A = A_new
                    current_val = val_new
                
                T *= cooling
        
        if current_val < best_val:
            best_val = current_val
            best_seq = s.copy()
    
    return best_seq
