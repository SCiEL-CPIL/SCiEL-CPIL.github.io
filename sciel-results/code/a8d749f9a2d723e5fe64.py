import time
import numpy as np

def _score_set(U):
    """Compute the lower bound 1 + log(|U-U|/|U+U|)/log(2 max U + 1)."""
    if not U or len(U) < 2:
        return 0.0
    arr = np.array(U, dtype=np.int64)
    if arr[0] != 0:
        arr = np.concatenate(([0], arr))
    arr = np.unique(arr)
    n = len(arr)
    if n < 2:
        return 0.0
    mx = int(arr.max())
    if mx == 0:
        return 0.0
    
    # Size of sumset U+U: values in [0, 2*mx]
    # Size of diffset U-U: values in [-mx, mx], shift by mx -> [0, 2*mx]
    N = 2 * mx + 1
    
    # Efficient counting via FFT
    sig = np.zeros(N, dtype=np.float64)
    sig[arr] = 1.0
    
    fft_size = 1
    while fft_size < 2 * N:
        fft_size *= 2
        
    F = np.fft.rfft(sig, n=fft_size)
    
    # Sumset
    S_sum = np.fft.irfft(F * F, n=fft_size)
    sumset_size = np.sum(S_sum[:N] > 1e-6) # Sums range 0..2*mx which is 0..N-1
    
    # Diffset
    flip_sig = np.zeros(N, dtype=np.float64)
    flip_sig[:] = sig[::-1]
    
    F_flip = np.fft.rfft(flip_sig, n=fft_size)
    S_diff = np.fft.irfft(F * F_flip, n=fft_size)
    
    diffset_size = np.sum(S_diff > 1e-6)
    
    if sumset_size == 0:
        return 0.0
        
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
        
    denom = np.log(2 * mx + 1)
    if denom == 0:
        return 0.0
        
    return 1.0 + np.log(ratio) / denom

def _score_fast(U):
    """Fast scoring for small sets using set operations."""
    if not U or len(U) < 2:
        return 0.0
    U_set = set(U)
    if 0 not in U_set:
        U_set.add(0)
    U_list = sorted(U_set)
    n = len(U_list)
    if n < 2:
        return 0.0
    mx = U_list[-1]
    if mx == 0:
        return 0.0
    
    sumset = set()
    diffset = set()
    for i in range(n):
        ui = U_list[i]
        for j in range(n):
            uj = U_list[j]
            sumset.add(ui + uj)
            diffset.add(ui - uj)
    
    s_size = len(sumset)
    d_size = len(diffset)
    if s_size == 0:
        return 0.0
    ratio = d_size / s_size
    if ratio <= 1:
        return 0.0
    return 1.0 + np.log(ratio) / np.log(2 * mx + 1)

def _greedy_construct():
    """
    Greedy construction: start with {0}, repeatedly add the element that maximizes
    the marginal improvement in score.
    """
    U = [0]
    # Try adding elements in order of increasing value, but with some heuristic
    # We'll try to add elements that spread out the set to maximize diffset/sumset ratio
    
    max_val = 200  # Limit search space for greedy
    
    for _ in range(20):  # Try to add up to 20 elements
        best_elem = None
        best_score = _score_fast(U)
        
        for candidate in range(1, max_val + 1):
            if candidate in U:
                continue
            new_U = U + [candidate]
            s = _score_fast(new_U)
            if s > best_score:
                best_score = s
                best_elem = candidate
        
        if best_elem is None:
            break
        U.append(best_elem)
        U.sort()
    
    return U

def _local_search(U, deadline):
    """
    Local search: try to improve U by adding, removing, or moving elements.
    """
    U = sorted(set(U))
    if 0 not in U:
        U = [0] + U
    current_score = _score_fast(U)
    best_score = current_score
    best_U = U[:]
    
    no_improve = 0
    max_no_improve = 50
    
    while time.time() < deadline:
        improved = False
        
        # Move 1: Add a random element
        if len(U) < 30:
            candidate = np.random.randint(1, 500)
            if candidate not in U:
                new_U = sorted(U + [candidate])
                s = _score_fast(new_U)
                if s > current_score:
                    U = new_U
                    current_score = s
                    improved = True
                    no_improve = 0
                    if s > best_score:
                        best_score = s
                        best_U = U[:]
        
        # Move 2: Remove a random element (not 0)
        if len(U) > 2:
            idx = np.random.randint(1, len(U))
            new_U = U[:idx] + U[idx+1:]
            s = _score_fast(new_U)
            if s > current_score:
                U = new_U
                current_score = s
                improved = True
                no_improve = 0
                if s > best_score:
                    best_score = s
                    best_U = U[:]
        
        # Move 3: Change a random element (not 0)
        if len(U) > 1:
            idx = np.random.randint(1, len(U))
            old_val = U[idx]
            new_val = old_val + np.random.choice([-2, -1, 1, 2, 3, 5])
            if new_val > 0 and new_val not in U:
                new_U = U[:]
                new_U[idx] = new_val
                new_U.sort()
                s = _score_fast(new_U)
                if s > current_score:
                    U = new_U
                    current_score = s
                    improved = True
                    no_improve = 0
                    if s > best_score:
                        best_score = s
                        best_U = U[:]
        
        if not improved:
            no_improve += 1
        
        if no_improve > max_no_improve:
            # Random restart from a perturbed version of best
            U = best_U[:]
            for _ in range(3):
                idx = np.random.randint(1, len(U))
                U[idx] += np.random.randint(-5, 6)
                if U[idx] <= 0:
                    U[idx] = 1
                U.sort()
            U = sorted(set(U))
            if 0 not in U:
                U = [0] + U
            current_score = _score_fast(U)
            no_improve = 0
    
    return best_U

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    np.random.seed(42)
    
    # Start with greedy construction
    best_U = _greedy_construct()
    best_score = _score_fast(best_U)
    
    # Run local search
    U = _local_search(best_U, deadline)
    s = _score_fast(U)
    if s > best_score:
        best_U = U
        best_score = s
    
    # Try a few more random restarts if time permits
    while time.time() < deadline - 5:
        # Generate random initial set
        n = np.random.randint(8, 20)
        U_rand = [0]
        for _ in range(n - 1):
            U_rand.append(np.random.randint(1, 300))
        U_rand = sorted(set(U_rand))
        
        U_opt = _local_search(U_rand, deadline)
        s = _score_fast(U_opt)
        if s > best_score:
            best_U = U_opt
            best_score = s
    
    return best_U

if __name__ == "__main__":
    s = construct()
    print(f"Result: {s[:10]}... Size: {len(s)}, Score: {_score_set(s)}")
