import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    n = 128
    min_sum = 128
    max_sum = 256
    num_sums = max_sum - min_sum + 1

    def compute_initial_U(A_set):
        """Compute unordered counts U[s] for s in [min_sum, max_sum]."""
        x_bin = np.zeros(257, dtype=np.float64)
        for x in A_set:
            x_bin[x] = 1.0
        conv = np.convolve(x_bin, x_bin, mode='full')
        U = np.zeros(num_sums, dtype=np.int32)
        for i, s in enumerate(range(min_sum, max_sum + 1)):
            ordered = conv[s]
            diag = 0
            if s % 2 == 0:
                mid = s // 2
                if 0 <= mid <= n:
                    diag = x_bin[mid]
            U[i] = int((ordered + diag) / 2)
        return U

    def get_peak(U):
        return int(np.max(U))

    def is_valid(U):
        return np.all(U > 0)

    # Initialization
    best_initial_U = None
    best_initial_A = None
    best_initial_peak = float('inf')

    for a in range(0, 65):
        for b in range(64, 129):
            if a >= b:
                continue
            A = set(range(0, a + 1)) | set(range(b, 129))
            U = compute_initial_U(A)
            if is_valid(U):
                peak = get_peak(U)
                if peak < best_initial_peak:
                    best_initial_peak = peak
                    best_initial_A = A
                    best_initial_U = U

    if best_initial_A is None:
        best_initial_A = set(range(64, 129))
        best_initial_U = compute_initial_U(best_initial_A)
        best_initial_peak = get_peak(best_initial_U)

    current_A = best_initial_A.copy()
    current_U = best_initial_U.copy()
    current_peak = best_initial_peak
    
    best_A = current_A.copy()
    best_U = current_U.copy()
    best_peak = current_peak

    random.seed(42)

    # Iterated Local Search
    while time.time() < deadline:
        # Local Search Phase
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try all possible moves
            best_move = None
            best_new_peak = current_peak
            
            # Add moves
            for x in range(129):
                if x in current_A:
                    continue
                new_U = current_U.copy()
                for y in current_A:
                    s = x + y
                    if min_sum <= s <= max_sum:
                        new_U[s - min_sum] += 1
                if np.any(new_U == 0):
                    continue
                new_peak = int(np.max(new_U))
                if new_peak < best_new_peak:
                    best_new_peak = new_peak
                    best_move = ('add', x, new_U)
            
            # Remove moves
            for x in current_A:
                new_U = current_U.copy()
                for y in current_A:
                    if y == x:
                        continue
                    s = x + y
                    if min_sum <= s <= max_sum:
                        new_U[s - min_sum] -= 1
                if np.any(new_U == 0):
                    continue
                new_peak = int(np.max(new_U))
                if new_peak < best_new_peak:
                    best_new_peak = new_peak
                    best_move = ('remove', x, new_U)
            
            if best_move is not None:
                move_type, x, new_U = best_move
                if move_type == 'add':
                    current_A.add(x)
                else:
                    current_A.discard(x)
                current_U = new_U
                current_peak = best_new_peak
                improved = True
                
                if current_peak < best_peak:
                    best_peak = current_peak
                    best_A = current_A.copy()
                    best_U = current_U.copy()
        
        # Perturbation Phase
        if time.time() >= deadline:
            break
            
        # Swap 2-3 elements
        num_swaps = random.randint(2, 4)
        
        for _ in range(num_swaps):
            if len(current_A) < 1:
                break
            x_remove = random.choice(list(current_A))
            candidates = [x for x in range(129) if x not in current_A]
            if not candidates:
                break
            x_add = random.choice(candidates)
            current_A.discard(x_remove)
            current_A.add(x_add)
        
        # Recompute U
        current_U = compute_initial_U(current_A)
        if not is_valid(current_U):
            current_A = best_A.copy()
            current_U = best_U.copy()
            current_peak = best_peak
        else:
            current_peak = get_peak(current_U)
            if current_peak < best_peak:
                best_peak = current_peak
                best_A = current_A.copy()
                best_U = current_U.copy()

    if best_A is None or len(best_A) == 0:
        return list(range(64, 129))

    return sorted(best_A)
