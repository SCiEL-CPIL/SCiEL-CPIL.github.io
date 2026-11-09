import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin (evaluator kills at 90s)
    random.seed(42)  # Reproducibility

    N_LOW = 128
    N_HIGH = 256
    RANGE_SIZE = N_HIGH - N_LOW + 1  # 129
    MAX_VAL = 128
    N_DOMAIN = MAX_VAL + 1  # 129

    # Precompute FFT sizes
    fft_size = 1
    while fft_size < 2 * N_DOMAIN - 1:
        fft_size *= 2

    # Precompute sums and diagonal masks for speed
    sums_arr = np.arange(N_LOW, N_HIGH + 1)
    even_mask = (sums_arr % 2 == 0)
    half_sums = sums_arr // 2
    # Mask for valid half sums (must be in [0, 128])
    valid_half_mask = (half_sums >= 0) & (half_sums <= MAX_VAL)
    # We need to check if half_sum is in A. We will do this dynamically or precompute for all possible A? 
    # Dynamic is better for memory, but we can precompute the indices.
    # For a given A (indicator vector), diag[i] = 1 if sums_arr[i] is even and sums_arr[i]//2 is in A.
    # Let A_ind be the indicator vector of A (length 129).
    # diag_count[i] = A_ind[sums_arr[i]//2] if sums_arr[i] is even else 0.

    def calc_reps(A_set):
        """Returns array of unordered pair counts for sums N_LOW..N_HIGH."""
        if not A_set:
            return np.zeros(RANGE_SIZE, dtype=np.int32)

        # Create indicator vector
        f = np.zeros(N_DOMAIN, dtype=np.float64)
        # Vectorized update of f is hard with a set, but loop is small (<=129)
        for x in A_set:
            f[x] = 1.0

        # FFT convolution
        F = np.fft.fft(f, n=fft_size)
        conv = np.fft.ifft(F * F).real
        
        # Extract relevant range
        lo_idx = N_LOW
        hi_idx = N_HIGH + 1
        ordered = conv[lo_idx:hi_idx]

        # Compute diagonal contribution
        # diag[i] = 1 if sums_arr[i] is even and f[sums_arr[i]//2] == 1
        # We can extract the diagonal values directly from f
        # half_sums is the index into f for the diagonal
        # We only care about indices where even_mask is True
        diag_vals = np.zeros(RANGE_SIZE, dtype=np.float64)
        # Get the values of f at the half-sum positions for even sums
        # half_sums[even_mask] gives the indices in f
        valid_indices = half_sums[even_mask]
        # Ensure valid_indices are within [0, 128]
        valid_mask = (valid_indices >= 0) & (valid_indices <= MAX_VAL)
        if np.any(valid_mask):
            # f[valid_indices[valid_mask]] gives 1.0 if in A, 0.0 otherwise
            diag_vals[even_mask][valid_mask] = f[valid_indices[valid_mask]]
        
        # Unordered count = (ordered + diag) / 2
        unordered = (ordered + diag_vals) / 2.0
        counts = np.round(unordered).astype(np.int32)
        return counts

    def is_covered(reps):
        return np.all(reps > 0)

    def get_peak(reps):
        return int(np.max(reps))

    def get_l2(reps):
        # Sum of squares, proxy for variance/peak flatness
        return float(np.sum(reps * reps))

    def get_total(reps):
        return int(np.sum(reps))

    def objective(reps):
        """Primary: L2 norm (flattening). Secondary: Peak. Tertiary: Total."""
        if not is_covered(reps):
            return (float('inf'), get_peak(reps), get_total(reps))
        return (get_l2(reps), get_peak(reps), get_total(reps))

    best_A = None
    best_obj = (float('inf'), float('inf'), float('inf'))

    def update_best(A, reps):
        nonlocal best_A, best_obj
        obj = objective(reps)
        if obj < best_obj:
            best_obj = obj
            best_A = list(A)

    def random_init():
        A = set()
        # Ensure some spread
        for x in range(0, 129, 4):
            A.add(x)
        # Add random
        while len(A) < 40:
            A.add(random.randint(0, 128))
        return A

    def repair_coverage(A):
        """Add elements until all sums in [128,256] are covered."""
        A = set(A)
        reps = calc_reps(A)
        tries = 0
        while not is_covered(reps) and tries < 100:
            missing = np.where(reps == 0)[0] + N_LOW
            if len(missing) == 0:
                break
            # Pick a random missing sum
            s = random.choice(missing)
            # Find a candidate x to add
            # x + y = s, y in A
            # y = s - x
            # x must be in [0, 128]
            # y must be in A
            candidates = []
            for y in A:
                x = s - y
                if 0 <= x <= 128 and x not in A:
                    candidates.append(x)
            if candidates:
                # Pick the candidate that adds the most coverage or lowest L2 increase
                # For speed, just pick random
                c = random.choice(candidates)
                A.add(c)
            else:
                # No direct fix, add random
                A.add(random.randint(0, 128))
            reps = calc_reps(A)
            tries += 1
        return A, reps

    # Phase 1: Multi-start Local Search with L2-aware moves
    num_restarts = 50

    for r in range(num_restarts):
        if time.time() > deadline:
            break

        current_A = random_init()
        current_A, current_reps = repair_coverage(current_A)

        if not is_covered(current_reps):
            continue

        update_best(current_A, current_reps)

        # Local Search: Simulated Annealing
        T = 5.0
        cooling = 0.995
        min_T = 0.01
        stuck_counter = 0
        max_stuck = 100
        iterations = 0

        while time.time() < deadline and T > min_T:
            if stuck_counter > max_stuck:
                break

            current_l2 = get_l2(current_reps)
            current_peak = get_peak(current_reps)
            
            # Choose move: Remove, Add, Swap
            move_type = random.random()

            if len(current_A) > 25 and move_type < 0.4:
                # Remove
                item_to_remove = random.choice(list(current_A))
                new_A = set(current_A)
                new_A.remove(item_to_remove)
                new_reps = calc_reps(new_A)
            elif move_type < 0.7:
                # Swap
                item_to_remove = random.choice(list(current_A))
                new_item = random.randint(0, 128)
                if new_item == item_to_remove:
                    T *= cooling
                    continue
                new_A = set(current_A)
                new_A.remove(item_to_remove)
                new_A.add(new_item)
                new_reps = calc_reps(new_A)
            else:
                # Add
                new_item = random.randint(0, 128)
                if new_item in current_A:
                    T *= cooling
                    continue
                new_A = set(current_A)
                new_A.add(new_item)
                new_reps = calc_reps(new_A)

            if is_covered(new_reps):
                new_l2 = get_l2(new_reps)
                new_peak = get_peak(new_reps)
                
                # Accept if L2 decreases, or if L2 same and Peak decreases
                # Or with probability if L2 increases slightly
                if new_l2 < current_l2 or (new_l2 == current_l2 and new_peak <= current_peak):
                    current_A = new_A
                    current_reps = new_reps
                    update_best(current_A, current_reps)
                    stuck_counter = 0
                else:
                    # Accept worse moves with probability exp(-delta/T)
                    delta = new_l2 - current_l2
                    if delta < 0:
                        prob = 1.0
                    else:
                        prob = np.exp(-delta / (T * 100.0)) # Scale T for L2
                    if random.random() < prob:
                        current_A = new_A
                        current_reps = new_reps
                        stuck_counter = 0
                    else:
                        stuck_counter += 1
            else:
                stuck_counter += 1

            T *= cooling
            iterations += 1

    # Final Polish: Greedy Removal
    if best_A:
        A = set(best_A)
        reps = calc_reps(A)
        
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for x in list(A):
                new_A = set(A)
                new_A.remove(x)
                new_reps = calc_reps(new_A)
                if is_covered(new_reps):
                    new_obj = objective(new_reps)
                    old_obj = objective(reps)
                    if new_obj < old_obj:
                        A = new_A
                        reps = new_reps
                        changed = True
                        update_best(A, reps)
                        break

    if best_A is None:
        return list(range(64, 129))

    return best_A
