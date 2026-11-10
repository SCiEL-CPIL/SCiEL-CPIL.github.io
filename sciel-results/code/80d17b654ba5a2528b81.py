import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    N = 129  # elements in [0, 128]
    MIN_SUM = 128
    MAX_SUM = 256
    NUM_SUMS = MAX_SUM - MIN_SUM + 1  # 129

    # Precompute FFT size
    fft_size = 512  # next power of 2 >= 2*129-1 = 257

    def calc_mult_fft(A_arr):
        """
        A_arr: boolean array of length 129, True if element is in A.
        Returns array of unordered pair multiplicities for sums in [128, 256].
        """
        sig = A_arr.astype(np.float64)
        sig_fft = np.fft.fft(sig, n=fft_size)
        conv_fft = sig_fft * sig_fft
        conv = np.fft.ifft(conv_fft).real
        M_ordered = np.round(conv[MIN_SUM:MAX_SUM + 1]).astype(np.int32)

        # Convert ordered to unordered
        M_unordered = np.zeros(NUM_SUMS, dtype=np.int32)
        for i in range(NUM_SUMS):
            s = MIN_SUM + i
            ordered_count = M_ordered[i]
            if s % 2 == 0:
                half = s // 2
                if 0 <= half < N and A_arr[half]:
                    M_unordered[i] = (ordered_count + 1) // 2
                else:
                    M_unordered[i] = ordered_count // 2
            else:
                M_unordered[i] = ordered_count // 2
        return M_unordered

    def calc_score(A_arr):
        """
        Returns (gaps, peak_mult). Lower is better. (0, 1) is optimal.
        """
        M = calc_mult_fft(A_arr)
        gaps = int(np.sum(M == 0))
        if gaps > 0:
            return (gaps, float('inf'))
        peak = int(np.max(M))
        return (0, peak)

    def get_contributors(A_arr, target_sum):
        """
        Returns list of elements a in A such that target_sum - a is also in A and a <= target_sum - a.
        """
        contributors = []
        for a in range(N):
            if A_arr[a]:
                b = target_sum - a
                if 0 <= b < N and A_arr[b] and a <= b:
                    contributors.append(a)
        return contributors

    def targeted_local_search(A_arr, deadline, max_iters=5000):
        """
        Local search that specifically attacks highest-multiplicity targets.
        """
        current_score = calc_score(A_arr)
        if current_score == (0, 1):
            return current_score, A_arr

        improved = True
        iters = 0

        while improved and time.time() < deadline and iters < max_iters:
            improved = False
            iters += 1

            M = calc_mult_fft(A_arr)
            max_mult = int(np.max(M))
            if max_mult <= 1:
                break

            # Get sums with max multiplicity
            high_mult_indices = np.where(M == max_mult)[0]

            # Pick a random bottleneck sum to focus on
            target_idx = random.choice(high_mult_indices)
            target_sum = MIN_SUM + target_idx

            # Get contributors to this sum
            contributors = get_contributors(A_arr, target_sum)

            if not contributors:
                continue

            A_list = [a for a in range(N) if A_arr[a]]
            available = [x for x in range(N) if not A_arr[x]]

            # Try removing one of the contributors
            random.shuffle(contributors)
            for a in contributors:
                if time.time() > deadline:
                    break
                new_A_arr = A_arr.copy()
                new_A_arr[a] = False
                if not np.any(new_A_arr):
                    continue
                new_score = calc_score(new_A_arr)
                if new_score < current_score:
                    current_score = new_score
                    A_arr = new_A_arr
                    improved = True
                    break

            if improved:
                continue

            # Try swapping a contributor with an available element
            if available:
                random.shuffle(contributors)
                for a in contributors:
                    if time.time() > deadline:
                        break
                    # Pick a few random candidates
                    n_candidates = min(8, len(available))
                    candidates = random.sample(available, n_candidates)
                    for b in candidates:
                        new_A_arr = A_arr.copy()
                        new_A_arr[a] = False
                        new_A_arr[b] = True
                        new_score = calc_score(new_A_arr)
                        if new_score < current_score:
                            current_score = new_score
                            A_arr = new_A_arr
                            improved = True
                            break
                    if improved:
                        break

        return current_score, A_arr

    best_score = (float('inf'), float('inf'))
    best_A_arr = None

    # Phase 1: Multi-restart local search
    # Start with random sets of various sizes
    for k in range(20, 61, 2):
        if time.time() > deadline - 5:
            break

        n_restarts = 5
        for restart in range(n_restarts):
            if time.time() > deadline:
                break

            # Random initialization
            A_arr = np.zeros(N, dtype=np.bool_)
            chosen = random.sample(range(N), k)
            for a in chosen:
                A_arr[a] = True

            # Local search
            current_score, A_arr = targeted_local_search(A_arr, deadline)

            if current_score < best_score:
                best_score = current_score
                best_A_arr = A_arr.copy()

            if best_score == (0, 1):
                return [a for a in range(N) if best_A_arr[a]]

    # Phase 2: If we found a valid solution, try to shrink it
    if best_A_arr is not None and best_score[0] == 0:
        A_arr = best_A_arr.copy()
        current_score = best_score
        improved = True
        while improved and time.time() < deadline:
            improved = False
            A_list = [a for a in range(N) if A_arr[a]]
            random.shuffle(A_list)
            for a in A_list:
                if time.time() > deadline:
                    break
                if np.sum(A_arr) <= 1:
                    break
                new_A_arr = A_arr.copy()
                new_A_arr[a] = False
                new_score = calc_score(new_A_arr)
                if new_score[0] == 0:
                    if new_score[1] < current_score[1]:
                        current_score = new_score
                        A_arr = new_A_arr
                        improved = True
                        break
                    elif new_score[1] == current_score[1] and np.sum(new_A_arr) < np.sum(A_arr):
                        current_score = new_score
                        A_arr = new_A_arr
                        improved = True
                        break

        if current_score < best_score:
            best_score = current_score
            best_A_arr = A_arr

    if best_A_arr is None:
        # Fallback: A set that is guaranteed to cover
        A_arr = np.zeros(N, dtype=np.bool_)
        for a in range(64, 129):
            A_arr[a] = True
        return [a for a in range(N) if A_arr[a]]

    return [a for a in range(N) if best_A_arr[a]]

# Run construct to test
if __name__ == "__main__":
    start = time.time()
    A = construct()
    end = time.time()
    print(f"Time: {end-start:.2f}s")
    print(f"Set size: {len(A)}")
    # Verify
    A_set = set(A)
    A_arr = np.zeros(129, dtype=np.bool_)
    for a in A:
        A_arr[a] = True
    M = calc_mult_fft(A_arr)
    print(f"Min mult: {np.min(M)}, Max mult: {np.max(M)}")
    print(f"Gaps: {np.sum(M==0)}")
    print(f"Score: {calc_score(A_arr)}")
