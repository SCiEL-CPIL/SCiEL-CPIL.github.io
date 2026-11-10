import time
import random
import numpy as np

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin for 60s limit

    n = 128
    targets = list(range(128, 257))
    T = 129  # number of targets

    def compute_counts(A_set):
        counts = np.zeros(257, dtype=np.int32)
        A_list = sorted(A_set)
        for i in range(len(A_list)):
            ai = A_list[i]
            for j in range(i, len(A_list)):
                aj = A_list[j]
                s = ai + aj
                if s <= 256:
                    counts[s] += 1
        return counts

    def get_metrics(counts):
        c = counts[128:257]
        peak = int(np.max(c))
        sq = int(np.sum(c.astype(np.int64)**2))
        covered = bool(np.all(c > 0))
        return peak, sq, covered

    def smooth_penalty(counts, t):
        """Sum of (max(0, c-t))^2 over targets - smooth surrogate for peak."""
        c = counts[128:257]
        excess = np.maximum(c - t, 0)
        return int(np.sum(excess**2))

    def greedy_init():
        A = set()
        # Start with some random elements to break symmetry
        for _ in range(15):
            A.add(random.randint(0, n))
        
        counts = compute_counts(A)
        peak, sq, covered = get_metrics(counts)
        
        while not covered and time.time() - start_time < time_limit - 5:
            # Find uncovered targets
            uncovered = [t for t in targets if counts[t] == 0]
            if not uncovered:
                break
            
            best_x = -1
            best_gain = -1
            best_pen = None
            
            # Try adding each x not in A
            for x in range(n + 1):
                if x in A:
                    continue
                # Estimate gain: how many new targets covered
                new_sums = set()
                for a in A:
                    s = x + a
                    if 128 <= s <= 256:
                        new_sums.add(s)
                s = 2 * x
                if 128 <= s <= 256:
                    new_sums.add(s)
                
                gain = len([s for s in new_sums if counts[s] == 0])
                
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
                    # Estimate penalty increase
                    pen = 0
                    for s in new_sums:
                        if counts[s] + 1 > peak:
                            pen += ((counts[s] + 1 - peak)**2)
                    best_pen = pen
                elif gain == best_gain and best_pen is not None:
                    pen = 0
                    for s in new_sums:
                        if counts[s] + 1 > peak:
                            pen += ((counts[s] + 1 - peak)**2)
                    if pen < best_pen:
                        best_x = x
                        best_pen = pen
                    
            if best_x == -1 or best_gain == 0:
                break
            
            A.add(best_x)
            counts = compute_counts(A)
            peak, sq, covered = get_metrics(counts)
            
        return A

    def spectral_init():
        N = 129
        adj = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if i == j: continue
                s = i + j
                if 128 <= s <= 256:
                    adj[i][j] = 1.0
        
        try:
            eigvals, eigvecs = np.linalg.eigh(adj)
            principal_vec = np.abs(eigvecs[:, -1])
        except:
            principal_vec = np.ones(N)
            
        k = 50
        indices = np.argsort(-principal_vec)
        A = set(indices[:k].tolist())
        
        counts = compute_counts(A)
        peak, sq, covered = get_metrics(counts)
        
        if not covered:
            while not covered and time.time() - start_time < time_limit - 5:
                uncovered = [t for t in targets if counts[t] == 0]
                if not uncovered: break
                best_x = -1
                best_gain = -1
                for x in range(n + 1):
                    if x in A: continue
                    new_sums = set()
                    for a in A:
                        s = x + a
                        if 128 <= s <= 256: new_sums.add(s)
                    s = 2*x
                    if 128 <= s <= 256: new_sums.add(s)
                    gain = len([s for s in new_sums if counts[s] == 0])
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x == -1 or best_gain == 0: break
                A.add(best_x)
                counts = compute_counts(A)
                peak, sq, covered = get_metrics(counts)
                
        return A

    # 1. Initialization
    A = greedy_init()
    counts = compute_counts(A)
    curr_peak, curr_sq, curr_covered = get_metrics(counts)
    
    best_peak = curr_peak
    best_sq = curr_sq
    best_A = set(A)
    
    # Try spectral init too, keep better one
    A2 = spectral_init()
    counts2 = compute_counts(A2)
    p2, s2, c2 = get_metrics(counts2)
    if c2 and (p2 < best_peak or (p2 == best_peak and s2 < best_sq)):
        A = A2
        counts = counts2
        curr_peak, curr_sq, curr_covered = p2, s2, c2
        best_peak, best_sq = p2, s2
        best_A = set(A)

    # 2. Local Search with incremental count updates
    
    def trial_remove(x, A, counts):
        """Fast incremental: remove x from A, update counts. Returns (new_counts, new_peak, new_sq, covered)."""
        new_counts = counts.copy()
        for a in A:
            if a == x:
                continue
            s = x + a
            if s <= 256:
                new_counts[s] -= 1
        # Handle 2x
        s2 = 2 * x
        if s2 <= 256:
            new_counts[s2] -= 1
        c = new_counts[128:257]
        if np.any(c < 0):
            return None
        peak = int(np.max(c))
        sq = int(np.sum(c.astype(np.int64)**2))
        covered = bool(np.all(c > 0))
        return new_counts, peak, sq, covered

    def trial_add(x, A, counts):
        """Fast incremental: add x to A, update counts. Returns (new_counts, new_peak, new_sq, covered)."""
        new_counts = counts.copy()
        for a in A:
            s = x + a
            if s <= 256:
                new_counts[s] += 1
        s2 = 2 * x
        if s2 <= 256:
            new_counts[s2] += 1
        c = new_counts[128:257]
        peak = int(np.max(c))
        sq = int(np.sum(c.astype(np.int64)**2))
        covered = bool(np.all(c > 0))
        return new_counts, peak, sq, covered

    def trial_swap(x_out, y_in, A, counts):
        """Fast incremental: remove x_out, add y_in. Returns (new_counts, new_peak, new_sq, covered)."""
        new_counts = counts.copy()
        # Remove x_out contributions
        for a in A:
            if a == x_out:
                continue
            s = x_out + a
            if s <= 256:
                new_counts[s] -= 1
        s2 = 2 * x_out
        if s2 <= 256:
            new_counts[s2] -= 1
        # Add y_in contributions (with A minus x_out, plus y_in itself)
        for a in A:
            if a == x_out:
                continue
            s = y_in + a
            if s <= 256:
                new_counts[s] += 1
        s2 = 2 * y_in
        if s2 <= 256:
            new_counts[s2] += 1
        c = new_counts[128:257]
        if np.any(c < 0):
            return None
        peak = int(np.max(c))
        sq = int(np.sum(c.astype(np.int64)**2))
        covered = bool(np.all(c > 0))
        return new_counts, peak, sq, covered

    def is_better(p, s, cov, curr_peak, curr_sq):
        if not cov:
            return False
        if p < curr_peak:
            return True
        if p == curr_peak and s < curr_sq:
            return True
        return False

    no_improve_count = 0
    max_no_improve = 150
    
    A_list = list(A)
    C_list = [i for i in range(n + 1) if i not in A]
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Refresh lists if they've changed
        if len(A) != len(A_list):
            A_list = list(A)
            C_list = [i for i in range(n + 1) if i not in A]
        
        # Try Removals
        random.shuffle(A_list)
        for x in A_list:
            if x not in A:
                continue
            res = trial_remove(x, A, counts)
            if res is None:
                continue
            new_counts, p, s, cov = res
            if is_better(p, s, cov, curr_peak, curr_sq):
                A = set(A)
                A.discard(x)
                counts = new_counts
                curr_peak, curr_sq, curr_covered = p, s, cov
                if curr_peak < best_peak or (curr_peak == best_peak and curr_sq < best_sq):
                    best_peak = curr_peak
                    best_sq = curr_sq
                    best_A = set(A)
                improved = True
                no_improve_count = 0
                A_list = list(A)
                C_list = [i for i in range(n + 1) if i not in A]
                break
        if improved: continue

        # Try Swaps (1 out, 1 in)
        random.shuffle(A_list)
        random.shuffle(C_list)
        for x in A_list:
            if x not in A:
                continue
            for y in C_list:
                if y in A:
                    continue
                res = trial_swap(x, y, A, counts)
                if res is None:
                    continue
                new_counts, p, s, cov = res
                if is_better(p, s, cov, curr_peak, curr_sq):
                    A = set(A)
                    A.discard(x)
                    A.add(y)
                    counts = new_counts
                    curr_peak, curr_sq, curr_covered = p, s, cov
                    if curr_peak < best_peak or (curr_peak == best_peak and curr_sq < best_sq):
                        best_peak = curr_peak
                        best_sq = curr_sq
                        best_A = set(A)
                    improved = True
                    no_improve_count = 0
                    A_list = list(A)
                    C_list = [i for i in range(n + 1) if i not in A]
                    break
            if improved: break
        if improved: continue

        # Try 2-Swaps when stuck
        if no_improve_count > 30:
            A_list = list(A)
            C_list = [i for i in range(n + 1) if i not in A]
            if len(A_list) >= 2 and len(C_list) >= 2:
                for _ in range(50):
                    if time.time() - start_time > time_limit - 1:
                        break
                    x1, x2 = random.sample(A_list, 2)
                    y1, y2 = random.sample(C_list, 2)
                    if x1 == x2 or y1 == y2:
                        continue
                    # Manual 2-swap
                    new_counts = counts.copy()
                    # Remove x1, x2
                    for a in A:
                        if a == x1 or a == x2:
                            continue
                        for xout in (x1, x2):
                            s = xout + a
                            if s <= 256:
                                new_counts[s] -= 1
                    for xout in (x1, x2):
                        s2 = 2 * xout
                        if s2 <= 256:
                            new_counts[s2] -= 1
                    # Also x1+x2 was counted once
                    s12 = x1 + x2
                    if s12 <= 256:
                        new_counts[s12] -= 1
                    # Add y1, y2
                    for a in A:
                        if a == x1 or a == x2:
                            continue
                        for yin in (y1, y2):
                            s = yin + a
                            if s <= 256:
                                new_counts[s] += 1
                    for yin in (y1, y2):
                        s2 = 2 * yin
                        if s2 <= 256:
                            new_counts[s2] += 1
                    s12 = y1 + y2
                    if s12 <= 256:
                        new_counts[s12] += 1
                    
                    c = new_counts[128:257]
                    if np.any(c < 0):
                        continue
                    p = int(np.max(c))
                    s = int(np.sum(c.astype(np.int64)**2))
                    cov = bool(np.all(c > 0))
                    if is_better(p, s, cov, curr_peak, curr_sq):
                        A = set(A)
                        A.discard(x1)
                        A.discard(x2)
                        A.add(y1)
                        A.add(y2)
                        counts = new_counts
                        curr_peak, curr_sq, curr_covered = p, s, cov
                        if curr_peak < best_peak or (curr_peak == best_peak and curr_sq < best_sq):
                            best_peak = curr_peak
                            best_sq = curr_sq
                            best_A = set(A)
                        improved = True
                        no_improve_count = 0
                        A_list = list(A)
                        C_list = [i for i in range(n + 1) if i not in A]
                        break
                        
        if not improved:
            no_improve_count += 1
            
            # Perturbation: random 3-swap to escape local minimum
            if no_improve_count > max_no_improve:
                found_valid = False
                A_list = list(A)
                C_list = [i for i in range(n + 1) if i not in A]
                if len(A_list) >= 3 and len(C_list) >= 3:
                    for _ in range(30):
                        if time.time() - start_time > time_limit - 1:
                            break
                        x1, x2, x3 = random.sample(A_list, 3)
                        y1, y2, y3 = random.sample(C_list, 3)
                        A_trial = set(A)
                        A_trial.discard(x1)
                        A_trial.discard(x2)
                        A_trial.discard(x3)
                        A_trial.add(y1)
                        A_trial.add(y2)
                        A_trial.add(y3)
                        
                        counts_trial = compute_counts(A_trial)
                        p, s, cov = get_metrics(counts_trial)
                        if cov:
                            A = A_trial
                            counts = counts_trial
                            curr_peak, curr_sq, curr_covered = p, s, cov
                            found_valid = True
                            break
                if found_valid:
                    no_improve_count = 0
                    A_list = list(A)
                    C_list = [i for i in range(n + 1) if i not in A]
                else:
                    A = set(best_A)
                    counts = compute_counts(A)
                    curr_peak, curr_sq, curr_covered = best_peak, best_sq, True
                    no_improve_count = 0
                    A_list = list(A)
                    C_list = [i for i in range(n + 1) if i not in A]

    return sorted(list(best_A))
