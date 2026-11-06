import numpy as np
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    N = 256
    
    # Precompute APs
    ap_list = []
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            ap_list.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
            
    aps = np.array(ap_list, dtype=np.int32)
    num_aps = len(aps)
    
    # Precompute for each index, the list of AP indices containing it
    aps_at = [[] for _ in range(N)]
    for idx in range(num_aps):
        for pos in aps[idx]:
            aps_at[pos].append(idx)
            
    # Precompute "others" for each index in each AP it belongs to
    others_for_pos = [None] * N
    for i in range(N):
        if not aps_at[i]:
            others_for_pos[i] = np.array([], dtype=np.int32).reshape(0, 3)
        else:
            others_list = []
            for ap_idx in aps_at[i]:
                row = aps[ap_idx]
                others = [p for p in row if p != i]
                others_list.append(others)
            others_for_pos[i] = np.array(others_list, dtype=np.int32)
    
    # Precompute flat arrays for fast delta
    others_flat = [None] * N
    others_ap_counts = [None] * N
    for i in range(N):
        others = others_for_pos[i]
        others_flat[i] = others.flatten()
        others_ap_counts[i] = others.shape[0]
        
    # Precompute shared APs for pairs (i, j)
    # shared_aps[i][j] = list of AP indices containing both i and j
    # To save memory, we can just compute on the fly or store a dictionary
    # Given N=256, N^2 is 65k. Storing lists is fine.
    shared_aps = [[[] for _ in range(N)] for _ in range(N)]
    for ap_idx in range(num_aps):
        row = aps[ap_idx]
        for k in range(4):
            for l in range(k+1, 4):
                i = row[k]
                j = row[l]
                shared_aps[i][j].append(ap_idx)
                shared_aps[j][i].append(ap_idx)

    def count_mono(col):
        c = col[aps]
        return int(np.sum((c[:, 0] == c[:, 1]) & (c[:, 1] == c[:, 2]) & (c[:, 2] == c[:, 3])))

    def fast_delta(col, i):
        n_aps_i = others_ap_counts[i]
        if n_aps_i == 0:
            return 0
        flat = others_flat[i]
        oc = col[flat]  # shape (3 * n_aps_i,)
        c_i = col[i]
        # Reshape to (n_aps_i, 3)
        oc2d = oc.reshape(n_aps_i, 3)
        # Check if all 3 are same
        all_same = (oc2d[:, 0] == oc2d[:, 1]) & (oc2d[:, 1] == oc2d[:, 2])
        if not np.any(all_same):
            return 0
        # For those where all same, check if they match c_i
        same_color = oc2d[:, 0]
        mask = all_same
        n_total = int(np.sum(mask))
        n_match = int(np.sum(same_color[mask] == c_i))
        n_mismatch = n_total - n_match
        return n_mismatch - n_match

    def delta_2flip(col, i, j):
        # Delta for flipping i and j simultaneously
        # Delta = Delta_1(i) + Delta_1(j) + Interaction(i, j)
        d_i = fast_delta(col, i)
        d_j = fast_delta(col, j)
        
        # Interaction term
        # Only APs containing both i and j matter for the interaction correction
        # Because Delta_1(i) assumes j is fixed, and Delta_1(j) assumes i is fixed.
        # If we flip both, the state of APs containing both changes differently.
        
        # Let's derive the interaction precisely.
        # Let S be the set of APs containing both i and j.
        # For any AP in S:
        # Let O be the other 2 elements.
        # c_i, c_j, c_o1, c_o2.
        # Before: Mono if c_i==c_j==c_o1==c_o2.
        # After: c'_i=1-c_i, c'_j=1-c_j. Mono if c'_i==c'_j==c_o1==c_o2.
        
        # Case 1: c_i != c_j.
        # Before: Not Mono (since c_i != c_j).
        # After: c'_i = 1-c_i, c'_j = 1-c_j. Since c_i != c_j, c'_i != c'_j. Not Mono.
        # Delta = 0.
        
        # Case 2: c_i == c_j.
        # Let c = c_i = c_j.
        # Before: Mono if c_o1==c and c_o2==c.
        # After: c'_i = 1-c, c'_j = 1-c. Mono if c_o1==1-c and c_o2==1-c.
        
        # If c_o1==c and c_o2==c:
        #   Before=1, After=0 (since c != 1-c). Delta = -1.
        # If c_o1==1-c and c_o2==1-c:
        #   Before=0, After=1. Delta = +1.
        # Else:
        #   Before=0, After=0. Delta = 0.
        
        inter_delta = 0
        if col[i] == col[j]:
            c = col[i]
            target = 1 - c
            for ap_idx in shared_aps[i][j]:
                row = aps[ap_idx]
                # Get the other 2 elements
                # row has 4 elements. i and j are in it.
                # We need to find the other two.
                # Since i, j are indices in row, we can just filter.
                # But row is small (4).
                others = [p for p in row if p != i and p != j]
                o1, o2 = others
                if col[o1] == c and col[o2] == c:
                    inter_delta -= 1
                elif col[o1] == target and col[o2] == target:
                    inter_delta += 1
        return d_i + d_j + inter_delta

    def local_search(col, max_time, seed):
        rng = random.Random(seed)
        current_score = count_mono(col)
        best_score = current_score
        best_col = col.copy()
        
        order = list(range(N))
        rng.shuffle(order)
        
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for i in order:
                if time.time() > deadline - 5.0:
                    break
                delta = fast_delta(col, i)
                if delta < 0:
                    col[i] = 1 - col[i]
                    current_score += delta
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_col = col.copy()
        return best_col, best_score

    def local_search_2flip(col, max_time, seed, max_pairs_per_pass=500):
        rng = random.Random(seed)
        current_score = count_mono(col)
        best_score = current_score
        best_col = col.copy()
        
        improved = True
        passes = 0
        while improved and time.time() < max_time:
            improved = False
            passes += 1
            # Try a subset of random pairs
            for _ in range(max_pairs_per_pass):
                if time.time() > max_time:
                    break
                i = rng.randint(0, N - 1)
                j = rng.randint(0, N - 1)
                if i == j:
                    continue
                # Quick check: if either i or j has no shared APs and single flip doesn't help, 
                # 2-flip is sum of singles. But we compute full delta.
                delta = delta_2flip(col, i, j)
                if delta < 0:
                    col[i] = 1 - col[i]
                    col[j] = 1 - col[j]
                    current_score += delta
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_col = col.copy()
        return best_col, best_score

    def sa_run(col_init, seed, max_steps, start_temp, end_temp):
        rng = random.Random(seed)
        
        col = col_init.copy()
        current_score = count_mono(col)
        best_score = current_score
        best_col = col.copy()
        
        for step in range(max_steps):
            if time.time() > deadline - 5.0:
                break
            t = start_temp * (end_temp / start_temp) ** (step / max_steps)
            if t < 0.001:
                t = 0.001
            
            i = rng.randint(0, N - 1)
            delta = fast_delta(col, i)
            
            if delta <= 0:
                col[i] = 1 - col[i]
                current_score += delta
                if current_score < best_score:
                    best_score = current_score
                    best_col = col.copy()
            else:
                if t > 0 and rng.random() < np.exp(-delta / t):
                    col[i] = 1 - col[i]
                    current_score += delta
        
        return best_col, best_score

    best_overall = None
    best_overall_score = float('inf')
    
    # 1. Structured starts
    structured = [
        np.array([ (i // 2) % 2 for i in range(N) ], dtype=np.int32),
        np.array([ (i // 4) % 2 for i in range(N) ], dtype=np.int32),
        np.array([ bin(i).count('1') % 2 for i in range(N) ], dtype=np.int32),  # Thue-Morse
        np.array([ (i % 3) % 2 for i in range(N) ], dtype=np.int32),
        np.array([ (i % 5) % 2 for i in range(N) ], dtype=np.int32),
        np.array([ bin(i).count('1') % 4 for i in range(N) ], dtype=np.int32) % 2,
        np.array([ ((i // 2) % 2) for i in range(N) ], dtype=np.int32),
        np.array([ ((i // 3) % 2) for i in range(N) ], dtype=np.int32),
        np.array([ (i % 7) % 2 for i in range(N) ], dtype=np.int32),
        np.array([ (i % 11) % 2 for i in range(N) ], dtype=np.int32),
        np.random.RandomState(42).randint(0, 2, size=N).astype(np.int32),
    ]
    
    for pat in structured:
        s = count_mono(pat)
        if s < best_overall_score:
            best_overall_score = s
            best_overall = pat.copy()
            
    # 2. SA + Local Search with restarts
    # Run 1: SA on best structured
    if time.time() < deadline - 30.0:
        b_col, b_score = sa_run(best_overall, 123, 50000, 2.0, 0.01)
        b_col, b_score = local_search(b_col, deadline - 10.0, 123)
        if b_score < best_overall_score:
            best_overall_score = b_score
            best_overall = b_col
    
    # Random restarts with SA + Local Search
    for r in range(2, 10):
        if time.time() > deadline - 20.0:
            break
        seed = 100 + r * 17
        init_col = np.random.RandomState(seed).randint(0, 2, size=N).astype(np.int32)
        b_col, b_score = sa_run(init_col, seed, 20000, 1.5, 0.005)
        b_col, b_score = local_search(b_col, deadline - 10.0, seed)
        
        if b_score < best_overall_score:
            best_overall_score = b_score
            best_overall = b_col.copy()
            
    # 3. 2-Flip Polish
    if best_overall is not None and time.time() < deadline - 2.0:
        # Use remaining time for 2-flip search
        remaining = deadline - time.time() - 1.0
        if remaining > 1.0:
            b_col, b_score = local_search_2flip(best_overall, start_time + remaining, 777, max_pairs_per_pass=1000)
            if b_score < best_overall_score:
                best_overall_score = b_score
                best_overall = b_col

    return [int(c) for c in best_overall]
