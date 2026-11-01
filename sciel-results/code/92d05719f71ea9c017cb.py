import time
import numpy as np
import random

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 110.0

    total = q ** n
    ball_size = 1 + n * (q - 1)

    # Build index arithmetic
    base = [1] * n
    for i in range(n - 2, -1, -1):
        base[i] = base[i + 1] * q

    # words: 2D array of all words
    words = np.zeros((total, n), dtype=np.int32)
    for i in range(total):
        temp = i
        for j in range(n - 1, -1, -1):
            words[i, j] = temp % q
            temp //= q

    # ball_indices[i]: list of word indices within Hamming distance 1 of word i
    ball_indices = [[] for _ in range(total)]
    for i in range(total):
        w = words[i]
        ball_indices[i].append(i)
        for coord in range(n):
            orig = w[coord]
            for v in range(q):
                if v == orig:
                    continue
                new_idx = i + (v - orig) * base[coord]
                ball_indices[i].append(new_idx)

    ball_arrays = [np.array(bi, dtype=np.int32) for bi in ball_indices]

    # --- Coset structure via [7,4,3] Hamming parity-check matrix ---
    # For q=3, n=7: the Hamming code over GF(3) has parity-check matrix H (3x7)
    # such that each column is a distinct nonzero vector in GF(3)^3 up to scalar.
    # Actually for ternary Hamming code: columns are representatives of 1-dim subspaces of GF(3)^3
    # There are (3^3-1)/(3-1) = 13 such subspaces, but we need 7 columns.
    # Let me use a standard [7,4,3]_3 Hamming code parity check matrix.
    # H = [[1,0,0,1,1,1,2],
    #      [0,1,0,1,2,1,1],
    #      [0,0,1,1,1,2,1]]
    # Each column is nonzero and no two columns are scalar multiples (in GF(3)).
    
    if q == 3 and n == 7:
        H = np.array([
            [1, 0, 0, 1, 1, 1, 2],
            [0, 1, 0, 1, 2, 1, 1],
            [0, 0, 1, 1, 1, 2, 1]
        ], dtype=np.int8)
        
        # Compute syndrome for each word: H @ w mod 3
        # syndrome is in {0,1,2}^3, so 27 cosets
        syndromes = np.zeros(total, dtype=np.int32)
        for i in range(total):
            s = np.dot(H, words[i]) % 3
            syndromes[i] = s[0] * 9 + s[1] * 3 + s[2]
        
        # Group words by syndrome
        cosets = {}
        for i in range(total):
            s = syndromes[i]
            if s not in cosets:
                cosets[s] = []
            cosets[s].append(i)
        
        # Number of cosets
        num_cosets = len(cosets)
        
        # For constraint propagation: if we pick a codeword from coset c,
        # it covers words in its ball. The ball of a word w in coset c contains:
        # w itself (coset c), and for each coordinate and each value change,
        # the resulting word is in a different coset (specifically, coset c + h_j * delta)
        # where h_j is column j of H and delta is the value change.
        
        # Precompute: for each word i, which coset each ball member belongs to
        # ball_coset_members[i][coset] = list of words in ball(i) that are in that coset
        # Actually simpler: for each word i, compute the set of (coset, count) pairs in its ball
        ball_coset_info = []
        for i in range(total):
            coset_counts = {}
            for p in ball_indices[i]:
                s = syndromes[p]
                coset_counts[s] = coset_counts.get(s, 0) + 1
            ball_coset_info.append(coset_counts)
        
        # For the constraint propagation approach:
        # Key insight: within a single coset, words are at distance >= 3 from each other
        # (since the code has minimum distance 3). So no two words in the same coset
        # are within distance 1. This means each codeword's ball intersects its own coset
        # in exactly 1 word (itself).
        
        # Strategy: Use backtracking with constraint propagation on cosets.
        # We process cosets one at a time. For each coset, we decide which words to include.
        # Constraint: every word must be covered.
        
        # However, a full exact cover via backtracking over 27 cosets with ~81 words each
        # is still large. Let's use a hybrid: greedy initialization + structured local search.
    
    # --- Greedy cover (fast initialization) ---
    def greedy_cover(order=None):
        if order is None:
            order = list(range(total))
        covered = np.zeros(total, dtype=bool)
        code = []
        remaining = np.ones(total, dtype=bool)
        
        while np.any(remaining):
            best_c = -1
            best_new = -1
            for c in order:
                ball = ball_arrays[c]
                new_cov = np.count_nonzero(remaining[ball])
                if new_cov > best_new:
                    best_new = new_cov
                    best_c = c
                    if best_new == ball_size:
                        break
            code.append(best_c)
            remaining[ball_arrays[best_c]] = False
        return code

    def delete_redundant(code_set, cnt):
        changed = False
        for c in list(code_set):
            ball = ball_arrays[c]
            if np.all(cnt[ball] >= 2):
                code_set.remove(c)
                cnt[ball] -= 1
                changed = True
        return changed

    # --- Initial greedy solutions ---
    best_code = None
    best_size = total + 1

    for trial in range(4):
        if time.time() > deadline - 20:
            break
        order = list(range(total))
        random.shuffle(order)
        code = greedy_cover(order)
        code_set = set(code)
        cnt = np.zeros(total, dtype=np.int32)
        for c in code_set:
            cnt[ball_arrays[c]] += 1
        while delete_redundant(code_set, cnt):
            pass
        if len(code_set) < best_size:
            best_size = len(code_set)
            best_code = list(code_set)
            best_cnt = cnt.copy()

    if best_code is None:
        best_code = list(range(0, total, max(1, total // 186)))
        best_size = len(best_code)
        best_cnt = np.zeros(total, dtype=np.int32)
        for c in best_code:
            best_cnt[ball_arrays[c]] += 1

    # --- Structured local search with coset-aware moves ---
    current_code = set(best_code)
    cnt = np.zeros(total, dtype=np.int32)
    for c in current_code:
        cnt[ball_arrays[c]] += 1
    best_size = len(current_code)
    best_code = list(current_code)

    # Precompute overlap matrix
    overlap = np.zeros((total, total), dtype=np.int8)
    balls_containing = [[] for _ in range(total)]
    for i in range(total):
        for p in ball_indices[i]:
            balls_containing[p].append(i)
    for p in range(total):
        bc = balls_containing[p]
        for a in range(len(bc)):
            for b in range(a + 1, len(bc)):
                i = bc[a]
                j = bc[b]
                overlap[i, j] += 1
                overlap[j, i] += 1

    # Coset membership for each word
    word_coset = np.zeros(total, dtype=np.int32)
    if q == 3 and n == 7:
        word_coset = syndromes.astype(np.int32)
    else:
        word_coset[:] = 0

    # --- Backtracking refinement: try to replace a subset of codewords with fewer ---
    # For each codeword, try to find a replacement that covers its "unique" points
    # and also helps cover points of another codeword, enabling deletion.
    
    # Build: for each codeword c, the set of points only covered by c (cnt==1 and in ball(c))
    # These are the "critical" points for c.
    
    no_improve = 0
    max_no_improve = 500

    while time.time() < deadline - 5.0 and no_improve < max_no_improve:
        improved = False
        code_list = list(current_code)
        
        # Phase 1: Deletion
        while delete_redundant(current_code, cnt):
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code = list(current_code)
            improved = True

        if not improved:
            no_improve += 1
            continue
        no_improve = 0
        code_list = list(current_code)

        # Phase 2: Coset-guided swaps
        # Group current codewords by coset
        coset_groups = {}
        for c in code_list:
            cs = word_coset[c]
            if cs not in coset_groups:
                coset_groups[cs] = []
            coset_groups[cs].append(c)

        # Try: for each pair of codewords in the same coset, try replacing both with one
        # (since words in same coset are far apart, their balls overlap little)
        # Or: try replacing one codeword with one from a nearby coset to enable a deletion.
        
        swaps_attempted = 0
        max_swaps = 200
        
        while swaps_attempted < max_swaps and time.time() < deadline - 5.0:
            swaps_attempted += 1
            if len(current_code) < 2:
                break
            
            # Random swap
            c_remove = random.choice(code_list)
            if c_remove not in current_code:
                continue
            c_add = random.randint(0, total - 1)
            if c_add == c_remove or c_add in current_code:
                continue

            ball_r = ball_arrays[c_remove]
            ball_a = ball_arrays[c_add]
            ov = overlap[c_remove, c_add]
            
            if ov >= ball_size:
                current_code.remove(c_remove)
                current_code.add(c_add)
                cnt[ball_r] -= 1
                cnt[ball_a] += 1
                improved = True
            else:
                mask_in_a = np.isin(ball_r, ball_a, assume_unique=False)
                points_to_check = ball_r[~mask_in_a]
                if len(points_to_check) > 0 and np.any(cnt[points_to_check] < 2):
                    continue
                current_code.remove(c_remove)
                current_code.add(c_add)
                cnt[ball_r] -= 1
                cnt[ball_a] += 1
                improved = True

            while delete_redundant(current_code, cnt):
                if len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code = list(current_code)
                improved = True

        # Phase 3: 2-for-1 replacement using coset structure
        # If two codewords are in the same coset, their balls are "aligned" in a useful way.
        # Try to find a single word whose ball covers the union of their critical points.
        two_for_one_attempts = 0
        max_2f1 = 80
        while two_for_one_attempts < max_2f1 and time.time() < deadline - 5.0:
            two_for_one_attempts += 1
            if len(current_code) < 2:
                break

            # Pick two codewords, preferably from the same coset
            if len(coset_groups) > 1:
                # Try same-coset pairs first
                same_coset_pairs = []
                for cs, members in coset_groups.items():
                    if len(members) >= 2:
                        same_coset_pairs.extend([(a, b) for a, b in zip(members[:-1], members[1:])])
                if same_coset_pairs and random.random() < 0.6:
                    c1, c2 = random.choice(same_coset_pairs)
                else:
                    c1, c2 = random.sample(list(current_code), 2)
            else:
                c1, c2 = random.sample(list(current_code), 2)

            if c1 not in current_code or c2 not in current_code:
                continue

            ball1 = ball_arrays[c1]
            ball2 = ball_arrays[c2]

            # Find candidate: a word whose ball covers all points in ball1∪ball2 with cnt<=2
            # that are not in ball_a
            for _ in range(3):
                c_add = random.randint(0, total - 1)
                if c_add in current_code:
                    continue
                ball_a = ball_arrays[c_add]

                mask1 = np.isin(ball1, ball_a, assume_unique=False)
                mask2 = np.isin(ball2, ball_a, assume_unique=False)
                b1_not_a = ball1[~mask1]
                b2_not_a = ball2[~mask2]

                valid = True
                if len(b1_not_a) > 0:
                    in_b2 = np.isin(b1_not_a, ball2, assume_unique=False)
                    if np.any(in_b2) and np.any(cnt[b1_not_a[in_b2]] < 3):
                        valid = False
                    if valid and np.any(~in_b2) and np.any(cnt[b1_not_a[~in_b2]] < 2):
                        valid = False
                if valid and len(b2_not_a) > 0:
                    in_b1 = np.isin(b2_not_a, ball1, assume_unique=False)
                    if np.any(in_b1) and np.any(cnt[b2_not_a[in_b1]] < 3):
                        valid = False
                    if valid and np.any(~in_b1) and np.any(cnt[b2_not_a[~in_b1]] < 2):
                        valid = False

                if not valid:
                    continue

                current_code.remove(c1)
                current_code.remove(c2)
                current_code.add(c_add)
                cnt[ball1] -= 1
                cnt[ball2] -= 1
                cnt[ball_a] += 1
                improved = True
                while delete_redundant(current_code, cnt):
                    if len(current_code) < best_size:
                        best_size = len(current_code)
                        best_code = list(current_code)
                    improved = True
                break

        if not improved:
            no_improve += 1
        else:
            no_improve = 0
            # Rebuild coset groups
            coset_groups = {}
            for c in current_code:
                cs = word_coset[c]
                if cs not in coset_groups:
                    coset_groups[cs] = []
                coset_groups[cs].append(c)

    # Final deletion
    while delete_redundant(current_code, cnt):
        if len(current_code) < best_size:
            best_size = len(current_code)
            best_code = list(current_code)

    # Convert to list of lists
    result = []
    for idx in best_code:
        w = []
        temp = idx
        for i in range(n):
            w.append(temp % q)
            temp //= q
        w.reverse()
        result.append(w)
    return result
