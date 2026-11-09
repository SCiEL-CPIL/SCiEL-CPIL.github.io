import time
import random

def construct():
    N = 1000
    max_sum = 2 * N
    deadline = time.time() + 140

    # Precompute for each x: the sums involved with x (2x and x+y for y=1..N, y!=x)
    # sums_with[x] = list of sums s such that there exists y in 1..N with x+y = s (y != x) or 2x=s
    # We'll use: for a given set S, the count of representations of sum s is the number of pairs (a,b) in S with a<=b and a+b=s.
    
    def build_sum_map(S):
        """Build a dict: sum -> list of (a,b) pairs in S."""
        sp = {}
        for i, a in enumerate(S):
            s = 2 * a
            sp[s] = sp.get(s, []) + [(a, a)]
            for b in S[i+1:]:
                s = a + b
                sp[s] = sp.get(s, []) + [(a, b)]
        return sp

    def validate_set(S):
        """Check if S satisfies B2[2] condition."""
        if not S:
            return True
        counts = [0] * (max_sum + 1)
        for i, a in enumerate(S):
            counts[2 * a] += 1
            for b in S[i+1:]:
                counts[a + b] += 1
        return all(c <= 2 for c in counts)

    def count_reps(S):
        """Return the counts array for set S."""
        counts = [0] * (max_sum + 1)
        for i, a in enumerate(S):
            counts[2 * a] += 1
            for b in S[i+1:]:
                counts[a + b] += 1
        return counts

    def can_add(S, counts, x):
        """Check if x can be added to S without violating B2[2]."""
        if x in S_set_global:
            return False
        # Check 2x
        if counts[2 * x] >= 2:
            return False
        # Check x + y for y in S
        for y in S:
            if counts[x + y] >= 2:
                return False
        return True

    def do_add(S, counts, x):
        """Add x to S and update counts."""
        S.append(x)
        counts[2 * x] += 1
        for y in S:
            if y != x:
                counts[x + y] += 1

    def do_remove(S, counts, x):
        """Remove x from S and update counts."""
        S.remove(x)
        counts[2 * x] -= 1
        for y in S:
            if y != x:
                counts[x + y] -= 1

    S_set_global = set()  # For fast lookup

    # --- Heuristic 1: Greedy ascending ---
    def greedy_ascending():
        S = []
        S_set = set()
        counts = [0] * (max_sum + 1)
        for x in range(1, N + 1):
            if x in S_set:
                continue
            if counts[2 * x] >= 2:
                continue
            valid = True
            for y in S:
                if counts[x + y] >= 2:
                    valid = False
                    break
            if valid:
                S.append(x)
                S_set.add(x)
                counts[2 * x] += 1
                for y in S:
                    if y != x:
                        counts[x + y] += 1
        return S

    # --- Heuristic 2: Randomized greedy ---
    def greedy_random(seed):
        rng = random.Random(seed)
        S = []
        S_set = set()
        counts = [0] * (max_sum + 1)
        candidates = list(range(1, N + 1))
        rng.shuffle(candidates)
        for x in candidates:
            if x in S_set:
                continue
            if counts[2 * x] >= 2:
                continue
            valid = True
            for y in S:
                if counts[x + y] >= 2:
                    valid = False
                    break
            if valid:
                S.append(x)
                S_set.add(x)
                counts[2 * x] += 1
                for y in S:
                    if y != x:
                        counts[x + y] += 1
        return S

    # --- Heuristic 3: Local search with targeted swaps ---
    def local_search(S_init, time_limit, seed=42):
        rng = random.Random(seed)
        S = S_init[:]
        S_set = set(S)
        counts = count_reps(S)
        
        # Build sum_pairs for blocker identification
        def build_sum_pairs():
            sp = {}
            for i, a in enumerate(S):
                sp[2 * a] = sp.get(2 * a, []) + [(a, a)]
                for j in range(i + 1, len(S)):
                    b = S[j]
                    s = a + b
                    sp[s] = sp.get(s, []) + [(a, b)]
            return sp

        def find_blockers_for_candidate(c):
            """Find elements in S that block candidate c."""
            blockers = set()
            
            # Check 2*c
            if counts[2 * c] >= 2:
                for a, b in sum_pairs.get(2 * c, []):
                    blockers.add(a)
                    blockers.add(b)
            
            # Check c + y for y in S
            for y in S:
                s = c + y
                if counts[s] >= 2:
                    for a, b in sum_pairs.get(s, []):
                        blockers.add(a)
                        blockers.add(b)
            
            return [b for b in blockers if b in S_set]

        sum_pairs = build_sum_pairs()
        candidates_not_in_S = [x for x in range(1, N + 1) if x not in S_set]
        rng.shuffle(candidates_not_in_S)
        
        best_size_in_search = len(S)
        best_S_snapshot = S[:]
        
        while time.time() < time_limit and candidates_not_in_S:
            idx = rng.randint(0, len(candidates_not_in_S) - 1)
            c = candidates_not_in_S[idx]
            
            if c in S_set:
                candidates_not_in_S.pop(idx)
                continue
            
            # Try direct add
            if counts[2 * c] < 2 and all(counts[c + y] < 2 for y in S):
                # Add c
                S.append(c)
                S_set.add(c)
                counts[2 * c] += 1
                for y in S:
                    if y != c:
                        counts[c + y] += 1
                sum_pairs = build_sum_pairs()
                candidates_not_in_S.pop(idx)
                if len(S) > best_size_in_search:
                    best_size_in_search = len(S)
                    best_S_snapshot = S[:]
                continue
            
            # Blocked. Find blockers.
            blockers = find_blockers_for_candidate(c)
            
            if not blockers:
                candidates_not_in_S.pop(idx)
                continue
            
            # Try 1-for-1 swaps
            found_swap = False
            for b in blockers:
                # Remove b
                S.remove(b)
                S_set.discard(b)
                counts[2 * b] -= 1
                for y in S:
                    if y != b:
                        counts[b + y] -= 1
                
                # Try adding c
                can_add_c = counts[2 * c] < 2 and all(counts[c + y] < 2 for y in S)
                
                if can_add_c:
                    S.append(c)
                    S_set.add(c)
                    counts[2 * c] += 1
                    for y in S:
                        if y != c:
                            counts[c + y] += 1
                    sum_pairs = build_sum_pairs()
                    candidates_not_in_S.pop(idx)
                    found_swap = True
                    if len(S) > best_size_in_search:
                        best_size_in_search = len(S)
                        best_S_snapshot = S[:]
                    break
                else:
                    # Restore b
                    S.append(b)
                    S_set.add(b)
                    counts[2 * b] += 1
                    for y in S:
                        if y != b:
                            counts[b + y] += 1
            
            if not found_swap and len(blockers) >= 2:
                # Try 2-for-1 moves
                tried_pairs = set()
                for i in range(len(blockers)):
                    for j in range(i + 1, len(blockers)):
                        b1, b2 = blockers[i], blockers[j]
                        pair_key = (min(b1, b2), max(b1, b2))
                        if pair_key in tried_pairs:
                            continue
                        tried_pairs.add(pair_key)
                        
                        # Remove b1 and b2
                        S.remove(b1)
                        S_set.discard(b1)
                        counts[2 * b1] -= 1
                        for y in S:
                            if y != b1:
                                counts[b1 + y] -= 1
                        
                        S.remove(b2)
                        S_set.discard(b2)
                        counts[2 * b2] -= 1
                        for y in S:
                            if y != b2:
                                counts[b2 + y] -= 1
                        
                        # Try adding c
                        can_add_c = counts[2 * c] < 2 and all(counts[c + y] < 2 for y in S)
                        
                        if can_add_c:
                            S.append(c)
                            S_set.add(c)
                            counts[2 * c] += 1
                            for y in S:
                                if y != c:
                                    counts[c + y] += 1
                            sum_pairs = build_sum_pairs()
                            candidates_not_in_S.pop(idx)
                            found_swap = True
                            if len(S) > best_size_in_search:
                                best_size_in_search = len(S)
                                best_S_snapshot = S[:]
                            break
                        else:
                            # Restore b1 and b2
                            S.append(b1)
                            S_set.add(b1)
                            counts[2 * b1] += 1
                            for y in S:
                                if y != b1:
                                    counts[b1 + y] += 1
                            
                            S.append(b2)
                            S_set.add(b2)
                            counts[2 * b2] += 1
                            for y in S:
                                if y != b2:
                                    counts[b2 + y] += 1
                        
                        if found_swap:
                            break
                    if found_swap:
                        break

        return best_S_snapshot

    # --- Main ---
    S_det = greedy_ascending()
    best_S = S_det[:]
    best_size = len(S_det)
    
    t_start = time.time()
    n_trials = 8
    for trial in range(n_trials):
        if time.time() > deadline - 10:
            break
        
        seed = trial * 1000 + 7
        S_rand = greedy_random(seed)
        
        if len(S_rand) > best_size:
            best_S = S_rand[:]
            best_size = len(S_rand)
        
        remaining = deadline - time.time() - 5
        if remaining < 5:
            continue
        
        ls_time = min(15, remaining)
        S_ls = local_search(S_rand, time.time() + ls_time, seed=seed)
        
        # Final polish: try to add remaining candidates
        S_final = S_ls[:]
        S_set_final = set(S_final)
        counts_final = count_reps(S_final)
        for x in range(1, N + 1):
            if x not in S_set_final:
                if counts_final[2 * x] < 2 and all(counts_final[x + y] < 2 for y in S_final):
                    S_final.append(x)
                    S_set_final.add(x)
                    counts_final[2 * x] += 1
                    for y in S_final:
                        if y != x:
                            counts_final[x + y] += 1
        
        if len(S_final) > best_size:
            best_size = len(S_final)
            best_S = S_final[:]
    
    # Final validation
    if not validate_set(best_S):
        best_S = S_det[:]
    
    return best_S
