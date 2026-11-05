import math
import time

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin

    N = 511
    weights = [0.0] * (N + 1)
    for a in range(1, N + 1):
        weights[a] = 1.0 / (512 - a)

    # Precompute conflicts
    conflicts = [set() for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            if math.gcd(a, b) > 1:
                conflicts[a].add(b)
                conflicts[b].add(a)

    # Greedy Initialization
    S = set()
    score = 0.0
    for a in range(N, 0, -1):
        if conflicts[a].isdisjoint(S):
            S.add(a)
            score += weights[a]

    best_S = set(S)
    best_score = score

    def is_compatible(u, S):
        return conflicts[u].isdisjoint(S)

    def local_search(S, score):
        nonlocal best_S, best_score, deadline, start_time
        
        while time.time() < deadline:
            improved = False
            
            # 1. Try Add
            # We want to add u if it's compatible and increases score (always true if u>0)
            best_add = -1
            best_add_gain = 0.0
            for u in range(1, N + 1):
                if u not in S and is_compatible(u, S):
                    gain = weights[u]
                    if gain > best_add_gain:
                        best_add_gain = gain
                        best_add = u
            
            if best_add != -1:
                S.add(best_add)
                score += best_add_gain
                improved = True
                if score > best_score:
                    best_score = score
                    best_S = set(S)
                continue # Restart loop after modification

            # 2. Try Swap 1-1
            # Replace v in S with u not in S
            best_swap = None
            best_swap_gain = 0.0
            for v in list(S):
                S_minus_v = S - {v}
                for u in range(1, N + 1):
                    if u not in S_minus_v and is_compatible(u, S_minus_v):
                        gain = weights[u] - weights[v]
                        if gain > best_swap_gain:
                            best_swap_gain = gain
                            best_swap = (v, u)
            
            if best_swap:
                v, u = best_swap
                S.remove(v)
                S.add(u)
                score += best_swap_gain
                improved = True
                if score > best_score:
                    best_score = score
                    best_S = set(S)
                continue

            # 3. Try Swap 1-2
            # Replace v in S with u1, u2 not in S
            # u1, u2 must be compatible with S\{v} and with each other
            best_swap_1_2 = None
            best_swap_1_2_gain = 0.0
            
            for v in list(S):
                S_minus_v = S - {v}
                # Find candidates u compatible with S_minus_v
                candidates = [u for u in range(1, N + 1) if u not in S_minus_v and is_compatible(u, S_minus_v)]
                # Sort by weight descending to check best pairs first
                candidates.sort(key=lambda x: -weights[x])
                
                # Check pairs
                # Limit check to top K candidates to save time, but N is small so full check is ok if optimized
                # However, O(M^2) where M is num candidates.
                # If M is large, this is slow.
                # But we only need the best pair.
                # We can break early if weights[candidates[0]] + weights[candidates[1]] <= weights[v]
                if len(candidates) < 2:
                    continue
                
                # Quick upper bound
                if weights[candidates[0]] + weights[candidates[1]] <= weights[v]:
                    continue

                # Check pairs from the top
                # We can just iterate all pairs in candidates
                # To be safe and fast, we can just check the top 50 or so?
                # No, let's check all, N=511 is small.
                for i in range(len(candidates)):
                    u1 = candidates[i]
                    for j in range(i + 1, len(candidates)):
                        u2 = candidates[j]
                        if math.gcd(u1, u2) == 1:
                            gain = weights[u1] + weights[u2] - weights[v]
                            if gain > best_swap_1_2_gain:
                                best_swap_1_2_gain = gain
                                best_swap_1_2 = (v, u1, u2)
            
            if best_swap_1_2:
                v, u1, u2 = best_swap_1_2
                S.remove(v)
                S.add(u1)
                S.add(u2)
                score += best_swap_1_2_gain
                improved = True
                if score > best_score:
                    best_score = score
                    best_S = set(S)
                continue

            # 4. Try Swap 2-1
            # Replace v1, v2 in S with u not in S
            best_swap_2_1 = None
            best_swap_2_1_gain = 0.0
            
            # This is O(|S|^2 * N). |S| is small (~50-100). N=511.
            # 100^2 * 511 ~ 5 million operations. Might be slow in Python if done every iteration.
            # We can optimize by only considering u that are "blocked" by v1 or v2.
            
            for i in range(len(list(S))):
                v1 = list(S)[i]
                for j in range(i + 1, len(list(S))):
                    v2 = list(S)[j]
                    S_minus_v1_v2 = S - {v1, v2}
                    
                    # Candidates u must be compatible with S_minus_v1_v2
                    # And u must be in conflicts[v1] or conflicts[v2] (otherwise it would have been added in "Add" step)
                    # Actually, if u was compatible with S, it would have been added.
                    # So u must be incompatible with S.
                    # Since u is compatible with S\{v1,v2}, it must be incompatible with v1 or v2.
                    
                    # We can iterate through all u not in S
                    for u in range(1, N + 1):
                        if u in S:
                            continue
                        if is_compatible(u, S_minus_v1_v2):
                            gain = weights[u] - weights[v1] - weights[v2]
                            if gain > best_swap_2_1_gain:
                                best_swap_2_1_gain = gain
                                best_swap_2_1 = (v1, v2, u)
            
            if best_swap_2_1:
                v1, v2, u = best_swap_2_1
                S.remove(v1)
                S.remove(v2)
                S.add(u)
                score += best_swap_2_1_gain
                improved = True
                if score > best_score:
                    best_score = score
                    best_S = set(S)
                continue

            if not improved:
                break

        return S, score

    S, score = local_search(S, score)
    
    return list(best_S)
