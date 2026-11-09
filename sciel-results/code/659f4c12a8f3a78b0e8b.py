import time
import random
import numpy as np

N = 64
M = N * N

def construct():
    start_time = time.time()
    deadline = start_time + 290

    points = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int32)
    
    try:
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        D = np.sum(diff**2, axis=2).astype(np.int32)
    except MemoryError:
        D = None

    def can_add(p, S_list):
        if not S_list:
            return True, []
            
        s_indices = np.array(S_list)
        
        if D is not None:
            d_p_S = D[p, s_indices]
            d_S_S = D[np.ix_(s_indices, s_indices)]
        else:
            diff_p = points[p] - points[s_indices]
            d_p_S = np.sum(diff_p**2, axis=1).astype(np.int32)
            d_S_S = np.zeros((len(s_indices), len(s_indices)), dtype=np.int32)
            for i in range(len(s_indices)):
                diff_i = points[s_indices[i]] - points[s_indices]
                d_S_S[i] = np.sum(diff_i**2, axis=1)

        # 1. Distances from p to all in S must be distinct
        if len(np.unique(d_p_S)) != len(s_indices):
            return False, []
            
        # 2. For each s in S, d(p, s) must not equal d(s, t) for any t in S
        # matches[i, j] is True if d_p_S[i] == d_S_S[i, j]
        matches = (d_S_S == d_p_S[:, np.newaxis])
        
        if matches.any():
            # Find conflicting points
            # For each i (point s_i), if any j has match, then s_i is conflicting?
            # Actually, if d_p_S[i] == d_S_S[i, j], then (p, s_i, s_j) is isosceles.
            # So s_i is "involved" in a conflict.
            # But we need to remove *some* points to make p valid.
            # The conflict set is not just the indices where matches.any(axis=1) is True.
            # If (p, s_i, s_j) is isosceles, removing s_i or s_j or p fixes it.
            # Since we want to add p, we must remove s_i or s_j.
            # Let's return the list of all indices involved in any match.
            conflict_indices = []
            for i in range(len(s_indices)):
                if matches[i].any():
                    # s_i is involved
                    if s_indices[i] not in conflict_indices:
                        conflict_indices.append(s_indices[i])
                    # s_j is also involved
                    for j in range(len(s_indices)):
                        if matches[i, j]:
                            if s_indices[j] not in conflict_indices:
                                conflict_indices.append(s_indices[j])
            return False, conflict_indices

        return True, []

    # Initial Greedy
    S = []
    candidates = list(range(M))
    random.shuffle(candidates)
    for p in candidates:
        if time.time() > deadline - 10:
            break
        valid, _ = can_add(p, S)
        if valid:
            S.append(p)

    best_S = S.copy()
    best_size = len(S)
    
    # Hill Climbing with Swaps
    stuck_counter = 0
    
    while time.time() < deadline - 5:
        # Try to add a random point
        p = random.randint(0, M - 1)
        if p in S:
            continue
            
        valid, conflicts = can_add(p, S)
        
        if valid:
            S.append(p)
            stuck_counter = 0
            if len(S) > best_size:
                best_size = len(S)
                best_S = S.copy()
        else:
            # Try to swap
            if conflicts:
                # Pick a random conflicting point to remove
                q = random.choice(conflicts)
                if q in S:
                    # Remove q
                    S.remove(q)
                    # Try to add p
                    valid2, _ = can_add(p, S)
                    if valid2:
                        S.append(p)
                        # Size is same, but structure changed
                        # We don't update best_size, but we accept the move
                    else:
                        # Revert
                        S.append(q)
                        stuck_counter += 1
                else:
                    stuck_counter += 1
            else:
                stuck_counter += 1
                
        # If stuck for a while, perturb
        if stuck_counter > 50:
            # Remove 1-3 random points
            remove_count = random.randint(1, 3)
            if len(S) > remove_count:
                to_remove = random.sample(S, remove_count)
                for r in to_remove:
                    S.remove(r)
                # Try to refill
                refill_candidates = [p for p in range(M) if p not in S]
                random.shuffle(refill_candidates)
                for p in refill_candidates:
                    if time.time() > deadline - 5:
                        break
                    valid, _ = can_add(p, S)
                    if valid:
                        S.append(p)
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = S.copy()
            stuck_counter = 0

    return [tuple(points[i]) for i in best_S]
