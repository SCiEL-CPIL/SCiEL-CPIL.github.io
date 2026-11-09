"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def construct():
    start_time = time.time()
    # Use 145s as deadline to stay well within 150s limit
    deadline = start_time + 145.0
    
    N = 512
    # weights[a] = 1/(512-a)
    weights = [0.0] * N
    for a in range(1, N):
        weights[a] = 1.0 / (N - a)

    # Precompute conflict masks
    # conflict[a] is a bitmask where bit b is set if gcd(a, b) > 1
    conflict = [0] * N
    for a in range(1, N):
        mask = 0
        for b in range(1, N):
            if a != b and gcd(a, b) > 1:
                mask |= (1 << b)
        conflict[a] = mask

    def score_mask(mask):
        """Calculate the score of a bitmask representation."""
        s = 0.0
        m = mask
        while m:
            lsb = m & (-m)
            a = lsb.bit_length() - 1
            s += weights[a]
            m ^= lsb
        return s

    def get_chosen_list(mask):
        """Convert bitmask to list of integers."""
        chosen = []
        m = mask
        while m:
            lsb = m & (-m)
            a = lsb.bit_length() - 1
            chosen.append(a)
            m ^= lsb
        return chosen

    def greedy_seed(order):
        """Construct a valid set using greedy addition in the given order."""
        mask = 0
        for a in order:
            if (conflict[a] & mask) == 0:
                mask |= (1 << a)
        return mask

    def local_search(mask, deadline):
        """
        Perform local search with moves:
        1. Add (always improves if possible)
        2. Swap 1-for-1
        3. Swap 1-for-2
        4. Swap 2-for-1
        5. Swap 2-for-2
        """
        # Phase 1: Maximize by adding
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Find the best candidate to add
            best_add = -1
            best_w = 0.0
            for a in range(1, N):
                if not (mask & (1 << a)):
                    if (conflict[a] & mask) == 0:
                        if weights[a] > best_w:
                            best_w = weights[a]
                            best_add = a
            if best_add > 0:
                mask |= (1 << best_add)
                improved = True
        
        # Phase 2: Swaps
        while time.time() < deadline:
            improved = False
            chosen = get_chosen_list(mask)
            chosen_set = set(chosen)
            nc = len(chosen)
            
            # --- 1-for-1 Swap ---
            best_gain = 1e-9
            best_move = None
            for a_out in chosen:
                base = mask ^ (1 << a_out)
                for a_in in range(1, N):
                    if a_in in chosen_set:
                        continue
                    if (conflict[a_in] & base) == 0:
                        gain = weights[a_in] - weights[a_out]
                        if gain > best_gain:
                            best_gain = gain
                            best_move = (a_out, a_in)
            if best_move:
                a_out, a_in = best_move
                mask = (mask ^ (1 << a_out)) | (1 << a_in)
                improved = True
                continue

            # --- 1-for-2 Swap ---
            best_gain = 1e-9
            best_move = None
            for a_out in chosen:
                base = mask ^ (1 << a_out)
                # Collect compatible candidates
                cands = []
                for a_in in range(1, N):
                    if a_in in chosen_set:
                        continue
                    if (conflict[a_in] & base) == 0:
                        cands.append(a_in)
                if len(cands) < 2:
                    continue
                # Sort by weight to find best pairs faster
                cands.sort(key=lambda x: weights[x], reverse=True)
                # Limit candidates to top 50 to keep O(K^2) manageable
                cands = cands[:50]
                for i in range(len(cands)):
                    a1 = cands[i]
                    for j in range(i + 1, len(cands)):
                        a2 = cands[j]
                        if (conflict[a1] & (1 << a2)) == 0:
                            gain = weights[a1] + weights[a2] - weights[a_out]
                            if gain > best_gain:
                                best_gain = gain
                                best_move = (a_out, a1, a2)
            if best_move:
                a_out, a1, a2 = best_move
                mask = (mask ^ (1 << a_out)) | (1 << a1) | (1 << a2)
                improved = True
                continue

            # --- 2-for-1 Swap ---
            best_gain = 1e-9
            best_move = None
            for i in range(nc):
                a1 = chosen[i]
                for j in range(i + 1, nc):
                    a2 = chosen[j]
                    base = mask ^ (1 << a1) ^ (1 << a2)
                    for a_in in range(1, N):
                        if a_in in chosen_set:
                            continue
                        if (conflict[a_in] & base) == 0:
                            gain = weights[a_in] - weights[a1] - weights[a2]
                            if gain > best_gain:
                                best_gain = gain
                                best_move = (a1, a2, a_in)
            if best_move:
                a1, a2, a_in = best_move
                mask = (mask ^ (1 << a1) ^ (1 << a2)) | (1 << a_in)
                improved = True
                continue

            # --- 2-for-2 Swap ---
            best_gain = 1e-9
            best_move = None
            for i in range(nc):
                a1 = chosen[i]
                for j in range(i + 1, nc):
                    a2 = chosen[j]
                    base = mask ^ (1 << a1) ^ (1 << a2)
                    cands = []
                    for a_in in range(1, N):
                        if a_in in chosen_set:
                            continue
                        if (conflict[a_in] & base) == 0:
                            cands.append(a_in)
                    if len(cands) < 2:
                        continue
                    cands.sort(key=lambda x: weights[x], reverse=True)
                    cands = cands[:40]
                    for k in range(len(cands)):
                        b1 = cands[k]
                        for l in range(k + 1, len(cands)):
                            b2 = cands[l]
                            if (conflict[b1] & (1 << b2)) == 0:
                                gain = weights[b1] + weights[b2] - weights[a1] - weights[a2]
                                if gain > best_gain:
                                    best_gain = gain
                                    best_move = (a1, a2, b1, b2)
            if best_move:
                a1, a2, b1, b2 = best_move
                mask = (mask ^ (1 << a1) ^ (1 << a2)) | (1 << b1) | (1 << b2)
                improved = True
                continue

            if not improved:
                break

        return mask

    # 1. Initial Greedy
    # Start with descending order (highest weights first)
    order = list(range(N - 1, 0, -1))
    mask = greedy_seed(order)
    
    # 2. Local Search
    mask = local_search(mask, deadline - 10.0)
    best_score = score_mask(mask)
    
    # 3. Multi-start Randomized Search
    rng = random.Random(12345)
    # Reserve 5s for final safety
    while time.time() < deadline - 5.0:
        # Create a randomized order
        # Bias towards higher weights to get good seeds
        rand_order = list(range(1, N))
        # Simple shuffle
        rng.shuffle(rand_order)
        
        m = greedy_seed(rand_order)
        m = local_search(m, deadline - 2.0)
        s = score_mask(m)
        
        if s > best_score:
            best_score = s
            mask = m

    return get_chosen_list(mask)
# EVOLVE-BLOCK-END
