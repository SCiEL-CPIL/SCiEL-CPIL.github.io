import time
import random
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    if n <= 1:
        return "1"
    
    if n < 7:
        # Standard construction is optimal for n < 7
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        return "".join(map(str, word))
    
    # n == 7
    # Get standard order
    def get_seed_order(m):
        if m == 1:
            return [(1,)]
        word = [1]
        for k in range(2, m + 1):
            perms_k, seen = [], set()
            for i in range(len(word) - k + 2):
                window = tuple(word[i:i + k - 1])
                if len(set(window)) == k - 1 and window not in seen:
                    seen.add(window)
                    perms_k.append(window)
            new = []
            for p in perms_k:
                block = list(p) + [k] + list(p)
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        return perms_k

    initial_order = get_seed_order(n - 1)
    
    # All perms of 1..6
    all_perms = list(permutations(range(1, n)))
    perm_to_idx = {p: i for i, p in enumerate(all_perms)}
    N = len(all_perms)
    
    # Precompute overlap matrix
    # L[i][j] = overlap(all_perms[i], all_perms[j])
    L = [[0]*N for _ in range(N)]
    for i in range(N):
        p1 = all_perms[i]
        for j in range(N):
            p2 = all_perms[j]
            # Max overlap 5
            for k in range(5, -1, -1):
                if p1[-k:] == p2[:k]:
                    L[i][j] = k
                    break
    
    # Initial order indices
    current_order = [perm_to_idx[p] for p in initial_order]
    
    def get_path_cost(order):
        cost = 13 # First block
        for i in range(len(order) - 1):
            cost += 13 - L[order[i]][order[i+1]]
        return cost
    
    current_cost = get_path_cost(current_order)
    best_order = current_order[:]
    best_cost = current_cost
    
    # Local Search: 2-Opt
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(N - 1):
            if time.time() > deadline:
                break
            for j in range(i + 1, N):
                # Try reversing i..j
                # Old cost contribution of edges (i-1,i), (i,i+1)...(j,j+1)
                # New cost contribution of edges (i-1,j), (j,j-1)...(i,i+1)
                
                # This is complex to update in O(1) without prefix sums.
                # Given N=720, O(N) update per pair is $720^2 \times 720 \approx 3 \times 10^8$.
                # Too slow.
                
                # Let's just do random swaps.
                pass
        
        # Do a batch of random swaps
        for _ in range(1000):
            if time.time() > deadline:
                break
            i, j = random.sample(range(N), 2)
            current_order[i], current_order[j] = current_order[j], current_order[i]
            new_cost = get_path_cost(current_order)
            if new_cost < best_cost:
                best_cost = new_cost
                best_order = current_order[:]
                improved = True
            else:
                current_order[i], current_order[j] = current_order[j], current_order[i]
    
    # Construct word from best_order
    word = []
    for idx in best_order:
        p = all_perms[idx]
        block = list(p) + [n] + list(p)
        overlap = next(o for o in range(min(len(word), len(block)), -1, -1) if word[len(word) - o:] == block[:o])
        word += block[overlap:]
        
    return "".join(map(str, word))
