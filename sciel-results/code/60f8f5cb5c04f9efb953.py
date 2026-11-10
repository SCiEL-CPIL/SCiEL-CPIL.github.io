import numpy as np
import time
import random

def construct(q, n, radius):
    if q == 3 and n == 8 and radius == 1:
        deadline = time.time() + 115.0
        N = q ** n  # 6561
        powers = [q ** i for i in range(n)]
        neigh_size = 1 + n * (q - 1)  # 17 for q=3,n=8
        
        # Precompute neighbors
        neighbors = np.zeros((N, neigh_size), dtype=np.int32)
        for w in range(N):
            neighbors[w, 0] = w
            idx = 1
            for i in range(n):
                d = (w // powers[i]) % q
                for delta in range(1, q):
                    new_d = (d + delta) % q
                    diff = new_d - d
                    new_w = w + diff * powers[i]
                    neighbors[w, idx] = new_w
                    idx += 1
        
        # Build bitset representation of neighbor sets
        # Each word w covers a set of words; represent as Python int bitmask
        N_bits = N  # 6561 bits
        # Build cover_mask[w] as a Python int where bit j is 1 if j is in neighbors[w]
        cover_mask = [0] * N
        for w in range(N):
            mask = 0
            for nb in neighbors[w, :]:
                mask |= (1 << nb)
            cover_mask[w] = mask
        
        # For efficiency, precompute neighbor lists as tuples
        neighbor_lists = [tuple(neighbors[w, :].tolist()) for w in range(N)]
        
        def greedy_init(rng, deadline):
            """Fast greedy initialization using bitsets."""
            uncovered = (1 << N) - 1  # all bits set
            code = set()
            
            while uncovered and time.time() < deadline - 5:
                # Find word with maximum uncovered coverage
                best_w = -1
                best_gain = -1
                # Sample candidates to avoid full scan (N=6561 is manageable)
                for w in range(N):
                    gain = bin(uncovered & cover_mask[w]).count('1')
                    if gain > best_gain:
                        best_gain = gain
                        best_w = w
                        if best_gain == neigh_size:
                            break
                if best_gain <= 0:
                    break
                code.add(best_w)
                uncovered &= ~cover_mask[best_w]
            
            return code
        
        def count_uncovered(code_set):
            """Count words not covered by any codeword."""
            if not code_set:
                return N
            # Union of all cover masks
            union_mask = 0
            for c in code_set:
                union_mask |= cover_mask[c]
            return N - bin(union_mask).count('1')
        
        def simulated_annealing(code_set, rng, deadline):
            """Simulated annealing with add/remove/swap moves."""
            code_set = set(code_set)
            current_cost = count_uncovered(code_set)
            current_size = len(code_set)
            
            T = 50.0  # initial temperature
            T_min = 0.1
            cooling = 0.995
            
            best_code = set(code_set)
            best_size = current_size
            
            no_improve = 0
            max_no_improve = 2000
            
            while time.time() < deadline - 3 and no_improve < max_no_improve:
                if T <= T_min:
                    break
                
                # Choose a random move type
                move_type = rng.random()
                code_list = list(code_set)
                
                if move_type < 0.3 and current_size > 10:
                    # Remove a random codeword
                    c = rng.choice(code_list)
                    new_set = code_set - {c}
                    new_cost = count_uncovered(new_set)
                    new_size = len(new_set)
                    
                    # Delta: cost change + size penalty
                    delta_cost = (new_cost - current_cost) + 10 * (new_size - current_size)
                    
                    if delta_cost < 0 or rng.random() < np.exp(-delta_cost / T):
                        code_set = new_set
                        current_cost = new_cost
                        current_size = new_size
                        no_improve = 0
                        if current_size < best_size and current_cost == 0:
                            best_code = set(code_set)
                            best_size = current_size
                            no_improve = 0
                    else:
                        no_improve += 1
                        
                elif move_type < 0.6:
                    # Add a random codeword (if not already present)
                    d = rng.integers(0, N)
                    if d not in code_set:
                        new_set = code_set | {d}
                        new_cost = count_uncovered(new_set)
                        new_size = len(new_set)
                        
                        delta_cost = (new_cost - current_cost) + 10 * (new_size - current_size)
                        
                        if delta_cost < 0 or rng.random() < np.exp(-delta_cost / T):
                            code_set = new_set
                            current_cost = new_cost
                            current_size = new_size
                            no_improve = 0
                        else:
                            no_improve += 1
                    else:
                        no_improve += 1
                        
                else:
                    # Swap: remove one, add one
                    if not code_list:
                        no_improve += 1
                        continue
                    c = rng.choice(code_list)
                    d = rng.integers(0, N)
                    if d == c or d in code_set:
                        no_improve += 1
                        continue
                    
                    new_set = code_set - {c} | {d}
                    new_cost = count_uncovered(new_set)
                    new_size = len(new_set)
                    
                    delta_cost = (new_cost - current_cost) + 10 * (new_size - current_size)
                    
                    if delta_cost < 0 or rng.random() < np.exp(-delta_cost / T):
                        code_set = new_set
                        current_cost = new_cost
                        current_size = new_size
                        no_improve = 0
                        if current_size < best_size and current_cost == 0:
                            best_code = set(code_set)
                            best_size = current_size
                    else:
                        no_improve += 1
                
                T *= cooling
            
            return best_code
        
        def local_polish(code_set, deadline):
            """Final local search: try deletions and 1-swaps."""
            code_set = set(code_set)
            improved = True
            while improved and time.time() < deadline - 2:
                improved = False
                code_list = list(code_set)
                
                # Try deletions
                for c in code_list:
                    if time.time() > deadline - 2:
                        break
                    new_set = code_set - {c}
                    if count_uncovered(new_set) == 0:
                        code_set = new_set
                        improved = True
                        break
                
                if improved:
                    continue
                
                # Try 1-swaps
                code_list = list(code_set)
                rng = np.random.default_rng()
                rng.shuffle(code_list)
                for c in code_list:
                    if time.time() > deadline - 2:
                        break
                    c_neighbors = neighbor_lists[c]
                    # Words only covered by c
                    cov_count = np.zeros(N, dtype=np.uint8)
                    for d in code_set:
                        if d == c:
                            continue
                        for nb in neighbor_lists[d]:
                            cov_count[nb] += 1
                    essential = [nb for nb in c_neighbors if cov_count[nb] == 0]
                    
                    if not essential:
                        continue
                    
                    # Find d that covers all essential words
                    # d must be in intersection of neighbors of all essential words
                    cand_mask = cover_mask[essential[0]]
                    for ew in essential[1:]:
                        cand_mask &= cover_mask[ew]
                        if cand_mask == 0:
                            break
                    
                    if cand_mask == 0:
                        continue
                    
                    # Get candidate indices
                    candidates = []
                    m = cand_mask
                    while m:
                        lb = m & (-m)
                        idx = lb.bit_length() - 1
                        candidates.append(idx)
                        m ^= lb
                    
                    valid = [d for d in candidates if d not in code_set]
                    if valid:
                        d = rng.choice(valid)
                        code_set.remove(c)
                        code_set.add(d)
                        improved = True
                        break
            
            return code_set
        
        best_code = None
        best_size = float('inf')
        
        # Run multiple restarts with simulated annealing
        for i in range(30):
            if time.time() > deadline - 15:
                break
            
            rng = np.random.default_rng(12345 + i * 7919)
            
            # Greedy init
            code = greedy_init(rng, deadline)
            
            # Simulated annealing
            code = simulated_annealing(code, rng, deadline)
            
            # Local polish
            code = local_polish(code, deadline)
            
            if len(code) < best_size:
                best_size = len(code)
                best_code = code
                
            if best_size <= 486:
                if best_size <= 480:
                    break
        
        if best_code is None:
            # Fallback: tetracode-based construction
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            best_code = set()
            import itertools
            for h in tetracode:
                for tail in itertools.product(range(3), repeat=n - 4):
                    word = h + list(tail)
                    val = 0
                    for i, d in enumerate(word):
                        val += d * (q ** i)
                    best_code.add(val)
        
        final_code = []
        for w in best_code:
            word = []
            temp = w
            for _ in range(n):
                word.append(temp % q)
                temp //= q
            final_code.append(word)
        
        return final_code
    
    else:
        N = q ** n
        deadline = time.time() + 110.0
        powers = [q ** i for i in range(n)]
        neigh_size = 1 + n * (q - 1)
        neighbors = np.zeros((N, neigh_size), dtype=np.int32)
        
        for w in range(N):
            neighbors[w, 0] = w
            idx = 1
            for i in range(n):
                d = (w // powers[i]) % q
                for delta in range(1, q):
                    new_d = (d + delta) % q
                    diff = new_d - d
                    new_w = w + diff * powers[i]
                    neighbors[w, idx] = new_w
                    idx += 1
        
        rng = np.random.default_rng()
        uncovered = np.ones(N, dtype=bool)
        code = set()
        while np.any(uncovered) and time.time() < deadline:
            gains = np.zeros(N)
            for i in range(N):
                gains[i] = np.sum(uncovered[neighbors[i, :]])
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            w = rng.choice(candidates)
            code.add(w)
            uncovered[neighbors[w, :]] = False
        
        final_code = []
        for w in code:
            word = []
            temp = w
            for _ in range(n):
                word.append(temp % q)
                temp //= q
            final_code.append(word)
        return final_code
