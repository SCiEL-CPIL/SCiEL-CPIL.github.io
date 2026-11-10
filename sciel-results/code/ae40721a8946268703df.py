import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    N = 1000
    max_sum = 2 * N
    
    def can_add(y, S, counts):
        """Check if y can be added to S."""
        if counts[2 * y] >= 2:
            return False
        if len(S) > 0:
            sums = np.array(S) + y
            if np.any(counts[sums] >= 2):
                return False
        return True

    def do_add(y, S, counts, in_set):
        """Add y to S and update counts."""
        if len(S) > 0:
            counts[np.array(S) + y] += 1
        counts[2 * y] += 1
        in_set[y] = True
        S.append(y)

    def do_remove(y, S, counts, in_set):
        """Remove y from S and update counts."""
        if len(S) > 1:
            others = np.array([v for v in S if v != y])
            counts[others + y] -= 1
        counts[2 * y] -= 1
        in_set[y] = False
        S.remove(y)

    def try_insert_all(S, counts, in_set, deadline):
        """Try to insert as many elements as possible."""
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            candidates = np.where(~in_set)[0]
            candidates = candidates[candidates >= 1]
            if len(candidates) == 0:
                break
            random.shuffle(candidates.tolist())
            for y in candidates:
                y = int(y)
                if can_add(y, S, counts):
                    do_add(y, S, counts, in_set)
                    improved = True
        return len(S)

    def local_search(S, counts, in_set, deadline):
        """Improve set via insertions and swaps."""
        stall_count = 0
        
        while time.time() < deadline - 5:
            # Phase 1: Try to insert any element
            size_before = len(S)
            try_insert_all(S, counts, in_set, deadline)
            if len(S) > size_before:
                stall_count = 0
                continue
            
            # Phase 2: Try swaps (remove x, add y)
            if len(S) < 2:
                break
                
            candidates = np.where(~in_set)[0]
            candidates = candidates[candidates >= 1]
            if len(candidates) == 0:
                break
            
            found_swap = False
            n_swap_attempts = min(200, max(50, len(S) // 2))
            
            for _ in range(n_swap_attempts):
                x_idx = random.randint(0, len(S) - 1)
                x = S[x_idx]
                
                # Pick a random candidate y different from x
                y = int(random.choice(candidates))
                if y == x:
                    continue
                
                # Backup
                S_backup = list(S)
                counts_backup = counts.copy()
                in_set_backup = in_set.copy()
                
                # Remove x
                do_remove(x, S, counts, in_set)
                
                # Try to add y
                if can_add(y, S, counts):
                    do_add(y, S, counts, in_set)
                    
                    # Now try to insert more elements
                    size_after_swap = len(S)
                    try_insert_all(S, counts, in_set, deadline)
                    
                    if len(S) > size_before:
                        # Size increased, keep this
                        found_swap = True
                        stall_count = 0
                        break
                    else:
                        # Size same or less, revert
                        S[:] = S_backup
                        counts[:] = counts_backup
                        in_set[:] = in_set_backup
                else:
                    # Revert
                    S[:] = S_backup
                    counts[:] = counts_backup
                    in_set[:] = in_set_backup
            
            if found_swap:
                continue
            
            # Phase 3: Perturbation - remove random elements and rebuild
            stall_count += 1
            
            if stall_count > 3:
                # More aggressive perturbation
                n_remove = min(3, len(S) // 4)
                for _ in range(n_remove):
                    if len(S) == 0:
                        break
                    x_idx = random.randint(0, len(S) - 1)
                    x = S[x_idx]
                    do_remove(x, S, counts, in_set)
                stall_count = 0
            else:
                # Single element removal
                if len(S) > 0:
                    x_idx = random.randint(0, len(S) - 1)
                    x = S[x_idx]
                    do_remove(x, S, counts, in_set)
            
            # After perturbation, try to insert
            try_insert_all(S, counts, in_set, deadline)
        
        return S, counts, in_set

    best_size = 0
    best_S = []
    
    # Try multiple strategies with diverse orderings
    strategies = [
        "asc", "desc", "rand1", "rand2", "rand3", "rand4", "rand5",
        "weighted_low", "weighted_high", "sparse"
    ]
    
    for strat in strategies:
        if time.time() > deadline - 20:
            break
            
        counts = np.zeros(max_sum + 1, dtype=np.int32)
        in_set = np.zeros(N + 1, dtype=bool)
        S = []
        
        # Greedy build with different orderings
        if strat == "asc":
            order = list(range(1, N + 1))
        elif strat == "desc":
            order = list(range(N, 0, -1))
        elif strat == "weighted_low":
            # Favor smaller numbers
            weights = [1.0 / (i + 1) for i in range(1, N + 1)]
            order = list(range(1, N + 1))
            random.shuffle(order)
            order.sort(key=lambda x: -weights[x - 1] + random.random() * 0.5)
        elif strat == "weighted_high":
            # Favor larger numbers
            weights = [float(i) for i in range(1, N + 1)]
            order = list(range(1, N + 1))
            random.shuffle(order)
            order.sort(key=lambda x: -weights[x - 1] + random.random() * 0.5)
        elif strat == "sparse":
            # Start with sparse selection
            order = list(range(1, N + 1, 2)) + list(range(2, N + 1, 2))
            random.shuffle(order)
        else:
            order = list(range(1, N + 1))
            random.shuffle(order)
            
        for y in order:
            if in_set[y]:
                continue
            if can_add(y, S, counts):
                do_add(y, S, counts, in_set)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = list(S)
            
        # Local search
        S, counts, in_set = local_search(S, counts, in_set, deadline)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = list(S)
            
    return best_S

if __name__ == "__main__":
    res = construct()
    print(len(res))
