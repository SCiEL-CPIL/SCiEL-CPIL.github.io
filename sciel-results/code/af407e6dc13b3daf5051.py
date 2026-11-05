import time
import random
import math

def construct():
    N = 256
    deadline = time.time() + 50.0  # Safety margin before 60s kill
    
    # 1. Precompute 4-AP conflicts for fast validity checking
    # conflicts[x] is a list of tuples (y, z, w) such that {x, y, z, w} is a 4-AP
    conflicts = [[] for _ in range(N)]
    for a in range(N):
        max_d = (N - 1 - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            conflicts[a].append((b, c, e))
            conflicts[b].append((a, c, e))
            conflicts[c].append((a, b, e))
            conflicts[e].append((a, b, c))

    def is_valid_add(mask, x):
        """Check if adding x to mask creates a 4-AP."""
        for (y, z, w) in conflicts[x]:
            if (mask >> y) & 1 and (mask >> z) & 1 and (mask >> w) & 1:
                return False
        return True

    def count_3aps(mask):
        """Count 3-term APs in the set represented by mask."""
        count = 0
        for i in range(N):
            if not ((mask >> i) & 1):
                continue
            max_d = (N - 1 - i) // 2
            for d in range(1, max_d + 1):
                j = i + d
                k = i + 2 * d
                if ((mask >> j) & 1) and ((mask >> k) & 1):
                    count += 1
        return count

    def delta_3aps_add(mask, x):
        """Calculate increase in 3-AP count if x is added to mask."""
        delta = 0
        # x is start
        max_d = (N - 1 - x) // 2
        for d in range(1, max_d + 1):
            if ((mask >> (x + d)) & 1) and ((mask >> (x + 2 * d)) & 1):
                delta += 1
        # x is middle
        max_d = min(x, N - 1 - x)
        for d in range(1, max_d + 1):
            if ((mask >> (x - d)) & 1) and ((mask >> (x + d)) & 1):
                delta += 1
        # x is end
        max_d = x // 2
        for d in range(1, max_d + 1):
            if ((mask >> (x - 2 * d)) & 1) and ((mask >> (x - d)) & 1):
                delta += 1
        return delta

    # 2. Greedy initialization
    def greedy_init():
        mask = 0
        elems = list(range(N))
        random.shuffle(elems)
        for x in elems:
            if is_valid_add(mask, x):
                if delta_3aps_add(mask, x) > 0 or random.random() < 0.2:
                    mask |= (1 << x)
        return mask

    # 3. Hill Climbing with Random Restarts
    # This is more aggressive exploitation than SA
    best_mask = 0
    best_score = -1

    # Multiple restarts
    num_restarts = 12  # Adjust based on time
    per_restart_time = (deadline - time.time()) / num_restarts

    for r in range(num_restarts):
        if time.time() > deadline - 2.0:
            break

        # Initialize
        current_mask = greedy_init()
        current_score = count_3aps(current_mask)

        if current_score > best_score:
            best_score = current_score
            best_mask = current_mask

        # Local search
        no_improve_count = 0
        max_no_improve = 500  # Max steps without improvement before giving up this restart
        
        while no_improve_count < max_no_improve and time.time() < deadline - 2.0:
            improved = False
            
            # Try adding a random element
            x = random.randint(0, N - 1)
            if not ((current_mask >> x) & 1):
                if is_valid_add(current_mask, x):
                    new_mask = current_mask | (1 << x)
                    new_score = current_score + delta_3aps_add(current_mask, x)
                    if new_score >= current_score:
                        current_mask = new_mask
                        current_score = new_score
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_mask = current_mask

            # Try removing a random element (less likely to improve, but can unlock)
            if not improved and random.random() < 0.1:
                x = random.randint(0, N - 1)
                if (current_mask >> x) & 1:
                    new_mask = current_mask & ~(1 << x)
                    new_score = current_score - delta_3aps_add(current_mask, x)
                    if new_score > current_score:  # Should never happen with pure removal
                        current_mask = new_mask
                        current_score = new_score
                        improved = True

            # Try swap
            if not improved:
                x = random.randint(0, N - 1)
                if (current_mask >> x) & 1:
                    y = random.randint(0, N - 1)
                    if not ((current_mask >> y) & 1):
                        temp_mask = current_mask & ~(1 << x)
                        if is_valid_add(temp_mask, y):
                            d_rem = delta_3aps_add(current_mask, x)
                            d_add = delta_3aps_add(temp_mask, y)
                            new_score = current_score - d_rem + d_add
                            new_mask = temp_mask | (1 << y)
                            if new_score > current_score:
                                current_mask = new_mask
                                current_score = new_score
                                improved = True
                                if current_score > best_score:
                                    best_score = current_score
                                    best_mask = current_mask

            if not improved:
                no_improve_count += 1
            else:
                no_improve_count = 0

    # Final check
    if best_score < 0:
        # Fallback
        best_mask = 1  # {0} -> {1}
        best_score = 0

    result = [i + 1 for i in range(N) if (best_mask >> i) & 1]
    if not result:
        result = [1]
    return sorted(result)
