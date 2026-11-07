"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 78.0  # safety margin before 90s kill
    N = 128
    
    # Build the graph: edge (a,b) exists iff a+b <= 128
    # adj[x] = bitmask of y such that edge (x,y) exists
    adj = [0] * (N + 1)
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            if a + b <= N:
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)
    
    # Precompute edge_count for ordering
    edge_count = [bin(adj[x]).count('1') for x in range(N + 1)]
    
    def is_valid_mask(mask):
        """Check if the set represented by mask is valid (no bad triangle)."""
        # For each pair (a,b) with edge, check if there's c > b with edges (a,c) and (b,c)
        # and c in mask
        for a in range(1, N + 1):
            if not (mask >> a) & 1:
                continue
            # For each b > a with edge (a,b) and b in mask
            b_mask = adj[a] & mask
            b_mask &= ~((1 << a) - 1)  # b > a
            while b_mask:
                lb = b_mask & (-b_mask)
                b = lb.bit_length() - 1
                b_mask ^= lb
                # Need c > b with edges (a,c) and (b,c), and c in mask
                common = adj[a] & adj[b] & mask
                common &= ~((1 << (b + 1)) - 1)  # c > b
                if common:
                    return False
        return True
    
    def is_valid_list(S):
        mask = 0
        for x in S:
            mask |= (1 << x)
        return is_valid_mask(mask)
    
    def greedy_add_to_mask(mask, order):
        """Greedy add vertices in given order to mask."""
        for x in order:
            if (mask >> x) & 1:
                continue
            new_mask = mask | (1 << x)
            if is_valid_mask(new_mask):
                mask = new_mask
        return mask
    
    def mask_to_list(mask):
        result = []
        for x in range(1, N + 1):
            if (mask >> x) & 1:
                result.append(x)
        return result
    
    # Phase 1: Greedy with different orderings
    best_mask = 0
    best_size = 0
    
    # Ordering 1: descending edge count (most constrained first)
    order1 = sorted(range(1, N + 1), key=lambda x: -edge_count[x])
    m = greedy_add_to_mask(0, order1)
    if bin(m).count('1') > best_size:
        best_mask = m
        best_size = bin(m).count('1')
    
    # Ordering 2: ascending (least constrained first)
    order2 = sorted(range(1, N + 1), key=lambda x: edge_count[x])
    m = greedy_add_to_mask(0, order2)
    if bin(m).count('1') > best_size:
        best_mask = m
        best_size = bin(m).count('1')
    
    # Ordering 3: natural
    m = greedy_add_to_mask(0, range(1, N + 1))
    if bin(m).count('1') > best_size:
        best_mask = m
        best_size = bin(m).count('1')
    
    # Ordering 4: reverse natural
    m = greedy_add_to_mask(0, range(N, 0, -1))
    if bin(m).count('1') > best_size:
        best_mask = m
        best_size = bin(m).count('1')
    
    # Phase 2: Local search - remove one, greedily re-add
    for iteration in range(20000):
        if time.time() > deadline:
            break
        # Remove a random vertex from best
        S = mask_to_list(best_mask)
        if not S:
            break
        idx = random.randint(0, len(S) - 1)
        removed = S[idx]
        S2 = S[:idx] + S[idx+1:]
        mask2 = 0
        for x in S2:
            mask2 |= (1 << x)
        
        # Greedily add back: try all candidates in random order
        candidates = [removed] + [x for x in range(1, N + 1) if not (mask2 >> x) & 1]
        random.shuffle(candidates)
        
        new_mask = mask2
        for x in candidates:
            if (new_mask >> x) & 1:
                continue
            trial = new_mask | (1 << x)
            if is_valid_mask(trial):
                new_mask = trial
        
        if bin(new_mask).count('1') > best_size:
            best_mask = new_mask
            best_size = bin(new_mask).count('1')
    
    # Phase 3: Simulated annealing
    T = 10.0
    cooling = 0.9999
    current_mask = best_mask
    current_size = best_size
    
    for iteration in range(50000):
        if time.time() > deadline:
            break
        if T < 0.01:
            break
        
        # Random move: add or remove a random vertex
        x = random.randint(1, N)
        if (current_mask >> x) & 1:
            # Remove
            new_mask = current_mask & ~(1 << x)
            new_size = current_size - 1
        else:
            # Add
            new_mask = current_mask | (1 << x)
            if not is_valid_mask(new_mask):
                continue
            new_size = current_size + 1
        
        delta = new_size - current_size
        if delta >= 0 or random.random() < pow(2.718281828, delta / T):
            current_mask = new_mask
            current_size = new_size
        
        if current_size > best_size:
            best_mask = current_mask
            best_size = current_size
        
        T *= cooling
    
    # Phase 4: Final greedy polish
    final_mask = best_mask
    for x in range(1, N + 1):
        if (final_mask >> x) & 1:
            continue
        trial = final_mask | (1 << x)
        if is_valid_mask(trial):
            final_mask = trial
    
    best_mask = final_mask
    best_size = bin(best_mask).count('1')
    
    # Convert to list
    result = mask_to_list(best_mask)
    
    # Final verification
    if not is_valid_list(result):
        # Fallback
        result = list(range(17, 33)) + list(range(65, 129))
    
    return [int(x) for x in result]
# EVOLVE-BLOCK-END
