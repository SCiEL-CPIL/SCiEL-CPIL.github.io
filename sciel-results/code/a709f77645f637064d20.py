import itertools
import random
import time
import math

def construct(q, n, radius):
    if q != 3 or n != 6 or radius != 1:
        return [list(t) for t in itertools.product(range(q), repeat=n)]

    # Precompute neighbors
    # Index to tuple
    def idx_to_tuple(idx):
        t = [0] * n
        for i in range(n):
            t[n - 1 - i] = idx % q
            idx //= q
        return tuple(t)
    
    def tuple_to_idx(t):
        idx = 0
        for i in range(n):
            idx = idx * q + t[i]
        return idx

    all_indices = list(range(q**n))
    neighbors = [[] for _ in range(q**n)]
    
    for idx in all_indices:
        w = idx_to_tuple(idx)
        neighbors[idx].append(idx)
        for i in range(n):
            for d in range(1, q):
                new_w = list(w)
                new_w[i] = (new_w[i] + d) % q
                neighbors[idx].append(tuple_to_idx(tuple(new_w)))
        # Unique
        neighbors[idx] = list(set(neighbors[idx]))
        
    # Precompute neighbor lists as tuples for faster iteration?
    # Lists are fine.

    # Initial Code: Tetracode * All Pairs
    # Tetracode: {(a,b,a+b,a+2b) : a,b in {0,1,2}}
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    code_indices = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=2):
            code_indices.append(tuple_to_idx(tuple(h + list(tail))))
            
    code_indices = list(set(code_indices)) # Ensure unique
    
    # Initialize cover counts
    cover_count = [0] * (q**n)
    for c in code_indices:
        for nb in neighbors[c]:
            cover_count[nb] += 1
            
    uncovered_count = sum(1 for c in cover_count if c == 0)
    # Should be 0 for valid code
    
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    best_code = list(code_indices)
    best_size = len(code_indices)
    
    # Greedy Pruning
    while True:
        if time.time() > deadline - 5: break
        min_loss = float('inf')
        min_idx = -1
        for i, c in enumerate(code_indices):
            loss = 0
            for nb in neighbors[c]:
                if cover_count[nb] == 1:
                    loss += 1
            if loss < min_loss:
                min_loss = loss
                min_idx = i
            if min_loss == 0:
                break
        if min_loss == 0:
            c_to_remove = code_indices[min_idx]
            for nb in neighbors[c_to_remove]:
                cover_count[nb] -= 1
                if cover_count[nb] == 0:
                    uncovered_count += 1
            code_indices.pop(min_idx)
            if len(code_indices) < best_size:
                best_size = len(code_indices)
                best_code = list(code_indices)
        else:
            break
            
    # Simulated Annealing
    T = 5.0
    cooling = 0.999
    M = 100.0 # Penalty for uncovered words
    
    while time.time() < deadline - 5:
        if T < 0.01:
            T = 0.01
            
        # Random Move
        move_type = random.random()
        if move_type < 0.5:
            # Remove
            if not code_indices:
                continue
            i = random.randint(0, len(code_indices) - 1)
            c = code_indices[i]
            
            # Calculate marginal loss
            loss = 0
            for nb in neighbors[c]:
                if cover_count[nb] == 1:
                    loss += 1
            
            if loss == 0:
                # Valid removal
                for nb in neighbors[c]:
                    cover_count[nb] -= 1
                    if cover_count[nb] == 0:
                        uncovered_count += 1
                code_indices.pop(i)
                if len(code_indices) < best_size:
                    best_size = len(code_indices)
                    best_code = list(code_indices)
            else:
                # Invalid removal
                delta_E = -1 + M * loss
                if random.random() < math.exp(-delta_E / T):
                    # Accept
                    for nb in neighbors[c]:
                        cover_count[nb] -= 1
                        if cover_count[nb] == 0:
                            uncovered_count += 1
                    code_indices.pop(i)
        else:
            # Swap
            if not code_indices:
                continue
            i = random.randint(0, len(code_indices) - 1)
            c_old = code_indices[i]
            c_new = random.choice(all_indices)
            
            if c_new == c_old:
                continue
            
            # Check if c_new is already in code
            # For efficiency, we can just try it. If it's in code, 
            # removing c_old and adding c_new is effectively a swap 
            # only if c_new != c_old.
            # But if c_new is already in code at index j, 
            # then we are removing c_old and adding c_new (which is already there).
            # This results in c_new being covered by 2 codewords? No.
            # The set of codewords changes from {c_old, ..., c_new, ...} to {c_new, ..., c_new, ...}
            # This is not a valid set operation if we treat it as a multiset.
            # We treat code as a set.
            # So if c_new is in code, the new set is the same as old set minus c_old.
            # This is just a removal.
            # So we should pick c_new not in code.
            
            # Efficient check: 
            # Since code size is ~73 and space is 729, 
            # a set lookup is fast.
            code_set = set(code_indices)
            if c_new in code_set:
                # Pick another or skip? 
                # Just skip this move.
                continue
                
            # Perform Swap
            # Remove c_old
            for nb in neighbors[c_old]:
                cover_count[nb] -= 1
                if cover_count[nb] == 0:
                    uncovered_count += 1
            # Add c_new
            for nb in neighbors[c_new]:
                cover_count[nb] += 1
                if cover_count[nb] == 1:
                    uncovered_count -= 1
                    
            code_indices[i] = c_new
            
            # Check validity/energy
            if uncovered_count == 0:
                # Valid swap
                if len(code_indices) < best_size:
                    best_size = len(code_indices)
                    best_code = list(code_indices)
            else:
                delta_E = M * uncovered_count
                if random.random() >= math.exp(-delta_E / T):
                    # Reject, revert
                    # Remove c_new
                    for nb in neighbors[c_new]:
                        cover_count[nb] -= 1
                        if cover_count[nb] == 0:
                            uncovered_count += 1
                    # Add c_old
                    for nb in neighbors[c_old]:
                        cover_count[nb] += 1
                        if cover_count[nb] == 1:
                            uncovered_count -= 1
                    code_indices[i] = c_old
                    
        T *= cooling
        
    # Return best code found
    return [list(idx_to_tuple(c)) for c in best_code]
