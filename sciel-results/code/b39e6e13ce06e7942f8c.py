import time
import random
import numpy as np

N = 512
DEADLINE = 55.0  # Safety margin

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Precompute 4-AP conflicts for fast checking
    # For each x, precompute all (a,b,c) such that x with a,b,c forms a 4-AP
    # where x is in position 1,2,3,4
    # We'll use a different approach: precompute for each x the list of "triples" to check
    
    def build_conflicts():
        """Precompute for each position x, the list of (p,q,r) triples to check."""
        conflicts = [[] for _ in range(N + 1)]
        for x in range(1, N + 1):
            # x is 1st: x, x+d, x+2d, x+3d
            max_d = (N - x) // 3
            for d in range(1, max_d + 1):
                conflicts[x].append((x + d, x + 2*d, x + 3*d))
            # x is 2nd: x-d, x, x+d, x+2d
            max_d2 = min(x - 1, (N - x) // 2)
            for d in range(1, max_d2 + 1):
                conflicts[x].append((x - d, x + d, x + 2*d))
            # x is 3rd: x-2d, x-d, x, x+d
            max_d3 = min((x - 1) // 2, N - x)
            for d in range(1, max_d3 + 1):
                conflicts[x].append((x - 2*d, x - d, x + d))
            # x is 4th: x-3d, x-2d, x-d, x
            max_d4 = (x - 1) // 3
            for d in range(1, max_d4 + 1):
                conflicts[x].append((x - 3*d, x - 2*d, x - d))
        return conflicts
    
    conflicts = build_conflicts()
    
    def can_add(mask, x):
        """Check if x can be added to mask without creating a 4-AP."""
        for (a, b, c) in conflicts[x]:
            if (mask >> a & 1) and (mask >> b & 1) and (mask >> c & 1):
                return False
        return True

    def fast_greedy(rng=None, order=None):
        """Fast greedy construction with optional randomization."""
        mask = 0
        if order is None:
            if rng is None:
                order = list(range(1, N + 1))
            else:
                order = list(range(1, N + 1))
                rng.shuffle(order)
        
        for x in order:
            if can_add(mask, x):
                mask |= (1 << x)
        
        return mask

    def local_search(mask, rng, deadline):
        """Local search with add and swap moves."""
        S_size = bin(mask).count('1')
        blocked = set(x for x in range(1, N + 1) if not (mask >> x & 1))
        
        no_improve_count = 0
        max_no_improve = 50
        
        while time.time() < deadline and no_improve_count < max_no_improve:
            improved = False
            
            # 1. Try adding any blocked element
            blocked_list = list(blocked)
            rng.shuffle(blocked_list)
            for x in blocked_list:
                if time.time() >= deadline:
                    return mask, S_size
                if can_add(mask, x):
                    mask |= (1 << x)
                    S_size += 1
                    blocked.discard(x)
                    improved = True
                    break
            
            if improved:
                no_improve_count = 0
                continue
            
            # 2. Try swaps: remove r, add a
            if not blocked:
                break
                
            # Try multiple random blocked elements with deeper search
            blocked_list = list(blocked)
            rng.shuffle(blocked_list)
            
            swap_found = False
            for a in blocked_list[:20]:  # Try up to 20 random blocked elements
                if time.time() >= deadline:
                    return mask, S_size
                
                S_list = [x for x in range(1, N + 1) if (mask >> x & 1)]
                rng.shuffle(S_list)
                
                # Try up to 80 random removals
                for r in S_list[:80]:
                    if time.time() >= deadline:
                        return mask, S_size
                    
                    mask_without_r = mask & ~(1 << r)
                    
                    if can_add(mask_without_r, a):
                        # Perform swap
                        mask = mask_without_r | (1 << a)
                        blocked.discard(a)
                        blocked.add(r)
                        swap_found = True
                        break
                
                if swap_found:
                    break
            
            if swap_found:
                no_improve_count = 0
            else:
                no_improve_count += 1

        return mask, S_size

    def polish_pass(mask, rng, deadline):
        """Try multiple ordering strategies to find more valid additions."""
        S_size = bin(mask).count('1')
        blocked = set(x for x in range(1, N + 1) if not (mask >> x & 1))
        
        # Try several different orderings
        orders = []
        # Ascending
        orders.append(list(range(1, N + 1)))
        # Descending
        orders.append(list(range(N, 0, -1)))
        # Random shuffles
        for _ in range(5):
            o = list(range(1, N + 1))
            rng.shuffle(o)
            orders.append(o)
        # Even then odd
        orders.append([x for x in range(1, N + 1) if x % 2 == 0] + [x for x in range(1, N + 1) if x % 2 == 1])
        # Odd then even
        orders.append([x for x in range(1, N + 1) if x % 2 == 1] + [x for x in range(1, N + 1) if x % 2 == 0])
        
        for order in orders:
            if time.time() >= deadline:
                break
            
            # Try to add elements in this order
            for x in order:
                if time.time() >= deadline:
                    break
                if x in blocked:
                    if can_add(mask, x):
                        mask |= (1 << x)
                        S_size += 1
                        blocked.discard(x)
        
        return mask, S_size

    # Main
    best_mask = 0
    best_size = 0
    
    # Multiple restarts with different random seeds
    for trial in range(25):
        if time.time() >= deadline - 10:
            break
        
        rng = random.Random(42 + trial)
        
        # Initial seed with randomization
        mask = fast_greedy(rng)
        
        # Run local search
        mask, size = local_search(mask, rng, deadline - 5)
        
        # Polish pass
        mask = polish_pass(mask, rng, deadline - 2)
        size = bin(mask).count('1')
        
        if size > best_size:
            best_size = size
            best_mask = mask
    
    # Convert to list
    result = [x for x in range(1, N + 1) if (best_mask >> x & 1)]
    
    return result

# Quick test
if __name__ == "__main__":
    t0 = time.time()
    res = construct()
    t1 = time.time()
    print(f"Size: {len(res)}, Time: {t1-t0:.2f}s")
