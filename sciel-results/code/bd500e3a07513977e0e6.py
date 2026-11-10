import numpy as np
import time
import sys
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    deadline = time.time() + 600  # Safety margin before 630s kill
    
    # 1. Build Forbidden Set F
    # F = {x^2 mod 205 for x in 1..204} \ {0}
    F = set()
    for x in range(1, m):
        sq = (x * x) % m
        if sq != 0:
            F.add(sq)
    
    # 2. Build Conflict Graph using Bitmasks
    # conflict_mask[i] is a bitmask where bit j is 1 if i and j are in conflict (difference in F)
    # We want Maximum Independent Set (MIS) in this graph.
    # Equivalent to Maximum Clique in the Complement Graph.
    # Let's work with the Conflict Graph directly.
    # Two vertices i, j are compatible if (i-j)%m not in F and (j-i)%m not in F.
    # Since F is symmetric (x^2 == (-x)^2), (i-j)%m in F iff (j-i)%m in F.
    
    # compat_mask[i] is a bitmask of vertices compatible with i.
    compat_mask = [0] * m
    for i in range(m):
        mask = 0
        for j in range(m):
            if i == j:
                continue
            diff = (i - j) % m
            if diff not in F:
                mask |= (1 << j)
        compat_mask[i] = mask

    # 3. Local Search for Maximum Independent Set
    # State: set of vertices (bitmask)
    # Objective: maximize popcount(state)
    
    def popcount(x):
        return bin(x).count('1')
    
    # Greedy Initialization
    def greedy_init():
        remaining = (1 << m) - 1
        current_set = 0
        while remaining:
            # Pick vertex with highest degree in remaining subgraph? Or just first?
            # Better: Pick vertex that allows most future additions?
            # Simple greedy: Pick vertex with max compat count in remaining.
            best_v = -1
            max_deg = -1
            # Iterate through bits in remaining
            temp = remaining
            while temp:
                # Get lowest set bit
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                # Count compatible vertices in remaining
                # compat_mask[v] & remaining
                deg = popcount(compat_mask[v] & remaining)
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
                temp ^= lsb
            
            if best_v == -1:
                break
            
            current_set |= (1 << best_v)
            # Remove best_v and its conflicts from remaining
            # Conflicts of best_v are NOT in compat_mask[best_v].
            # So we keep only vertices in remaining that are compatible with best_v.
            # Actually, we remove best_v and all its INCOMPATIBLE neighbors.
            # Incompatible neighbors = remaining & ~compat_mask[best_v] & ~(1<<best_v)
            # Compatible neighbors = remaining & compat_mask[best_v]
            # We must remove best_v. We can keep compatible neighbors.
            # So new remaining = (remaining & compat_mask[best_v])
            # Wait, if u is compatible with v, u can stay. If u is incompatible, u must go.
            # So yes, remaining &= compat_mask[best_v]
            # But we also need to remove best_v itself. compat_mask[best_v] does not have bit best_v.
            # So remaining &= compat_mask[best_v] is correct.
            remaining &= compat_mask[best_v]
            
        return current_set

    def local_search(initial_set):
        current_set = initial_set
        current_size = popcount(current_set)
        best_set = current_set
        best_size = current_size
        
        # Precompute list of vertices for iteration
        vertices = list(range(m))
        
        # Local Search Loop
        while time.time() < deadline - 10:
            improved = False
            
            # Try adding a vertex
            # Candidates: vertices not in current_set
            # Check if they are compatible with ALL vertices in current_set
            # A vertex v is compatible with current_set if (compat_mask[v] & current_set) == current_set
            # i.e., current_set is a subset of compat_mask[v]
            
            # Optimization: Only try vertices that are "almost" compatible?
            # For simplicity, iterate all not in set.
            not_in_set = ((1 << m) - 1) ^ current_set
            temp = not_in_set
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                # Check compatibility
                if (compat_mask[v] & current_set) == current_set:
                    # Can add v
                    new_set = current_set | lsb
                    new_size = current_size + 1
                    if new_size > best_size:
                        best_size = new_size
                        best_set = new_set
                    current_set = new_set
                    current_size = new_size
                    improved = True
                    # Restart loop? Or continue? 
                    # Restarting is safer to re-evaluate.
                    # But for speed, we can just break and restart the while loop.
                    break
                temp ^= lsb
            
            if improved:
                continue
            
            # Try swapping: remove one vertex u, add multiple vertices
            # Or remove one, add one (if better, but size same, so only if it enables more later?)
            # For MIS, size is the only metric. So swap must increase size.
            # Remove u, add v (size same) is not useful unless it leads to more additions.
            # Let's try "Remove u, then Greedy Add"
            
            # Pick a vertex u to remove.
            # Heuristic: Remove vertex that has fewest conflicts with potential additions?
            # Or just random.
            
            # To avoid O(N^2) checks, let's limit the number of swap attempts per iteration.
            # If we haven't improved in a while, do a random restart?
            
            # Let's try a simple "kick": remove a random vertex, then greedy fill.
            if random.random() < 0.1: # 10% chance to kick
                if current_size > 0:
                    # Pick random vertex from current_set
                    temp = current_set
                    lsb = temp & -temp
                    # Random bit selection
                    # Convert to list? No, use random index.
                    # List of set bits
                    bits = []
                    t = current_set
                    while t:
                        b = t & -t
                        bits.append(b)
                        t ^= b
                    if bits:
                        remove_bit = bits[random.randint(0, len(bits)-1)]
                        current_set ^= remove_bit
                        current_size -= 1
                        
                        # Greedy Fill
                        remaining = ((1 << m) - 1) ^ current_set
                        # We want to add vertices compatible with current_set
                        # Greedy: pick best from remaining
                        # This is O(N^2) worst case, but N=205, so ~40k ops. Fast.
                        while remaining:
                            best_v = -1
                            max_deg = -1
                            t = remaining
                            while t:
                                lsb = t & -t
                                v = lsb.bit_length() - 1
                                # Check compat with current_set
                                if (compat_mask[v] & current_set) == current_set:
                                    deg = popcount(compat_mask[v] & remaining)
                                    if deg > max_deg:
                                        max_deg = deg
                                        best_v = v
                                t ^= lsb
                            
                            if best_v == -1:
                                break
                            
                            current_set |= (1 << best_v)
                            current_size += 1
                            remaining &= compat_mask[best_v]
                        
                        if current_size > best_size:
                            best_size = current_size
                            best_set = current_set
                        # If we didn't improve, we might revert? 
                        # Usually keep the new state to explore further.
                        # But if it's worse, we might get stuck.
                        # Let's just keep it.
            else:
                # If no improvement and no kick, we might be stuck.
                # Break to restart?
                # Or try a different swap?
                # For simplicity, break and restart with a new greedy init?
                # No, let's just break the inner loop and let the outer while handle it?
                # The outer while is `while time < deadline`.
                # If we don't improve and don't kick, we loop forever.
                # So we must either kick or break.
                # Let's force a kick if no improvement for a while?
                # Simpler: If no improvement, do a kick.
                if current_size > 0:
                    temp = current_set
                    bits = []
                    t = current_set
                    while t:
                        b = t & -t
                        bits.append(b)
                        t ^= b
                    if bits:
                        remove_bit = bits[random.randint(0, len(bits)-1)]
                        current_set ^= remove_bit
                        current_size -= 1
                        
                        # Greedy Fill
                        remaining = ((1 << m) - 1) ^ current_set
                        while remaining:
                            best_v = -1
                            max_deg = -1
                            t = remaining
                            while t:
                                lsb = t & -t
                                v = lsb.bit_length() - 1
                                if (compat_mask[v] & current_set) == current_set:
                                    deg = popcount(compat_mask[v] & remaining)
                                    if deg > max_deg:
                                        max_deg = deg
                                        best_v = v
                                t ^= lsb
                            
                            if best_v == -1:
                                break
                            
                            current_set |= (1 << best_v)
                            current_size += 1
                            remaining &= compat_mask[best_v]
                        
                        if current_size > best_size:
                            best_size = current_size
                            best_set = current_set

    # Main Loop: Multiple Restarts
    best_global_set = 0
    best_global_size = 0
    
    # Initial Greedy
    s = greedy_init()
    size = popcount(s)
    if size > best_global_size:
        best_global_size = size
        best_global_set = s
    
    # Run Local Search
    # We can run it for a fixed number of iterations or time.
    # Let's run it until deadline - 10s.
    
    # To ensure we explore different basins, we can do random restarts of the greedy init.
    # But local search from one good start is usually good enough.
    # Let's run local search on the initial greedy set.
    
    if time.time() < deadline - 20:
        best_set, best_size = local_search(s)
        if best_size > best_global_size:
            best_global_size = best_size
            best_global_set = best_set
    
    # Try a few more random restarts if time permits
    restarts = 0
    while time.time() < deadline - 20 and restarts < 5:
        # Randomized Greedy
        remaining = (1 << m) - 1
        current_set = 0
        # Shuffle vertices?
        # Just pick random vertex with max degree?
        # For simplicity, use standard greedy but with random tie-breaking.
        while remaining:
            best_v = -1
            max_deg = -1
            temp = remaining
            # Collect candidates with max degree
            # To save time, just pick first max
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                deg = popcount(compat_mask[v] & remaining)
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
                temp ^= lsb
            
            if best_v == -1:
                break
            
            current_set |= (1 << best_v)
            remaining &= compat_mask[best_v]
        
        size = popcount(current_set)
        if size > best_global_size:
            best_global_size = size
            best_global_set = current_set
            
        # Run local search on this new start
        if time.time() < deadline - 10:
            bs, bsz = local_search(current_set)
            if bsz > best_global_size:
                best_global_size = bsz
                best_global_set = bs
                
        restarts += 1
        
    # Convert bitmask to list
    result = []
    t = best_global_set
    while t:
        lsb = t & -t
        v = lsb.bit_length() - 1
        result.append(v)
        t ^= lsb
        
    # Verification
    res_set = set(result)
    for a in res_set:
        for b in res_set:
            if a == b:
                continue
            d = (a - b) % m
            if d in F:
                # Invalid! 
                # This should not happen. 
                # If it does, remove one.
                res_set.remove(b)
                break
        if b in res_set:
            pass # continue
    
    # Re-verify after removal if any
    final_set = set()
    for a in res_set:
        valid = True
        for b in final_set:
            d = (a - b) % m
            if d in F:
                valid = False
                break
        if valid:
            final_set.add(a)
            
    return np.array(sorted(final_set), dtype=np.int64)

if __name__ == "__main__":
    res = furstenberg_sarkozy_205()
    print(f"Size: {len(res)}")
    print(res)
