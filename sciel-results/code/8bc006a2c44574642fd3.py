import math
import time

def construct():
    start_time = time.time()
    TIME_LIMIT = 85.0  # Safety margin

    n = 256
    max_sum = 2 * n
    
    # 1. Identify squares
    squares = set()
    i = 1
    while i * i <= max_sum:
        squares.add(i * i)
        i += 1
        
    # 2. Filter candidates: a is valid if 2*a is not a square
    candidates = []
    for a in range(1, n + 1):
        if 2 * a not in squares:
            candidates.append(a)
            
    m = len(candidates)
    
    # Map original values to indices 0..m-1
    val_to_idx = {val: i for i, val in enumerate(candidates)}
    
    # 3. Build Conflict Graph G (edges if sum is square)
    # We will build the Complement Graph H (edges if sum is NOT square)
    # H[i] is a bitmask of vertices j such that candidates[i] + candidates[j] is NOT a square
    # Note: H should not have self-loops.
    
    H = [0] * m
    for i in range(m):
        vi = candidates[i]
        mask_i = 0
        for j in range(m):
            if i == j:
                continue
            vj = candidates[j]
            if (vi + vj) not in squares:
                mask_i |= (1 << j)
        H[i] = mask_i
        
    # 4. Local Search for Maximum Independent Set in G (Complement of H)
    # G[i] bitmask: neighbors in G (conflicts). H[i] is non-neighbors (compatible).
    # We want Max Independent Set in G.
    
    # Precompute G bitmasks from H
    # G[i] = AllBits ^ H[i] ^ (1 << i)  (Remove self, invert complement)
    all_bits = (1 << m) - 1
    G = [0] * m
    for i in range(m):
        # H[i] has 1s for compatible vertices (excluding i)
        # G[i] should have 1s for incompatible vertices (excluding i)
        # Incompatible = All - H[i] - (1<<i)
        G[i] = all_bits ^ H[i] ^ (1 << i)
        
    best_size = 0
    best_set_mask = 0
    
    # Helper to count bits
    def popcount(x):
        return bin(x).count('1')
        
    # Local Search Function
    def local_search(initial_mask):
        S = initial_mask
        if S == 0:
            return 0, 0
            
        while True:
            improved = False
            # 1. Try to Add any vertex that is compatible with all in S
            # Compatible with S means: (G[v] & S) == 0
            remaining = all_bits ^ S
            # Iterate through remaining vertices
            temp_rem = remaining
            while temp_rem:
                if time.time() - start_time > TIME_LIMIT:
                    return popcount(S), S
                v_bit = temp_rem & -temp_rem
                v_idx = v_bit.bit_length() - 1
                # Check if v is compatible with S
                if (G[v_idx] & S) == 0:
                    S |= v_bit
                    improved = True
                    # Restart scan? Or just continue? 
                    # Continuing is faster, but might miss interactions. 
                    # Since adding v only adds constraints, we can just continue scanning the rest.
                    # However, adding v might make previously "incompatible" vertices compatible? No, it makes more incompatible.
                    # So we just remove v from remaining and continue.
                    remaining &= ~v_bit
                    temp_rem &= ~v_bit
                    continue
                temp_rem &= temp_rem - 1
                
            if improved:
                continue
                
            # 2. If no Add, try Swap
            # Find a swap (u in S, v not in S) such that removing u and adding v 
            # allows adding at least one more vertex later.
            # Heuristic: Swap u->v if v is blocked ONLY by u, and removing u 
            # unblocks some other vertex w.
            
            best_swap_gain = 0
            best_swap_u = -1
            best_swap_v = -1
            
            # Iterate v not in S
            temp_v = remaining
            while temp_v:
                if time.time() - start_time > TIME_LIMIT:
                    return popcount(S), S
                v_bit = temp_v & -temp_v
                v_idx = v_bit.bit_length() - 1
                # Neighbors of v in S
                blockers = G[v_idx] & S
                # If v is blocked by exactly 1 vertex u
                if blockers and (blockers & (blockers - 1)) == 0:
                    u_bit = blockers
                    u_idx = u_bit.bit_length() - 1
                    
                    # If we remove u and add v, what is the new state?
                    # S_new = (S ^ u_bit) | v_bit
                    # How many new vertices can we add?
                    # A vertex w (not in S, not v) becomes addable if (G[w] & S_new) == 0
                    # (G[w] & S_new) = (G[w] & (S ^ u_bit) | v_bit)
                    # = ((G[w] & S) ^ (G[w] & u_bit)) | (G[w] & v_bit)
                    # Since w is not in S, and v is not in S, we don't care about w=v.
                    # We want to count w in `remaining ^ v_bit` such that:
                    # 1. G[w] & v_bit == 0 (w is compatible with v)
                    # 2. G[w] & S has u_bit as the ONLY blocker? No.
                    #    Condition: (G[w] & S_new) == 0
                    #    => (G[w] & S) & ~u_bit == 0 AND (G[w] & v_bit) == 0
                    #    => (G[w] & S) is either 0 or u_bit.
                    
                    # Let's just calculate the gain by simulation for the most promising candidates.
                    # To speed up, we can estimate.
                    # But m is small (~240). We can just try the swap if it looks good.
                    
                    # Let's count how many w in `remaining ^ v_bit` satisfy:
                    # (G[w] & v_bit) == 0 AND ((G[w] & S) == 0 OR (G[w] & S) == u_bit)
                    
                    gain = 0
                    temp_w = remaining ^ v_bit
                    # Optimization: Only check w that are compatible with v
                    # w_compat = ~G[v_idx] & remaining ^ v_bit  (vertices compatible with v)
                    # Then filter by S constraints.
                    
                    # Actually, let's just do the swap if gain > 0.
                    # We need to be careful not to do O(N^2) per swap candidate.
                    # There are O(N) candidates for v. For each, O(N) check. O(N^2) per local search step.
                    # N=240, N^2=57k. Local search steps maybe 100? 5.7M ops. Fast in Python? 
                    # Python is slow. 5.7M simple ops might take ~1s. 
                    # We have 60s. So we can afford a few hundred local search steps.
                    
                    # Let's optimize the gain calculation.
                    # Vertices w that become addable must be compatible with v.
                    # compat_with_v = all_bits ^ G[v_idx] ^ (1 << v_idx)
                    # Candidates for w: temp_w & compat_with_v
                    # For each such w, check if (G[w] & S) is 0 or u_bit.
                    
                    candidates_w = temp_w & (all_bits ^ G[v_idx] ^ (1 << v_idx))
                    cw = candidates_w
                    while cw:
                        w_bit = cw & -cw
                        w_idx = w_bit.bit_length() - 1
                        # Check condition
                        # G[w] & S should be 0 or u_bit
                        # Since w is in remaining, G[w] & S is not 0? 
                        # Wait, if G[w] & S == 0, w would have been added in the "Add" phase.
                        # So for w in `remaining`, G[w] & S is NOT 0.
                        # Therefore, the condition simplifies to: (G[w] & S) == u_bit
                        if (G[w_idx] & S) == u_bit:
                            gain += 1
                        cw &= cw - 1
                        
                    if gain > best_swap_gain:
                        best_swap_gain = gain
                        best_swap_u = u_idx
                        best_swap_v = v_idx
                        
                temp_v &= temp_v - 1
                
            if best_swap_gain > 0:
                # Perform the swap
                u_bit = 1 << best_swap_u
                v_bit = 1 << best_swap_v
                S = (S ^ u_bit) | v_bit
                improved = True
                continue
                
            # No improvement possible
            break
            
        return popcount(S), S

    # Run multiple restarts with different greedy initializations
    # Strategy: Randomize the order of vertices or use different degree heuristics
    
    import random
    random.seed(42)
    
    # Pre-compute degrees in G for greedy
    degrees = [popcount(G[i]) for i in range(m)]
    
    for restart in range(50):
        if time.time() - start_time > TIME_LIMIT:
            break
            
        # Greedy Initialization
        # Pick vertices in order of increasing degree in G (least restrictive first)
        # Or random order?
        # Let's use a mix.
        
        if restart % 2 == 0:
            # Greedy by degree (ascending)
            order = sorted(range(m), key=lambda x: degrees[x])
        else:
            # Random order
            order = list(range(m))
            random.shuffle(order)
            
        S = 0
        for v in order:
            if (G[v] & S) == 0:
                S |= (1 << v)
                
        # Local Search
        size, final_S = local_search(S)
        
        if size > best_size:
            best_size = size
            best_set_mask = final_S
            # Print progress? No, keep silent.
            
    # Convert best_set_mask to indices
    best_vertices = []
    temp = best_set_mask
    while temp:
        v_bit = temp & -temp
        v_idx = v_bit.bit_length() - 1
        best_vertices.append(v_idx)
        temp &= temp - 1
    
    # Convert indices back to values
    result = [candidates[i] for i in best_vertices]
    
    # Sort for consistent output
    result.sort()
    
    # Verification
    for i, a in enumerate(result):
        for b in result[i:]:
            s = a + b
            root = math.isqrt(s)
            if root * root == s:
                raise ValueError(f"Conflict: {a} + {b} = {s}")
                
    return result
